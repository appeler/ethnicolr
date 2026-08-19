#!/usr/bin/env python3
"""Fit post-hoc calibration and conformal statistics for a trained model.

For each model this script recreates the deterministic held-out split used in
training (same loaders and seed, without retraining), splits it into a
calibration half and an evaluation half, then:

1. fits a temperature T on the calibration half (Guo et al. 2017): the shipped
   probabilities become softmax(logits / T);
2. computes split-conformal APS quantiles for coverage levels 0.80/0.90/0.95;
3. verifies on the evaluation half: ECE before/after, Brier before/after, and
   empirical conformal coverage;
4. writes `<model>_stats_pt.json` next to the model weights and a reliability
   diagram PNG under docs/source/_static/.

Weighting ("re-inflate back to n"): models trained on deduplicated names would
otherwise be calibrated per *unique name*, not per *person*. Where the raw data
allows (NC), each held-out row is weighted by its pre-deduplication frequency,
so the guarantees speak about a randomly chosen person. The weighting used is
recorded in the JSON and surfaced in the model cards.

Usage:
    python calibrate_model.py wikipedia_surname
    python calibrate_model.py north_carolina_voter_full_name \
        --data-dir ../data-acquisition/raw
    python calibrate_model.py census --year 2010
"""

import argparse
import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from train_name_lstm import (  # noqa: E402
    MODEL_CONFIGS,
    NGRAMS,
    build_name_features,
    file_sha256,
    prepare_data_partitions,
)

from ethnicolr.model_metadata import load_class_labels, load_vocabulary  # noqa: E402
from ethnicolr.neural_name_model import NeuralNameModel  # noqa: E402
from ethnicolr.torch_utils import (  # noqa: E402
    load_character_ngram_model,
    pad_name_sequences,
)

COVERAGE_LEVELS = (0.80, 0.90, 0.95)
EXPECTED_CALIBRATION_ERROR_BIN_COUNT = 15

# 2020 Census population shares renormalized over the four census-model
# categories (white alone NH 57.8%, hispanic 18.7%, black alone NH 12.1%,
# asian+NHPI alone NH 6.4%). Used by the census models' prior="census" preset
# and recorded here so every stats file carries its own copy.
CENSUS_US_PRIOR = {"api": 0.067, "black": 0.127, "hispanic": 0.197, "white": 0.609}


@dataclass(frozen=True)
class CalibrationContext:
    """Model artifacts and held-out data needed for calibration."""

    model: torch.nn.Module
    categories: list[str]
    test_sequences: np.ndarray
    observed_categories: np.ndarray
    row_weights: np.ndarray
    reference_weighting: str
    training_distribution: dict[str, float]
    statistics_path: Path
    model_id: str
    diagram_name: str


def weighted_quantile(
    values: np.ndarray, weights: np.ndarray, quantile: float
) -> float:
    sorted_indices = np.argsort(values)
    sorted_values = values[sorted_indices]
    sorted_weights = weights[sorted_indices]
    cumulative_weight = np.cumsum(sorted_weights) / sorted_weights.sum()
    quantile_index = np.searchsorted(cumulative_weight, quantile, side="left").clip(
        0, len(sorted_values) - 1
    )
    return float(sorted_values[quantile_index])


def compute_category_logits(
    model: torch.nn.Module, input_sequences: np.ndarray
) -> np.ndarray:
    model.eval()
    category_logit_batches = []
    with torch.no_grad():
        for batch_start in range(0, len(input_sequences), 4096):
            input_batch = torch.from_numpy(
                input_sequences[batch_start : batch_start + 4096]
            )
            category_logit_batches.append(model(input_batch).numpy())
    return np.concatenate(category_logit_batches)


def fit_temperature_scaling(
    category_logits: np.ndarray,
    observed_categories: np.ndarray,
    row_weights: np.ndarray,
) -> float:
    logits_tensor = torch.from_numpy(category_logits)
    observed_category_tensor = torch.from_numpy(observed_categories)
    normalized_weight_tensor = torch.from_numpy(row_weights / row_weights.sum()).float()
    log_temperature = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=100)

    def closure():
        optimizer.zero_grad()
        loss = (
            F.cross_entropy(
                logits_tensor / log_temperature.exp(),
                observed_category_tensor,
                reduction="none",
            )
            * normalized_weight_tensor
        ).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.exp().item())


def expected_calibration_error(
    category_probabilities: np.ndarray,
    observed_categories: np.ndarray,
    row_weights: np.ndarray,
) -> tuple[float, list[list[float]]]:
    predicted_confidence = category_probabilities.max(axis=1)
    correct_prediction = (
        category_probabilities.argmax(axis=1) == observed_categories
    ).astype(float)
    bin_edges = np.linspace(0, 1, EXPECTED_CALIBRATION_ERROR_BIN_COUNT + 1)
    total_weight = row_weights.sum()
    calibration_error = 0.0
    reliability_bins = []
    for lower_edge, upper_edge in itertools.pairwise(bin_edges):
        rows_in_bin = (predicted_confidence > lower_edge) & (
            predicted_confidence <= upper_edge
        )
        bin_weight = row_weights[rows_in_bin].sum()
        if bin_weight == 0:
            continue
        mean_confidence = float(
            np.average(
                predicted_confidence[rows_in_bin], weights=row_weights[rows_in_bin]
            )
        )
        mean_accuracy = float(
            np.average(
                correct_prediction[rows_in_bin], weights=row_weights[rows_in_bin]
            )
        )
        relative_bin_weight = float(bin_weight / total_weight)
        calibration_error += relative_bin_weight * abs(mean_confidence - mean_accuracy)
        reliability_bins.append(
            [
                float(lower_edge),
                float(upper_edge),
                mean_confidence,
                mean_accuracy,
                relative_bin_weight,
            ]
        )
    return float(calibration_error), reliability_bins


def multiclass_brier_score(
    category_probabilities: np.ndarray,
    observed_categories: np.ndarray,
    row_weights: np.ndarray,
) -> float:
    observed_category_indicators = np.zeros_like(category_probabilities)
    observed_category_indicators[
        np.arange(len(observed_categories)), observed_categories
    ] = 1.0
    row_scores = ((category_probabilities - observed_category_indicators) ** 2).sum(
        axis=1
    )
    return float(np.average(row_scores, weights=row_weights))


def adaptive_prediction_scores(
    category_probabilities: np.ndarray, observed_categories: np.ndarray
) -> np.ndarray:
    """Adaptive-prediction-set score: cumulative mass down to the true class."""
    descending_category_indices = np.argsort(-category_probabilities, axis=1)
    sorted_probabilities = np.take_along_axis(
        category_probabilities, descending_category_indices, axis=1
    )
    cumulative_probabilities = np.cumsum(sorted_probabilities, axis=1)
    observed_category_ranks = np.argmax(
        descending_category_indices == observed_categories[:, None], axis=1
    )
    return cumulative_probabilities[
        np.arange(len(observed_categories)), observed_category_ranks
    ]


def adaptive_prediction_set_sizes(
    category_probabilities: np.ndarray, conformal_quantile: float
) -> np.ndarray:
    sorted_probabilities = -np.sort(-category_probabilities, axis=1)
    cumulative_probabilities = np.cumsum(sorted_probabilities, axis=1)
    return (cumulative_probabilities < conformal_quantile).sum(axis=1) + 1


def empirical_prediction_set_coverage(
    category_probabilities: np.ndarray,
    observed_categories: np.ndarray,
    row_weights: np.ndarray,
    conformal_quantile: float,
) -> float:
    covered = (
        adaptive_prediction_scores(category_probabilities, observed_categories)
        <= conformal_quantile
    )
    return float(np.average(covered, weights=row_weights))


def top_k_accuracy(
    category_probabilities: np.ndarray,
    observed_categories: np.ndarray,
    row_weights: np.ndarray,
    k_value: int,
) -> float:
    top_category_indices = np.argsort(-category_probabilities, axis=1)[:, :k_value]
    correct = (top_category_indices == observed_categories[:, None]).any(axis=1)
    return float(np.average(correct, weights=row_weights))


def north_carolina_person_weights(
    test_data: pd.DataFrame, data_directory: Path
) -> np.ndarray:
    """Re-inflate deduplicated NC names to person frequency."""
    source_data = pd.read_csv(
        data_directory / "nc_voter_name_race.csv.gz",
        dtype=str,
        keep_default_na=False,
    )
    source_data = source_data[
        source_data.race_code.isin(list("ABIMOW"))
        & source_data.ethnic_code.isin(["HL", "NL"])
    ]
    source_data["race"] = source_data["ethnic_code"] + "+" + source_data["race_code"]
    source_data["name_last"] = source_data["last_name"].str.strip().str.title()
    source_data["name_first"] = source_data["first_name"].str.strip().str.title()
    name_category_counts = source_data.groupby(
        ["name_last", "name_first", "race"]
    ).size()

    test_name_categories = pd.MultiIndex.from_arrays(
        [
            test_data["name_last"].str.strip().str.title(),
            test_data["name_first"].str.strip().str.title(),
            test_data["race"],
        ]
    )
    return (
        name_category_counts.reindex(test_name_categories)
        .fillna(1)
        .to_numpy(dtype=float)
    )


def load_trained_model_context(
    model_name: str, arguments: argparse.Namespace
) -> CalibrationContext:
    model_config = MODEL_CONFIGS[model_name]
    data_path = arguments.data_dir / model_config.data_file
    artifact_base_path = REPO_ROOT / "ethnicolr" / "models" / model_config.artifact_path
    training_manifest_path = Path(f"{artifact_base_path}_training_pt.json")
    requested_source_rows = 1_000_000
    if training_manifest_path.exists():
        training_manifest = json.loads(training_manifest_path.read_text())
        if arguments.seed != training_manifest["seed"]:
            raise ValueError(
                "calibration seed must match the training manifest: "
                f"{training_manifest['seed']}"
            )
        expected_data_hash = training_manifest["training_data"]["sha256"]
        actual_data_hash = file_sha256(data_path)
        if actual_data_hash != expected_data_hash:
            raise ValueError("training data does not match the training manifest")
        requested_source_rows = training_manifest["requested_source_rows"]
    training_data, _, test_data = prepare_data_partitions(
        model_config,
        data_path,
        seed=arguments.seed,
        source_row_limit=requested_source_rows,
    )

    vocabulary = load_vocabulary(Path(f"{artifact_base_path}_vocab_pt.json"))
    vocabulary_index = {
        token: token_index for token_index, token in enumerate(vocabulary)
    }
    categories = load_class_labels(Path(f"{artifact_base_path}_labels_pt.json"))

    features = build_name_features(test_data, model_config.name_scope)
    test_sequences = pad_name_sequences(
        [
            NeuralNameModel.encode_ngrams(vocabulary_index, name, NGRAMS)
            for name in features
        ],
        model_config.max_sequence_length,
    )
    category_index = {category: index for index, category in enumerate(categories)}
    observed_categories = test_data["race"].map(category_index).to_numpy()

    if model_name == "north_carolina_voter_full_name":
        row_weights = north_carolina_person_weights(test_data, arguments.data_dir)
        reference_weighting = (
            "person-reinflated (source-disjoint pre-dedup NC voter frequency)"
        )
    else:
        row_weights = np.ones(len(test_data))
        reference_weighting = {
            "wiki": "unique notable person (Wikipedia/Wikidata)",
            "origin": "unique notable person (Wikipedia/Wikidata)",
            "florida": "person (source-disjoint registered Florida voter sample)",
        }[model_config.source_loader]

    model = load_character_ngram_model(
        model_path=Path(f"{artifact_base_path}_lstm_pt.pt"),
        vocabulary_size=len(vocabulary),
        category_count=len(categories),
        device=torch.device("cpu"),
    )

    training_proportions = (
        training_data["race"]
        .value_counts(normalize=True)
        .reindex(categories)
        .to_numpy()
    )
    training_distribution = {
        category: float(proportion)
        for category, proportion in zip(categories, training_proportions, strict=True)
    }
    return CalibrationContext(
        model=model,
        categories=categories,
        test_sequences=test_sequences,
        observed_categories=observed_categories,
        row_weights=row_weights,
        reference_weighting=reference_weighting,
        training_distribution=training_distribution,
        statistics_path=Path(f"{artifact_base_path}_stats_pt.json"),
        model_id=model_name.replace("_", "-"),
        diagram_name=model_name,
    )


def load_census_model_context(
    year: int, arguments: argparse.Namespace
) -> CalibrationContext:
    census_training_directory = Path(__file__).parent / "census"
    sys.path.insert(0, str(census_training_directory))
    import train_census_lstm_pytorch as census

    census_data = census.load_census_data(year)
    evaluation_data = census.sample_and_assign_race(
        census_data, n_samples=200_000, seed=arguments.seed + 81
    )

    artifact_directory = REPO_ROOT / "ethnicolr" / "models" / "census" / "lstm"
    vocabulary = load_vocabulary(
        artifact_directory / f"census{year}_ln_vocab_pytorch.json"
    )
    vocabulary_index = {
        token: token_index for token_index, token in enumerate(vocabulary)
    }
    categories = load_class_labels(
        artifact_directory / f"census{year}_labels_pytorch.json"
    )

    test_sequences = pad_name_sequences(
        [
            NeuralNameModel.encode_ngrams(vocabulary_index, name, NGRAMS)
            for name in evaluation_data["name_title"]
        ],
        20,
    )
    category_index = {category: index for index, category in enumerate(categories)}
    observed_categories = evaluation_data["race"].map(category_index).to_numpy()

    model = load_character_ngram_model(
        model_path=artifact_directory / f"census{year}_ln_lstm_pytorch.pt",
        vocabulary_size=len(vocabulary),
        category_count=len(categories),
        device=torch.device("cpu"),
    )
    training_distribution = (
        CENSUS_US_PRIOR
        if categories == sorted(CENSUS_US_PRIOR)
        else {category: 1 / len(categories) for category in categories}
    )
    return CalibrationContext(
        model=model,
        categories=categories,
        test_sequences=test_sequences,
        observed_categories=observed_categories,
        row_weights=np.ones(len(observed_categories)),
        reference_weighting=(
            "person (census count-weighted sample, labels drawn from "
            "census race shares)"
        ),
        training_distribution=training_distribution,
        statistics_path=(artifact_directory / f"census{year}_ln_stats_pytorch.json"),
        model_id=f"census-surname-{year}",
        diagram_name=f"census_{year}",
    )


def write_reliability_diagram(
    uncalibrated_bins: list[list[float]],
    calibrated_bins: list[list[float]],
    model_name: str,
    output_directory: Path,
) -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for axis, reliability_bins, title in [
        (axes[0], uncalibrated_bins, "Before calibration"),
        (axes[1], calibrated_bins, "After calibration"),
    ]:
        mean_confidence = [reliability_bin[2] for reliability_bin in reliability_bins]
        mean_accuracy = [reliability_bin[3] for reliability_bin in reliability_bins]
        axis.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
        axis.plot(mean_confidence, mean_accuracy, "o-")
        axis.set_xlabel("Predicted confidence")
        axis.set_title(title)
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
    axes[0].set_ylabel("Empirical accuracy")
    figure.suptitle(f"Reliability: {model_name}")
    figure.tight_layout()
    output_directory.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_directory / f"reliability_{model_name}.png", dpi=120)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=[*sorted(MODEL_CONFIGS), "census"])
    parser.add_argument("--year", type=int, default=None, choices=[2000, 2010, 2020])
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data-acquisition" / "raw",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if args.model == "census":
        if args.year is None:
            parser.error("--year required for census")
        calibration_context = load_census_model_context(args.year, args)
    else:
        calibration_context = load_trained_model_context(args.model, args)

    category_logits = compute_category_logits(
        calibration_context.model, calibration_context.test_sequences
    )
    observed_categories = calibration_context.observed_categories
    row_weights = calibration_context.row_weights.astype(float)

    random_number_generator = np.random.default_rng(args.seed)
    shuffled_indices = random_number_generator.permutation(len(observed_categories))
    split_position = len(observed_categories) // 2
    calibration_rows = shuffled_indices[:split_position]
    evaluation_rows = shuffled_indices[split_position:]

    temperature = fit_temperature_scaling(
        category_logits[calibration_rows],
        observed_categories[calibration_rows],
        row_weights[calibration_rows],
    )
    uncalibrated_probabilities = torch.softmax(
        torch.from_numpy(category_logits), dim=1
    ).numpy()
    calibrated_probabilities = torch.softmax(
        torch.from_numpy(category_logits) / temperature, dim=1
    ).numpy()

    calibration_scores = adaptive_prediction_scores(
        calibrated_probabilities[calibration_rows],
        observed_categories[calibration_rows],
    )
    calibration_population_size = row_weights[calibration_rows].sum()
    conformal_quantiles = {
        f"{coverage_level:.2f}": weighted_quantile(
            calibration_scores,
            row_weights[calibration_rows],
            min(
                1.0,
                np.ceil((calibration_population_size + 1) * coverage_level)
                / calibration_population_size,
            ),
        )
        for coverage_level in COVERAGE_LEVELS
    }

    uncalibrated_error, uncalibrated_bins = expected_calibration_error(
        uncalibrated_probabilities[evaluation_rows],
        observed_categories[evaluation_rows],
        row_weights[evaluation_rows],
    )
    calibrated_error, calibrated_bins = expected_calibration_error(
        calibrated_probabilities[evaluation_rows],
        observed_categories[evaluation_rows],
        row_weights[evaluation_rows],
    )
    empirical_coverages = {
        coverage_level: empirical_prediction_set_coverage(
            calibrated_probabilities[evaluation_rows],
            observed_categories[evaluation_rows],
            row_weights[evaluation_rows],
            conformal_quantile,
        )
        for coverage_level, conformal_quantile in conformal_quantiles.items()
    }
    mean_prediction_set_sizes = {
        coverage_level: float(
            np.average(
                adaptive_prediction_set_sizes(
                    calibrated_probabilities[evaluation_rows], conformal_quantile
                ),
                weights=row_weights[evaluation_rows],
            )
        )
        for coverage_level, conformal_quantile in conformal_quantiles.items()
    }

    model_statistics = {
        "model": calibration_context.model_id,
        "calibration_status": "validated-source-disjoint",
        "temperature": temperature,
        "train_class_distribution": calibration_context.training_distribution,
        "classes": calibration_context.categories,
        "calibration_weighting": calibration_context.reference_weighting,
        "conformal_quantiles": conformal_quantiles,
        "metrics": {
            "n_calibration": len(calibration_rows),
            "n_evaluation": len(evaluation_rows),
            "accuracy": top_k_accuracy(
                calibrated_probabilities[evaluation_rows],
                observed_categories[evaluation_rows],
                row_weights[evaluation_rows],
                1,
            ),
            "top2": top_k_accuracy(
                calibrated_probabilities[evaluation_rows],
                observed_categories[evaluation_rows],
                row_weights[evaluation_rows],
                2,
            ),
            "top3": top_k_accuracy(
                calibrated_probabilities[evaluation_rows],
                observed_categories[evaluation_rows],
                row_weights[evaluation_rows],
                3,
            ),
            "ece_pre": uncalibrated_error,
            "ece_post": calibrated_error,
            "brier_pre": multiclass_brier_score(
                uncalibrated_probabilities[evaluation_rows],
                observed_categories[evaluation_rows],
                row_weights[evaluation_rows],
            ),
            "brier_post": multiclass_brier_score(
                calibrated_probabilities[evaluation_rows],
                observed_categories[evaluation_rows],
                row_weights[evaluation_rows],
            ),
            "conformal_empirical_coverage": empirical_coverages,
            "conformal_mean_set_size": mean_prediction_set_sizes,
        },
        "reliability_bins_pre": uncalibrated_bins,
        "reliability_bins_post": calibrated_bins,
    }

    calibration_context.statistics_path.write_text(
        json.dumps(model_statistics, indent=2) + "\n"
    )
    write_reliability_diagram(
        uncalibrated_bins,
        calibrated_bins,
        calibration_context.diagram_name,
        REPO_ROOT / "docs" / "source" / "_static",
    )

    metrics = model_statistics["metrics"]
    print(
        f"model: {model_statistics['model']}  "
        f"weighting: {calibration_context.reference_weighting}"
    )
    print(f"temperature: {temperature:.3f}")
    print(
        f"acc: {metrics['accuracy']:.4f}  top2: {metrics['top2']:.4f}  "
        f"top3: {metrics['top3']:.4f}\n"
        f"ECE: {metrics['ece_pre']:.4f} -> {metrics['ece_post']:.4f}   "
        f"Brier: {metrics['brier_pre']:.4f} -> {metrics['brier_post']:.4f}"
    )
    for coverage_level, conformal_quantile in conformal_quantiles.items():
        print(
            f"coverage@{coverage_level}: nominal={coverage_level}  "
            f"empirical="
            f"{metrics['conformal_empirical_coverage'][coverage_level]:.4f}  "
            f"mean set size="
            f"{metrics['conformal_mean_set_size'][coverage_level]:.2f}  "
            f"(quantile={conformal_quantile:.4f})"
        )
    print(f"wrote {calibration_context.statistics_path}")


if __name__ == "__main__":
    main()
