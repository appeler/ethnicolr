#!/usr/bin/env python3
"""Train the bundled PyTorch name models.

Replaces the legacy Keras training notebooks with one parameterized trainer.
Census models have their own script (census/train_census_lstm_pytorch.py) because
their labels are sampled probabilistically from aggregate counts.

Usage:
    python train_name_lstm.py wikipedia_surname
    python train_name_lstm.py florida_voter_full_name --data-dir ../data-acquisition/raw
    python train_name_lstm.py north_carolina_voter_full_name --device mps

Data expectations (see scripts/data-acquisition/):
    wiki:  <data-dir>/wiki_name_race_2026.csv.gz (built by wiki/prepare_wiki_data.py;
           merges the 2009-era in-repo CSV with fresh Wikidata people)
    fl:    <data-dir>/fl_reg_name_race_2022.csv.gz
    nc:    <data-dir>/nc_voter_name_race.csv.gz
"""

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ethnicolr.model_metadata import (  # noqa: E402
    write_class_labels,
    write_vocabulary,
)
from ethnicolr.neural_name_model import NeuralNameModel  # noqa: E402
from ethnicolr.torch_utils import (  # noqa: E402
    CharacterNgramLSTM,
    pad_name_sequences,
    select_inference_device,
)

NGRAMS = 2

FLORIDA_CATEGORY_MAP = {
    "native_indian": "other",
    "asian": "asian",
    "nh_black": "nh_black",
    "hispanic": "hispanic",
    "nh_white": "nh_white",
    "other": "other",
    "multi_racial": "other",
    "unknown": "unknown",
}
NC_RACE_CODES = ["A", "B", "I", "M", "O", "W"]


@dataclass
class ModelConfig:
    source_loader: str
    name_scope: str
    max_sequence_length: int
    epochs: int
    artifact_path: str
    data_file: str
    expected_category_count: int


MODEL_CONFIGS = {
    "wikipedia_surname": ModelConfig(
        "wiki",
        "surname",
        20,
        20,
        "wiki/lstm/wiki_ln",
        "wiki_name_race_2026.csv.gz",
        13,
    ),
    "wikipedia_full_name": ModelConfig(
        "wiki",
        "full-name",
        25,
        20,
        "wiki/lstm/wiki_name",
        "wiki_name_race_2026.csv.gz",
        13,
    ),
    "florida_voter_surname": ModelConfig(
        "florida",
        "surname",
        20,
        20,
        "fl_voter_reg/lstm/fl_ln_five_cat_2022",
        "fl_reg_name_race_2022.csv.gz",
        5,
    ),
    "florida_voter_full_name": ModelConfig(
        "florida",
        "full-name",
        25,
        20,
        "fl_voter_reg/lstm/fl_name_five_cat_2022",
        "fl_reg_name_race_2022.csv.gz",
        5,
    ),
    "north_carolina_voter_full_name": ModelConfig(
        "nc",
        "full-name",
        25,
        15,
        "nc_voter_reg/lstm/nc_name",
        "nc_voter_name_race.csv.gz",
        12,
    ),
    "wikipedia_origin": ModelConfig(
        "origin",
        "full-name",
        25,
        8,
        "wiki/lstm/wiki_origin",
        "wiki_origin.csv.gz",
        0,
    ),
}


def load_wiki(
    path: Path,
    random_number_generator: np.random.Generator,
    max_rows_per_category: int = 300_000,
) -> pd.DataFrame:
    data = pd.read_csv(path)
    data["__source_row"] = data.index
    data = data.dropna(subset=["name_first", "name_last"])
    sampling_seed = int(random_number_generator.integers(2**31))
    data = data.groupby("race", group_keys=False)[data.columns].apply(
        lambda category_data: category_data.sample(
            min(len(category_data), max_rows_per_category),
            random_state=sampling_seed,
        )
    )
    return data[["__source_row", "name_last", "name_first", "race"]]


def load_florida_voters(
    path: Path,
    random_number_generator: np.random.Generator,
    source_row_limit: int = 1_000_000,
) -> pd.DataFrame:
    data = pd.read_csv(path)
    data["__source_row"] = data.index
    data = data.dropna(subset=["name_first", "name_last"])
    data["race"] = data.race.map(FLORIDA_CATEGORY_MAP)
    data = data[data.race != "unknown"]
    if len(data) > source_row_limit:
        data = data.sample(
            source_row_limit,
            random_state=int(random_number_generator.integers(2**31)),
        )
    return data[["__source_row", "name_last", "name_first", "race"]]


def load_nc(
    path: Path,
    random_number_generator: np.random.Generator,
    source_row_limit: int = 1_000_000,
) -> pd.DataFrame:
    data = pd.read_csv(path)
    data["__source_row"] = data.index
    data = data.dropna(subset=["first_name", "last_name"])
    data = data[
        data.race_code.isin(NC_RACE_CODES) & data.ethnic_code.isin(["HL", "NL"])
    ]
    data["race"] = data["ethnic_code"] + "+" + data["race_code"]
    data = data.drop_duplicates(subset=["last_name", "first_name", "race"])

    if len(data) > source_row_limit:
        data = data.sample(
            source_row_limit,
            random_state=int(random_number_generator.integers(2**31)),
        )

    data = data.rename(columns={"last_name": "name_last", "first_name": "name_first"})
    return data[["__source_row", "name_last", "name_first", "race"]]


def load_origin(
    path: Path,
    random_number_generator: np.random.Generator,
    source_row_limit: int = 1_000_000,
) -> pd.DataFrame:
    """Load origin data with a tighter per-category limit."""
    del source_row_limit
    return load_wiki(path, random_number_generator, max_rows_per_category=50_000)


LOADERS = {
    "wiki": load_wiki,
    "origin": load_origin,
    "florida": load_florida_voters,
    "nc": load_nc,
}


def prepare_data_partitions(
    model_config: ModelConfig,
    data_path: Path,
    seed: int,
    source_row_limit: int = 1_000_000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create source-disjoint train, validation, and test partitions."""
    random_number_generator = np.random.default_rng(seed)
    source_loader = LOADERS[model_config.source_loader]
    source_data = (
        source_loader(data_path, random_number_generator)
        if model_config.source_loader in {"wiki", "origin"}
        else source_loader(
            data_path,
            random_number_generator,
            source_row_limit=source_row_limit,
        )
    )
    development, test = train_test_split(
        source_data,
        test_size=0.2,
        random_state=seed,
        stratify=source_data["race"],
    )
    train, validation = train_test_split(
        development,
        test_size=0.125,
        random_state=seed + 1,
        stratify=development["race"],
    )

    if model_config.source_loader in {"florida", "nc"}:
        target_rows = len(train)
        per_class = target_rows // train["race"].nunique()
        balance_seed = int(random_number_generator.integers(2**31))
        train = pd.concat(
            group.sample(
                per_class,
                replace=len(group) < per_class,
                random_state=balance_seed,
            )
            for _, group in train.groupby("race")
        )

    source_sets = {
        "training": set(train["__source_row"]),
        "validation": set(validation["__source_row"]),
        "test": set(test["__source_row"]),
    }
    partition_pairs = (
        ("training", "validation"),
        ("training", "test"),
        ("validation", "test"),
    )
    for left_name, right_name in partition_pairs:
        overlap = source_sets[left_name] & source_sets[right_name]
        if overlap:
            raise RuntimeError(
                f"{left_name}/{right_name} source overlap detected: "
                f"{len(overlap)} source rows"
            )
    return (
        train.reset_index(drop=True),
        validation.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def build_name_features(data: pd.DataFrame, name_scope: str) -> pd.Series:
    """Build normalized surname or full-name model inputs."""
    surnames = data["name_last"].astype(str).str.strip().str.title()
    if name_scope == "surname":
        return surnames
    if name_scope != "full-name":
        raise ValueError(f"Unsupported name scope: {name_scope!r}")
    first_names = data["name_first"].astype(str).str.strip().str.title()
    return (surnames + " " + first_names).str.strip()


def build_vocabulary(names: pd.Series) -> list[str]:
    """Build a frequency-ordered character n-gram vocabulary."""
    vectorizer = CountVectorizer(
        analyzer="char",
        max_df=0.3,
        min_df=3,
        ngram_range=(NGRAMS, NGRAMS),
        lowercase=False,
    )
    token_counts = np.asarray(vectorizer.fit_transform(names).sum(axis=0)).flatten()
    sorted_items = sorted(
        (
            (token_counts[token_index], token)
            for token, token_index in vectorizer.vocabulary_.items()
        ),
        reverse=True,
    )
    return ["UNK"] + [token for _, token in sorted_items]


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a training-data file."""
    digest = hashlib.sha256()
    with path.open("rb") as data_file:
        for chunk in iter(lambda: data_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def top_k_accuracy(model, data_loader, device, requested_k_values=(2, 3)):
    """Fraction of samples whose true label is within the top-k predictions."""
    model.eval()
    correct_counts = dict.fromkeys(requested_k_values, 0)
    row_count = 0
    with torch.no_grad():
        for input_batch, category_batch in data_loader:
            input_batch = input_batch.to(device)
            category_batch = category_batch.to(device)
            top_category_indices = (
                model(input_batch).topk(max(requested_k_values), dim=1).indices
            )
            for k_value in requested_k_values:
                correct_counts[k_value] += (
                    (top_category_indices[:, :k_value] == category_batch.unsqueeze(1))
                    .any(dim=1)
                    .sum()
                    .item()
                )
            row_count += category_batch.size(0)
    return {
        k_value: correct_counts[k_value] / row_count for k_value in requested_k_values
    }


def run_model_epoch(model, data_loader, loss_function, device, optimizer=None):
    """Train or evaluate one model epoch."""
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss = 0.0
    correct_count = 0
    row_count = 0
    predicted_categories = []
    observed_categories = []
    with torch.set_grad_enabled(training):
        for input_batch, category_batch in data_loader:
            input_batch = input_batch.to(device)
            category_batch = category_batch.to(device)
            category_logits = model(input_batch)
            loss = loss_function(category_logits, category_batch)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * input_batch.size(0)
            predicted_batch = category_logits.argmax(1)
            correct_count += predicted_batch.eq(category_batch).sum().item()
            row_count += category_batch.size(0)
            if not training:
                predicted_categories.extend(predicted_batch.cpu().numpy())
                observed_categories.extend(category_batch.cpu().numpy())
    return (
        total_loss / row_count,
        correct_count / row_count,
        np.array(predicted_categories),
        np.array(observed_categories),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=sorted(MODEL_CONFIGS))
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data-acquisition" / "raw",
    )
    parser.add_argument("--samples", type=int, default=1_000_000)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "ethnicolr" / "models",
    )
    args = parser.parse_args()

    model_config = MODEL_CONFIGS[args.model]
    device = (
        select_inference_device()
        if args.device == "auto"
        else torch.device(args.device)
    )
    epochs = args.epochs or model_config.epochs
    print(f"Training {args.model} on {device} for {epochs} epochs")

    torch.manual_seed(args.seed)
    data_path = args.data_dir / model_config.data_file
    if not data_path.exists():
        raise FileNotFoundError(
            f"Training data not found: {data_path}\n"
            "Run the scripts in scripts/data-acquisition/ first."
        )

    training_data, validation_data, test_data = prepare_data_partitions(
        model_config,
        data_path,
        seed=args.seed,
        source_row_limit=args.samples,
    )
    print(
        f"Train: {len(training_data):,}, Validation: {len(validation_data):,}, "
        f"Test: {len(test_data):,}"
    )
    print(training_data["race"].value_counts())

    categories = sorted(pd.concat([training_data["race"], test_data["race"]]).unique())
    if (
        model_config.expected_category_count
        and len(categories) != model_config.expected_category_count
    ):
        raise ValueError(
            f"Expected {model_config.expected_category_count} categories, "
            f"got {len(categories)}: {categories}"
        )

    training_names = build_name_features(training_data, model_config.name_scope)
    validation_names = build_name_features(validation_data, model_config.name_scope)
    test_names = build_name_features(test_data, model_config.name_scope)
    vocabulary = build_vocabulary(training_names)
    vocabulary_index = {
        token: token_index for token_index, token in enumerate(vocabulary)
    }
    print(f"Vocabulary size: {len(vocabulary)}")

    training_sequences = pad_name_sequences(
        [
            NeuralNameModel.encode_ngrams(vocabulary_index, name, NGRAMS)
            for name in training_names
        ],
        model_config.max_sequence_length,
    )
    test_sequences = pad_name_sequences(
        [
            NeuralNameModel.encode_ngrams(vocabulary_index, name, NGRAMS)
            for name in test_names
        ],
        model_config.max_sequence_length,
    )
    validation_sequences = pad_name_sequences(
        [
            NeuralNameModel.encode_ngrams(vocabulary_index, name, NGRAMS)
            for name in validation_names
        ],
        model_config.max_sequence_length,
    )
    category_index = {category: index for index, category in enumerate(categories)}
    training_categories = (
        training_data["race"].map(category_index).to_numpy(dtype=np.int64, copy=True)
    )
    validation_categories = (
        validation_data["race"].map(category_index).to_numpy(dtype=np.int64, copy=True)
    )
    test_categories = (
        test_data["race"].map(category_index).to_numpy(dtype=np.int64, copy=True)
    )

    training_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(training_sequences),
            torch.from_numpy(training_categories),
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(test_sequences), torch.from_numpy(test_categories)
        ),
        batch_size=args.batch_size * 4,
    )
    validation_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(validation_sequences),
            torch.from_numpy(validation_categories),
        ),
        batch_size=args.batch_size * 4,
    )

    model = CharacterNgramLSTM(
        vocabulary_size=len(vocabulary), category_count=len(categories)
    ).to(device)
    print(f"Parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_epoch = 0
    best_validation_loss = float("inf")
    best_validation_accuracy = 0.0
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, epochs + 1):
        training_loss, training_accuracy, _, _ = run_model_epoch(
            model, training_loader, loss_function, device, optimizer
        )
        validation_loss, validation_accuracy, _, _ = run_model_epoch(
            model, validation_loader, loss_function, device
        )
        if validation_loss < best_validation_loss:
            best_epoch = epoch
            best_validation_loss = validation_loss
            best_validation_accuracy = validation_accuracy
            best_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.state_dict().items()
            }
        print(
            f"Epoch {epoch:2d}/{epochs}: train_loss={training_loss:.4f} "
            f"train_acc={training_accuracy:.4f} val_loss={validation_loss:.4f} "
            f"val_acc={validation_accuracy:.4f}"
        )

    if best_state is None:
        raise RuntimeError("training did not produce a model checkpoint")
    cpu_model = CharacterNgramLSTM(
        vocabulary_size=len(vocabulary), category_count=len(categories)
    )
    cpu_model.load_state_dict(best_state)
    test_loss, test_accuracy, predicted_categories, observed_categories = (
        run_model_epoch(cpu_model, test_loader, loss_function, torch.device("cpu"))
    )
    print(f"\nTest accuracy (CPU eval): {test_accuracy:.4f}")
    top_k_results = top_k_accuracy(cpu_model, test_loader, torch.device("cpu"))
    print(
        " ".join(
            f"top-{k_value} accuracy: {accuracy:.4f}"
            for k_value, accuracy in sorted(top_k_results.items())
        )
    )
    print(
        classification_report(
            observed_categories, predicted_categories, target_names=categories
        )
    )
    print(confusion_matrix(observed_categories, predicted_categories))

    artifact_base_path = args.output_root / model_config.artifact_path
    artifact_base_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cpu_model.state_dict(), f"{artifact_base_path}_lstm_pt.pt")
    write_vocabulary(Path(f"{artifact_base_path}_vocab_pt.json"), vocabulary)
    write_class_labels(Path(f"{artifact_base_path}_labels_pt.json"), categories)
    training_manifest = {
        "schema_version": 1,
        "model": args.model,
        "seed": args.seed,
        "epochs": epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "requested_source_rows": args.samples,
        "training_device": str(device),
        "training_data": {
            "file": data_path.name,
            "sha256": file_sha256(data_path),
        },
        "split": {
            "training_rows": len(training_data),
            "training_source_rows": int(training_data["__source_row"].nunique()),
            "validation_rows": len(validation_data),
            "validation_source_rows": int(validation_data["__source_row"].nunique()),
            "test_rows": len(test_data),
            "test_source_rows": int(test_data["__source_row"].nunique()),
            "source_overlap_rows": 0,
            "split_before_balancing": True,
            "vocabulary_from_training_only": True,
            "training_balanced": model_config.source_loader in {"florida", "nc"},
        },
        "features": {
            "input_scope": model_config.name_scope,
            "ngram_size": NGRAMS,
            "sequence_length": model_config.max_sequence_length,
            "vocabulary_size": len(vocabulary),
            "classes": categories,
        },
        "evaluation": {
            "selected_epoch": best_epoch,
            "validation_loss": best_validation_loss,
            "validation_accuracy": best_validation_accuracy,
            "loss": test_loss,
            "accuracy": test_accuracy,
            "top_2_accuracy": top_k_results[2],
            "top_3_accuracy": top_k_results[3],
        },
    }
    Path(f"{artifact_base_path}_training_pt.json").write_text(
        json.dumps(training_manifest, indent=2) + "\n"
    )
    print(f"Saved model bundle at {artifact_base_path}_*")


if __name__ == "__main__":
    main()
