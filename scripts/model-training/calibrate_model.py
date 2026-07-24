#!/usr/bin/env python3
"""Fit post-hoc calibration and conformal statistics for a trained model.

For each model this script recreates the deterministic held-out split used in
training (same loaders, same seed — no retraining), splits it into a
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
    python calibrate_model.py wiki_ln
    python calibrate_model.py nc_name --data-dir ../data-acquisition/raw
    python calibrate_model.py census --year 2010
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from train_name_lstm import (  # noqa: E402
    LOADERS,
    MODEL_CONFIGS,
    NGRAMS,
    build_features,
)

from ethnicolr.ethnicolr_class import EthnicolrModelClass  # noqa: E402
from ethnicolr.torch_utils import NameLSTM, pad_sequences  # noqa: E402

COVERAGE_LEVELS = (0.80, 0.90, 0.95)
ECE_BINS = 15

# 2020 Census population shares renormalized over the four census-model
# categories (white alone NH 57.8%, hispanic 18.7%, black alone NH 12.1%,
# asian+NHPI alone NH 6.4%). Used by the census models' prior="census" preset
# and recorded here so every stats file carries its own copy.
CENSUS_US_PRIOR = {"api": 0.067, "black": 0.127, "hispanic": 0.197, "white": 0.609}


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cum = np.cumsum(weights) / weights.sum()
    return float(values[np.searchsorted(cum, q, side="left").clip(0, len(values) - 1)])


def compute_logits(model: torch.nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 4096):
            out.append(model(torch.from_numpy(X[i : i + 4096])).numpy())
    return np.concatenate(out)


def fit_temperature(logits: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    logits_t = torch.from_numpy(logits)
    y_t = torch.from_numpy(y)
    w_t = torch.from_numpy(w / w.sum()).float()
    log_temp = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temp], lr=0.1, max_iter=100)

    def closure():
        optimizer.zero_grad()
        loss = (
            F.cross_entropy(logits_t / log_temp.exp(), y_t, reduction="none") * w_t
        ).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temp.exp().item())


def ece_and_bins(
    probs: np.ndarray, y: np.ndarray, w: np.ndarray
) -> tuple[float, list[list[float]]]:
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0, 1, ECE_BINS + 1)
    total_w = w.sum()
    ece, bins = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        mask = (conf > lo) & (conf <= hi)
        bw = w[mask].sum()
        if bw == 0:
            continue
        avg_conf = float(np.average(conf[mask], weights=w[mask]))
        avg_acc = float(np.average(correct[mask], weights=w[mask]))
        ece += (bw / total_w) * abs(avg_conf - avg_acc)
        bins.append([float(lo), float(hi), avg_conf, avg_acc, float(bw / total_w)])
    return float(ece), bins


def brier(probs: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y)), y] = 1.0
    per_row = ((probs - onehot) ** 2).sum(axis=1)
    return float(np.average(per_row, weights=w))


def aps_scores(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Adaptive-prediction-set score: cumulative mass down to the true class."""
    order = np.argsort(-probs, axis=1)
    sorted_probs = np.take_along_axis(probs, order, axis=1)
    cum = np.cumsum(sorted_probs, axis=1)
    true_rank = np.argmax(order == y[:, None], axis=1)
    return cum[np.arange(len(y)), true_rank]


def aps_set_sizes(probs: np.ndarray, qhat: float) -> np.ndarray:
    sorted_probs = -np.sort(-probs, axis=1)
    cum = np.cumsum(sorted_probs, axis=1)
    return (cum < qhat).sum(axis=1) + 1


def empirical_coverage(
    probs: np.ndarray, y: np.ndarray, w: np.ndarray, qhat: float
) -> float:
    return float(np.average(aps_scores(probs, y) <= qhat, weights=w))


def topk_acc(probs: np.ndarray, y: np.ndarray, w: np.ndarray, k: int) -> float:
    topk = np.argsort(-probs, axis=1)[:, :k]
    return float(np.average((topk == y[:, None]).any(axis=1), weights=w))


def nc_weights(df: pd.DataFrame, data_dir: Path) -> np.ndarray:
    """Re-inflate deduplicated NC names to person frequency."""
    raw = pd.read_csv(
        data_dir / "nc_voter_name_race.csv.gz", dtype=str, keep_default_na=False
    )
    raw = raw[raw.race_code.isin(list("ABIMOW")) & raw.ethnic_code.isin(["HL", "NL"])]
    raw["race"] = raw["ethnic_code"] + "+" + raw["race_code"]
    raw["name_last"] = raw["last_name"].str.strip().str.title()
    raw["name_first"] = raw["first_name"].str.strip().str.title()
    counts = raw.groupby(["name_last", "name_first", "race"]).size()

    keys = pd.MultiIndex.from_arrays(
        [
            df["name_last"].str.strip().str.title(),
            df["name_first"].str.strip().str.title(),
            df["race"],
        ]
    )
    return counts.reindex(keys).fillna(1).to_numpy(dtype=float)


def load_trainer_model(key: str, args) -> dict:
    cfg = MODEL_CONFIGS[key]
    rng = np.random.default_rng(args.seed)
    data_path = args.data_dir / cfg.data_file
    loader = LOADERS[cfg.loader]
    df = (
        loader(data_path, rng)
        if cfg.loader == "wiki"
        else loader(data_path, rng, n_samples=1_000_000)
    )
    df = df.reset_index(drop=True)

    base = REPO_ROOT / "ethnicolr" / "models" / cfg.basename
    vocab = pd.read_csv(f"{base}_vocab_pt.csv").vocab.tolist()
    vocab_dict = {word: idx for idx, word in enumerate(vocab)}
    races = pd.read_csv(f"{base}_race_pt.csv").race.tolist()

    features = build_features(df, cfg.feature)
    X = pad_sequences(
        [
            EthnicolrModelClass.find_ngrams(vocab_dict, name, NGRAMS)
            for name in features
        ],
        cfg.feature_len,
    )
    y = df["race"].map({r: i for i, r in enumerate(races)}).to_numpy()

    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        idx, test_size=0.2, random_state=args.seed, stratify=y
    )

    if key == "nc_name":
        weights = nc_weights(df.iloc[test_idx], args.data_dir)
        weighting = "person-reinflated (pre-dedup NC voter frequency)"
    else:
        weights = np.ones(len(test_idx))
        weighting = {
            "wiki": "unique notable person (Wikipedia/Wikidata)",
            "fl_4cat": "person (registered FL voter sample)",
            "fl_5cat": "person, class-balanced (200k/class FL voter sample)",
        }[cfg.loader]

    model = NameLSTM(vocab_size=len(vocab), num_classes=len(races))
    model.load_state_dict(
        torch.load(f"{base}_lstm_pt.pt", map_location="cpu", weights_only=True)
    )

    train_dist = (
        pd.Series(y[train_idx]).value_counts(normalize=True).sort_index().to_numpy()
    )
    return {
        "model": model,
        "races": races,
        "X_test": X[test_idx],
        "y_test": y[test_idx],
        "weights": weights,
        "weighting": weighting,
        "train_dist": {r: float(p) for r, p in zip(races, train_dist, strict=True)},
        "stats_path": Path(f"{base}_stats_pt.json"),
        "png_name": key,
    }


def load_census_model(year: int, args) -> dict:
    census_dir = Path(__file__).parent / "census"
    sys.path.insert(0, str(census_dir))
    import train_census_lstm_pytorch as census  # noqa: PLC0415

    df = census.load_census_data(year)
    # Fresh probabilistically-labeled sample with a non-training seed: labels
    # are draws from each name's census race distribution, names drawn by
    # population count, so the evaluation is person-weighted by construction.
    sdf = census.sample_and_assign_race(df, n_samples=200_000, seed=args.seed + 81)

    base = REPO_ROOT / "ethnicolr" / "models" / "census" / "lstm"
    vocab = pd.read_csv(base / f"census{year}_ln_vocab_pytorch.csv").vocab.tolist()
    vocab_dict = {word: idx for idx, word in enumerate(vocab)}
    races = pd.read_csv(base / f"census{year}_race_pytorch.csv").race.tolist()

    X = pad_sequences(
        [
            EthnicolrModelClass.find_ngrams(vocab_dict, name, NGRAMS)
            for name in sdf["name_title"]
        ],
        20,
    )
    y = sdf["race"].map({r: i for i, r in enumerate(races)}).to_numpy()

    model = NameLSTM(vocab_size=len(vocab), num_classes=len(races))
    model.load_state_dict(
        torch.load(
            base / f"census{year}_ln_lstm_pytorch.pt",
            map_location="cpu",
            weights_only=True,
        )
    )
    return {
        "model": model,
        "races": races,
        "X_test": X,
        "y_test": y,
        "weights": np.ones(len(y)),
        "weighting": "person (census count-weighted sample, labels drawn from "
        "census race shares)",
        "train_dist": CENSUS_US_PRIOR
        if races == sorted(CENSUS_US_PRIOR)
        else {r: 1 / len(races) for r in races},
        "stats_path": base / f"census{year}_ln_stats_pytorch.json",
        "png_name": f"census_{year}",
    }


def reliability_png(bins_pre, bins_post, name: str, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, bins, title in [
        (axes[0], bins_pre, "Before calibration"),
        (axes[1], bins_post, "After calibration"),
    ]:
        conf = [b[2] for b in bins]
        acc = [b[3] for b in bins]
        ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
        ax.plot(conf, acc, "o-")
        ax.set_xlabel("Predicted confidence")
        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Empirical accuracy")
    fig.suptitle(f"Reliability: {name}")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"reliability_{name}.png", dpi=120)
    plt.close(fig)


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
        ctx = load_census_model(args.year, args)
    else:
        ctx = load_trainer_model(args.model, args)

    races = ctx["races"]
    logits = compute_logits(ctx["model"], ctx["X_test"])
    y, w = ctx["y_test"], ctx["weights"].astype(float)

    # Split held-out data into a calibration half and an evaluation half
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(y))
    half = len(y) // 2
    cal, ev = perm[:half], perm[half:]

    T = fit_temperature(logits[cal], y[cal], w[cal])
    probs_pre = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    probs_post = torch.softmax(torch.from_numpy(logits) / T, dim=1).numpy()

    scores_cal = aps_scores(probs_post[cal], y[cal])
    n_cal = len(cal)
    quantiles = {
        f"{level:.2f}": weighted_quantile(
            scores_cal, w[cal], min(1.0, np.ceil((n_cal + 1) * level) / n_cal)
        )
        for level in COVERAGE_LEVELS
    }

    ece_pre, bins_pre = ece_and_bins(probs_pre[ev], y[ev], w[ev])
    ece_post, bins_post = ece_and_bins(probs_post[ev], y[ev], w[ev])
    coverage = {
        level_key: empirical_coverage(probs_post[ev], y[ev], w[ev], qhat)
        for level_key, qhat in quantiles.items()
    }
    set_sizes = {
        level_key: float(np.average(aps_set_sizes(probs_post[ev], qhat), weights=w[ev]))
        for level_key, qhat in quantiles.items()
    }

    stats = {
        "model": ctx["png_name"],
        "temperature": T,
        "train_class_distribution": ctx["train_dist"],
        "classes": races,
        "calibration_weighting": ctx["weighting"],
        "conformal_quantiles": quantiles,
        "metrics": {
            "n_calibration": int(len(cal)),
            "n_evaluation": int(len(ev)),
            "accuracy": topk_acc(probs_post[ev], y[ev], w[ev], 1),
            "top2": topk_acc(probs_post[ev], y[ev], w[ev], 2),
            "top3": topk_acc(probs_post[ev], y[ev], w[ev], 3),
            "ece_pre": ece_pre,
            "ece_post": ece_post,
            "brier_pre": brier(probs_pre[ev], y[ev], w[ev]),
            "brier_post": brier(probs_post[ev], y[ev], w[ev]),
            "conformal_empirical_coverage": coverage,
            "conformal_mean_set_size": set_sizes,
        },
        "reliability_bins_pre": bins_pre,
        "reliability_bins_post": bins_post,
    }

    ctx["stats_path"].write_text(json.dumps(stats, indent=2) + "\n")
    reliability_png(
        bins_pre, bins_post, ctx["png_name"], REPO_ROOT / "docs" / "source" / "_static"
    )

    m = stats["metrics"]
    print(f"model: {stats['model']}  weighting: {ctx['weighting']}")
    print(f"temperature: {T:.3f}")
    print(
        f"acc: {m['accuracy']:.4f}  top2: {m['top2']:.4f}  top3: {m['top3']:.4f}\n"
        f"ECE: {m['ece_pre']:.4f} -> {m['ece_post']:.4f}   "
        f"Brier: {m['brier_pre']:.4f} -> {m['brier_post']:.4f}"
    )
    for level_key, qhat in quantiles.items():
        print(
            f"coverage@{level_key}: nominal={level_key}  "
            f"empirical={m['conformal_empirical_coverage'][level_key]:.4f}  "
            f"mean set size={m['conformal_mean_set_size'][level_key]:.2f}  "
            f"(qhat={qhat:.4f})"
        )
    print(f"wrote {ctx['stats_path']}")


if __name__ == "__main__":
    main()
