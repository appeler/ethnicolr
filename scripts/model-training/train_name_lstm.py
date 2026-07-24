#!/usr/bin/env python3
"""Train PyTorch LSTM name models for ethnicolr (wiki, FL voter reg, NC voter reg).

Replaces the legacy Keras training notebooks with one parameterized trainer.
Census models have their own script (census/train_census_lstm_pytorch.py) because
their labels are sampled probabilistically from aggregate counts.

Usage:
    python train_name_lstm.py wiki_ln
    python train_name_lstm.py fl_name_five_cat_2022 --data-dir ../data-acquisition/raw
    python train_name_lstm.py nc_name --device mps

Data expectations (see scripts/data-acquisition/):
    wiki:  <data-dir>/wiki_name_race_2026.csv.gz (built by wiki/prepare_wiki_data.py;
           merges the 2009-era in-repo CSV with fresh Wikidata people)
    fl:    <data-dir>/fl_reg_name_race.csv.gz and fl_reg_name_race_2022.csv.gz
    nc:    <data-dir>/nc_voter_name_race.csv.gz
"""

import argparse
import csv
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

from ethnicolr.ethnicolr_class import EthnicolrModelClass  # noqa: E402
from ethnicolr.torch_utils import NameLSTM, get_device, pad_sequences  # noqa: E402

NGRAMS = 2

FL_FIVE_CAT_MAP = {
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
    loader: str  # wiki | fl_4cat | fl_5cat | nc
    feature: str  # ln | name
    feature_len: int
    epochs: int
    basename: str  # artifact basename, e.g. "wiki/lstm/wiki_ln"
    data_file: str  # relative to repo (wiki) or --data-dir (fl/nc)
    n_classes: int


MODEL_CONFIGS = {
    "wiki_ln": ModelConfig(
        "wiki",
        "ln",
        20,
        20,
        "wiki/lstm/wiki_ln",
        "wiki_name_race_2026.csv.gz",
        13,
    ),
    "wiki_name": ModelConfig(
        "wiki",
        "name",
        25,
        20,
        "wiki/lstm/wiki_name",
        "wiki_name_race_2026.csv.gz",
        13,
    ),
    "fl_ln": ModelConfig(
        "fl_4cat", "ln", 20, 15, "fl_voter_reg/lstm/fl_ln", "fl_reg_name_race.csv.gz", 4
    ),
    "fl_name": ModelConfig(
        "fl_4cat",
        "name",
        25,
        15,
        "fl_voter_reg/lstm/fl_name",
        "fl_reg_name_race.csv.gz",
        4,
    ),
    "fl_ln_five_cat": ModelConfig(
        "fl_5cat",
        "ln",
        20,
        20,
        "fl_voter_reg/lstm/fl_ln_five_cat",
        "fl_reg_name_race.csv.gz",
        5,
    ),
    "fl_name_five_cat": ModelConfig(
        "fl_5cat",
        "name",
        25,
        20,
        "fl_voter_reg/lstm/fl_name_five_cat",
        "fl_reg_name_race.csv.gz",
        5,
    ),
    "fl_ln_five_cat_2022": ModelConfig(
        "fl_5cat",
        "ln",
        20,
        20,
        "fl_voter_reg/lstm/fl_ln_five_cat_2022",
        "fl_reg_name_race_2022.csv.gz",
        5,
    ),
    "fl_name_five_cat_2022": ModelConfig(
        "fl_5cat",
        "name",
        25,
        20,
        "fl_voter_reg/lstm/fl_name_five_cat_2022",
        "fl_reg_name_race_2022.csv.gz",
        5,
    ),
    "nc_name": ModelConfig(
        "nc",
        "name",
        25,
        15,
        "nc_voter_reg/lstm/nc_name",
        "nc_voter_name_race.csv.gz",
        12,
    ),
}


def load_wiki(
    path: Path, rng: np.random.Generator, max_per_class: int = 300_000
) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["name_first", "name_last"])
    # Cap dominant categories so 13-class balance stays workable
    seed = int(rng.integers(2**31))
    df = df.groupby("race", group_keys=False)[df.columns].apply(
        lambda g: g.sample(min(len(g), max_per_class), random_state=seed)
    )
    return df[["name_last", "name_first", "race"]]


def load_fl_4cat(
    path: Path, rng: np.random.Generator, n_samples: int = 1_000_000
) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["name_first", "name_last"])
    df = df[df.race.isin(["asian", "hispanic", "nh_black", "nh_white"])]
    if len(df) > n_samples:
        df = df.sample(n_samples, random_state=int(rng.integers(2**31)))
    return df[["name_last", "name_first", "race"]]


def load_fl_5cat(
    path: Path, rng: np.random.Generator, n_samples: int = 1_000_000
) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["name_first", "name_last"])
    df["race"] = df.race.map(FL_FIVE_CAT_MAP)
    df = df[df.race != "unknown"]
    per_class = n_samples // df.race.nunique()
    df = df.groupby("race").sample(
        per_class, random_state=int(rng.integers(2**31)), replace=True
    )
    return df[["name_last", "name_first", "race"]]


def load_nc(
    path: Path, rng: np.random.Generator, n_samples: int = 1_000_000
) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["first_name", "last_name"])
    df = df[df.race_code.isin(NC_RACE_CODES) & df.ethnic_code.isin(["HL", "NL"])]
    df["race"] = df["ethnic_code"] + "+" + df["race_code"]
    df = df.drop_duplicates(subset=["last_name", "first_name", "race"])

    # Oversample every class to the majority class size, then draw the sample
    max_size = df["race"].value_counts().max()
    seed = int(rng.integers(2**31))
    parts = [df]
    for _, group in df.groupby("race"):
        parts.append(
            group.sample(max_size - len(group), replace=True, random_state=seed)
        )
    df = pd.concat(parts)
    df = df.sample(min(n_samples, len(df)), random_state=seed)

    df = df.rename(columns={"last_name": "name_last", "first_name": "name_first"})
    return df[["name_last", "name_first", "race"]]


LOADERS = {
    "wiki": load_wiki,
    "fl_4cat": load_fl_4cat,
    "fl_5cat": load_fl_5cat,
    "nc": load_nc,
}


def build_features(df: pd.DataFrame, feature: str) -> pd.Series:
    last = df["name_last"].astype(str).str.strip().str.title()
    if feature == "ln":
        return last
    first = df["name_first"].astype(str).str.strip().str.title()
    return (last + " " + first).str.strip()


def build_vocab(names: pd.Series) -> list[str]:
    vect = CountVectorizer(
        analyzer="char",
        max_df=0.3,
        min_df=3,
        ngram_range=(NGRAMS, NGRAMS),
        lowercase=False,
    )
    counts = np.asarray(vect.fit_transform(names).sum(axis=0)).flatten()
    sorted_items = sorted(
        ((counts[idx], word) for word, idx in vect.vocabulary_.items()), reverse=True
    )
    return ["UNK"] + [word for _, word in sorted_items]


def topk_accuracy(model, loader, device, ks=(2, 3)):
    """Fraction of samples whose true label is within the top-k predictions."""
    model.eval()
    hits = dict.fromkeys(ks, 0)
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            topk = model(X_batch).topk(max(ks), dim=1).indices
            for k in ks:
                hits[k] += (topk[:, :k] == y_batch.unsqueeze(1)).any(dim=1).sum().item()
            total += y_batch.size(0)
    return {k: hits[k] / total for k in ks}


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.set_grad_enabled(training):
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * X_batch.size(0)
            predicted = outputs.argmax(1)
            correct += predicted.eq(y_batch).sum().item()
            total += y_batch.size(0)
            if not training:
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
    return (
        total_loss / total,
        correct / total,
        np.array(all_preds),
        np.array(all_labels),
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
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model]
    device = get_device() if args.device == "auto" else torch.device(args.device)
    epochs = args.epochs or cfg.epochs
    print(f"Training {args.model} on {device} for {epochs} epochs")

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    data_path = args.data_dir / cfg.data_file
    if not data_path.exists():
        raise FileNotFoundError(
            f"Training data not found: {data_path}\n"
            "Run the scripts in scripts/data-acquisition/ first."
        )

    loader_fn = LOADERS[cfg.loader]
    df = (
        loader_fn(data_path, rng)
        if cfg.loader == "wiki"
        else loader_fn(data_path, rng, n_samples=args.samples)
    )
    print(f"Loaded {len(df):,} rows")
    print(df["race"].value_counts())

    races = sorted(df["race"].unique())
    if len(races) != cfg.n_classes:
        raise ValueError(f"Expected {cfg.n_classes} classes, got {len(races)}: {races}")

    features = build_features(df, cfg.feature)
    vocab = build_vocab(features)
    vocab_dict = {word: idx for idx, word in enumerate(vocab)}
    print(f"Vocabulary size: {len(vocab)}")

    X = pad_sequences(
        [
            EthnicolrModelClass.find_ngrams(vocab_dict, name, NGRAMS)
            for name in features
        ],
        cfg.feature_len,
    )
    race_to_idx = {race: idx for idx, race in enumerate(races)}
    y = df["race"].map(race_to_idx).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )
    print(f"Train: {len(X_train):,}, Test: {len(X_test):,}")

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
        batch_size=args.batch_size * 4,
    )

    model = NameLSTM(vocab_size=len(vocab), num_classes=len(races)).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc, _, _ = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss, val_acc, _, _ = run_epoch(model, test_loader, criterion, device)
        print(
            f"Epoch {epoch:2d}/{epochs}: train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    # Final evaluation always on CPU: guards against any device-specific
    # numerical issue in the saved artifact.
    cpu_model = NameLSTM(vocab_size=len(vocab), num_classes=len(races))
    cpu_model.load_state_dict({k: v.cpu() for k, v in model.state_dict().items()})
    _, test_acc, y_pred, y_true = run_epoch(
        cpu_model, test_loader, criterion, torch.device("cpu")
    )
    print(f"\nTest accuracy (CPU eval): {test_acc:.4f}")
    topk = topk_accuracy(cpu_model, test_loader, torch.device("cpu"))
    print(" ".join(f"top-{k} accuracy: {v:.4f}" for k, v in sorted(topk.items())))
    print(classification_report(y_true, y_pred, target_names=races))
    print(confusion_matrix(y_true, y_pred))

    out_base = REPO_ROOT / "ethnicolr" / "models" / cfg.basename
    out_base.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cpu_model.state_dict(), f"{out_base}_lstm_pt.pt")
    # QUOTE_ALL keeps n-grams with leading/trailing spaces intact
    pd.DataFrame({"vocab": vocab}).to_csv(
        f"{out_base}_vocab_pt.csv", index=False, quoting=csv.QUOTE_ALL
    )
    pd.DataFrame({"race": races}).to_csv(f"{out_base}_race_pt.csv", index=False)
    print(f"Saved {out_base}_lstm_pt.pt / _vocab_pt.csv / _race_pt.csv")


if __name__ == "__main__":
    main()
