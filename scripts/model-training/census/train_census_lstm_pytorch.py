#!/usr/bin/env python3
"""
PyTorch LSTM training script for Census surname → race/ethnicity prediction.

Trains a character-bigram LSTM model on US Census surname data.
Compatible with ethnicolr inference pipeline.

Usage:
    python train_census_lstm_pytorch.py --year 2020 --epochs 15
    python train_census_lstm_pytorch.py --year 2020 --epochs 15 --device cuda
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent.parent.parent / "ethnicolr" / "data" / "census"
MODEL_DIR = SCRIPT_DIR.parent.parent.parent / "ethnicolr" / "models" / "census" / "lstm"

RACES = ["api", "black", "hispanic", "white"]
RACE_PCT_COLS = ["pctapi", "pctblack", "pcthispanic", "pctwhite"]


class CensusLSTM(nn.Module):
    """Character-bigram LSTM for race/ethnicity prediction."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 32,
        hidden_dim: int = 128,
        num_classes: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len) of bigram indices
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        _, (hidden, _) = self.lstm(embedded)  # hidden: (1, batch, hidden_dim)
        hidden = self.dropout(hidden.squeeze(0))  # (batch, hidden_dim)
        return self.fc(hidden)  # (batch, num_classes)


def load_census_data(year: int) -> pd.DataFrame:
    """Load census CSV data."""
    csv_path = DATA_DIR / f"census_{year}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Census data not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df.dropna(subset=["name"], inplace=True)
    df.replace("(S)", 0, inplace=True)

    for col in RACE_PCT_COLS + ["count"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(0)

    print(f"Loaded {len(df):,} surnames from {csv_path}")
    return df


def sample_and_assign_race(
    df: pd.DataFrame, n_samples: int = 1_000_000, seed: int = 42
) -> pd.DataFrame:
    """Sample names weighted by frequency and assign race probabilistically."""
    rng = np.random.default_rng(seed)

    # Sample weighted by count
    weights = df["count"].values.astype(float)
    weights = weights / weights.sum()
    indices = rng.choice(len(df), size=n_samples, replace=True, p=weights)
    sdf = df.iloc[indices].copy().reset_index(drop=True)

    # Assign race based on percentages
    def assign_race(row):
        probs = np.array([row[col] for col in RACE_PCT_COLS], dtype=float)
        if probs.sum() == 0:
            return "white"
        probs = probs / probs.sum()
        return rng.choice(RACES, p=probs)

    sdf["race"] = sdf.apply(assign_race, axis=1)
    sdf["name_title"] = sdf["name"].str.title()

    print(f"Sampled {len(sdf):,} names")
    print("Race distribution:")
    print(sdf["race"].value_counts())

    return sdf


def build_vocab(names: pd.Series, ngram: int = 2) -> tuple[list[str], dict[str, int]]:
    """Build bigram vocabulary sorted by frequency."""
    vect = CountVectorizer(
        analyzer="char",
        max_df=0.3,
        min_df=3,
        ngram_range=(ngram, ngram),
        lowercase=False,
    )
    vect.fit(names)

    vocab = vect.vocabulary_
    counts = np.asarray(vect.transform(names).sum(axis=0)).flatten()

    # Sort by frequency (highest first), prepend UNK token
    sorted_items = sorted(
        [(counts[idx], word) for word, idx in vocab.items()], reverse=True
    )
    words_list = ["UNK"] + [word for _, word in sorted_items]
    word_to_idx = {word: idx for idx, word in enumerate(words_list)}

    print(f"Vocabulary size: {len(words_list)}")
    return words_list, word_to_idx


def names_to_sequences(
    names: pd.Series, word_to_idx: dict[str, int], ngram: int = 2
) -> list[list[int]]:
    """Convert names to sequences of bigram indices."""
    sequences = []
    for name in names:
        seq = []
        for i in range(len(name) - ngram + 1):
            bigram = name[i : i + ngram]
            seq.append(word_to_idx.get(bigram, 0))  # 0 = UNK
        sequences.append(seq)
    return sequences


def pad_sequences(sequences: list[list[int]], maxlen: int = 20) -> np.ndarray:
    """Pad sequences to fixed length (pre-padding with zeros)."""
    padded = np.zeros((len(sequences), maxlen), dtype=np.int64)
    for i, seq in enumerate(sequences):
        if len(seq) > maxlen:
            padded[i] = seq[:maxlen]
        else:
            padded[i, -len(seq) :] = seq
    return padded


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(y_batch).sum().item()
        total += y_batch.size(0)

    return total_loss / total, correct / total


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            total_loss += loss.item() * X_batch.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(y_batch).sum().item()
            total += y_batch.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    return (
        total_loss / total,
        correct / total,
        np.array(all_preds),
        np.array(all_labels),
    )


def save_model_for_ethnicolr(
    model: nn.Module,
    words_list: list[str],
    year: int,
    output_dir: Path,
):
    """Save model and vocab in ethnicolr-compatible format."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save PyTorch model
    model_path = output_dir / f"census{year}_ln_lstm_pytorch.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")

    # Save vocabulary
    vocab_path = output_dir / f"census{year}_ln_vocab_pytorch.csv"
    vocab_df = pd.DataFrame(words_list, columns=["vocab"])
    vocab_df.to_csv(vocab_path, index=False, encoding="utf-8")
    print(f"Saved vocabulary to {vocab_path}")

    # Save race labels
    race_path = output_dir / f"census{year}_race_pytorch.csv"
    race_df = pd.DataFrame({"race": RACES})
    race_df.to_csv(race_path, index=False)
    print(f"Saved race labels to {race_path}")


def export_to_onnx(
    model: nn.Module,
    vocab_size: int,
    seq_len: int,
    year: int,
    output_dir: Path,
    device: torch.device,
):
    """Export model to ONNX format for cross-framework compatibility."""
    model.eval()
    dummy_input = torch.zeros(1, seq_len, dtype=torch.long, device=device)
    onnx_path = output_dir / f"census{year}_ln_lstm.onnx"

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=14,
    )
    print(f"Exported ONNX model to {onnx_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Census LSTM model in PyTorch")
    parser.add_argument("--year", type=int, default=2020, choices=[2000, 2010, 2020])
    parser.add_argument("--samples", type=int, default=1_000_000)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seq-len", type=int, default=20)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--export-onnx", action="store_true")
    args = parser.parse_args()

    # Set device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load and prepare data
    print(f"\n=== Loading Census {args.year} data ===")
    df = load_census_data(args.year)

    print(f"\n=== Sampling {args.samples:,} names ===")
    sdf = sample_and_assign_race(df, n_samples=args.samples, seed=args.seed)

    print("\n=== Building vocabulary ===")
    words_list, word_to_idx = build_vocab(sdf["name_title"], ngram=2)

    print("\n=== Converting to sequences ===")
    X = names_to_sequences(sdf["name_title"], word_to_idx, ngram=2)
    X = pad_sequences(X, maxlen=args.seq_len)

    # Encode labels
    race_to_idx = {race: idx for idx, race in enumerate(RACES)}
    y = np.array([race_to_idx[race] for race in sdf["race"]])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )
    print(f"Train: {len(X_train):,}, Test: {len(X_test):,}")

    # Create data loaders
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # Create model
    print("\n=== Building model ===")
    model = CensusLSTM(
        vocab_size=len(words_list),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_classes=len(RACES),
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(model)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    print(f"\n=== Training for {args.epochs} epochs ===")
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validation on subset of test set for speed
        val_loader = DataLoader(
            test_dataset, batch_size=args.batch_size * 4, shuffle=False
        )
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:2d}/{args.epochs}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc

    # Final evaluation
    print("\n=== Final Evaluation ===")
    test_loss, test_acc, y_pred, y_true = evaluate(
        model, test_loader, criterion, device
    )
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=RACES))

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    # Save model
    print("\n=== Saving model ===")
    save_model_for_ethnicolr(model, words_list, args.year, MODEL_DIR)

    if args.export_onnx:
        export_to_onnx(
            model, len(words_list), args.seq_len, args.year, MODEL_DIR, device
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
