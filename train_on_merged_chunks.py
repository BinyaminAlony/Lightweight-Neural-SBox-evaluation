from __future__ import annotations

import argparse
import os
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

# ============================================================================
# Data Loading
# ============================================================================

CHUNK_PATTERN = re.compile(r"^chunk_(\d+)(?:_fin)?\.pt$")
RECALL_BIT_INDICES = {1, 3}
DEFAULT_RECALL_THRESHOLD = 0.5


class ChunkDataset(Dataset):
    """Lightweight dataset wrapper for a single chunk."""

    def __init__(self, data: torch.Tensor, labels: torch.Tensor):
        self.data = data.float()
        self.labels = labels.float()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]


def list_chunk_files(chunks_dir: Path) -> list[Path]:
    """List all chunk files sorted by index."""
    if not chunks_dir.exists() or not chunks_dir.is_dir():
        raise FileNotFoundError(f"Chunk directory not found: {chunks_dir}")

    files = [p for p in chunks_dir.glob("chunk_*.pt") if CHUNK_PATTERN.match(p.name)]
    if not files:
        raise FileNotFoundError(f"No chunk files found in: {chunks_dir}")

    return sorted(files, key=lambda p: int(CHUNK_PATTERN.match(p.name).group(1)))


def load_chunk(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Load data and labels from a chunk file."""
    chunk = torch.load(path, weights_only=True)
    data = chunk["data"].float()
    labels = chunk["labels"].float()
    return data, labels


# ============================================================================
# Model Architectures
# ============================================================================

class SimpleCNN(nn.Module):
    """Convolutional neural network for 1D inputs."""

    def __init__(self, input_size: int = 32, output_dim: int = 5, hidden_dim: int = 32):
        super().__init__()
        self.conv1 = nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(hidden_dim * 2 * input_size, output_dim)
        self.model_type = "CNN"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # [batch, 1, input_size]
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class SimpleMLP(nn.Module):
    """Multi-layer perceptron for 1D inputs."""

    def __init__(self, input_size: int = 32, output_dim: int = 5, hidden_dim: int = 256):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.model_type = "MLP"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        return x


# ============================================================================
# Training Functions
# ============================================================================

def mse_loss_fn(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean squared error loss."""
    return nn.functional.mse_loss(predictions, targets)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: str,
) -> float:
    """Train for one epoch and return average loss."""
    model.train()
    total_loss = 0.0
    batch_count = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = mse_loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        batch_count += y.size(0)

    return total_loss / batch_count if batch_count > 0 else 0.0


def evaluate_epoch_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    output_dim: int,
    recall_threshold: float,
) -> dict[str, object]:
    """Evaluate model and return overall loss plus per-bit MSE/recall statistics."""
    model.eval()
    total_loss = 0.0
    batch_count = 0
    squared_error_sum = torch.zeros(output_dim, dtype=torch.float64)
    tp = torch.zeros(output_dim, dtype=torch.float64)
    fn = torch.zeros(output_dim, dtype=torch.float64)

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            predictions = model(x)
            loss = mse_loss_fn(predictions, y)

            batch_size = y.size(0)
            total_loss += loss.item() * batch_size
            batch_count += batch_size

            error = predictions - y
            squared_error_sum += torch.sum(error.pow(2), dim=0).detach().cpu().to(torch.float64)

            positive_predictions = predictions >= recall_threshold
            positive_targets = y >= recall_threshold
            tp += torch.sum(positive_predictions & positive_targets, dim=0).detach().cpu().to(torch.float64)
            fn += torch.sum((~positive_predictions) & positive_targets, dim=0).detach().cpu().to(torch.float64)

    avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
    per_bit_mse = (squared_error_sum / batch_count).tolist() if batch_count > 0 else [0.0] * output_dim
    per_bit_recall = []
    for bit_idx in range(output_dim):
        if bit_idx in RECALL_BIT_INDICES:
            denominator = tp[bit_idx] + fn[bit_idx]
            per_bit_recall.append(float((tp[bit_idx] / denominator * 100.0).item()) if denominator > 0 else 0.0)
        else:
            per_bit_recall.append(None)

    return {
        "avg_loss": avg_loss,
        "sample_count": batch_count,
        "squared_error_sum": squared_error_sum.tolist(),
        "tp": tp.tolist(),
        "fn": fn.tolist(),
        "per_bit_mse": per_bit_mse,
        "per_bit_recall": per_bit_recall,
    }


# ============================================================================
# Main Training Script
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a model on merged chunks (streaming all chunks each epoch, no full dataset in memory)."
    )
    parser.add_argument(
        "--chunks-dir",
        type=str,
        default="merged_chunks_2026-05-04",
        help="Directory containing merged chunk files.",
    )
    parser.add_argument(
        "--model-type",
        choices=["MLP", "CNN"],
        default="MLP",
        help="Model architecture to use.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="Hidden dimension size for the model.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of epochs (each epoch streams through all chunks).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size.",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=128,
        help="Test batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="L2 weight decay.",
    )
    parser.add_argument(
        "--train-test-ratio",
        type=float,
        default=0.8,
        help="Train/test split ratio within each chunk.",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="Disable CUDA.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Maximum number of chunks to process.",
    )
    parser.add_argument(
        "--recall-threshold",
        type=float,
        default=DEFAULT_RECALL_THRESHOLD,
        help="Threshold used to convert predictions and labels to positives for recall bits.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Setup device
    device = "cpu" if args.no_cuda else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # List chunk files
    print(f"\nLoading chunk files from: {args.chunks_dir}")
    chunk_files = list_chunk_files(Path(args.chunks_dir))
    print(f"Found {len(chunk_files)} chunks")

    if args.max_chunks is not None:
        chunk_files = chunk_files[: args.max_chunks]
        print(f"Limited to first {len(chunk_files)} chunks")

    # Determine output dimension from first chunk
    data_first, labels_first = load_chunk(chunk_files[0])
    output_dim = labels_first.shape[1]
    input_size = data_first.shape[1]
    print(f"Input size: {input_size}, Output dimension: {output_dim}")

    # Create model
    if args.model_type == "MLP":
        model = SimpleMLP(input_size=input_size, output_dim=output_dim, hidden_dim=args.hidden_dim)
    else:
        model = SimpleCNN(input_size=input_size, output_dim=output_dim, hidden_dim=args.hidden_dim)

    model = model.to(device)
    print(f"Model type: {model.model_type}")

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(f"Optimizer: Adam (lr={args.lr}, weight_decay={args.weight_decay})")

    # Training loop: epoch-by-epoch, streaming through all chunks each epoch
    print(f"\nTraining for {args.epochs} epochs (streaming all chunks each epoch)...")
    all_metrics = {
        "epoch": [],
        "avg_train_loss": [],
        "avg_test_loss": [],
        "bit_metrics": {bit_idx: [] for bit_idx in range(output_dim)},
    }
    bit_metric_types = {
        bit_idx: "recall" if bit_idx in RECALL_BIT_INDICES else "mse"
        for bit_idx in range(output_dim)
    }

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*70}")

        epoch_train_loss_sum = 0.0
        epoch_train_sample_count = 0
        epoch_test_loss_sum = 0.0
        epoch_test_sample_count = 0
        epoch_squared_error_sum = np.zeros(output_dim, dtype=np.float64)
        epoch_tp_sum = np.zeros(output_dim, dtype=np.float64)
        epoch_fn_sum = np.zeros(output_dim, dtype=np.float64)

        for chunk_idx, chunk_file in enumerate(chunk_files):
            # Load chunk
            data, labels = load_chunk(chunk_file)

            # Split within the chunk
            dataset = ChunkDataset(data, labels)
            train_size = int(len(dataset) * args.train_test_ratio)
            test_size = len(dataset) - train_size
            train_ds, test_ds = random_split(dataset, [train_size, test_size])

            train_loader = DataLoader(
                train_ds,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
            )
            test_loader = DataLoader(
                test_ds,
                batch_size=args.test_batch_size,
                shuffle=False,
                num_workers=args.num_workers,
            )

            # Train one pass through this chunk
            train_loss = train_epoch(model, train_loader, optimizer, device)
            test_stats = evaluate_epoch_metrics(
                model,
                test_loader,
                device,
                output_dim,
                args.recall_threshold,
            )

            epoch_train_loss_sum += train_loss * train_size
            epoch_train_sample_count += train_size
            epoch_test_loss_sum += float(test_stats["avg_loss"]) * int(test_stats["sample_count"])
            epoch_test_sample_count += int(test_stats["sample_count"])
            epoch_squared_error_sum += np.array(test_stats["squared_error_sum"], dtype=np.float64)
            epoch_tp_sum += np.array(test_stats["tp"], dtype=np.float64)
            epoch_fn_sum += np.array(test_stats["fn"], dtype=np.float64)

            if (chunk_idx + 1) % max(1, len(chunk_files) // 10) == 0 or chunk_idx == 0:
                chunk_metric_parts = []
                for bit_idx in range(output_dim):
                    if bit_metric_types[bit_idx] == "recall":
                        chunk_metric_parts.append(
                            f"b{bit_idx}=recall:{test_stats['per_bit_recall'][bit_idx]:.2f}%"
                        )
                    else:
                        chunk_metric_parts.append(f"b{bit_idx}=mse:{test_stats['per_bit_mse'][bit_idx]:.6f}")
                print(
                    f"  Chunk {chunk_idx + 1:4d}/{len(chunk_files)}: "
                    f"train_loss={train_loss:.6f}, test_loss={float(test_stats['avg_loss']):.6f}, "
                    + ", ".join(chunk_metric_parts)
                )

        # Log epoch metrics
        avg_train_loss = epoch_train_loss_sum / epoch_train_sample_count if epoch_train_sample_count > 0 else 0.0
        avg_test_loss = epoch_test_loss_sum / epoch_test_sample_count if epoch_test_sample_count > 0 else 0.0

        epoch_bit_metrics: dict[int, float] = {}
        for bit_idx in range(output_dim):
            if bit_metric_types[bit_idx] == "recall":
                denominator = epoch_tp_sum[bit_idx] + epoch_fn_sum[bit_idx]
                epoch_bit_metrics[bit_idx] = float(epoch_tp_sum[bit_idx] / denominator * 100.0) if denominator > 0 else 0.0
            else:
                epoch_bit_metrics[bit_idx] = (
                    float(epoch_squared_error_sum[bit_idx] / epoch_test_sample_count)
                    if epoch_test_sample_count > 0
                    else 0.0
                )

        all_metrics["epoch"].append(epoch)
        all_metrics["avg_train_loss"].append(avg_train_loss)
        all_metrics["avg_test_loss"].append(avg_test_loss)
        for bit_idx, value in epoch_bit_metrics.items():
            all_metrics["bit_metrics"][bit_idx].append(value)

        bit_summary = []
        for bit_idx in range(output_dim):
            if bit_metric_types[bit_idx] == "recall":
                bit_summary.append(f"b{bit_idx}=recall:{epoch_bit_metrics[bit_idx]:.2f}%")
            else:
                bit_summary.append(f"b{bit_idx}=mse:{epoch_bit_metrics[bit_idx]:.6f}")

        print(
            f"\nEpoch {epoch} Summary: "
            f"avg_train_loss={avg_train_loss:.6f}, "
            f"avg_test_loss={avg_test_loss:.6f}, "
            + ", ".join(bit_summary)
        )

    print(f"\n{'='*70}")
    print("Training complete!")
    print(f"{'='*70}")

    # Save model
    os.makedirs("models", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"model_{args.model_type}_{timestamp}.pt"
    model_path = os.path.join("models", model_name)
    torch.save(model.state_dict(), model_path)
    print(f"\nSaved model state_dict to: {model_path}")

    # Plot overall loss results
    if len(all_metrics["epoch"]) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        epochs = all_metrics["epoch"]

        axes[0].plot(epochs, all_metrics["avg_train_loss"], marker="o", label="Train Loss")
        axes[0].plot(epochs, all_metrics["avg_test_loss"], marker="s", label="Test Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss (MSE)")
        axes[0].set_title("Average Loss per Epoch")
        axes[0].grid(True)
        axes[0].legend()

        axes[1].axis("off")
        axes[1].text(
            0.5,
            0.5,
            "Per-bit plots are saved separately",
            ha="center",
            va="center",
            fontsize=11,
        )

        os.makedirs("plots", exist_ok=True)
        plot_name = f"training_{args.model_type}_{timestamp}.png"
        plot_path = os.path.join("plots", plot_name)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=100)
        print(f"Saved plot to: {plot_path}")
        plt.close()

        # Plot per-bit metrics separately.
        for bit_idx in range(output_dim):
            fig, ax = plt.subplots(1, 1, figsize=(7, 4))
            metric_values = all_metrics["bit_metrics"][bit_idx]
            metric_type = bit_metric_types[bit_idx]

            if metric_type == "recall":
                ax.plot(epochs, metric_values, marker="o", color="C3")
                ax.set_ylabel("Recall %")
                ax.set_title(f"Bit {bit_idx} Recall per Epoch")
            else:
                ax.plot(epochs, metric_values, marker="o", color="C0")
                ax.set_ylabel("MSE")
                ax.set_title(f"Bit {bit_idx} MSE per Epoch")

            ax.set_xlabel("Epoch")
            ax.grid(True)

            bit_plot_name = f"training_{args.model_type}_bit_{bit_idx}_{timestamp}.png"
            bit_plot_path = os.path.join("plots", bit_plot_name)
            plt.tight_layout()
            plt.savefig(bit_plot_path, dpi=100)
            print(f"Saved bit {bit_idx} plot to: {bit_plot_path}")
            plt.close()


if __name__ == "__main__":
    main()
