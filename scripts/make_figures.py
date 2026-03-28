import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_datasets
from src.model import RNAPairTransformer, RNAPairTransformerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate training and PR-curve figures for a run.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("assets"))
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def load_model_from_run(run_dir: Path) -> RNAPairTransformer:
    checkpoint = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=False)
    config = RNAPairTransformerConfig(**checkpoint["config"])
    model = RNAPairTransformer(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def collect_predictions(
    model: RNAPairTransformer,
    dataloader: DataLoader,
) -> tuple[torch.Tensor, torch.Tensor]:
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for batch in dataloader:
            outputs = model(
                target_ids=batch["target_ids"],
                target_mask=batch["target_mask"],
                mirna_ids=batch["mirna_ids"],
                mirna_mask=batch["mirna_mask"],
            )
            all_labels.append(batch["label"])
            all_probabilities.append(outputs["probabilities"])

    labels = torch.cat(all_labels).cpu()
    probabilities = torch.cat(all_probabilities).cpu()
    return labels, probabilities


def plot_training_curves(history_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(history_df["epoch"], history_df["train_loss"], label="Train Loss", linewidth=2)
    axes[0].plot(history_df["epoch"], history_df["val_loss"], label="Val Loss", linewidth=2)
    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(history_df["epoch"], history_df["train_auprc"], label="Train AUPRC", linewidth=2)
    axes[1].plot(history_df["epoch"], history_df["val_auprc"], label="Val AUPRC", linewidth=2)
    axes[1].set_title("AUPRC Curves")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUPRC")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_pr_curves(
    model: RNAPairTransformer,
    processed_dir: Path,
    batch_size: int,
    output_path: Path,
) -> None:
    datasets = load_datasets(processed_dir)
    test_loader = DataLoader(datasets["test"], batch_size=batch_size, shuffle=False)
    external_loader = DataLoader(datasets["external_test"], batch_size=batch_size, shuffle=False)

    test_labels, test_probabilities = collect_predictions(model, test_loader)
    ext_labels, ext_probabilities = collect_predictions(model, external_loader)

    test_precision, test_recall, _ = precision_recall_curve(test_labels.numpy(), test_probabilities.numpy())
    ext_precision, ext_recall, _ = precision_recall_curve(ext_labels.numpy(), ext_probabilities.numpy())

    test_ap = average_precision_score(test_labels.numpy(), test_probabilities.numpy())
    ext_ap = average_precision_score(ext_labels.numpy(), ext_probabilities.numpy())

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(test_recall, test_precision, linewidth=2, label=f"Hejret test (AP={test_ap:.3f})")
    ax.plot(ext_recall, ext_precision, linewidth=2, label=f"Klimentova external (AP={ext_ap:.3f})")
    ax.set_title("Precision-Recall Curves")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    history_df = pd.read_csv(args.run_dir / "history.csv")
    model = load_model_from_run(args.run_dir)

    training_curves_path = args.output_dir / "training_curves.png"
    pr_curve_path = args.output_dir / "pr_curve.png"

    plot_training_curves(history_df, training_curves_path)
    plot_pr_curves(model, args.processed_dir, args.batch_size, pr_curve_path)

    print(f"Saved {training_curves_path}")
    print(f"Saved {pr_curve_path}")


if __name__ == "__main__":
    main()
