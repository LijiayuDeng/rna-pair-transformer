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

BACKGROUND = "#fffaf2"
PANEL = "#fffdf8"
TEXT = "#1f2937"
MUTED = "#6b7280"
GRID = "#d6d3d1"
TRAIN_COLOR = "#1d4ed8"
VAL_COLOR = "#c2410c"
TEST_COLOR = "#0f766e"
EXTERNAL_COLOR = "#b45309"


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


def style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(PANEL)
    ax.grid(alpha=0.35, color=GRID, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9ca3af")
    ax.spines["bottom"].set_color("#9ca3af")
    ax.tick_params(colors=TEXT)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)


def plot_training_curves(history_df: pd.DataFrame, output_path: Path) -> None:
    best_epoch = int(history_df.loc[history_df["val_auprc"].idxmax(), "epoch"])
    best_val_auprc = float(history_df["val_auprc"].max())
    epoch_ticks = history_df["epoch"].astype(int).tolist()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    fig.patch.set_facecolor(BACKGROUND)

    for ax in axes:
        style_axes(ax)

    axes[0].plot(
        history_df["epoch"],
        history_df["train_loss"],
        label="Train loss",
        linewidth=2.6,
        color=TRAIN_COLOR,
        marker="o",
        markersize=3.5,
    )
    axes[0].plot(
        history_df["epoch"],
        history_df["val_loss"],
        label="Validation loss",
        linewidth=2.6,
        color=VAL_COLOR,
        marker="o",
        markersize=3.5,
    )
    axes[0].axvline(best_epoch, color=MUTED, linestyle="--", linewidth=1.2, alpha=0.9)
    axes[0].set_title("Loss During Training", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_xticks(epoch_ticks)
    axes[0].legend(frameon=False, loc="upper right")

    axes[1].plot(
        history_df["epoch"],
        history_df["train_auprc"],
        label="Train AUPRC",
        linewidth=2.6,
        color=TRAIN_COLOR,
        marker="o",
        markersize=3.5,
    )
    axes[1].plot(
        history_df["epoch"],
        history_df["val_auprc"],
        label="Validation AUPRC",
        linewidth=2.6,
        color=VAL_COLOR,
        marker="o",
        markersize=3.5,
    )
    axes[1].axvline(best_epoch, color=MUTED, linestyle="--", linewidth=1.2, alpha=0.9)
    axes[1].scatter(
        [best_epoch],
        [best_val_auprc],
        color="#111827",
        s=28,
        zorder=5,
    )
    axes[1].annotate(
        f"Best epoch {best_epoch}\nVal AUPRC {best_val_auprc:.3f}",
        xy=(best_epoch, best_val_auprc),
        xytext=(best_epoch + 1.2, best_val_auprc - 0.055),
        fontsize=9,
        color=TEXT,
        arrowprops={"arrowstyle": "->", "color": MUTED, "lw": 1.0},
    )
    axes[1].set_title("AUPRC During Training", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUPRC")
    axes[1].set_xticks(epoch_ticks)
    axes[1].legend(frameon=False, loc="lower right")

    fig.suptitle(
        "RNA Pair Transformer Training Summary",
        fontsize=17,
        fontweight="bold",
        color=TEXT,
        y=1.02,
    )
    fig.text(
        0.5,
        0.985,
        "Best checkpoint selected by validation AUPRC",
        ha="center",
        va="top",
        fontsize=10,
        color=MUTED,
    )
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
    random_baseline = float(test_labels.float().mean().item())

    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    fig.patch.set_facecolor(BACKGROUND)
    style_axes(ax)
    ax.plot(
        test_recall,
        test_precision,
        linewidth=2.8,
        color=TEST_COLOR,
        label=f"Hejret test  AP={test_ap:.3f}",
    )
    ax.plot(
        ext_recall,
        ext_precision,
        linewidth=2.8,
        color=EXTERNAL_COLOR,
        label=f"Klimentova external  AP={ext_ap:.3f}",
    )
    ax.axhline(
        random_baseline,
        color=MUTED,
        linestyle="--",
        linewidth=1.2,
        alpha=0.8,
        label=f"Random baseline  P={random_baseline:.2f}",
    )
    ax.set_title("Precision-Recall Curves", fontsize=14, fontweight="bold")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.legend(frameon=False, loc="lower left")
    fig.text(
        0.5,
        0.96,
        "Held-out in-distribution test versus external benchmark",
        ha="center",
        va="top",
        fontsize=10,
        color=MUTED,
    )
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
