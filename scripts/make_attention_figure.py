import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Rectangle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import RNAPairDataset, decode_sequence, load_processed_dataframe
from src.model import RNAPairTransformer, RNAPairTransformerConfig

BACKGROUND = "#fffaf2"
PANEL = "#fffdf8"
TEXT = "#1f2937"
MUTED = "#6b7280"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an attention heatmap for a positive example.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split-path", type=Path, default=Path("data/processed/test.tsv"))
    parser.add_argument("--output-path", type=Path, default=Path("assets/attention_example.png"))
    parser.add_argument("--top-k", type=int, default=50)
    return parser.parse_args()


def load_model(run_dir: Path) -> RNAPairTransformer:
    checkpoint = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=False)
    config = RNAPairTransformerConfig(**checkpoint["config"])
    model = RNAPairTransformer(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def choose_example(
    model: RNAPairTransformer,
    dataset: RNAPairDataset,
    dataframe: pd.DataFrame,
    top_k: int,
) -> tuple[int, dict[str, torch.Tensor], float]:
    positive_indices = dataframe.index[dataframe["label"] == 1].tolist()
    candidates: list[tuple[float, int, dict[str, torch.Tensor]]] = []

    with torch.no_grad():
        for idx in positive_indices:
            sample = dataset[idx]
            outputs = model(
                target_ids=sample["target_ids"].unsqueeze(0),
                target_mask=sample["target_mask"].unsqueeze(0),
                mirna_ids=sample["mirna_ids"].unsqueeze(0),
                mirna_mask=sample["mirna_mask"].unsqueeze(0),
                return_attention=True,
            )
            probability = float(outputs["probabilities"][0].item())
            candidates.append((probability, idx, outputs))

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected_probability, selected_index, selected_outputs = candidates[min(top_k - 1, len(candidates) - 1)]
    return selected_index, selected_outputs, selected_probability


def plot_attention(
    dataset: RNAPairDataset,
    dataframe: pd.DataFrame,
    sample_index: int,
    outputs: dict[str, torch.Tensor],
    probability: float,
    output_path: Path,
) -> None:
    sample = dataset[sample_index]
    row = dataframe.iloc[sample_index]

    target_len = int(sample["target_mask"].sum().item())
    mirna_len = int(sample["mirna_mask"].sum().item())
    target_seq = decode_sequence(sample["target_ids"].tolist())[:target_len]
    mirna_seq = decode_sequence(sample["mirna_ids"].tolist())[:mirna_len]

    attention = outputs["attention_weights"][0].mean(dim=0).cpu().numpy()
    attention = attention[:mirna_len, :target_len]

    fig, ax = plt.subplots(figsize=(max(9, target_len * 0.22), max(4.8, mirna_len * 0.36)))
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(PANEL)
    image = ax.imshow(attention, aspect="auto", cmap="magma")
    ax.set_title("Cross-attention Heatmap", fontsize=14, fontweight="bold", color=TEXT)
    ax.set_xlabel("Target-site positions", color=TEXT)
    ax.set_ylabel("miRNA positions", color=TEXT)
    ax.set_xticks(range(target_len))
    ax.set_xticklabels(list(target_seq), fontsize=8, color=TEXT)
    ax.set_yticks(range(mirna_len))
    ax.set_yticklabels([f"{idx + 1} {base}" for idx, base in enumerate(mirna_seq)], fontsize=9, color=TEXT)
    ax.tick_params(axis="x", colors=TEXT)
    ax.tick_params(axis="y", colors=TEXT)
    ax.set_xticks(np.arange(-0.5, target_len, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, mirna_len, 1), minor=True)
    ax.grid(which="minor", color="#f8fafc", linestyle="-", linewidth=0.55, alpha=0.45)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_color("#9ca3af")
    plt.setp(ax.get_xticklabels(), rotation=90)

    seed_start_idx = 1
    seed_end_idx = min(7, mirna_len - 1)
    if seed_end_idx >= seed_start_idx:
        ax.add_patch(
            Rectangle(
                (-0.5, seed_start_idx - 0.5),
                target_len,
                seed_end_idx - seed_start_idx + 1,
                fill=False,
                edgecolor="#facc15",
                linewidth=1.8,
                linestyle="--",
            )
        )
        ax.text(
            target_len - 0.5,
            seed_start_idx - 0.65,
            f"Seed region ({seed_start_idx + 1}-{seed_end_idx + 1})",
            ha="right",
            va="bottom",
            fontsize=9,
            color="#f59e0b",
            fontweight="bold",
        )

    fig.text(
        0.5,
        0.955,
        f"Positive held-out sample  |  predicted probability = {probability:.3f}",
        ha="center",
        va="top",
        fontsize=10,
        color=MUTED,
    )
    fig.text(
        0.5,
        0.925,
        "Average over attention heads; brighter regions indicate stronger target focus",
        ha="center",
        va="top",
        fontsize=9,
        color=MUTED,
    )
    fig.colorbar(image, ax=ax, fraction=0.028, pad=0.02, label="Attention weight")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    model = load_model(args.run_dir)
    dataset = RNAPairDataset(args.split_path)
    dataframe = load_processed_dataframe(args.split_path)

    sample_index, outputs, probability = choose_example(
        model=model,
        dataset=dataset,
        dataframe=dataframe,
        top_k=args.top_k,
    )
    plot_attention(
        dataset=dataset,
        dataframe=dataframe,
        sample_index=sample_index,
        outputs=outputs,
        probability=probability,
        output_path=args.output_path,
    )
    print(f"Saved {args.output_path}")


if __name__ == "__main__":
    main()
