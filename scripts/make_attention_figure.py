import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import RNAPairDataset, decode_sequence, load_processed_dataframe
from src.model import RNAPairTransformer, RNAPairTransformerConfig


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

    fig, ax = plt.subplots(figsize=(max(8, target_len * 0.18), max(4, mirna_len * 0.3)))
    image = ax.imshow(attention, aspect="auto", cmap="viridis")
    ax.set_title(f"Cross-attention Heatmap (label={int(row['label'])}, prob={probability:.3f})")
    ax.set_xlabel("Target-site position")
    ax.set_ylabel("miRNA position")
    ax.set_xticks(range(target_len))
    ax.set_xticklabels(list(target_seq), fontsize=8)
    ax.set_yticks(range(mirna_len))
    ax.set_yticklabels(list(mirna_seq), fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=90)
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02, label="Attention weight")
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
