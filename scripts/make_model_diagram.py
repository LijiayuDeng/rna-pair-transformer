import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a simple model diagram for the RNA Pair Transformer.")
    parser.add_argument("--output-path", type=Path, default=Path("assets/model_diagram.png"))
    return parser.parse_args()


def add_box(ax, x: float, y: float, w: float, h: float, label: str, facecolor: str) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.5,
        edgecolor="#1f2937",
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        label,
        ha="center",
        va="center",
        fontsize=11,
        color="#111827",
        wrap=True,
    )


def add_arrow(ax, start: tuple[float, float], end: tuple[float, float]) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.8,
        color="#374151",
        connectionstyle="arc3",
    )
    ax.add_patch(arrow)


def main() -> None:
    args = parse_args()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    fig.patch.set_facecolor("#fffdf8")
    ax.set_facecolor("#fffdf8")

    add_box(ax, 0.6, 3.6, 1.7, 0.9, "miRNA\n30 tokens", "#dbeafe")
    add_box(ax, 0.6, 1.4, 1.7, 0.9, "Target site\n50 tokens", "#fde68a")

    add_box(ax, 2.8, 3.6, 1.7, 0.9, "Shared token\nembedding", "#e5e7eb")
    add_box(ax, 2.8, 1.4, 1.7, 0.9, "Shared token\nembedding", "#e5e7eb")

    add_box(ax, 5.0, 3.6, 1.9, 0.9, "miRNA encoder\n2-3 self-attn layers", "#bfdbfe")
    add_box(ax, 5.0, 1.4, 1.9, 0.9, "Target encoder\n2-3 self-attn layers", "#fde68a")

    add_box(ax, 7.6, 2.5, 1.9, 1.0, "Cross-attention\nmiRNA queries\ntarget", "#fecaca")
    add_box(ax, 10.0, 2.5, 1.4, 1.0, "Masked mean\npooling", "#ddd6fe")
    add_box(ax, 10.0, 0.9, 1.4, 1.0, "MLP\nclassifier", "#c7f9cc")

    add_arrow(ax, (2.3, 4.05), (2.8, 4.05))
    add_arrow(ax, (2.3, 1.85), (2.8, 1.85))
    add_arrow(ax, (4.5, 4.05), (5.0, 4.05))
    add_arrow(ax, (4.5, 1.85), (5.0, 1.85))
    add_arrow(ax, (6.9, 4.05), (7.6, 3.15))
    add_arrow(ax, (6.9, 1.85), (7.6, 2.85))
    add_arrow(ax, (9.5, 3.0), (10.0, 3.0))
    add_arrow(ax, (10.7, 2.5), (10.7, 1.9))

    ax.text(
        6.0,
        5.3,
        "RNA Pair Transformer Demo",
        ha="center",
        va="center",
        fontsize=17,
        fontweight="bold",
        color="#111827",
    )
    ax.text(
        6.0,
        4.9,
        "Lightweight dual-encoder architecture with one cross-attention block",
        ha="center",
        va="center",
        fontsize=11,
        color="#4b5563",
    )
    ax.text(
        10.7,
        0.45,
        "Output: interaction logit / probability",
        ha="center",
        va="center",
        fontsize=10,
        color="#374151",
    )

    fig.tight_layout()
    fig.savefig(args.output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.output_path}")


if __name__ == "__main__":
    main()
