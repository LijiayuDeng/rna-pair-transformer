import argparse
import json
from pathlib import Path

import torch

from src.data import DEFAULT_MAX_MIRNA_LEN, DEFAULT_MAX_TARGET_LEN, encode_sequence, normalize_sequence
from src.model import RNAPairTransformer, RNAPairTransformerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-example prediction with the RNA Pair Transformer.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--target-seq", type=str, required=True)
    parser.add_argument("--mirna-seq", type=str, required=True)
    parser.add_argument("--return-attention", action="store_true")
    return parser.parse_args()


def build_inputs(
    target_seq: str,
    mirna_seq: str,
    max_target_len: int,
    max_mirna_len: int,
) -> dict[str, torch.Tensor]:
    target_ids, target_mask = encode_sequence(target_seq, max_target_len)
    mirna_ids, mirna_mask = encode_sequence(mirna_seq, max_mirna_len)

    return {
        "target_ids": torch.tensor([target_ids], dtype=torch.long),
        "target_mask": torch.tensor([target_mask], dtype=torch.bool),
        "mirna_ids": torch.tensor([mirna_ids], dtype=torch.long),
        "mirna_mask": torch.tensor([mirna_mask], dtype=torch.bool),
    }


def main() -> None:
    args = parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = RNAPairTransformerConfig(**checkpoint["config"])
    model = RNAPairTransformer(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    target_seq = normalize_sequence(args.target_seq)
    mirna_seq = normalize_sequence(args.mirna_seq)

    inputs = build_inputs(
        target_seq=target_seq,
        mirna_seq=mirna_seq,
        max_target_len=config.max_target_len or DEFAULT_MAX_TARGET_LEN,
        max_mirna_len=config.max_mirna_len or DEFAULT_MAX_MIRNA_LEN,
    )

    with torch.no_grad():
        outputs = model(
            target_ids=inputs["target_ids"],
            target_mask=inputs["target_mask"],
            mirna_ids=inputs["mirna_ids"],
            mirna_mask=inputs["mirna_mask"],
            return_attention=args.return_attention,
        )

    prediction = {
        "target_seq": target_seq,
        "mirna_seq": mirna_seq,
        "logit": float(outputs["logits"][0].item()),
        "probability": float(outputs["probabilities"][0].item()),
    }
    if args.return_attention and "attention_weights" in outputs:
        prediction["attention_shape"] = list(outputs["attention_weights"].shape)

    print(json.dumps(prediction, indent=2))


if __name__ == "__main__":
    main()
