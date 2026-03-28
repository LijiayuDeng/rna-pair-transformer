from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset


PAD_TOKEN = "<pad>"
PAD_ID = 0
TOKEN_TO_ID = {
    PAD_TOKEN: PAD_ID,
    "A": 1,
    "C": 2,
    "G": 3,
    "U": 4,
    "N": 5,
}
ID_TO_TOKEN = {idx: token for token, idx in TOKEN_TO_ID.items()}

DEFAULT_MAX_TARGET_LEN = 50
DEFAULT_MAX_MIRNA_LEN = 30
REQUIRED_COLUMNS = ["target_seq", "mirna_seq", "label"]


def normalize_sequence(sequence: str) -> str:
    sequence = sequence.upper().replace("T", "U")
    return "".join(base if base in {"A", "C", "G", "U", "N"} else "N" for base in sequence)


def encode_sequence(
    sequence: str,
    max_len: int,
    token_to_id: dict[str, int] | None = None,
) -> tuple[list[int], list[int]]:
    token_to_id = token_to_id or TOKEN_TO_ID
    normalized = normalize_sequence(sequence)

    token_ids = [token_to_id.get(base, token_to_id["N"]) for base in normalized[:max_len]]
    attention_mask = [1] * len(token_ids)

    pad_length = max_len - len(token_ids)
    if pad_length > 0:
        token_ids.extend([token_to_id[PAD_TOKEN]] * pad_length)
        attention_mask.extend([0] * pad_length)

    return token_ids, attention_mask


def decode_sequence(token_ids: list[int]) -> str:
    tokens = [ID_TO_TOKEN[token_id] for token_id in token_ids if token_id != PAD_ID]
    return "".join(tokens)


def load_processed_dataframe(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path, sep="\t")

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required columns in {path}: {missing}")

    df = df[REQUIRED_COLUMNS].copy()
    df["target_seq"] = df["target_seq"].map(normalize_sequence)
    df["mirna_seq"] = df["mirna_seq"].map(normalize_sequence)
    df["label"] = df["label"].astype(float)

    return df


class RNAPairDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        max_target_len: int = DEFAULT_MAX_TARGET_LEN,
        max_mirna_len: int = DEFAULT_MAX_MIRNA_LEN,
    ) -> None:
        self.path = Path(path)
        self.max_target_len = max_target_len
        self.max_mirna_len = max_mirna_len
        self.dataframe = load_processed_dataframe(self.path)

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.dataframe.iloc[index]

        target_ids, target_mask = encode_sequence(row["target_seq"], self.max_target_len)
        mirna_ids, mirna_mask = encode_sequence(row["mirna_seq"], self.max_mirna_len)

        return {
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
            "target_mask": torch.tensor(target_mask, dtype=torch.bool),
            "mirna_ids": torch.tensor(mirna_ids, dtype=torch.long),
            "mirna_mask": torch.tensor(mirna_mask, dtype=torch.bool),
            "label": torch.tensor(row["label"], dtype=torch.float32),
        }


def build_split_paths(processed_dir: str | Path) -> dict[str, Path]:
    processed_dir = Path(processed_dir)
    return {
        "train": processed_dir / "train.tsv",
        "val": processed_dir / "val.tsv",
        "test": processed_dir / "test.tsv",
        "external_test": processed_dir / "external_test.tsv",
    }


def load_datasets(
    processed_dir: str | Path,
    max_target_len: int = DEFAULT_MAX_TARGET_LEN,
    max_mirna_len: int = DEFAULT_MAX_MIRNA_LEN,
) -> dict[str, RNAPairDataset]:
    split_paths = build_split_paths(processed_dir)
    return {
        split_name: RNAPairDataset(
            path,
            max_target_len=max_target_len,
            max_mirna_len=max_mirna_len,
        )
        for split_name, path in split_paths.items()
    }
