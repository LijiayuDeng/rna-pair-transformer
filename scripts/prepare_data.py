from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

HEJRET_TRAIN = RAW_DIR / "AGO2_CLASH_Hejret2023_train.tsv"
HEJRET_TEST = RAW_DIR / "AGO2_CLASH_Hejret2023_test.tsv"
KLIMENTOVA_TEST = RAW_DIR / "AGO2_eCLIP_Klimentova2022_test.tsv"
VAL_SIZE = 0.2


def load_raw_tsv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")

    df = df[["gene", "noncodingRNA", "label"]].copy()
    df = df.rename(
        columns={
            "gene": "target_seq",
            "noncodingRNA": "mirna_seq",
        }
    )

    df["target_seq"] = df["target_seq"].str.upper().str.replace("T", "U", regex=False)
    df["mirna_seq"] = df["mirna_seq"].str.upper().str.replace("T", "U", regex=False)
    df["label"] = df["label"].astype(int)

    return df


def remove_external_overlaps(
    train_df: pd.DataFrame, external_df: pd.DataFrame
) -> pd.DataFrame:
    overlap_keys = set(
        zip(
            external_df["target_seq"],
            external_df["mirna_seq"],
            external_df["label"],
        )
    )

    keep_mask = [
        (target, mirna, label) not in overlap_keys
        for target, mirna, label in zip(
            train_df["target_seq"],
            train_df["mirna_seq"],
            train_df["label"],
        )
    ]

    filtered_df = train_df.loc[keep_mask].reset_index(drop=True)
    return filtered_df


def print_summary(name: str, df: pd.DataFrame) -> None:
    label_counts = df["label"].value_counts().sort_index().to_dict()
    print(f"{name}: n={len(df)} labels={label_counts}")


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train_df = load_raw_tsv(HEJRET_TRAIN)
    test_df = load_raw_tsv(HEJRET_TEST)
    external_test_df = load_raw_tsv(KLIMENTOVA_TEST)

    print_summary("raw_train", train_df)
    print_summary("raw_test", test_df)
    print_summary("raw_external_test", external_test_df)

    train_df = remove_external_overlaps(train_df, external_test_df)
    print_summary("train_after_overlap_removal", train_df)

    train_df, val_df = train_test_split(
        train_df,
        test_size=VAL_SIZE,
        random_state=42,
        stratify=train_df["label"],
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    print_summary("train", train_df)
    print_summary("val", val_df)
    print_summary("test", test_df)
    print_summary("external_test", external_test_df)

    train_df.to_csv(PROCESSED_DIR / "train.tsv", sep="\t", index=False)
    val_df.to_csv(PROCESSED_DIR / "val.tsv", sep="\t", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.tsv", sep="\t", index=False)
    external_test_df.to_csv(PROCESSED_DIR / "external_test.tsv", sep="\t", index=False)

    print(f"Saved processed files to: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
