import argparse
import json
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, matthews_corrcoef, roc_auc_score
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from src.data import load_datasets
from src.model import RNAPairTransformer, RNAPairTransformerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the RNA Pair Transformer demo model.")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-encoder-layers", type=int, default=2)
    parser.add_argument("--dim-feedforward", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_run_dir(output_dir: Path, run_name: str | None) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = run_name or f"rna_pair_transformer_{timestamp}"
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def build_dataloaders(
    processed_dir: Path,
    batch_size: int,
    num_workers: int,
) -> dict[str, DataLoader]:
    datasets = load_datasets(processed_dir)
    pin_memory = torch.cuda.is_available()

    return {
        split_name: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        for split_name, dataset in datasets.items()
    }


def compute_binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = (probabilities >= 0.5).astype(int)
    has_both_classes = np.unique(labels).size > 1

    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions),
        "mcc": matthews_corrcoef(labels, predictions),
    }

    try:
        metrics["auprc"] = average_precision_score(labels, probabilities)
    except ValueError:
        metrics["auprc"] = float("nan")

    try:
        if not has_both_classes:
            raise ValueError("ROC-AUC is undefined when only one class is present.")
        metrics["roc_auc"] = roc_auc_score(labels, probabilities)
    except ValueError:
        metrics["roc_auc"] = float("nan")

    return metrics


def run_epoch(
    model: RNAPairTransformer,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    max_batches: int | None = None,
) -> dict[str, float]:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_examples = 0
    all_labels: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []

    for batch_index, batch in enumerate(dataloader):
        if max_batches is not None and batch_index >= max_batches:
            break

        batch = move_batch_to_device(batch, device)
        labels = batch["label"]

        if is_training:
            optimizer.zero_grad()

        outputs = model(
            target_ids=batch["target_ids"],
            target_mask=batch["target_mask"],
            mirna_ids=batch["mirna_ids"],
            mirna_mask=batch["mirna_mask"],
        )
        logits = outputs["logits"]
        loss = criterion(logits, labels)

        if is_training:
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

        all_labels.append(labels.detach().cpu().numpy())
        all_probabilities.append(outputs["probabilities"].detach().cpu().numpy())

    labels_array = np.concatenate(all_labels)
    probabilities_array = np.concatenate(all_probabilities)
    metrics = compute_binary_metrics(labels_array, probabilities_array)
    metrics["loss"] = total_loss / max(total_examples, 1)
    return metrics


def collect_predictions(
    model: RNAPairTransformer,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_labels: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []

    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break

            batch = move_batch_to_device(batch, device)
            outputs = model(
                target_ids=batch["target_ids"],
                target_mask=batch["target_mask"],
                mirna_ids=batch["mirna_ids"],
                mirna_mask=batch["mirna_mask"],
            )
            all_labels.append(batch["label"].detach().cpu().numpy())
            all_probabilities.append(outputs["probabilities"].detach().cpu().numpy())

    labels_array = np.concatenate(all_labels).astype(int)
    probabilities_array = np.concatenate(all_probabilities)
    return labels_array, probabilities_array


def compute_metrics_at_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(int)
    has_both_classes = np.unique(labels).size > 1

    metrics = {
        "threshold": threshold,
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions),
        "mcc": matthews_corrcoef(labels, predictions),
        "auprc": average_precision_score(labels, probabilities),
    }
    if has_both_classes:
        metrics["roc_auc"] = roc_auc_score(labels, probabilities)
    else:
        metrics["roc_auc"] = float("nan")
    metrics["positive_rate"] = float(predictions.mean())
    return metrics


def find_best_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    objective: str = "mcc",
    threshold_min: float = 0.1,
    threshold_max: float = 0.9,
    num_thresholds: int = 161,
) -> tuple[float, dict[str, float]]:
    thresholds = np.linspace(threshold_min, threshold_max, num_thresholds)
    best_threshold = 0.5
    best_metrics = compute_metrics_at_threshold(labels, probabilities, best_threshold)
    best_score = best_metrics[objective]

    for threshold in thresholds:
        metrics = compute_metrics_at_threshold(labels, probabilities, float(threshold))
        score = metrics[objective]
        if score > best_score:
            best_threshold = float(threshold)
            best_metrics = metrics
            best_score = score

    return best_threshold, best_metrics


def evaluate_splits(
    model: RNAPairTransformer,
    dataloaders: dict[str, DataLoader],
    criterion: nn.Module,
    device: torch.device,
    max_eval_batches: int | None = None,
) -> dict[str, dict[str, float]]:
    evaluation_results: dict[str, dict[str, float]] = {}
    model.eval()

    with torch.no_grad():
        for split_name in ["val", "test", "external_test"]:
            evaluation_results[split_name] = run_epoch(
                model=model,
                dataloader=dataloaders[split_name],
                criterion=criterion,
                device=device,
                optimizer=None,
                max_batches=max_eval_batches,
            )

    return evaluation_results


def format_metrics(metrics: dict[str, float]) -> str:
    ordered_keys = ["threshold", "loss", "auprc", "roc_auc", "f1", "mcc", "accuracy", "positive_rate"]
    parts = [f"{key}={metrics[key]:.4f}" for key in ordered_keys if key in metrics]
    return " ".join(parts)


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = get_device()
    run_dir = create_run_dir(args.output_dir, args.run_name)
    dataloaders = build_dataloaders(
        processed_dir=args.processed_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    config = RNAPairTransformerConfig(
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_encoder_layers=args.num_encoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    )
    model = RNAPairTransformer(config).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    print(f"device={device}")
    print(f"run_dir={run_dir}")
    print(f"train_size={len(dataloaders['train'].dataset)} val_size={len(dataloaders['val'].dataset)}")

    history: list[dict[str, float | int]] = []
    best_val_auprc = float("-inf")
    best_epoch = -1
    best_checkpoint_path = run_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            dataloader=dataloaders["train"],
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            max_batches=args.max_train_batches,
        )

        model.eval()
        with torch.no_grad():
            val_metrics = run_epoch(
                model=model,
                dataloader=dataloaders["val"],
                criterion=criterion,
                device=device,
                optimizer=None,
                max_batches=args.max_eval_batches,
            )

        epoch_record: dict[str, float | int] = {"epoch": epoch}
        epoch_record.update({f"train_{key}": value for key, value in train_metrics.items()})
        epoch_record.update({f"val_{key}": value for key, value in val_metrics.items()})
        history.append(epoch_record)

        print(
            f"epoch={epoch:02d} "
            f"train[{format_metrics(train_metrics)}] "
            f"val[{format_metrics(val_metrics)}]"
        )

        if val_metrics["auprc"] > best_val_auprc:
            best_val_auprc = val_metrics["auprc"]
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": asdict(config),
                    "args": vars(args),
                    "val_metrics": val_metrics,
                },
                best_checkpoint_path,
            )

    checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    final_results = evaluate_splits(
        model=model,
        dataloaders=dataloaders,
        criterion=criterion,
        device=device,
        max_eval_batches=args.max_eval_batches,
    )

    val_labels, val_probabilities = collect_predictions(
        model=model,
        dataloader=dataloaders["val"],
        device=device,
        max_batches=args.max_eval_batches,
    )
    selected_threshold, tuned_val_metrics = find_best_threshold(
        labels=val_labels,
        probabilities=val_probabilities,
        objective="mcc",
    )
    tuned_results = {
        "val": tuned_val_metrics,
    }
    for split_name in ["test", "external_test"]:
        labels, probabilities = collect_predictions(
            model=model,
            dataloader=dataloaders[split_name],
            device=device,
            max_batches=args.max_eval_batches,
        )
        tuned_results[split_name] = compute_metrics_at_threshold(
            labels=labels,
            probabilities=probabilities,
            threshold=selected_threshold,
        )

    print(f"best_epoch={best_epoch} best_val_auprc={best_val_auprc:.4f}")
    print(f"test[{format_metrics(final_results['test'])}]")
    print(f"external_test[{format_metrics(final_results['external_test'])}]")
    print(f"selected_threshold={selected_threshold:.4f} objective=mcc")
    print(f"tuned_val[{format_metrics(tuned_results['val'])}]")
    print(f"tuned_test[{format_metrics(tuned_results['test'])}]")
    print(f"tuned_external_test[{format_metrics(tuned_results['external_test'])}]")

    history_df = pd.DataFrame(history)
    history_df.to_csv(run_dir / "history.csv", index=False)
    save_json(
        run_dir / "metrics.json",
        {
            "best_epoch": best_epoch,
            "best_val_auprc": best_val_auprc,
            "selected_threshold": selected_threshold,
            "val": final_results["val"],
            "test": final_results["test"],
            "external_test": final_results["external_test"],
            "tuned_val": tuned_results["val"],
            "tuned_test": tuned_results["test"],
            "tuned_external_test": tuned_results["external_test"],
        },
    )
    save_json(
        run_dir / "config.json",
        {
            "model_config": asdict(config),
            "train_args": vars(args),
        },
    )


if __name__ == "__main__":
    main()
