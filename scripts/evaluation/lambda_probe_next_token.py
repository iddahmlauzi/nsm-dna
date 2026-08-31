import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from nsm_dna.data import encode_sequence
from nsm_dna.models.next_token import NextTokenModel

DNA_BASES = frozenset("ACGT")


@dataclass(frozen=True)
class LambdaSplit:
    segment_ids: list[str]
    sequences: list[str]
    labels: np.ndarray
    sources: list[str]
    num_excluded_ambiguous: int


def sha256(path: Path) -> str:
    """Calculate the SHA-256 digest of one input artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_lambda_split(path: Path, expected_sequence_length: int) -> LambdaSplit:
    """Read one official LAMBDA binary-classification split."""
    with path.open(encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))

    sequences = [row["sequence"].upper() for row in rows]
    unexpected_lengths = sorted(
        {len(sequence) for sequence in sequences if len(sequence) != expected_sequence_length}
    )
    if unexpected_lengths:
        raise ValueError(
            f"{path} contains sequence lengths other than "
            f"{expected_sequence_length}: {unexpected_lengths}"
        )

    supported_rows = [
        row
        for row, sequence in zip(rows, sequences, strict=True)
        if set(sequence) <= DNA_BASES
    ]
    num_excluded_ambiguous = len(rows) - len(supported_rows)
    sequences = [row["sequence"].upper() for row in supported_rows]

    labels = np.asarray(
        [int(row["label"]) for row in supported_rows],
        dtype=np.int64,
    )
    if not set(labels.tolist()).issubset({0, 1}):
        raise ValueError(f"{path} contains labels other than 0 and 1.")

    return LambdaSplit(
        segment_ids=[row["segment_id"] for row in supported_rows],
        sequences=sequences,
        labels=labels,
        sources=[row["source"] for row in supported_rows],
        num_excluded_ambiguous=num_excluded_ambiguous,
    )


@torch.inference_mode()
def extract_segment_embeddings(
    model: NextTokenModel,
    sequences: list[str],
    batch_size: int,
    device: torch.device,
    *,
    description: str,
) -> np.ndarray:
    """Mean-pool hidden states over native-context chunks of each segment."""
    embeddings = np.empty((len(sequences), model.model_dim), dtype=np.float32)

    for start in tqdm(
        range(0, len(sequences), batch_size),
        desc=description,
        unit="batch",
    ):
        batch_sequences = sequences[start : start + batch_size]
        batch_ids = torch.stack(
            [encode_sequence(sequence) for sequence in batch_sequences]
        )
        embedding_sum = torch.zeros(
            (len(batch_sequences), model.model_dim),
            dtype=torch.float32,
            device=device,
        )

        for chunk_ids in batch_ids.split(model.max_sequence_length, dim=1):
            chunk_ids = chunk_ids.to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                hidden_states = model.encode(chunk_ids)
            embedding_sum += hidden_states.float().sum(dim=1)

        embeddings[start : start + len(batch_sequences)] = (
            embedding_sum.div(batch_ids.shape[1]).cpu().numpy()
        )

    return embeddings


def calculate_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float | int]:
    """Calculate the classification metrics reported by LAMBDA."""
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "auc": float(roc_auc_score(labels, probabilities)),
        "sensitivity": (
            float(true_positive / (true_positive + false_negative))
            if true_positive + false_negative
            else 0.0
        ),
        "specificity": (
            float(true_negative / (true_negative + false_positive))
            if true_negative + false_positive
            else 0.0
        ),
        "true_positive": int(true_positive),
        "true_negative": int(true_negative),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
    }


def fit_linear_probe(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    seed: int,
) -> tuple[dict[str, float | int], np.ndarray, np.ndarray]:
    """Fit the StandardScaler and logistic-regression probe used by LAMBDA."""
    scaler = StandardScaler()
    scaled_train_embeddings = scaler.fit_transform(train_embeddings)
    scaled_test_embeddings = scaler.transform(test_embeddings)

    classifier = LogisticRegression(
        max_iter=1_000,
        random_state=seed,
        solver="lbfgs",
    )
    classifier.fit(scaled_train_embeddings, train_labels)
    probabilities = classifier.predict_proba(scaled_test_embeddings)[:, 1]
    predictions = classifier.predict(scaled_test_embeddings)
    return (
        calculate_metrics(test_labels, predictions, probabilities),
        predictions,
        probabilities,
    )


class ThreeLayerProbe(nn.Module):
    """The nonlinear probe used by the official LAMBDA evaluation."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.network(embeddings)


def fit_three_layer_probe(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    validation_embeddings: np.ndarray,
    validation_labels: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    config: DictConfig,
    device: torch.device,
) -> tuple[dict[str, float | int], np.ndarray, np.ndarray, int]:
    """Fit the official nonlinear probe with validation-F1 early stopping."""
    seed = int(config.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_embeddings).astype(np.float32)
    scaled_validation = scaler.transform(validation_embeddings).astype(np.float32)
    scaled_test = scaler.transform(test_embeddings).astype(np.float32)

    train_features = torch.from_numpy(scaled_train)
    train_targets = torch.from_numpy(train_labels)
    validation_features = torch.from_numpy(scaled_validation).to(device)
    test_features = torch.from_numpy(scaled_test).to(device)

    train_loader = DataLoader(
        TensorDataset(train_features, train_targets),
        batch_size=int(config.batch_size),
        shuffle=True,
        pin_memory=device.type == "cuda",
    )
    model = ThreeLayerProbe(
        input_dim=train_embeddings.shape[1],
        hidden_dim=int(config.hidden_dim),
        dropout=float(config.dropout),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=int(config.scheduler_patience),
    )
    loss_function = nn.CrossEntropyLoss()

    best_validation_f1 = -1.0
    best_model_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    completed_epochs = 0

    for epoch in range(int(config.max_epochs)):
        model.train()
        for features, targets in train_loader:
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(features), targets)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.inference_mode():
            validation_predictions = model(validation_features).argmax(dim=1).cpu().numpy()
        validation_f1 = float(
            f1_score(validation_labels, validation_predictions, zero_division=0)
        )
        scheduler.step(validation_f1)
        completed_epochs = epoch + 1

        if validation_f1 > best_validation_f1:
            best_validation_f1 = validation_f1
            best_model_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= int(config.early_stopping_patience):
            break

    assert best_model_state is not None
    model.load_state_dict(best_model_state)
    model.eval()
    with torch.inference_mode():
        logits = model(test_features)
        probabilities = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        predictions = logits.argmax(dim=1).cpu().numpy()

    return (
        calculate_metrics(test_labels, predictions, probabilities),
        predictions,
        probabilities,
        completed_epochs,
    )


def write_predictions(
    path: Path,
    split: LambdaSplit,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    """Write test-set predictions without copying the long DNA sequences."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=("segment_id", "source", "label", "probability", "prediction"),
        )
        writer.writeheader()
        for segment_id, source, label, probability, prediction in zip(
            split.segment_ids,
            split.sources,
            split.labels,
            probabilities,
            predictions,
            strict=True,
        ):
            writer.writerow(
                {
                    "segment_id": segment_id,
                    "source": source,
                    "label": int(label),
                    "probability": float(probability),
                    "prediction": int(prediction),
                }
            )


def evaluate_representation(
    name: str,
    model: NextTokenModel,
    splits: dict[str, LambdaSplit],
    config: DictConfig,
    output_directory: Path,
    device: torch.device,
) -> dict[str, object]:
    """Extract one representation and run both official LAMBDA probes."""
    embeddings = {
        split_name: extract_segment_embeddings(
            model,
            split.sequences,
            int(config.embedding_batch_size),
            device,
            description=f"{name} {split_name}",
        )
        for split_name, split in splits.items()
    }

    linear_metrics, linear_predictions, linear_probabilities = fit_linear_probe(
        embeddings["train"],
        splits["train"].labels,
        embeddings["test"],
        splits["test"].labels,
        int(config.probe.seed),
    )
    nn_metrics, nn_predictions, nn_probabilities, completed_epochs = (
        fit_three_layer_probe(
            embeddings["train"],
            splits["train"].labels,
            embeddings["validation"],
            splits["validation"].labels,
            embeddings["test"],
            splits["test"].labels,
            config.probe,
            device,
        )
    )

    write_predictions(
        output_directory / "predictions" / f"{name}_linear.csv",
        splits["test"],
        linear_predictions,
        linear_probabilities,
    )
    write_predictions(
        output_directory / "predictions" / f"{name}_three_layer.csv",
        splits["test"],
        nn_predictions,
        nn_probabilities,
    )
    return {
        "linear_probe": linear_metrics,
        "three_layer_probe": {
            **nn_metrics,
            "completed_epochs": completed_epochs,
        },
    }


@hydra.main(
    version_base=None,
    config_path="../../configs/evaluation",
    config_name="lambda_probe_next_token",
)
def main(config: DictConfig) -> None:
    """Compare trained and random next-token representations on LAMBDA."""
    device = torch.device(config.device)
    dataset_directory = Path(config.dataset_directory)
    checkpoint_path = Path(config.checkpoint)
    output_directory = Path(config.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    split_paths = {
        "train": dataset_directory / "train.csv",
        "validation": dataset_directory / "dev.csv",
        "test": dataset_directory / "test.csv",
    }
    splits = {
        name: read_lambda_split(path, int(config.expected_sequence_length))
        for name, path in split_paths.items()
    }

    trained_model, checkpoint_step = NextTokenModel.from_checkpoint(
        checkpoint_path,
        device,
        frozen=True,
    )
    parameter_count = sum(parameter.numel() for parameter in trained_model.parameters())
    trained_results = evaluate_representation(
        "trained",
        trained_model,
        splits,
        config,
        output_directory,
        device,
    )
    del trained_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_config = OmegaConf.create(checkpoint["config"])
    random_seed = int(config.random_model_seed)
    torch.manual_seed(random_seed)
    random_model = NextTokenModel.from_config(model_config).to(device)
    random_model.eval()
    random_model.requires_grad_(False)
    random_results = evaluate_representation(
        f"random_seed_{random_seed}",
        random_model,
        splits,
        config,
        output_directory,
        device,
    )

    results = {
        "trained": trained_results,
        f"random_seed_{random_seed}": random_results,
        "delta_mcc": {
            "linear_probe": (
                trained_results["linear_probe"]["mcc"]
                - random_results["linear_probe"]["mcc"]
            ),
            "three_layer_probe": (
                trained_results["three_layer_probe"]["mcc"]
                - random_results["three_layer_probe"]["mcc"]
            ),
        },
    }
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_step": checkpoint_step,
        "parameter_count": parameter_count,
        "dataset": "LAMBDA binary_segments_2k",
        "dataset_files": {
            name: {
                "path": str(path),
                "sha256": sha256(path),
                "num_sequences": len(splits[name].sequences),
                "num_excluded_ambiguous": splits[name].num_excluded_ambiguous,
            }
            for name, path in split_paths.items()
        },
        "representation": (
            "mean of final normalized nucleotide hidden states across "
            f"non-overlapping native-context chunks of at most "
            f"{random_model.max_sequence_length} bases"
        ),
        "embedding_batch_size": int(config.embedding_batch_size),
        "random_model_seed": random_seed,
        "probe_config": OmegaConf.to_container(config.probe, resolve=True),
        "results": results,
    }
    (output_directory / "results.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
