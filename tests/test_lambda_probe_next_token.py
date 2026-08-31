from pathlib import Path

import numpy as np
import torch

from nsm_dna.data import encode_sequence
from nsm_dna.models.next_token import NextTokenModel
from scripts.evaluation.lambda_probe_next_token import (
    extract_segment_embeddings,
    fit_linear_probe,
    read_lambda_split,
)


def _build_model() -> NextTokenModel:
    model = NextTokenModel(
        vocab_size=4,
        model_dim=8,
        num_layers=1,
        num_heads=2,
        max_sequence_length=3,
        dropout=0.0,
    )
    model.eval()
    return model


def test_read_lambda_split_preserves_official_fields(tmp_path: Path) -> None:
    split_path = tmp_path / "train.csv"
    split_path.write_text(
        "segment_id,sequence,label,source\n"
        "phage_1,ACGT,1,inphared\n"
        "bacteria_1,TGCA,0,gtdb\n"
        "ambiguous_1,ACGN,1,inphared\n",
        encoding="utf-8",
    )

    split = read_lambda_split(split_path, expected_sequence_length=4)

    assert split.segment_ids == ["phage_1", "bacteria_1"]
    assert split.sequences == ["ACGT", "TGCA"]
    assert split.labels.tolist() == [1, 0]
    assert split.sources == ["inphared", "gtdb"]
    assert split.num_excluded_ambiguous == 1


def test_segment_embeddings_mean_pool_all_native_context_chunks() -> None:
    model = _build_model()
    sequences = ["ACGTACG", "TGCATGC"]
    input_ids = torch.stack([encode_sequence(sequence) for sequence in sequences])

    embeddings = extract_segment_embeddings(
        model,
        sequences,
        batch_size=2,
        device=torch.device("cpu"),
        description="test",
    )

    expected_hidden_states = torch.cat(
        [
            model.encode(input_ids[:, start : start + model.max_sequence_length])
            for start in range(0, input_ids.shape[1], model.max_sequence_length)
        ],
        dim=1,
    )
    np.testing.assert_allclose(
        embeddings,
        expected_hidden_states.mean(dim=1).detach().numpy(),
        rtol=1e-5,
        atol=1e-6,
    )


def test_linear_probe_separates_a_simple_signal() -> None:
    train_embeddings = np.asarray([[-3.0], [-2.0], [2.0], [3.0]])
    train_labels = np.asarray([0, 0, 1, 1])
    test_embeddings = np.asarray([[-4.0], [-1.0], [1.0], [4.0]])
    test_labels = np.asarray([0, 0, 1, 1])

    metrics, predictions, _ = fit_linear_probe(
        train_embeddings,
        train_labels,
        test_embeddings,
        test_labels,
        seed=42,
    )

    assert predictions.tolist() == test_labels.tolist()
    assert metrics["mcc"] == 1.0
