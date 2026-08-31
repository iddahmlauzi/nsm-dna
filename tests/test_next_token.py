import math
from pathlib import Path

import pytest
import torch

from nsm_dna.models.common import RMSNorm
from nsm_dna.models.next_token import NextTokenModel


def _build_model(
    *,
    num_layers: int = 1,
    max_sequence_length: int = 8,
) -> NextTokenModel:
    return NextTokenModel(
        vocab_size=4,
        model_dim=8,
        num_layers=num_layers,
        num_heads=2,
        max_sequence_length=max_sequence_length,
        dropout=0.0,
    )


def test_next_token_model_returns_one_prediction_per_context_position() -> None:
    model = _build_model()
    input_ids = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])

    logits = model(input_ids)

    assert logits.shape == (2, 4, 4)


def test_next_token_model_encode_returns_the_states_used_for_prediction() -> None:
    model = _build_model()
    input_ids = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])

    hidden_states = model.encode(input_ids)

    assert hidden_states.shape == (2, 4, model.model_dim)
    torch.testing.assert_close(model(input_ids), model.output_projection(hidden_states))


def test_next_token_model_uses_causal_attention() -> None:
    model = _build_model()
    model.eval()
    input_ids = torch.tensor([[0, 1, 2, 3]])
    changed_input_ids = torch.tensor([[0, 2, 3, 3]])

    logits = model(input_ids)
    changed_logits = model(changed_input_ids)

    # Changing position one cannot affect the prediction made at position zero.
    torch.testing.assert_close(logits[:, :1], changed_logits[:, :1])


def test_next_token_model_ties_input_and_output_embeddings() -> None:
    model = _build_model()

    assert model.output_projection.weight is model.token_embedding.weight


def test_next_token_model_normalizes_queries_and_keys() -> None:
    model = _build_model()

    assert isinstance(model.blocks[0].attn.q_norm, RMSNorm)
    assert isinstance(model.blocks[0].attn.k_norm, RMSNorm)


def test_next_token_model_uses_pre_rms_norm() -> None:
    model = _build_model()

    assert isinstance(model.blocks[0].attn_norm, RMSNorm)
    assert isinstance(model.blocks[0].mlp_norm, RMSNorm)
    assert isinstance(model.final_norm, RMSNorm)
    assert model.blocks[0].attn_norm.eps == pytest.approx(1e-5)
    assert model.blocks[0].mlp_norm.eps == pytest.approx(1e-5)
    assert model.final_norm.eps == pytest.approx(1e-5)


def test_next_token_model_scales_residual_projection_initialization() -> None:
    num_layers = 4
    model = NextTokenModel(
        vocab_size=4,
        model_dim=64,
        num_layers=num_layers,
        num_heads=4,
        max_sequence_length=8,
        dropout=0.0,
    )
    expected_standard_deviation = 0.02 / math.sqrt(2 * num_layers)

    for block in model.blocks:
        assert block.attn.out_proj.weight.std().item() == pytest.approx(
            expected_standard_deviation,
            rel=0.05,
        )
        assert block.mlp.down_proj.weight.std().item() == pytest.approx(
            expected_standard_deviation,
            rel=0.05,
        )


def test_next_token_model_rejects_sequences_above_its_maximum() -> None:
    model = _build_model(max_sequence_length=3)

    with pytest.raises(ValueError, match="Sequence length 4"):
        model(torch.tensor([[0, 1, 2, 3]]))


def test_next_token_model_from_checkpoint_restores_model_and_step(
    tmp_path: Path,
) -> None:
    model = _build_model(max_sequence_length=3)
    checkpoint_path = tmp_path / "next_token.pt"
    torch.save(
        {
            "step": 17,
            "model": model.state_dict(),
            "config": {
                "data": {"sequence_length": 4},
                "model": {
                    "vocab_size": 4,
                    "model_dim": 8,
                    "num_layers": 1,
                    "num_heads": 2,
                    "dropout": 0.0,
                    "bias": False,
                    "use_qk_norm": True,
                    "rope_base": 10000.0,
                },
            },
        },
        checkpoint_path,
    )

    restored_model, checkpoint_step = NextTokenModel.from_checkpoint(
        checkpoint_path,
        torch.device("cpu"),
        frozen=True,
    )

    assert checkpoint_step == 17
    assert not restored_model.training
    assert all(not parameter.requires_grad for parameter in restored_model.parameters())
    for parameter, restored_parameter in zip(
        model.parameters(),
        restored_model.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(restored_parameter, parameter)
