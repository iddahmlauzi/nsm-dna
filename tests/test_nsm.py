import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from nsm_dna.models.common import RMSNorm
from nsm_dna.models.next_scale import MultiscaleOutputHead, NSM


def _build_model(
    *,
    num_layers: int = 1,
    max_prefix_length: int = 0,
) -> NSM:
    return NSM(
        prefix_dim=3,
        model_dim=8,
        scale_lengths=[1, 2, 4],
        codebook_sizes=[4, 5, 6],
        codebook_vectors=[torch.randn(4, 3), torch.randn(5, 3), torch.randn(6, 3)],
        num_layers=num_layers,
        num_heads=2,
        dropout=0.0,
        max_prefix_length=max_prefix_length,
    )


def _indices() -> list[torch.Tensor]:
    return [
        torch.tensor([[1], [2]]),
        torch.tensor([[2, 3], [3, 4]]),
        torch.tensor([[3, 4, 5, 0], [4, 5, 0, 1]]),
    ]


def test_nsm_from_checkpoint_restores_model_and_step(tmp_path: Path) -> None:
    codebook_vectors = [torch.randn(4, 3), torch.randn(5, 3)]
    tokenizer = SimpleNamespace(
        quantization_dim=3,
        scale_lengths=[1, 2],
        codebook_sizes=[4, 5],
        context_length=4,
        quantizer=SimpleNamespace(
            codebooks=[
                SimpleNamespace(codebook=vectors) for vectors in codebook_vectors
            ]
        ),
    )
    model = NSM(
        prefix_dim=tokenizer.quantization_dim,
        model_dim=8,
        scale_lengths=tokenizer.scale_lengths,
        codebook_sizes=tokenizer.codebook_sizes,
        codebook_vectors=codebook_vectors,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        max_prefix_length=2,
    )
    checkpoint_path = tmp_path / "nsm.pt"
    torch.save(
        {
            "step": 17,
            "model": model.state_dict(),
            "config": {
                "data": {"sequence_length": 6},
                "model": {
                    "model_dim": 8,
                    "num_layers": 1,
                    "num_heads": 2,
                    "dropout": 0.0,
                    "bias": False,
                    "use_qk_norm": True,
                    "rope_base": 10000.0,
                    "head_num_blocks": 2,
                    "head_hidden_multiplier": 2.0,
                },
            },
        },
        checkpoint_path,
    )

    restored_model, checkpoint_step = NSM.from_checkpoint(
        checkpoint_path,
        tokenizer,
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


def test_multiscale_output_head_starts_as_linear_classifiers() -> None:
    head = MultiscaleOutputHead(
        model_dim=4,
        scale_lengths=[2, 3],
        codebook_sizes=[5, 7],
        num_blocks=2,
        hidden_multiplier=2.0,
        dropout=0.0,
    )
    x = torch.randn(2, 5, 4)

    logits = head(x)
    expected_logits = [
        projection(hidden_states)
        for projection, hidden_states in zip(
            head.codebook_projections,
            torch.split(x, [2, 3], dim=1),
            strict=True,
        )
    ]

    assert [value.shape for value in logits] == [(2, 2, 5), (2, 3, 7)]
    for actual, expected in zip(logits, expected_logits, strict=True):
        torch.testing.assert_close(actual, expected)


def test_hierarchy_inputs_are_cumulative_and_shifted_within_each_scale() -> None:
    model = _build_model(num_layers=0)
    indices_by_scale = _indices()

    inputs = model._embed_hierarchy_inputs(indices_by_scale)
    first_codes = model.codebook_vectors_0[indices_by_scale[0]]
    second_codes = model.codebook_vectors_1[indices_by_scale[1]]
    third_codes = model.codebook_vectors_2[indices_by_scale[2]]
    expected_inputs = torch.cat(
        [
            torch.zeros(2, 1, 3),
            first_codes.repeat_interleave(2, dim=1)
            + torch.cat(
                [torch.zeros_like(second_codes[:, :1]), second_codes[:, :1]],
                dim=1,
            ),
            first_codes.repeat_interleave(4, dim=1)
            + second_codes.repeat_interleave(2, dim=1)
            + torch.cat(
                [torch.zeros_like(third_codes[:, :1]), third_codes[:, :-1]],
                dim=1,
            ),
        ],
        dim=1,
    )
    torch.testing.assert_close(inputs, model.input_projection(expected_inputs))


def test_hierarchy_attention_is_causal() -> None:
    model = NSM(
        prefix_dim=3,
        model_dim=8,
        scale_lengths=[1, 2],
        codebook_sizes=[4, 5],
        codebook_vectors=[torch.randn(4, 3), torch.randn(5, 3)],
        num_layers=1,
        num_heads=2,
        dropout=0.0,
    )

    expected_mask = torch.tensor(
        [
            [True, False, False],
            [False, True, False],
            [False, True, True],
        ]
    ).reshape(1, 1, 3, 3)

    torch.testing.assert_close(model.hierarchy_attention_mask, expected_mask)


def test_prefix_is_visible_to_every_causal_hierarchy_position() -> None:
    model = NSM(
        prefix_dim=3,
        model_dim=8,
        scale_lengths=[1, 2],
        codebook_sizes=[4, 5],
        codebook_vectors=[torch.randn(4, 3), torch.randn(5, 3)],
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        max_prefix_length=2,
    )

    expected_mask = torch.tensor(
        [
            [True, True, False, False, False],
            [True, True, False, False, False],
            [True, True, True, False, False],
            [True, True, False, True, False],
            [True, True, False, True, True],
        ]
    ).reshape(1, 1, 5, 5)

    torch.testing.assert_close(model._build_attention_mask(2), expected_mask)


def test_scale_ids_match_hierarchy_sections() -> None:
    model = _build_model()
    torch.testing.assert_close(model.scale_ids, torch.tensor([0, 1, 1, 2, 2, 2, 2]))


def test_rope_positions_restart_at_each_scale() -> None:
    model = _build_model()

    cosine, sine = model._get_rotary_embeddings(prefix_length=0)
    torch.testing.assert_close(cosine, model.rope_cosine)
    torch.testing.assert_close(sine, model.rope_sine)
    torch.testing.assert_close(model.rope_cosine[0], model.rope_cosine[1])
    torch.testing.assert_close(model.rope_cosine[1], model.rope_cosine[3])
    assert not torch.equal(model.rope_cosine[1], model.rope_cosine[2])


def test_prefix_rope_precedes_hierarchy_positions() -> None:
    model = _build_model(max_prefix_length=2)
    cosine, sine = model._get_rotary_embeddings(prefix_length=2)

    torch.testing.assert_close(cosine[:2], model.prefix_rope_cosine)
    torch.testing.assert_close(sine[:2], model.prefix_rope_sine)
    torch.testing.assert_close(cosine[2:], model.rope_cosine)
    torch.testing.assert_close(sine[2:], model.rope_sine)


def test_nsm_uses_rms_norm_and_qk_norm() -> None:
    model = _build_model()

    assert isinstance(model.blocks[0].attn.q_norm, RMSNorm)
    assert isinstance(model.blocks[0].attn.k_norm, RMSNorm)
    assert isinstance(model.blocks[0].attn_norm, RMSNorm)
    assert isinstance(model.blocks[0].mlp_norm, RMSNorm)
    assert isinstance(model.final_norm, RMSNorm)
    assert model.final_norm.eps == pytest.approx(1e-5)


def test_nsm_scales_residual_projection_initialization() -> None:
    num_layers = 4
    model = NSM(
        prefix_dim=8,
        model_dim=64,
        scale_lengths=[1, 2, 4],
        codebook_sizes=[8, 8, 8],
        codebook_vectors=[torch.randn(8, 8) for _ in range(3)],
        num_layers=num_layers,
        num_heads=4,
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


def test_nsm_predicts_every_codebook_including_scale_one() -> None:
    model = _build_model(num_layers=0)
    indices_by_scale = _indices()

    logits = model(indices_by_scale)
    hierarchy_inputs = model._embed_hierarchy_inputs(indices_by_scale)
    expected_hidden_states = model.final_norm(
        hierarchy_inputs + model.scale_embedding(model.scale_ids)
    )
    expected_logits = model.output_head(expected_hidden_states)

    assert [value.shape for value in logits] == [
        (2, 1, 4),
        (2, 2, 5),
        (2, 4, 6),
    ]
    for actual, expected in zip(logits, expected_logits, strict=True):
        torch.testing.assert_close(actual, expected)


def test_nsm_returns_prefix_and_hierarchy_hidden_states() -> None:
    model = _build_model(num_layers=0, max_prefix_length=4)
    prefix = torch.randn(2, 4, 3)
    indices_by_scale = _indices()

    hidden_states = model.encode(indices_by_scale, prefix=prefix)
    logits = model(indices_by_scale, prefix=prefix)

    assert hidden_states.shape == (2, 11, 8)
    assert [value.shape for value in logits] == [
        (2, 1, 4),
        (2, 2, 5),
        (2, 4, 6),
    ]
    expected_logits = model.output_head(hidden_states[:, 4:])
    for actual, expected in zip(logits, expected_logits, strict=True):
        torch.testing.assert_close(actual, expected)


def test_current_code_cannot_change_its_own_or_earlier_logits() -> None:
    torch.manual_seed(0)
    model = _build_model(num_layers=2).eval()
    indices_by_scale = _indices()
    changed_indices = [indices.clone() for indices in indices_by_scale]
    changed_indices[1][:, 0] = (changed_indices[1][:, 0] + 1) % 5

    logits = model(indices_by_scale)
    changed_logits = model(changed_indices)

    # A code first enters its own scale at the next position.
    torch.testing.assert_close(logits[0], changed_logits[0])
    torch.testing.assert_close(logits[1][:, :1], changed_logits[1][:, :1])
    assert not torch.equal(logits[1][:, 1], changed_logits[1][:, 1])


def test_first_scale_is_predicted_without_supplying_its_code() -> None:
    torch.manual_seed(0)
    model = _build_model(num_layers=1).eval()
    indices_by_scale = _indices()
    changed_indices = [indices.clone() for indices in indices_by_scale]
    changed_indices[0][:, 0] = (changed_indices[0][:, 0] + 1) % 4

    logits = model(indices_by_scale)
    changed_logits = model(changed_indices)

    torch.testing.assert_close(logits[0], changed_logits[0])
    assert not torch.equal(logits[1], changed_logits[1])
