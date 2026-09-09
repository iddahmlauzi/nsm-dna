import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from nsm_dna.models.common import RMSNorm
from nsm_dna.models.next_scale import NSM, SharedOutputHead


def test_nsm_from_checkpoint_restores_model_and_step(tmp_path: Path) -> None:
    tokenizer = SimpleNamespace(
        embed_dim=3,
        scale_lengths=[1, 2],
        codebook_sizes=[5, 5],
        context_length=4,
    )
    model = NSM(
        vq_embed_dim=tokenizer.embed_dim,
        model_dim=8,
        scale_lengths=tokenizer.scale_lengths,
        codebook_size=tokenizer.codebook_sizes[0],
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
                    "input_refinement_kernel_size": 3,
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


def test_shared_output_head_starts_as_a_linear_classifier() -> None:
    head = SharedOutputHead(
        model_dim=4,
        codebook_size=7,
        num_blocks=2,
        hidden_multiplier=2.0,
        dropout=0.0,
    )
    x = torch.randn(2, 3, 4)

    logits = head(x)
    expected_logits = head.codebook_projection(x)

    torch.testing.assert_close(logits, expected_logits)


def test_scale_input_refinement_starts_as_identity() -> None:
    model = NSM(
        vq_embed_dim=3,
        model_dim=8,
        scale_lengths=[1, 2, 3],
        codebook_size=5,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
    )
    x = [torch.randn(2, 2, 3), torch.randn(2, 3, 3)]

    refined_inputs = model._refine_scale_inputs(x)

    torch.testing.assert_close(refined_inputs[0], x[0])
    torch.testing.assert_close(refined_inputs[1], x[1])

    with torch.no_grad():
        model.scale_input_convs[0].weight[:, :, 1].copy_(torch.eye(3))

    refined_inputs = model._refine_scale_inputs(x)

    torch.testing.assert_close(refined_inputs[0], 2 * x[0])
    torch.testing.assert_close(refined_inputs[1], x[1])


def test_scale_attention_mask_allows_only_the_current_scale() -> None:
    model = NSM(
        vq_embed_dim=3,
        model_dim=8,
        scale_lengths=[1, 2, 3],
        codebook_size=5,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
    )

    expected_mask = torch.tensor(
        [
            [True, False, False, False, False, False],
            [False, True, True, False, False, False],
            [False, True, True, False, False, False],
            [False, False, False, True, True, True],
            [False, False, False, True, True, True],
            [False, False, False, True, True, True],
        ]
    ).reshape(1, 1, 6, 6)

    torch.testing.assert_close(model.scale_attention_mask, expected_mask)


def test_prefix_attention_mask_is_visible_to_every_scale() -> None:
    model = NSM(
        vq_embed_dim=3,
        model_dim=8,
        scale_lengths=[1, 2],
        codebook_size=5,
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
            [True, True, True, True, True],
            [True, True, True, True, True],
        ]
    ).reshape(1, 1, 5, 5)

    torch.testing.assert_close(model._build_attention_mask(2), expected_mask)


def test_scale_ids_match_hierarchy_sections() -> None:
    model = NSM(
        vq_embed_dim=3,
        model_dim=8,
        scale_lengths=[1, 2, 3],
        codebook_size=5,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
    )

    torch.testing.assert_close(model.scale_ids, torch.tensor([0, 1, 1, 2, 2, 2]))


def test_rope_positions_reset_at_each_scale() -> None:
    model = NSM(
        vq_embed_dim=3,
        model_dim=8,
        scale_lengths=[1, 2, 3],
        codebook_size=5,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
    )

    zero_positions = torch.tensor([0, 1, 3])

    torch.testing.assert_close(
        model.rope_cosine[zero_positions],
        torch.ones(3, 4),
    )
    torch.testing.assert_close(
        model.rope_sine[zero_positions],
        torch.zeros(3, 4),
    )


def test_prefix_rope_precedes_reset_scale_positions() -> None:
    model = NSM(
        vq_embed_dim=3,
        model_dim=8,
        scale_lengths=[1, 2, 3],
        codebook_size=5,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        max_prefix_length=2,
    )

    cosine, sine = model._get_rotary_embeddings(prefix_length=2)

    torch.testing.assert_close(cosine[:2], model.prefix_rope_cosine)
    torch.testing.assert_close(sine[:2], model.prefix_rope_sine)
    torch.testing.assert_close(cosine[2:], model.rope_cosine)
    torch.testing.assert_close(sine[2:], model.rope_sine)


def test_nsm_normalizes_queries_and_keys() -> None:
    model = NSM(
        vq_embed_dim=3,
        model_dim=8,
        scale_lengths=[1, 2, 3],
        codebook_size=5,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
    )

    assert isinstance(model.blocks[0].attn.q_norm, RMSNorm)
    assert isinstance(model.blocks[0].attn.k_norm, RMSNorm)


def test_nsm_uses_pre_rms_norm() -> None:
    model = NSM(
        vq_embed_dim=3,
        model_dim=8,
        scale_lengths=[1, 2, 3],
        codebook_size=5,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
    )

    assert isinstance(model.blocks[0].attn_norm, RMSNorm)
    assert isinstance(model.blocks[0].mlp_norm, RMSNorm)
    assert isinstance(model.final_norm, RMSNorm)
    assert model.blocks[0].attn_norm.eps == pytest.approx(1e-5)
    assert model.blocks[0].mlp_norm.eps == pytest.approx(1e-5)
    assert model.final_norm.eps == pytest.approx(1e-5)


def test_nsm_scales_residual_projection_initialization() -> None:
    num_layers = 4
    model = NSM(
        vq_embed_dim=8,
        model_dim=64,
        scale_lengths=[1, 2, 4],
        codebook_size=8,
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


def test_nsm_prepends_learned_bos() -> None:
    model = NSM(
        vq_embed_dim=3,
        model_dim=8,
        scale_lengths=[1, 2, 3],
        codebook_size=5,
        num_layers=0,
        num_heads=2,
        dropout=0.0,
    )
    scale_inputs = [torch.randn(2, 2, 3), torch.randn(2, 3, 3)]

    model_inputs = model(scale_inputs)

    expected_bos = model.bos.expand(2, -1, -1)
    projected_scale_inputs = torch.cat(
        [model.input_projection(scale_input) for scale_input in scale_inputs],
        dim=1,
    )
    expected_hidden_states = model.final_norm(
        torch.cat([expected_bos, projected_scale_inputs], dim=1)
        + model.scale_embedding(model.scale_ids)
    )
    expected_logits = model.output_head(expected_hidden_states)

    assert model_inputs.shape == (2, 6, 5)
    torch.testing.assert_close(model_inputs, expected_logits)


def test_nsm_returns_only_hierarchy_logits_when_prefix_is_present() -> None:
    model = NSM(
        vq_embed_dim=3,
        model_dim=8,
        scale_lengths=[1, 2, 3],
        codebook_size=5,
        num_layers=0,
        num_heads=2,
        dropout=0.0,
        max_prefix_length=4,
    )
    prefix = torch.randn(2, 4, 3)
    scale_inputs = [torch.randn(2, 2, 3), torch.randn(2, 3, 3)]

    hidden_states = model.encode(scale_inputs, prefix=prefix)
    logits = model(scale_inputs, prefix=prefix)

    assert hidden_states.shape == (2, 10, 8)
    assert logits.shape == (2, 6, 5)
    torch.testing.assert_close(logits, model.output_head(hidden_states[:, 4:]))


def test_nsm_transformer_keeps_scale_sections_isolated() -> None:
    model = NSM(
        vq_embed_dim=3,
        model_dim=8,
        scale_lengths=[1, 2],
        codebook_size=5,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
    )
    model.eval()

    x = [torch.randn(1, 2, 3)]
    x_with_changed_second_scale = [x[0] + 10.0]

    output = model(x)
    output_with_changed_second_scale = model(x_with_changed_second_scale)

    torch.testing.assert_close(
        output[:, :1],
        output_with_changed_second_scale[:, :1],
    )
