import math
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from nsm_dna.models.common import RMSNorm
from nsm_dna.models.next_scale import NSMDNA, NextScaleTransformer
from nsm_dna.models.next_scale.transformer import SharedOutputHead


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
    model = NextScaleTransformer(
        input_dim=3,
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
    model = NextScaleTransformer(
        input_dim=3,
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


def test_prefix_attention_mask_connects_prefix_directly_to_every_scale() -> None:
    model = NextScaleTransformer(
        input_dim=3,
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
            [True, True, False, True, True],
            [True, True, False, True, True],
        ]
    ).reshape(1, 1, 5, 5)

    torch.testing.assert_close(model._build_attention_mask(2), expected_mask)


def test_scale_ids_match_hierarchy_sections() -> None:
    model = NextScaleTransformer(
        input_dim=3,
        model_dim=8,
        scale_lengths=[1, 2, 3],
        codebook_size=5,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
    )

    torch.testing.assert_close(model.scale_ids, torch.tensor([0, 1, 1, 2, 2, 2]))


def test_rope_positions_reset_at_each_scale() -> None:
    model = NextScaleTransformer(
        input_dim=3,
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
    model = NextScaleTransformer(
        input_dim=3,
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
    model = NextScaleTransformer(
        input_dim=3,
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
    model = NextScaleTransformer(
        input_dim=3,
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
    model = NextScaleTransformer(
        input_dim=8,
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
    model = NextScaleTransformer(
        input_dim=3,
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
    model = NextScaleTransformer(
        input_dim=3,
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
    model = NextScaleTransformer(
        input_dim=3,
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


def _build_end_to_end_config():
    return OmegaConf.create(
        {
            "data": {"sequence_length": 8},
            "model": {
                "tokenizer": {
                    "vocab_size": 4,
                    "context_length": 4,
                    "embed_dim": 8,
                    "num_heads": 2,
                    "scale_lengths": [1, 2, 4],
                    "codebook_size": 8,
                    "encoder_dropout": 0.0,
                    "decoder_dropout": 0.0,
                    "bias": False,
                    "rope_base": 10_000.0,
                    "pre_quant_num_groups": 2,
                    "commitment_cost": 0.25,
                    "decay": 0.99,
                    "eps": 1e-5,
                    "temperature": 1.0,
                    "refinement_ratio": 0.5,
                    "refinement_kernel_size": 3,
                },
                "transformer": {
                    "model_dim": 8,
                    "num_layers": 1,
                    "num_heads": 2,
                    "dropout": 0.0,
                    "bias": False,
                    "use_qk_norm": True,
                    "rope_base": 10_000.0,
                    "head_num_blocks": 2,
                    "head_hidden_multiplier": 2.0,
                    "input_refinement_kernel_size": 3,
                },
            }
        }
    )


def test_forward_runs_the_complete_pipeline_in_one_model_call() -> None:
    model = NSMDNA.from_config(_build_end_to_end_config())
    encoder_calls = 0

    def count_encoder_call(_module, _inputs, _output) -> None:
        nonlocal encoder_calls
        encoder_calls += 1

    hook = model.tokenizer.encoder.register_forward_hook(count_encoder_call)
    output = model(
        torch.tensor(
            [
                [0, 1, 2, 3, 3, 2, 1, 0],
                [3, 2, 1, 0, 0, 1, 2, 3],
            ]
        ),
        corruption_probability=0.1,
    )
    hook.remove()

    assert encoder_calls == 1
    assert output.reconstruction_logits.shape == (2, 4, 4)
    assert output.autoregressive_reconstruction_logits is None
    assert [logits.shape for logits in output.next_scale_logits_by_scale] == [
        (2, 1, 8),
        (2, 2, 8),
        (2, 4, 8),
    ]
    assert len(output.quantizer.cumulative_latents) == 3
    assert len(output.quantizer.next_scale_inputs) == 2


def test_forward_decodes_hard_predictions_with_soft_gradients_for_apr() -> None:
    model = NSMDNA.from_config(_build_end_to_end_config())
    output = model(
        torch.tensor(
            [
                [0, 1, 2, 3, 3, 2, 1, 0],
                [3, 2, 1, 0, 0, 1, 2, 3],
            ]
        ),
        return_autoregressive_reconstruction=True,
    )

    assert output.autoregressive_reconstruction_logits is not None
    assert output.autoregressive_reconstruction_logits.shape == (2, 4, 4)
    output.autoregressive_reconstruction_logits.square().mean().backward()
    assert any(
        parameter.grad is not None and parameter.grad.count_nonzero() > 0
        for parameter in model.transformer.parameters()
    )
    assert any(
        parameter.grad is not None and parameter.grad.count_nonzero() > 0
        for parameter in model.tokenizer.decoder.parameters()
    )
    assert all(
        codebook.codebook.grad is None
        for codebook in model.tokenizer.quantizer.codebooks
    )


def test_forward_can_condition_on_detached_model_predictions() -> None:
    model = NSMDNA.from_config(_build_end_to_end_config()).eval()
    sequence_ids = torch.tensor(
        [
            [0, 1, 2, 3, 3, 2, 1, 0],
            [3, 2, 1, 0, 0, 1, 2, 3],
        ]
    )
    transformer_calls = 0

    def count_transformer_call(_module, _inputs, _output) -> None:
        nonlocal transformer_calls
        transformer_calls += 1

    hook = model.transformer.register_forward_hook(count_transformer_call)
    output = model(sequence_ids, self_conditioning_probability=1.0)
    hook.remove()

    assert transformer_calls == 2
    assert [logits.shape for logits in output.next_scale_logits_by_scale] == [
        (2, 1, 8),
        (2, 2, 8),
        (2, 4, 8),
    ]


def test_forward_rejects_invalid_self_conditioning_probability() -> None:
    model = NSMDNA.from_config(_build_end_to_end_config())
    sequence_ids = torch.tensor([[0, 1, 2, 3, 3, 2, 1, 0]])

    with pytest.raises(ValueError, match="Self-conditioning probability"):
        model(sequence_ids, self_conditioning_probability=1.1)


def test_joint_loss_reaches_every_trainable_submodule() -> None:
    model = NSMDNA.from_config(_build_end_to_end_config())
    sequence_ids = torch.tensor(
        [
            [0, 1, 2, 3, 3, 2, 1, 0],
            [3, 2, 1, 0, 0, 1, 2, 3],
        ]
    )

    output = model(sequence_ids, corruption_probability=0.1)
    loss = (
        output.reconstruction_logits.square().mean()
        + torch.stack(
            [logits.square().mean() for logits in output.next_scale_logits_by_scale]
        ).mean()
        + output.quantizer.vq_loss
    )
    loss.backward()

    parameters_without_gradients = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert parameters_without_gradients == []
    assert all(
        codebook.codebook.grad is None
        for codebook in model.tokenizer.quantizer.codebooks
    )


def test_complete_model_checkpoint_is_restored_and_frozen(tmp_path: Path) -> None:
    config = _build_end_to_end_config()
    model = NSMDNA.from_config(config)
    checkpoint_path = tmp_path / "nsm-dna.pt"
    torch.save(
        {
            "step": 17,
            "model": model.state_dict(),
            "config": OmegaConf.to_container(config, resolve=True),
        },
        checkpoint_path,
    )

    restored_model, checkpoint_step = NSMDNA.from_checkpoint(
        checkpoint_path,
        torch.device("cpu"),
        frozen=True,
    )

    assert checkpoint_step == 17
    assert not restored_model.training
    assert all(not parameter.requires_grad for parameter in restored_model.parameters())
    for name, expected_value in model.state_dict().items():
        torch.testing.assert_close(
            restored_model.state_dict()[name],
            expected_value,
        )


def test_generate_predicts_and_decodes_a_hard_hierarchy() -> None:
    model = NSMDNA.from_config(_build_end_to_end_config()).eval()

    generation = model.generate(torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]]))

    assert generation.nucleotide_logits.shape == (2, 4, 4)
    assert [indices.shape for indices in generation.indices_by_scale] == [
        (2, 1),
        (2, 2),
        (2, 4),
    ]


def test_forward_accepts_a_prefix_shorter_than_the_target() -> None:
    model = NSMDNA.from_config(_build_end_to_end_config())

    output = model(torch.zeros(1, 7, dtype=torch.long))

    assert output.reconstruction_logits.shape == (1, 4, 4)


def test_forward_allows_a_target_without_a_prefix() -> None:
    config = _build_end_to_end_config()
    config.data.sequence_length = 4
    model = NSMDNA.from_config(config)

    output = model(torch.zeros(1, 4, dtype=torch.long))

    assert model.transformer.max_prefix_length == 0
    assert output.reconstruction_logits.shape == (1, 4, 4)
