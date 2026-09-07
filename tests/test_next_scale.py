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
            [True, True, False, False, False],
            [True, True, False, False, False],
            [False, False, True, True, True],
            [False, False, True, True, True],
            [False, False, True, True, True],
        ]
    ).reshape(1, 1, 5, 5)

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
            [True, True, False, False],
            [True, True, False, False],
            [True, True, True, True],
            [True, True, True, True],
        ]
    ).reshape(1, 1, 4, 4)

    torch.testing.assert_close(model._build_attention_mask(2), expected_mask)


def test_prefix_memory_mask_makes_memory_the_only_prefix_to_scale_route() -> None:
    model = NextScaleTransformer(
        input_dim=3,
        model_dim=8,
        scale_lengths=[1, 2],
        codebook_size=5,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        max_prefix_length=2,
        use_prefix_memory=True,
    )

    expected_mask = torch.tensor(
        [
            [True, True, False, False, False],
            [True, True, False, False, False],
            [True, True, True, False, False],
            [False, False, True, True, True],
            [False, False, True, True, True],
        ]
    ).reshape(1, 1, 5, 5)

    torch.testing.assert_close(model._build_attention_mask(2), expected_mask)


def test_prefix_memory_is_included_in_hidden_states_but_not_logits() -> None:
    model = NextScaleTransformer(
        input_dim=3,
        model_dim=8,
        scale_lengths=[1, 2, 3],
        codebook_size=5,
        num_layers=0,
        num_heads=2,
        dropout=0.0,
        max_prefix_length=2,
        use_prefix_memory=True,
    )
    prefix = torch.randn(2, 2, 3)
    scale_inputs = [torch.randn(2, 2, 3), torch.randn(2, 3, 3)]

    hidden_states = model.encode(scale_inputs, prefix=prefix)
    logits = model(scale_inputs, prefix=prefix)

    assert model.prefix_context_length(prefix.shape[1]) == 3
    assert hidden_states.shape == (2, 8, 8)
    assert logits.shape == (2, 5, 5)
    torch.testing.assert_close(logits, model.output_head(hidden_states[:, 3:]))


def test_prefix_memory_carries_prefix_gradients_to_target_scales() -> None:
    model = NextScaleTransformer(
        input_dim=3,
        model_dim=8,
        scale_lengths=[1, 2],
        codebook_size=5,
        num_layers=2,
        num_heads=2,
        dropout=0.0,
        max_prefix_length=2,
        use_prefix_memory=True,
    )
    prefix = torch.randn(2, 2, 3, requires_grad=True)
    scale_inputs = [torch.randn(2, 2, 3)]

    model(scale_inputs, prefix=prefix).square().mean().backward()

    assert model.prefix_memory_token is not None
    assert model.prefix_memory_token.grad is not None
    assert model.prefix_memory_token.grad.count_nonzero() > 0
    assert prefix.grad is not None
    assert prefix.grad.count_nonzero() > 0


def test_scale_one_bos_memory_routes_prefix_to_later_scales() -> None:
    model = NextScaleTransformer(
        input_dim=3,
        model_dim=8,
        scale_lengths=[1, 2],
        codebook_size=5,
        num_layers=2,
        num_heads=2,
        dropout=0.0,
        max_prefix_length=2,
        predict_first_scale=True,
    )
    prefix = torch.randn(2, 2, 3, requires_grad=True)
    scale_inputs = [torch.randn(2, 2, 3)]

    expected_mask = torch.tensor(
        [
            [True, True, False, False, False],
            [True, True, False, False, False],
            [True, True, True, False, False],
            [False, False, True, True, True],
            [False, False, True, True, True],
        ]
    ).reshape(1, 1, 5, 5)
    torch.testing.assert_close(model._build_attention_mask(2), expected_mask)

    logits = model(scale_inputs, prefix=prefix)
    assert logits.shape == (2, 3, 5)
    logits[:, 1:].square().mean().backward()

    assert model.first_scale_bos is not None
    assert model.first_scale_bos.grad is not None
    assert model.first_scale_bos.grad.count_nonzero() > 0
    assert prefix.grad is not None
    assert prefix.grad.count_nonzero() > 0


def test_scale_one_bos_predictions_match_parallel_hierarchy() -> None:
    model = NextScaleTransformer(
        input_dim=6,
        model_dim=16,
        scale_lengths=[1, 2, 4],
        codebook_size=7,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        max_prefix_length=3,
        predict_first_scale=True,
    ).eval()
    prefix = torch.randn(2, 3, 6)
    scale_inputs = [torch.randn(2, 2, 6), torch.randn(2, 4, 6)]

    parallel_logits = torch.split(
        model(scale_inputs, prefix=prefix),
        model.predicted_scale_lengths,
        dim=1,
    )
    isolated_logits = [
        model.predict_first_scale(prefix),
        *[
            model.predict_scale(
                scale_input,
                prediction_index=prediction_index,
                prefix=prefix,
            )
            for prediction_index, scale_input in enumerate(scale_inputs, start=1)
        ],
    ]

    for actual, expected in zip(isolated_logits, parallel_logits, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_prefix_memory_modes_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="not both"):
        NextScaleTransformer(
            input_dim=3,
            model_dim=8,
            scale_lengths=[1, 2],
            codebook_size=5,
            num_layers=1,
            num_heads=2,
            use_prefix_memory=True,
            predict_first_scale=True,
        )


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

    torch.testing.assert_close(model.scale_ids, torch.tensor([0, 0, 1, 1, 1]))


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

    zero_positions = torch.tensor([0, 2])

    torch.testing.assert_close(
        model.rope_cosine[zero_positions],
        torch.ones(2, 4),
    )
    torch.testing.assert_close(
        model.rope_sine[zero_positions],
        torch.zeros(2, 4),
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


def test_nsm_uses_supplied_first_scale_to_predict_later_scales() -> None:
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

    projected_scale_inputs = torch.cat(
        [model.input_projection(scale_input) for scale_input in scale_inputs],
        dim=1,
    )
    expected_hidden_states = model.final_norm(
        projected_scale_inputs + model.scale_embedding(model.scale_ids)
    )
    expected_logits = model.output_head(expected_hidden_states)

    assert model_inputs.shape == (2, 5, 5)
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

    assert hidden_states.shape == (2, 9, 8)
    assert logits.shape == (2, 5, 5)
    torch.testing.assert_close(logits, model.output_head(hidden_states[:, 4:]))


def test_nsm_transformer_keeps_scale_sections_isolated() -> None:
    model = NextScaleTransformer(
        input_dim=3,
        model_dim=8,
        scale_lengths=[1, 2, 3],
        codebook_size=5,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
    )
    model.eval()

    x = [torch.randn(1, 2, 3), torch.randn(1, 3, 3)]
    x_with_changed_second_scale = [x[0], x[1] + 10.0]

    output = model(x)
    output_with_changed_second_scale = model(x_with_changed_second_scale)

    torch.testing.assert_close(
        output[:, :2],
        output_with_changed_second_scale[:, :2],
    )


def test_single_scale_prediction_matches_parallel_scale_section() -> None:
    model = NextScaleTransformer(
        input_dim=6,
        model_dim=16,
        scale_lengths=[1, 2, 4],
        codebook_size=7,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        max_prefix_length=3,
    ).eval()
    prefix = torch.randn(2, 3, 6)
    scale_inputs = [torch.randn(2, 2, 6), torch.randn(2, 4, 6)]

    parallel_logits = torch.split(
        model(scale_inputs, prefix=prefix),
        model.predicted_scale_lengths,
        dim=1,
    )
    single_scale_logits = [
        model.predict_scale(
            scale_input,
            prediction_index,
            prefix=prefix,
        )
        for prediction_index, scale_input in enumerate(scale_inputs)
    ]

    for actual, expected in zip(
        single_scale_logits,
        parallel_logits,
        strict=True,
    ):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


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
            },
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
    assert output.partial_reconstruction_logits is None
    assert output.teacher_forced_prediction_reconstruction_logits is None
    assert [logits.shape for logits in output.next_scale_logits_by_scale] == [
        (2, 2, 8),
        (2, 4, 8),
    ]


def test_scale_one_bos_model_predicts_and_rolls_out_every_scale() -> None:
    config = _build_end_to_end_config()
    config.model.transformer.predict_first_scale = True
    model = NSMDNA.from_config(config)
    sequence_ids = torch.tensor(
        [
            [0, 1, 2, 3, 3, 2, 1, 0],
            [3, 2, 1, 0, 0, 1, 2, 3],
        ]
    )

    output = model(
        sequence_ids,
        return_teacher_forced_prediction_reconstruction=True,
        return_rollout=True,
        return_rollout_reconstructions=True,
    )

    assert model.transformer.predicted_scale_lengths == [1, 2, 4]
    assert [logits.shape for logits in output.next_scale_logits_by_scale] == [
        (2, 1, 8),
        (2, 2, 8),
        (2, 4, 8),
    ]
    assert output.teacher_forced_prediction_reconstruction_logits is not None
    assert output.teacher_forced_prediction_reconstruction_logits.shape == (2, 4, 4)
    assert output.rollout is not None
    assert [logits.shape for logits in output.rollout.prediction_logits_by_scale] == [
        (2, 1, 8),
        (2, 2, 8),
        (2, 4, 8),
    ]
    assert [indices.shape for indices in output.rollout.indices_by_scale] == [
        (2, 1),
        (2, 2),
        (2, 4),
    ]

    generation = model.eval().generate(sequence_ids[:, :4])
    assert [indices.shape for indices in generation.indices_by_scale] == [
        (2, 1),
        (2, 2),
        (2, 4),
    ]


def test_forward_runs_without_a_prefix() -> None:
    config = _build_end_to_end_config()
    config.data.sequence_length = config.model.tokenizer.context_length
    model = NSMDNA.from_config(config)
    sequence_ids = torch.tensor(
        [
            [0, 1, 2, 3],
            [3, 2, 1, 0],
        ]
    )

    prefix_latent, target_latent = model.tokenizer.encode_pair(sequence_ids)
    output = model(sequence_ids)

    assert model.transformer.max_prefix_length == 0
    assert prefix_latent.shape == (2, 0, model.tokenizer.embed_dim)
    assert target_latent.shape == (2, 4, model.tokenizer.embed_dim)
    assert output.reconstruction_logits.shape == (2, 4, 4)
    assert [logits.shape for logits in output.next_scale_logits_by_scale] == [
        (2, 2, 8),
        (2, 4, 8),
    ]
    assert len(output.quantizer.cumulative_latents) == 3
    assert len(output.quantizer.next_scale_inputs) == 2


def test_forward_decodes_teacher_forced_predictions_with_soft_gradients() -> None:
    model = NSMDNA.from_config(_build_end_to_end_config())
    output = model(
        torch.tensor(
            [
                [0, 1, 2, 3, 3, 2, 1, 0],
                [3, 2, 1, 0, 0, 1, 2, 3],
            ]
        ),
        return_teacher_forced_prediction_reconstruction=True,
    )

    reconstruction_logits = output.teacher_forced_prediction_reconstruction_logits
    assert reconstruction_logits is not None
    assert reconstruction_logits.shape == (2, 4, 4)
    reconstruction_logits.square().mean().backward()
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


def test_forward_collects_a_detached_rollout() -> None:
    config = _build_end_to_end_config()
    config.model.transformer.dropout = 0.2
    model = NSMDNA.from_config(config)
    sequence_ids = torch.tensor(
        [
            [0, 1, 2, 3, 3, 2, 1, 0],
            [3, 2, 1, 0, 0, 1, 2, 3],
        ]
    )

    output = model(
        sequence_ids,
        return_rollout=True,
        return_rollout_reconstructions=True,
    )

    rollout = output.rollout
    assert rollout is not None
    assert [logits.shape for logits in rollout.prediction_logits_by_scale] == [
        (2, 2, 8),
        (2, 4, 8),
    ]
    assert [indices.shape for indices in rollout.indices_by_scale] == [
        (2, 1),
        (2, 2),
        (2, 4),
    ]
    torch.testing.assert_close(
        rollout.indices_by_scale[0],
        output.quantizer.indices_by_scale[0],
    )
    assert all(
        not logits.requires_grad for logits in rollout.prediction_logits_by_scale
    )
    assert all(not indices.requires_grad for indices in rollout.indices_by_scale)
    for replay_logits, collected_indices in zip(
        rollout.prediction_logits_by_scale,
        rollout.indices_by_scale[1:],
        strict=True,
    ):
        torch.testing.assert_close(
            replay_logits.argmax(dim=-1),
            collected_indices,
        )
    assert rollout.final_reconstruction_logits is not None
    assert rollout.final_reconstruction_logits.shape == (2, 4, 4)
    assert rollout.cumulative_reconstruction_logits_by_scale is not None
    assert len(rollout.cumulative_reconstruction_logits_by_scale) == 3


def test_tokenizer_stability_snapshot_is_detached_and_preserves_training_mode() -> None:
    model = NSMDNA.from_config(_build_end_to_end_config()).train()
    sequence_ids = torch.tensor(
        [
            [0, 1, 2, 3, 3, 2, 1, 0],
            [3, 2, 1, 0, 0, 1, 2, 3],
        ]
    )
    ema_counts_before = [
        codebook.ema_counts.clone() for codebook in model.tokenizer.quantizer.codebooks
    ]

    snapshot = model.tokenizer_stability_snapshot(sequence_ids)

    assert model.training
    assert model.tokenizer.training
    assert model.transformer.training
    assert not snapshot.encoder_latent.requires_grad
    assert len(snapshot.indices_by_scale) == 3
    assert len(snapshot.codebooks_by_scale) == 3
    for codebook, expected_counts in zip(
        model.tokenizer.quantizer.codebooks,
        ema_counts_before,
        strict=True,
    ):
        torch.testing.assert_close(codebook.ema_counts, expected_counts)


def test_rollout_advances_with_its_own_predictions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = NSMDNA.from_config(_build_end_to_end_config())
    quantizer = model.tokenizer.quantizer
    predicted_code_by_scale = [1, 2]
    contribution_indices = []
    rollout_transformer_modes = []
    original_scale_contribution = quantizer.scale_contribution_from_indices

    def record_scale_contribution(indices, scale_index):
        contribution_indices.append(indices.detach().clone())
        return original_scale_contribution(indices, scale_index)

    def fixed_predictions(scale_input, prediction_index, *, prefix=None):
        del prefix
        rollout_transformer_modes.append(model.transformer.training)
        logits = scale_input.new_zeros(
            scale_input.shape[0],
            scale_input.shape[1],
            model.transformer.codebook_size,
        )
        logits[..., predicted_code_by_scale[prediction_index]] = 1
        return logits

    monkeypatch.setattr(
        quantizer,
        "scale_contribution_from_indices",
        record_scale_contribution,
    )
    monkeypatch.setattr(model.transformer, "predict_scale", fixed_predictions)

    model.train()
    output = model(
        torch.tensor(
            [
                [0, 1, 2, 3, 3, 2, 1, 0],
                [3, 2, 1, 0, 0, 1, 2, 3],
            ]
        ),
        return_rollout=True,
    )

    rollout = output.rollout
    assert rollout is not None
    assert rollout_transformer_modes == [False, False]
    assert model.transformer.training is True
    assert torch.equal(
        rollout.indices_by_scale[1],
        torch.full((2, 2), 1),
    )
    assert torch.equal(
        rollout.indices_by_scale[2],
        torch.full((2, 4), 2),
    )
    assert torch.equal(contribution_indices[1], torch.full((2, 2), 1))
    assert torch.equal(contribution_indices[2], torch.full((2, 4), 2))


def test_rollout_does_not_add_an_ema_update() -> None:
    config = _build_end_to_end_config()
    teacher_forced_model = NSMDNA.from_config(config)
    rollout_model = NSMDNA.from_config(config)
    rollout_model.load_state_dict(teacher_forced_model.state_dict())
    sequence_ids = torch.tensor(
        [
            [0, 1, 2, 3, 3, 2, 1, 0],
            [3, 2, 1, 0, 0, 1, 2, 3],
        ]
    )

    teacher_forced_model(sequence_ids)
    rollout_model(sequence_ids, return_rollout=True)

    for expected_codebook, actual_codebook in zip(
        teacher_forced_model.tokenizer.quantizer.codebooks,
        rollout_model.tokenizer.quantizer.codebooks,
        strict=True,
    ):
        torch.testing.assert_close(
            actual_codebook.ema_counts,
            expected_codebook.ema_counts,
        )
        torch.testing.assert_close(
            actual_codebook.ema_vector_sums,
            expected_codebook.ema_vector_sums,
        )
        torch.testing.assert_close(
            actual_codebook.codebook_hits,
            expected_codebook.codebook_hits,
        )
        torch.testing.assert_close(
            actual_codebook.codebook,
            expected_codebook.codebook,
        )


def test_forward_decodes_one_true_partial_cumulative_latent() -> None:
    model = NSMDNA.from_config(_build_end_to_end_config())
    output = model(
        torch.tensor(
            [
                [0, 1, 2, 3, 3, 2, 1, 0],
                [3, 2, 1, 0, 0, 1, 2, 3],
            ]
        ),
        return_partial_reconstruction=True,
    )

    assert output.partial_reconstruction_logits is not None
    assert output.partial_reconstruction_logits.shape == (2, 4, 4)
    output.partial_reconstruction_logits.square().mean().backward()
    assert any(
        parameter.grad is not None and parameter.grad.count_nonzero() > 0
        for parameter in model.tokenizer.encoder.parameters()
    )
    assert any(
        parameter.grad is not None and parameter.grad.count_nonzero() > 0
        for parameter in model.tokenizer.quantizer.parameters()
    )
    assert any(
        parameter.grad is not None and parameter.grad.count_nonzero() > 0
        for parameter in model.tokenizer.decoder.parameters()
    )


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
    first_scale_indices = torch.tensor([[1], [2]])

    generation = model.generate(
        torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]]),
        first_scale_indices,
    )

    assert generation.nucleotide_logits.shape == (2, 4, 4)
    assert [indices.shape for indices in generation.indices_by_scale] == [
        (2, 1),
        (2, 2),
        (2, 4),
    ]
    torch.testing.assert_close(
        generation.indices_by_scale[0],
        first_scale_indices,
    )


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
