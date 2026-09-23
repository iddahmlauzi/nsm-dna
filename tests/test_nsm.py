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
    max_prefix_length: int = 4,
) -> NSM:
    return NSM(
        prefix_dim=3,
        model_dim=8,
        scale_lengths=[1, 2, 4],
        codebook_sizes=[4, 5, 6],
        target_length=12,
        vocab_size=4,
        num_layers=num_layers,
        num_heads=2,
        dropout=0.0,
        max_prefix_length=max_prefix_length,
    )


def _scale_vectors() -> list[torch.Tensor]:
    return [
        torch.randn(2, 1, 3),
        torch.randn(2, 2, 3),
        torch.randn(2, 4, 3),
    ]


def test_nsm_from_checkpoint_restores_model_and_step(tmp_path: Path) -> None:
    tokenizer = SimpleNamespace(
        quantization_dim=3,
        latent_length=2,
        scale_lengths=[1, 2],
        codebook_sizes=[4, 5],
        context_length=4,
        vocab_size=4,
    )
    model = NSM(
        prefix_dim=tokenizer.quantization_dim,
        model_dim=8,
        scale_lengths=tokenizer.scale_lengths,
        codebook_sizes=tokenizer.codebook_sizes,
        target_length=tokenizer.context_length,
        vocab_size=4,
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


def test_scale_inputs_repeat_prior_scale_vectors_to_the_target_length() -> None:
    model = _build_model(num_layers=0)
    vectors_by_scale = _scale_vectors()

    inputs = model._build_scale_inputs(
        vectors_by_scale[:-1],
        batch_size=2,
    )
    expected_inputs = torch.cat(
        [
            model.bos.expand(2, 1, -1) + model.scale_embedding.weight[0],
            model.input_projection(vectors_by_scale[0]).repeat_interleave(2, dim=1)
            + model.scale_embedding.weight[1],
            model.input_projection(vectors_by_scale[1]).repeat_interleave(2, dim=1)
            + model.scale_embedding.weight[2],
        ],
        dim=1,
    )
    torch.testing.assert_close(inputs, expected_inputs)


def test_next_scale_loss_backpropagates_to_context_vectors() -> None:
    model = _build_model()
    vectors_by_scale = [
        vectors.requires_grad_() for vectors in _scale_vectors()[:-1]
    ]

    logits_by_scale = model(
        vectors_by_scale,
        prefix=torch.randn(2, 4, 3),
    )
    loss = sum(logits.square().mean() for logits in logits_by_scale)
    loss.backward()

    assert all(
        vectors.grad is not None and vectors.grad.count_nonzero() > 0
        for vectors in vectors_by_scale
    )


def test_attention_is_markovian_between_scale_blocks() -> None:
    model = _build_model()
    expected_mask = torch.tensor(
        [
            [True, True, False, False, False],
            [True, True, False, False, False],
            [True, True, True, False, False],
            [True, True, False, True, True],
            [True, True, False, True, True],
        ]
    ).reshape(1, 1, 5, 5)

    torch.testing.assert_close(model._build_attention_mask(2, 2), expected_mask)


def test_prefix_is_visible_to_every_causal_hierarchy_position() -> None:
    model = NSM(
        prefix_dim=3,
        model_dim=8,
        scale_lengths=[1, 2],
        codebook_sizes=[4, 5],
        target_length=4,
        vocab_size=4,
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

    torch.testing.assert_close(model._build_attention_mask(2, 2), expected_mask)


def test_rope_positions_are_centers_in_the_finest_scale_coordinates() -> None:
    model = _build_model()
    torch.testing.assert_close(
        model.hierarchy_positions,
        torch.tensor([1.5, 0.5, 2.5, 0.0, 1.0, 2.0, 3.0]),
    )
    torch.testing.assert_close(
        model.prefix_positions,
        torch.tensor([-4.0, -3.0, -2.0, -1.0]),
    )
    torch.testing.assert_close(
        model.nucleotide_positions,
        (torch.arange(12) + 0.5) / 3 - 0.5,
    )


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
        target_length=8,
        vocab_size=4,
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
    vectors_by_scale = _scale_vectors()
    prefix = torch.randn(2, 4, 3)

    logits = model(vectors_by_scale[:-1], prefix=prefix)
    expected_hidden_states = model.encode(vectors_by_scale[:-1], prefix=prefix)
    expected_logits = model.output_head(expected_hidden_states[:, 4:])

    assert [value.shape for value in logits] == [
        (2, 1, 4),
        (2, 2, 5),
        (2, 4, 6),
    ]
    for actual, expected in zip(logits, expected_logits, strict=True):
        torch.testing.assert_close(actual, expected)


def test_first_scale_uses_the_learned_bos_input() -> None:
    model = _build_model(num_layers=0)
    prefix = torch.randn(2, 4, 3)
    hidden_states = model.encode(_scale_vectors()[:-1], prefix=prefix)

    torch.testing.assert_close(
        hidden_states[:, 4:5],
        model.final_norm(
            model.bos.expand(2, 1, -1) + model.scale_embedding.weight[0]
        ),
    )


def test_nsm_returns_prefix_and_hierarchy_hidden_states() -> None:
    model = _build_model(num_layers=0, max_prefix_length=4)
    prefix = torch.randn(2, 4, 3)
    vectors_by_scale = _scale_vectors()

    hidden_states = model.encode(vectors_by_scale[:-1], prefix=prefix)
    logits = model(vectors_by_scale[:-1], prefix=prefix)

    assert hidden_states.shape == (2, 11, 8)
    assert [value.shape for value in logits] == [
        (2, 1, 4),
        (2, 2, 5),
        (2, 4, 6),
    ]
    expected_logits = model.output_head(hidden_states[:, 4:])
    for actual, expected in zip(logits, expected_logits, strict=True):
        torch.testing.assert_close(actual, expected)


def test_a_scale_cannot_change_its_own_or_earlier_logits() -> None:
    torch.manual_seed(0)
    model = _build_model(num_layers=2).eval()
    prefix = torch.randn(2, 4, 3)
    vectors_by_scale = _scale_vectors()
    changed_vectors = [vectors.clone() for vectors in vectors_by_scale]
    changed_vectors[1][:, 0] += 1

    logits = model(vectors_by_scale[:-1], prefix=prefix)
    changed_logits = model(
        changed_vectors[:-1],
        prefix=prefix,
    )

    # Scale 2 is only supplied as the input used to predict scale 4.
    torch.testing.assert_close(logits[0], changed_logits[0])
    torch.testing.assert_close(logits[1], changed_logits[1])
    assert not torch.equal(logits[2], changed_logits[2])


def test_scale_inputs_do_not_skip_over_intermediate_scales() -> None:
    torch.manual_seed(0)
    model = _build_model(num_layers=1).eval()
    prefix = torch.randn(2, 4, 3)
    vectors_by_scale = _scale_vectors()
    changed_vectors = [vectors.clone() for vectors in vectors_by_scale]
    changed_vectors[0][:, 0] += 1

    logits = model(vectors_by_scale[:-1], prefix=prefix)
    changed_logits = model(
        changed_vectors[:-1],
        prefix=prefix,
    )

    torch.testing.assert_close(logits[0], changed_logits[0])
    assert not torch.equal(logits[1], changed_logits[1])
    torch.testing.assert_close(logits[2], changed_logits[2])


def test_nucleotide_prediction_uses_upsampled_final_scale() -> None:
    torch.manual_seed(0)
    model = _build_model(num_layers=0).eval()
    prefix = torch.randn(2, 4, 3)
    vectors_by_scale = _scale_vectors()
    changed_final_vectors = vectors_by_scale[-1].clone()
    changed_final_vectors[:, 0] += 1

    nucleotide_logits = model.predict_nucleotides(
        vectors_by_scale[-1],
        prefix=prefix,
    )
    changed_nucleotide_logits = model.predict_nucleotides(
        changed_final_vectors,
        prefix=prefix,
    )

    assert nucleotide_logits.shape == (2, 12, 4)
    assert not torch.equal(
        nucleotide_logits[:, :3],
        changed_nucleotide_logits[:, :3],
    )
    torch.testing.assert_close(
        nucleotide_logits[:, 3:],
        changed_nucleotide_logits[:, 3:],
    )


def test_packed_teacher_forcing_matches_scale_by_scale_prediction() -> None:
    model = _build_model(num_layers=1).eval()
    prefix = torch.randn(2, 4, 3)
    vectors_by_scale = _scale_vectors()

    teacher_forced_logits = model(vectors_by_scale[:-1], prefix=prefix)
    for scale_index in range(len(vectors_by_scale)):
        scale_logits = model.predict_scale(
            prefix,
            vectors_by_scale[:scale_index],
        )
        torch.testing.assert_close(
            scale_logits,
            teacher_forced_logits[scale_index],
        )
