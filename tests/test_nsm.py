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
    prefix_conditioning: str = "full_resolution",
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
        prefix_conditioning=prefix_conditioning,
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
        latent_length=2,
        scale_lengths=[1, 2],
        codebook_sizes=[4, 5],
        context_length=4,
        quantizer=SimpleNamespace(
            codebooks=[
                SimpleNamespace(codebook=codebook_vectors[0]),
                SimpleNamespace(codebook=codebook_vectors[1]),
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


def test_scale_inputs_repeat_prior_scale_codes_to_the_target_length() -> None:
    model = _build_model(num_layers=0)
    indices_by_scale = _indices()

    inputs = model._build_scale_inputs(
        indices_by_scale[:-1],
        batch_size=2,
    )
    first_codes = model.codebook_vectors_0[indices_by_scale[0]]
    second_codes = model.codebook_vectors_1[indices_by_scale[1]]
    expected_inputs = torch.cat(
        [
            model.bos.expand(2, 1, -1) + model.scale_embedding.weight[0],
            model.input_projection(first_codes).repeat_interleave(2, dim=1)
            + model.scale_embedding.weight[1],
            model.input_projection(second_codes).repeat_interleave(2, dim=1)
            + model.scale_embedding.weight[2],
        ],
        dim=1,
    )
    torch.testing.assert_close(inputs, expected_inputs)


def test_attention_is_bidirectional_within_each_scale_block() -> None:
    model = _build_model()
    expected_mask = torch.tensor(
        [
            [True, True, False, False, False],
            [True, True, False, False, False],
            [True, True, True, False, False],
            [True, True, True, True, True],
            [True, True, True, True, True],
        ]
    ).reshape(1, 1, 5, 5)

    torch.testing.assert_close(model._build_attention_mask(2, 2), expected_mask)


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
            [True, True, True, True, True],
            [True, True, True, True, True],
        ]
    ).reshape(1, 1, 5, 5)

    torch.testing.assert_close(model._build_attention_mask(2, 2), expected_mask)


def test_previous_scale_only_attention_hides_older_scale_blocks() -> None:
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
        hierarchy_attention="previous_scale_only",
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


def test_multiscale_prefix_resolutions_are_revealed_progressively() -> None:
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
        prefix_conditioning="all_previous_resolutions",
    )

    expected_mask = torch.tensor(
        [
            [True, False, False, False, False, False],
            [True, True, True, False, False, False],
            [True, True, True, False, False, False],
            [True, False, False, True, False, False],
            [True, True, True, True, True, True],
            [True, True, True, True, True, True],
        ]
    ).reshape(1, 1, 6, 6)

    torch.testing.assert_close(model._build_attention_mask(3, 2), expected_mask)


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
    prefix = torch.randn(2, 4, 3)

    logits = model(indices_by_scale[:-1], prefix_by_scale=[prefix])
    expected_hidden_states = model.encode(
        indices_by_scale[:-1],
        prefix_by_scale=[prefix],
    )
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
    hidden_states = model.encode(_indices()[:-1], prefix_by_scale=[prefix])

    torch.testing.assert_close(
        hidden_states[:, 4:5],
        model.final_norm(
            model.bos.expand(2, 1, -1) + model.scale_embedding.weight[0]
        ),
    )


def test_nsm_returns_prefix_and_hierarchy_hidden_states() -> None:
    model = _build_model(num_layers=0, max_prefix_length=4)
    prefix = torch.randn(2, 4, 3)
    indices_by_scale = _indices()

    hidden_states = model.encode(
        indices_by_scale[:-1],
        prefix_by_scale=[prefix],
    )
    logits = model(indices_by_scale[:-1], prefix_by_scale=[prefix])

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
    indices_by_scale = _indices()
    changed_indices = [indices.clone() for indices in indices_by_scale]
    changed_indices[1][:, 0] = (changed_indices[1][:, 0] + 1) % 5

    logits = model(indices_by_scale[:-1], prefix_by_scale=[prefix])
    changed_logits = model(changed_indices[:-1], prefix_by_scale=[prefix])

    # Scale 2 is only supplied as the input used to predict scale 4.
    torch.testing.assert_close(logits[0], changed_logits[0])
    torch.testing.assert_close(logits[1], changed_logits[1])
    assert not torch.equal(logits[2], changed_logits[2])


def test_first_scale_is_predicted_without_supplying_its_code() -> None:
    torch.manual_seed(0)
    model = _build_model(num_layers=1).eval()
    prefix = torch.randn(2, 4, 3)
    indices_by_scale = _indices()
    changed_indices = [indices.clone() for indices in indices_by_scale]
    changed_indices[0][:, 0] = (changed_indices[0][:, 0] + 1) % 4

    logits = model(indices_by_scale[:-1], prefix_by_scale=[prefix])
    changed_logits = model(changed_indices[:-1], prefix_by_scale=[prefix])

    torch.testing.assert_close(logits[0], changed_logits[0])
    assert not torch.equal(logits[1], changed_logits[1])


@pytest.mark.parametrize(
    "prefix_conditioning",
    ["full_resolution", "all_previous_resolutions", "nucleotide_sequence"],
)
def test_packed_teacher_forcing_matches_scale_by_scale_prediction(
    prefix_conditioning: str,
) -> None:
    model = _build_model(
        num_layers=1,
        prefix_conditioning=prefix_conditioning,
    ).eval()
    prefix_by_scale = [
        torch.randn(2, 1, 3),
        torch.randn(2, 2, 3),
        torch.randn(2, 4, 3),
    ]
    prefix_token_ids = torch.randint(0, 4, (2, 4))
    indices_by_scale = _indices()

    teacher_forced_logits = model(
        indices_by_scale[:-1],
        prefix_by_scale=prefix_by_scale,
        prefix_token_ids=prefix_token_ids,
    )
    for scale_index in range(len(indices_by_scale)):
        scale_logits = model.predict_scale(
            prefix_by_scale,
            indices_by_scale[:scale_index],
            prefix_token_ids=prefix_token_ids,
        )
        torch.testing.assert_close(
            scale_logits,
            teacher_forced_logits[scale_index],
        )


def test_nucleotide_prefix_uses_every_raw_base_with_aligned_positions() -> None:
    model = _build_model(
        num_layers=0,
        max_prefix_length=8,
        prefix_conditioning="nucleotide_sequence",
    )
    prefix_by_scale = [
        torch.randn(2, 1, 3),
        torch.randn(2, 2, 3),
        torch.randn(2, 4, 3),
    ]
    prefix_token_ids = torch.randint(0, 4, (2, 8))

    hidden_states = model.encode(
        _indices()[:-1],
        prefix_by_scale=prefix_by_scale,
        prefix_token_ids=prefix_token_ids,
    )

    assert hidden_states.shape == (2, 15, 8)
    torch.testing.assert_close(
        model.hierarchy_positions,
        torch.tensor([3.5, 1.5, 5.5, 0.5, 2.5, 4.5, 6.5]),
    )


def test_nucleotide_prefix_requires_raw_token_ids() -> None:
    model = _build_model(prefix_conditioning="nucleotide_sequence")

    with pytest.raises(ValueError, match="prefix_token_ids are required"):
        model(_indices()[:-1], prefix_by_scale=[torch.randn(2, 4, 3)])
