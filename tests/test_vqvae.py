from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from nsm_dna.models.autoencoder import Decoder
from nsm_dna.models.common import RMSNorm
from nsm_dna.models.quantization import (
    DeterministicCodebook,
    MultiscaleVectorQuantizer,
)
from nsm_dna.models.vqvae import VQVAE, VQVAEHierarchyOutput
from scripts.training.train_vqvae import (
    calculate_hierarchy_prediction_losses,
    evaluate,
)


def _build_model(
    *,
    decoder_num_layers: int = 1,
) -> VQVAE:
    return VQVAE(
        vocab_size=4,
        context_length=8,
        latent_length=4,
        embed_dim=8,
        quantization_dim=4,
        num_heads=2,
        scale_lengths=[1, 2, 4],
        codebook_sizes=[8, 8, 16],
        decoder_num_layers=decoder_num_layers,
        use_qk_norm=True,
    )


def test_default_config_uses_left_grouped_k64_hierarchy() -> None:
    config_path = Path(__file__).parents[1] / "configs" / "vqvae.yaml"
    config = OmegaConf.load(config_path)

    assert config.wandb.name == "vqvae-256-left-grouped-hierarchy-k64"
    assert list(config.model.codebook_sizes) == [64] * 7 + [16]
    assert config.model.group_codes_by_left_child is True


def test_decoder_supplies_rope_to_attention() -> None:
    class RecordingBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.rotary_embeddings = None

        def forward(
            self,
            x: torch.Tensor,
            *,
            rotary_embeddings=None,
            is_causal: bool = False,
        ) -> torch.Tensor:
            self.rotary_embeddings = rotary_embeddings
            return x

    decoder = Decoder(
        vocab_size=4,
        context_length=4,
        latent_length=2,
        embed_dim=8,
        quantization_dim=4,
        num_heads=2,
    )
    recording_block = RecordingBlock()
    decoder.blocks[0] = recording_block

    logits = decoder(torch.randn(1, 2, 4))

    assert logits.shape == (1, 4, 4)
    cosine, sine = recording_block.rotary_embeddings
    torch.testing.assert_close(cosine, decoder.rope_cosine)
    torch.testing.assert_close(sine, decoder.rope_sine)


def test_decoder_uses_configured_transformer_blocks_and_qk_norm() -> None:
    decoder = Decoder(
        vocab_size=4,
        context_length=4,
        latent_length=2,
        embed_dim=8,
        quantization_dim=4,
        num_heads=2,
        num_layers=4,
        use_qk_norm=True,
    )

    logits = decoder(torch.randn(1, 2, 4))

    assert len(decoder.blocks) == 4
    assert logits.shape == (1, 4, 4)
    for block in decoder.blocks:
        assert isinstance(block.attn.q_norm, RMSNorm)
        assert isinstance(block.attn.k_norm, RMSNorm)


def test_quantizer_builds_and_quantizes_every_scale() -> None:
    quantizer = MultiscaleVectorQuantizer(
        scale_lengths=[1, 2, 4],
        codebook_sizes=[4, 6, 8],
        quantization_dim=2,
        latent_length=4,
    ).eval()
    finest_indices = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])

    result = quantizer.quantize(finest_indices)

    assert result.quantized_latent.shape == (2, 4, 2)
    assert [indices.shape for indices in result.indices_by_scale] == [
        (2, 1),
        (2, 2),
        (2, 4),
    ]
    assert [
        scale.shape
        for scale in quantizer._downsample_to_scales(finest_indices)
    ] == [
        (2, 1, 2),
        (2, 2, 2),
        (2, 4, 2),
    ]


def test_structured_hierarchy_reserves_parent_codes_by_left_child() -> None:
    quantizer = MultiscaleVectorQuantizer(
        scale_lengths=[1, 2, 4],
        codebook_sizes=[8, 8, 4],
        quantization_dim=2,
        latent_length=4,
        group_codes_by_left_child=True,
    ).eval()
    finest_indices = torch.tensor([[0, 1, 2, 3], [3, 0, 1, 2]])

    result = quantizer.quantize(finest_indices)
    scale_one, scale_two, _ = result.indices_by_scale

    # Scale 2 has two possible parent codes for each finest-scale left child.
    torch.testing.assert_close(
        scale_two // 2,
        finest_indices[:, 0::2],
    )
    # Scale 1 has one possible parent code for each scale-2 left child.
    torch.testing.assert_close(scale_one, scale_two[:, 0::2])


def test_one_parent_per_left_child_ignores_right_child_for_assignment() -> None:
    quantizer = MultiscaleVectorQuantizer(
        scale_lengths=[1, 2, 4],
        codebook_sizes=[4, 4, 4],
        quantization_dim=2,
        latent_length=4,
        group_codes_by_left_child=True,
    ).eval()
    finest_indices = torch.tensor([[0, 1, 2, 3], [0, 3, 2, 1]])

    result = quantizer.quantize(finest_indices)

    torch.testing.assert_close(
        result.indices_by_scale[1][0],
        result.indices_by_scale[1][1],
    )
    torch.testing.assert_close(
        result.indices_by_scale[0][0],
        result.indices_by_scale[0][1],
    )


def test_discrete_hierarchy_downsamples_quantized_children() -> None:
    quantizer = MultiscaleVectorQuantizer(
        scale_lengths=[1, 2, 4],
        codebook_sizes=[4, 6, 8],
        quantization_dim=2,
        latent_length=4,
    ).eval()
    finest_indices = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
    downsampler_inputs = []
    hook = quantizer.downsamplers[0].register_forward_pre_hook(
        lambda _module, inputs: downsampler_inputs.append(inputs[0].detach())
    )

    result = quantizer.quantize(finest_indices)
    hook.remove()
    expected_scale_two = quantizer.downsamplers[0](
        result.quantized_latents_by_scale[-1]
    )

    torch.testing.assert_close(
        downsampler_inputs[0], result.quantized_latents_by_scale[-1]
    )
    torch.testing.assert_close(result.latents_by_scale[-2], expected_scale_two)


def test_decode_scales_matches_full_reconstruction_at_final_scale() -> None:
    model = _build_model().eval()
    token_ids = torch.tensor(
        [[0, 1, 2, 3, 3, 2, 1, 0], [3, 2, 1, 0, 0, 1, 2, 3]]
    )

    output = model(token_ids)
    scale_logits = model.decode_scales(output.indices_by_scale)

    assert len(scale_logits) == 3
    assert all(value.shape == (2, 8, 4) for value in scale_logits)
    torch.testing.assert_close(scale_logits[-1], output.logits)


def test_finest_scale_uses_exact_dinucleotide_ids() -> None:
    model = _build_model()
    dinucleotides = torch.tensor(
        [[left, right] for left in range(4) for right in range(4)]
    )
    token_ids = dinucleotides.reshape(4, 8)

    output = model(token_ids)
    finest_codebook = model.quantizer.codebooks[-1]

    assert isinstance(finest_codebook, DeterministicCodebook)
    torch.testing.assert_close(
        output.indices_by_scale[-1].flatten(),
        torch.arange(16),
    )
    assert finest_codebook.codebook.requires_grad
    assert finest_codebook.codebook_hits.count_nonzero() == 16


def test_full_reconstruction_trains_dinucleotide_vectors() -> None:
    model = _build_model()
    token_ids = torch.tensor(
        [[0, 1, 2, 3, 3, 2, 1, 0], [3, 2, 1, 0, 0, 1, 2, 3]]
    )

    output = model(token_ids)
    assert isinstance(output, VQVAEHierarchyOutput)
    F.cross_entropy(
        output.logits.flatten(0, 1), token_ids.flatten()
    ).backward()

    finest_codebook = model.quantizer.codebooks[-1]
    assert isinstance(finest_codebook, DeterministicCodebook)
    assert finest_codebook.codebook.grad is not None
    assert finest_codebook.codebook.grad.count_nonzero() > 0


def test_hierarchy_losses_predict_ordered_children_and_backpropagate() -> None:
    model = _build_model()
    token_ids = torch.tensor(
        [[0, 1, 2, 3, 3, 2, 1, 0], [3, 2, 1, 0, 0, 1, 2, 3]]
    )

    output = model(token_ids)
    assert isinstance(output, VQVAEHierarchyOutput)
    assert [
        (left.shape, right.shape)
        for left, right in output.child_logits_by_scale
    ] == [
        ((2, 1, 8), (2, 1, 8)),
        ((2, 2, 16), (2, 2, 16)),
    ]

    hierarchy_losses = calculate_hierarchy_prediction_losses(output)
    assert hierarchy_losses.shape == (2,)
    assert output.commitment_losses_by_scale.shape == (2,)
    total_hierarchy_loss = (
        hierarchy_losses.mean()
        + 0.25 * output.commitment_losses_by_scale.mean()
    )
    total_hierarchy_loss.backward()

    assert all(
        downsampler.convolution.weight.grad is not None
        and downsampler.convolution.weight.grad.count_nonzero() > 0
        for downsampler in model.quantizer.downsamplers
    )
    assert all(
        predictor.left.weight.grad is not None
        and predictor.left.weight.grad.count_nonzero() > 0
        and predictor.right.weight.grad is not None
        and predictor.right.weight.grad.count_nonzero() > 0
        for predictor in model.child_predictors
    )


def test_parent_prediction_loss_cannot_change_child_representation() -> None:
    model = _build_model()
    token_ids = torch.tensor(
        [[0, 1, 2, 3, 3, 2, 1, 0], [3, 2, 1, 0, 0, 1, 2, 3]]
    )

    output = model(token_ids)
    assert isinstance(output, VQVAEHierarchyOutput)
    scale_two_prediction_loss = calculate_hierarchy_prediction_losses(
        output
    )[1]
    scale_two_prediction_loss.backward()

    finest_codebook = model.quantizer.codebooks[-1]
    assert isinstance(finest_codebook, DeterministicCodebook)
    assert (
        finest_codebook.codebook.grad is None
        or finest_codebook.codebook.grad.count_nonzero() == 0
    )
    assert (
        model.quantizer.downsamplers[0].convolution.weight.grad is not None
    )
    assert (
        model.quantizer.downsamplers[0]
        .convolution.weight.grad.count_nonzero()
        > 0
    )
    assert model.child_predictors[1].left.weight.grad is not None
    assert model.child_predictors[1].right.weight.grad is not None


def test_evaluate_reports_diagnostics_by_scale() -> None:
    model = _build_model()
    batches = [
        {
            "input_ids": torch.tensor(
                [[0, 1, 2, 3, 3, 2, 1, 0], [3, 2, 1, 0, 0, 1, 2, 3]]
            )
        },
    ]

    metrics = evaluate(
        model,
        batches,
        use_mixed_precision=False,
        hierarchy_prediction_weight=1.0,
        commitment_cost=0.25,
    )

    assert "finest_latent_rms" in metrics
    assert "hierarchy_prediction_loss" in metrics
    assert "commitment_loss" in metrics
    for scale_length in model.scale_lengths:
        assert f"latent_mse_scale_{scale_length}" in metrics
        assert f"scale_latent_rms_scale_{scale_length}" in metrics
        assert f"codebook_perplexity_scale_{scale_length}" in metrics
        assert f"codebook_rms_scale_{scale_length}" in metrics
        if scale_length != model.scale_lengths[-1]:
            assert f"child_prediction_loss_scale_{scale_length}" in metrics
            assert f"child_prediction_accuracy_scale_{scale_length}" in metrics
            assert f"left_child_accuracy_scale_{scale_length}" in metrics
            assert f"right_child_accuracy_scale_{scale_length}" in metrics
            assert f"commitment_loss_scale_{scale_length}" in metrics
    assert "child_prediction_loss_scale_4" not in metrics
    assert "commitment_loss_scale_4" not in metrics
