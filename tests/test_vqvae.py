import torch
import torch.nn as nn
import torch.nn.functional as F

from nsm_dna.models.autoencoder import Decoder, Encoder
from nsm_dna.models.common import RMSNorm
from nsm_dna.models.quantization import Codebook, MultiscaleVectorQuantizer
from nsm_dna.models.vqvae import VQVAE
from scripts.training.train_vqvae import evaluate


def _build_model(*, decoder_num_layers: int = 1) -> VQVAE:
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


def test_encoder_uses_configured_non_overlapping_sampling_factor() -> None:
    encoder = Encoder(
        vocab_size=4,
        context_length=9,
        latent_length=3,
        embed_dim=4,
        quantization_dim=2,
    )
    token_ids = torch.tensor([[3, 1, 0, 2, 2, 0, 1, 3, 2]])

    latent = encoder(token_ids)

    assert latent.shape == (1, 3, 2)
    assert isinstance(encoder.downsampler, nn.Conv1d)
    assert encoder.downsampler.kernel_size == (3,)
    assert encoder.downsampler.stride == (3,)


def test_codebook_uses_hard_assignments_with_soft_gradients() -> None:
    codebook = Codebook(codebook_size=4, quantization_dim=3)
    latent = torch.randn(2, 3, 3, requires_grad=True)

    quantized, indices, assignments = codebook(latent)

    expected_assignments = F.one_hot(indices, num_classes=4).float()
    torch.testing.assert_close(assignments, expected_assignments)

    quantized.square().sum().backward()
    assert latent.grad is not None
    assert latent.grad.count_nonzero() > 0
    assert codebook.codebook.grad is not None
    assert codebook.codebook.grad.count_nonzero() > 0


def test_single_scale_triplet_vqvae() -> None:
    model = VQVAE(
        vocab_size=4,
        context_length=9,
        latent_length=3,
        embed_dim=8,
        quantization_dim=4,
        num_heads=2,
        scale_lengths=[3],
        codebook_sizes=[21],
    )
    batches = [{"input_ids": torch.randint(0, 4, (2, 9))}]

    logits, partial_logits, indices_by_scale, assignments_by_scale = model(
        batches[0]["input_ids"], include_partial_reconstruction=True
    )
    metrics = evaluate(
        model,
        batches,
        use_mixed_precision=False,
        partial_reconstruction_weight=0.25,
    )

    assert logits.shape == (2, 9, 4)
    assert partial_logits is None
    assert indices_by_scale[0].shape == (2, 3)
    assert assignments_by_scale[0].shape == (2, 3, 21)
    assert metrics["partial_reconstruction_loss"] == 0.0
    assert metrics["total_loss"] == metrics["full_reconstruction_loss"]


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
    latent = torch.randn(2, 4, 2)

    (
        quantized_latent,
        partial_latent,
        indices_by_scale,
        assignments_by_scale,
    ) = quantizer(latent)

    assert quantized_latent.shape == latent.shape
    assert partial_latent is None
    assert [indices.shape for indices in indices_by_scale] == [
        (2, 1),
        (2, 2),
        (2, 4),
    ]
    assert [assignments.shape for assignments in assignments_by_scale] == [
        (2, 1, 4),
        (2, 2, 6),
        (2, 4, 8),
    ]
    assert [scale.shape for scale in quantizer._downsample_to_scales(latent)] == [
        (2, 1, 2),
        (2, 2, 2),
        (2, 4, 2),
    ]


def test_decode_scales_matches_full_reconstruction_at_final_scale() -> None:
    model = _build_model().eval()
    token_ids = torch.tensor(
        [[0, 1, 2, 3, 3, 2, 1, 0], [3, 2, 1, 0, 0, 1, 2, 3]]
    )

    logits, partial_logits, indices_by_scale, _ = model(token_ids)
    scale_logits = model.decode_scales(indices_by_scale)

    assert partial_logits is None
    assert len(scale_logits) == 3
    assert all(value.shape == (2, 8, 4) for value in scale_logits)
    torch.testing.assert_close(scale_logits[-1], logits)


def test_encode_returns_continuous_latents() -> None:
    model = _build_model().eval()
    token_ids = torch.tensor(
        [[0, 1, 2, 3, 3, 2, 1, 0], [3, 2, 1, 0, 0, 1, 2, 3]]
    )

    latent = model.encode(token_ids)

    assert latent.shape == (2, 4, 4)
    torch.testing.assert_close(latent, model.encoder(token_ids))


def test_partial_reconstruction_backpropagates_through_encoder() -> None:
    model = _build_model()
    token_ids = torch.tensor(
        [[0, 1, 2, 3, 3, 2, 1, 0], [3, 2, 1, 0, 0, 1, 2, 3]]
    )

    _, partial_logits, _, _ = model(
        token_ids,
        include_partial_reconstruction=True,
    )

    assert partial_logits is not None
    F.cross_entropy(
        partial_logits.flatten(0, 1),
        token_ids.flatten(),
    ).backward()
    assert any(
        parameter.grad is not None and parameter.grad.count_nonzero() > 0
        for parameter in model.encoder.parameters()
    )
    assert any(
        parameter.grad is not None and parameter.grad.count_nonzero() > 0
        for parameter in model.quantizer.downsamplers.parameters()
    )


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
        partial_reconstruction_weight=0.25,
    )

    assert "encoder_latent_rms" in metrics
    for scale_length in model.scale_lengths:
        assert f"latent_mse_scale_{scale_length}" in metrics
        assert f"scale_latent_rms_scale_{scale_length}" in metrics
        assert f"codebook_perplexity_scale_{scale_length}" in metrics
        assert f"codebook_rms_scale_{scale_length}" in metrics
