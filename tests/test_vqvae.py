import torch
import torch.nn as nn
import torch.nn.functional as F

from nsm_dna.models.autoencoder import Decoder, Encoder
from nsm_dna.models.common import RMSNorm
from nsm_dna.models.quantization import MultiscaleVectorQuantizer
from nsm_dna.models.vqvae import VQVAE
from scripts.training.train_vqvae import evaluate


def test_encoder_uses_one_stride_four_sampler_for_fourfold_reduction() -> None:
    encoder = Encoder(
        vocab_size=4,
        context_length=8,
        latent_length=2,
        embed_dim=4,
        quantization_dim=2,
        dropout=0.0,
    )
    token_ids = torch.tensor([[3, 1, 0, 2, 2, 0, 1, 3]])

    latent = encoder(token_ids)
    assert latent.shape == (1, 2, 2)
    assert isinstance(encoder.downsampler, nn.Conv1d)
    assert encoder.downsampler.stride == (4,)


def test_encoder_contextualizes_before_downsampling() -> None:
    class RecordingBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_shape = None

        def forward(
            self,
            x: torch.Tensor,
            **kwargs,
        ) -> torch.Tensor:
            self.input_shape = x.shape
            return x

    encoder = Encoder(
        vocab_size=4,
        context_length=8,
        latent_length=2,
        embed_dim=8,
        quantization_dim=2,
        num_heads=2,
        num_layers=1,
        dropout=0.0,
    )
    recording_block = RecordingBlock()
    encoder.context_blocks[0] = recording_block

    latent = encoder(torch.tensor([[3, 1, 0, 2, 2, 0, 1, 3]]))

    assert recording_block.input_shape == (1, 8, 8)
    assert latent.shape == (1, 2, 2)


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
        dropout=0.0,
    )
    recording_block = RecordingBlock()
    decoder.blocks[0] = recording_block

    logits = decoder(torch.randn(1, 2, 4))

    assert logits.shape == (1, 4, 4)
    assert recording_block.rotary_embeddings is not None
    cosine, sine = recording_block.rotary_embeddings
    torch.testing.assert_close(cosine, decoder.rope_cosine)
    torch.testing.assert_close(sine, decoder.rope_sine)


def test_decoder_uses_configured_number_of_transformer_blocks() -> None:
    decoder = Decoder(
        vocab_size=4,
        context_length=4,
        latent_length=2,
        embed_dim=8,
        quantization_dim=4,
        num_heads=2,
        num_layers=4,
        dropout=0.0,
    )

    logits = decoder(torch.randn(1, 2, 4))

    assert len(decoder.blocks) == 4
    assert logits.shape == (1, 4, 4)


def test_decoder_normalizes_queries_and_keys() -> None:
    decoder = Decoder(
        vocab_size=4,
        context_length=4,
        latent_length=2,
        embed_dim=8,
        quantization_dim=4,
        num_heads=2,
        num_layers=4,
        use_qk_norm=True,
        dropout=0.0,
    )

    for block in decoder.blocks:
        assert isinstance(block.attn.q_norm, RMSNorm)
        assert isinstance(block.attn.k_norm, RMSNorm)


def test_vqvae_applies_qk_norm_to_encoder_and_decoder() -> None:
    model = VQVAE(
        vocab_size=4,
        context_length=4,
        latent_length=4,
        embed_dim=8,
        quantization_dim=4,
        num_heads=2,
        scale_lengths=[1, 2, 4],
        codebook_sizes=[8, 8, 8],
        encoder_num_layers=1,
        decoder_num_layers=2,
        use_qk_norm=True,
        dropout=0.0,
        pre_quant_num_groups=2,
    )

    blocks = [*model.encoder.context_blocks, *model.decoder.blocks]
    for block in blocks:
        assert isinstance(block.attn.q_norm, RMSNorm)
        assert isinstance(block.attn.k_norm, RMSNorm)


def test_first_scale_sampler_uses_a_cascade() -> None:
    quantizer = MultiscaleVectorQuantizer(
        scale_lengths=[4, 256],
        codebook_sizes=[4, 4],
        embed_dim=2,
        latent_length=256,
    )
    channels_first = torch.arange(2 * 256, dtype=torch.float32).reshape(1, 2, 256)

    downsampled = quantizer.first_scale_downsampler(channels_first)
    normalized = quantizer._resize_to_scale(
        channels_first.transpose(1, 2),
        scale_index=0,
    )
    upsampled = quantizer.first_scale_upsampler(normalized.transpose(1, 2))

    assert len(quantizer.first_scale_downsampler) == 6
    assert len(quantizer.first_scale_upsampler) == 3
    internal_norms = [
        module
        for module in quantizer.first_scale_downsampler
        if hasattr(module, "normalization")
    ]
    assert len(internal_norms) == 3
    assert all(
        norm.normalization.elementwise_affine is False for norm in internal_norms
    )
    assert downsampled.shape == (1, 2, 4)
    assert upsampled.shape == channels_first.shape
    torch.testing.assert_close(
        upsampled,
        normalized.transpose(1, 2).repeat_interleave(64, dim=-1),
    )
    torch.testing.assert_close(normalized, downsampled.transpose(1, 2))


def test_code_corruption_only_changes_training_decoder_input(monkeypatch) -> None:
    quantizer = MultiscaleVectorQuantizer(
        scale_lengths=[1, 2],
        codebook_sizes=[2, 2],
        embed_dim=1,
        latent_length=2,
        code_corruption=True,
        refinement_ratio=0.0,
    )
    with torch.no_grad():
        for codebook in quantizer.codebooks:
            codebook.codebook.copy_(torch.tensor([[0.0], [1.0]]))

    latent = torch.zeros(1, 2, 1)
    quantizer.eval()
    clean_decoder_latent, _, clean_vq_loss, clean_indices = quantizer(latent)

    monkeypatch.setattr(torch, "rand", lambda *_, **__: torch.ones(1, 1))
    monkeypatch.setattr(
        torch,
        "rand_like",
        lambda tensor, **__: torch.zeros_like(tensor, dtype=torch.float32),
    )
    monkeypatch.setattr(
        torch,
        "randint_like",
        lambda tensor, _: torch.ones_like(tensor),
    )
    quantizer.train()
    for codebook in quantizer.codebooks:
        codebook.eval()
    corrupted_decoder_latent, _, corrupted_vq_loss, corrupted_indices = quantizer(
        latent
    )

    for clean_scale_indices, corrupted_scale_indices in zip(
        clean_indices,
        corrupted_indices,
        strict=True,
    ):
        torch.testing.assert_close(corrupted_scale_indices, clean_scale_indices)
    torch.testing.assert_close(corrupted_vq_loss, clean_vq_loss)
    torch.testing.assert_close(clean_decoder_latent, torch.zeros_like(latent))
    torch.testing.assert_close(corrupted_decoder_latent, torch.ones_like(latent))

    quantizer.eval()
    torch.testing.assert_close(quantizer(latent)[0], clean_decoder_latent)


def test_vq_loss_averages_every_cumulative_scale() -> None:
    quantizer = MultiscaleVectorQuantizer(
        scale_lengths=[1, 2, 4],
        codebook_sizes=[4, 4, 4],
        embed_dim=2,
        latent_length=4,
    ).eval()
    latent = torch.randn(2, 4, 2)

    _, _, vq_loss, indices_by_scale = quantizer(latent)
    cumulative_latents = quantizer.indices_to_cumulative_latents(indices_by_scale)
    expected_loss = torch.stack(
        [
            (1 + quantizer.commitment_cost)
            * F.mse_loss(cumulative_latent, latent)
            for cumulative_latent in cumulative_latents
        ]
    ).mean()

    torch.testing.assert_close(vq_loss, expected_loss)


def test_cumulative_decode_matches_full_reconstruction() -> None:
    model = VQVAE(
        vocab_size=4,
        context_length=4,
        latent_length=4,
        embed_dim=8,
        quantization_dim=4,
        num_heads=2,
        scale_lengths=[1, 2, 4],
        codebook_sizes=[8, 8, 8],
        dropout=0.0,
        pre_quant_num_groups=2,
    )
    model.eval()
    token_ids = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])

    logits, truncated_logits, _, indices_by_scale = model(token_ids)
    cumulative_logits = model.decode_cumulative(indices_by_scale)

    assert truncated_logits is None
    assert len(cumulative_logits) == 3
    assert all(scale_logits.shape == (2, 4, 4) for scale_logits in cumulative_logits)
    torch.testing.assert_close(cumulative_logits[-1], logits)


def test_quantization_operates_at_shorter_learned_latent_length() -> None:
    model = VQVAE(
        vocab_size=4,
        context_length=8,
        latent_length=4,
        embed_dim=8,
        quantization_dim=4,
        num_heads=2,
        scale_lengths=[1, 2, 4],
        codebook_sizes=[8, 8, 8],
        dropout=0.0,
        pre_quant_num_groups=2,
    ).eval()
    token_ids = torch.tensor(
        [[0, 1, 2, 3, 3, 2, 1, 0], [3, 2, 1, 0, 0, 1, 2, 3]]
    )

    latent = model.encode(token_ids)
    logits, _, _, indices_by_scale = model(token_ids)
    cumulative_logits = model.decode_cumulative(indices_by_scale)

    assert latent.shape == (2, 4, 4)
    assert model.quantizer.codebooks[0].codebook.shape == (8, 4)
    assert logits.shape == (2, 8, 4)
    assert [indices.shape for indices in indices_by_scale] == [
        (2, 1),
        (2, 2),
        (2, 4),
    ]
    assert all(scale_logits.shape == (2, 8, 4) for scale_logits in cumulative_logits)
    torch.testing.assert_close(cumulative_logits[-1], logits)


def test_encode_returns_continuous_prefix_latents() -> None:
    model = VQVAE(
        vocab_size=4,
        context_length=4,
        latent_length=4,
        embed_dim=8,
        quantization_dim=4,
        num_heads=2,
        scale_lengths=[1, 2, 4],
        codebook_sizes=[8, 8, 8],
        dropout=0.0,
        pre_quant_num_groups=2,
    ).eval()
    token_ids = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])

    prefix_latents = model.encode(token_ids)
    encoder_latents = model.encoder(token_ids)
    expected_latents = model.pre_quant_norm(
        encoder_latents.transpose(1, 2)
    ).transpose(1, 2)

    assert prefix_latents.shape == (2, 4, 4)
    torch.testing.assert_close(prefix_latents, expected_latents)


def test_truncated_reconstruction_uses_straight_through_encoder_gradient() -> None:
    model = VQVAE(
        vocab_size=4,
        context_length=8,
        latent_length=4,
        embed_dim=8,
        quantization_dim=4,
        num_heads=2,
        scale_lengths=[1, 2, 4],
        codebook_sizes=[8, 8, 8],
        dropout=0.0,
        pre_quant_num_groups=2,
    )
    token_ids = torch.tensor(
        [[0, 1, 2, 3, 3, 2, 1, 0], [3, 2, 1, 0, 0, 1, 2, 3]]
    )

    logits, truncated_logits, _, _ = model(
        token_ids,
        include_truncated_reconstruction=True,
    )

    assert logits.shape == (2, 8, 4)
    assert truncated_logits is not None
    assert truncated_logits.shape == logits.shape

    truncated_loss = F.cross_entropy(
        truncated_logits.flatten(0, 1),
        token_ids.flatten(),
    )
    truncated_loss.backward()
    assert any(
        parameter.grad is not None and parameter.grad.count_nonzero() > 0
        for parameter in model.encoder.parameters()
    )
    assert all(
        parameter.grad is None or parameter.grad.count_nonzero() == 0
        for parameter in model.quantizer.parameters()
    )
    assert any(
        parameter.grad is not None and parameter.grad.count_nonzero() > 0
        for parameter in model.decoder.parameters()
    )


def test_evaluate_reports_vq_diagnostics_by_scale() -> None:
    model = VQVAE(
        vocab_size=4,
        context_length=4,
        latent_length=4,
        embed_dim=8,
        quantization_dim=4,
        num_heads=2,
        scale_lengths=[1, 2, 4],
        codebook_sizes=[8, 8, 8],
        dropout=0.0,
        pre_quant_num_groups=2,
    )
    batches = [
        {"input_ids": torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])},
    ]

    metrics = evaluate(
        model,
        batches,
        use_mixed_precision=False,
    )

    assert "encoder_latent_rms" in metrics
    mean_cumulative_latent_mse = sum(
        metrics[f"cumulative_latent_mse_scale_{scale_length}"]
        for scale_length in model.scale_lengths
    ) / len(model.scale_lengths)
    torch.testing.assert_close(
        torch.tensor(metrics["vq_loss"]),
        torch.tensor(
            (1 + model.quantizer.commitment_cost) * mean_cumulative_latent_mse
        ),
    )

    for scale_length in model.scale_lengths:
        assert f"cumulative_latent_mse_scale_{scale_length}" in metrics
        assert f"contribution_rms_scale_{scale_length}" in metrics
        assert f"codebook_perplexity_scale_{scale_length}" in metrics
        assert f"codebook_rms_scale_{scale_length}" in metrics
