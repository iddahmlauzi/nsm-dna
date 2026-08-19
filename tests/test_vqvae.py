import torch
import torch.nn.functional as F

from nsm_dna.models.quantization import MultiscaleResidualVectorQuantizer
from nsm_dna.models.vqvae import VQVAE
from scripts.train_vqvae import evaluate


def test_first_scale_sampler_uses_a_cascade() -> None:
    quantizer = MultiscaleResidualVectorQuantizer(
        scale_lengths=[4, 256],
        codebook_sizes=[4, 4],
        embed_dim=2,
    )
    channels_first = torch.arange(2 * 256, dtype=torch.float32).reshape(1, 2, 256)

    downsampled = quantizer.first_scale_downsampler(channels_first)
    normalized = quantizer._downsample_to_scale(
        channels_first.transpose(1, 2),
        scale_index=0,
    )
    expected_normalized = F.layer_norm(
        downsampled.transpose(1, 2),
        normalized_shape=(2,),
    )
    upsampled = quantizer.first_scale_upsampler(normalized.transpose(1, 2))

    assert len(quantizer.first_scale_downsampler) == 5
    assert len(quantizer.first_scale_upsampler) == 3
    internal_norms = [
        module
        for module in quantizer.first_scale_downsampler
        if hasattr(module, "normalization")
    ]
    assert len(internal_norms) == 2
    assert all(
        norm.normalization.elementwise_affine is False for norm in internal_norms
    )
    assert quantizer.first_scale_norm.elementwise_affine is False
    assert downsampled.shape == (1, 2, 4)
    assert upsampled.shape == channels_first.shape
    torch.testing.assert_close(
        upsampled,
        normalized.transpose(1, 2).repeat_interleave(64, dim=-1),
    )
    torch.testing.assert_close(normalized, expected_normalized)


def test_cumulative_decode_matches_full_reconstruction() -> None:
    model = VQVAE(
        vocab_size=4,
        context_length=4,
        embed_dim=8,
        num_heads=2,
        scale_lengths=[1, 2, 4],
        codebook_sizes=[8, 8, 8],
        encoder_dropout=0.0,
        decoder_dropout=0.0,
        pre_quant_num_groups=2,
    )
    model.eval()
    token_ids = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])

    logits, partial_logits, _, indices_by_scale = model(token_ids)
    cumulative_logits = model.decode_cumulative(indices_by_scale)

    assert partial_logits is None
    assert len(cumulative_logits) == 3
    assert all(scale_logits.shape == (2, 4, 4) for scale_logits in cumulative_logits)
    torch.testing.assert_close(cumulative_logits[-1], logits)


def test_partial_reconstruction_trains_quantizer_without_moving_encoder() -> None:
    model = VQVAE(
        vocab_size=4,
        context_length=4,
        embed_dim=8,
        num_heads=2,
        scale_lengths=[1, 2, 4],
        codebook_sizes=[8, 8, 8],
        encoder_dropout=0.0,
        decoder_dropout=0.0,
        pre_quant_num_groups=2,
    )
    token_ids = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])

    logits, partial_logits, _, _ = model(
        token_ids,
        include_partial_reconstruction=True,
    )

    assert logits.shape == (2, 4, 4)
    assert partial_logits is not None
    assert partial_logits.shape == logits.shape

    partial_loss = F.cross_entropy(
        partial_logits.flatten(0, 1),
        token_ids.flatten(),
    )
    partial_loss.backward()
    assert all(
        parameter.grad is None or parameter.grad.count_nonzero() == 0
        for parameter in model.encoder.parameters()
    )
    assert any(
        parameter.grad is not None and parameter.grad.count_nonzero() > 0
        for parameter in model.quantizer.parameters()
    )
    assert any(
        parameter.grad is not None and parameter.grad.count_nonzero() > 0
        for parameter in model.decoder.parameters()
    )


def test_partial_reconstruction_uses_separate_gradient_scales() -> None:
    def build_model() -> VQVAE:
        return VQVAE(
            vocab_size=4,
            context_length=4,
            embed_dim=8,
            num_heads=2,
            scale_lengths=[1, 2, 4],
            codebook_sizes=[8, 8, 8],
            encoder_dropout=0.0,
            decoder_dropout=0.0,
            pre_quant_num_groups=2,
        ).eval()

    def partial_gradients(
        model: VQVAE,
        token_ids: torch.Tensor,
        *,
        loss_weight: float,
        latent_gradient_scale: float,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        torch.manual_seed(0)
        _, partial_logits, _, _ = model(
            token_ids,
            include_partial_reconstruction=True,
            partial_latent_gradient_scale=latent_gradient_scale,
        )
        assert partial_logits is not None
        partial_loss = F.cross_entropy(
            partial_logits.flatten(0, 1),
            token_ids.flatten(),
        )
        (loss_weight * partial_loss).backward()
        quantizer_gradients = [
            parameter.grad.detach().clone()
            for parameter in model.quantizer.parameters()
            if parameter.grad is not None
        ]
        decoder_gradients = [
            parameter.grad.detach().clone()
            for parameter in model.decoder.parameters()
            if parameter.grad is not None
        ]
        return quantizer_gradients, decoder_gradients

    baseline_model = build_model()
    split_model = build_model()
    split_model.load_state_dict(baseline_model.state_dict())
    token_ids = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])

    baseline_quantizer, baseline_decoder = partial_gradients(
        baseline_model,
        token_ids,
        loss_weight=1.0,
        latent_gradient_scale=1.0,
    )
    split_quantizer, split_decoder = partial_gradients(
        split_model,
        token_ids,
        loss_weight=0.01,
        latent_gradient_scale=20.0,
    )

    assert len(baseline_quantizer) == len(split_quantizer)
    assert len(baseline_decoder) == len(split_decoder)
    for baseline_gradient, split_gradient in zip(
        baseline_quantizer, split_quantizer
    ):
        torch.testing.assert_close(split_gradient, 0.2 * baseline_gradient)
    for baseline_gradient, split_gradient in zip(baseline_decoder, split_decoder):
        torch.testing.assert_close(split_gradient, 0.01 * baseline_gradient)


def test_evaluate_reports_vq_diagnostics_by_scale() -> None:
    model = VQVAE(
        vocab_size=4,
        context_length=4,
        embed_dim=8,
        num_heads=2,
        scale_lengths=[1, 2, 4],
        codebook_sizes=[8, 8, 8],
        encoder_dropout=0.0,
        decoder_dropout=0.0,
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
    assert "first_scale_pre_norm_rms" in metrics
    assert "first_scale_post_norm_rms" in metrics
    mean_cumulative_latent_mse = sum(
        metrics[f"cumulative_latent_mse_scale_{scale_length}"]
        for scale_length in model.scale_lengths
    ) / len(model.scale_lengths)
    torch.testing.assert_close(
        torch.tensor(metrics["vq_loss"]),
        torch.tensor(1.25 * mean_cumulative_latent_mse),
    )

    for scale_length in model.scale_lengths:
        assert f"cumulative_latent_mse_scale_{scale_length}" in metrics
        assert f"contribution_rms_scale_{scale_length}" in metrics
        assert f"codebook_perplexity_scale_{scale_length}" in metrics
        assert f"codebook_rms_scale_{scale_length}" in metrics
