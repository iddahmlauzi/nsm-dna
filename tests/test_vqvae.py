import torch
import torch.nn.functional as F

from nsm_dna.models.quantization import MultiscaleResidualVectorQuantizer
from nsm_dna.models.vqvae import VQVAE
from scripts.train_vqvae import evaluate


def test_first_scale_learned_sampling_initialization() -> None:
    quantizer = MultiscaleResidualVectorQuantizer(
        scale_lengths=[1, 4],
        codebook_sizes=[4, 4],
        embed_dim=2,
    )
    residual = torch.arange(8, dtype=torch.float32).reshape(1, 4, 2)
    contribution = torch.tensor([[[2.0, 4.0]]])

    downsampled = quantizer._downsample_to_scale(residual, scale_index=0)
    upsampled = quantizer._upsample_to_full_length(
        contribution.transpose(1, 2),
        scale_index=0,
    )

    expected_downsampled = F.layer_norm(
        residual.mean(dim=1, keepdim=True),
        normalized_shape=(2,),
    )
    torch.testing.assert_close(downsampled, expected_downsampled)
    torch.testing.assert_close(
        upsampled,
        contribution.transpose(1, 2).expand(1, 2, 4),
    )


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
        fine_dropout_prob=0.0,
    )
    model.eval()
    token_ids = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])

    logits, _, indices_by_scale = model(token_ids)
    cumulative_logits = model.decode_cumulative(indices_by_scale)

    assert len(cumulative_logits) == 3
    assert all(scale_logits.shape == (2, 4, 4) for scale_logits in cumulative_logits)
    torch.testing.assert_close(cumulative_logits[-1], logits)


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
        fine_dropout_prob=0.0,
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
        torch.tensor(1.25 * mean_cumulative_latent_mse),
    )

    for scale_length in model.scale_lengths:
        assert f"cumulative_latent_mse_scale_{scale_length}" in metrics
        assert f"contribution_rms_scale_{scale_length}" in metrics
        assert f"codebook_perplexity_scale_{scale_length}" in metrics
        assert f"codebook_rms_scale_{scale_length}" in metrics
