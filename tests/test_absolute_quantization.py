import torch

from nsm_dna.models.quantization import MultiscaleVectorQuantizer
from nsm_dna.models.vqvae import VQVAE
from scripts.training.train_vqvae import evaluate


def test_learned_downsampling_preserves_left_right_order() -> None:
    quantizer = MultiscaleVectorQuantizer(
        scale_lengths=[1, 2, 4],
        codebook_sizes=[2, 2, 4],
        quantization_dim=2,
        latent_length=4,
    ).eval()
    with torch.no_grad():
        quantizer.codebooks[-1].codebook.copy_(
            torch.tensor([[1.0, 0.0], [3.0, 0.0], [2.0, 0.0], [4.0, 0.0]])
        )
        first_downsampler = quantizer.downsamplers[0].convolution
        first_downsampler.weight.zero_()
        first_downsampler.weight[0, 0, 0] = 1.0
        first_downsampler.weight[1, 0, 1] = 1.0

    finest_indices = torch.tensor([[0, 1, 2, 3]])
    swapped_indices = torch.tensor([[1, 0, 2, 3]])

    scale_two_latent = quantizer._downsample_to_scales(finest_indices)[1]
    swapped_scale_two_latent = quantizer._downsample_to_scales(swapped_indices)[1]

    assert not torch.allclose(
        scale_two_latent[:, 0],
        swapped_scale_two_latent[:, 0],
    )
    torch.testing.assert_close(
        scale_two_latent[:, 1],
        swapped_scale_two_latent[:, 1],
    )


def test_vqvae_decodes_each_scale_independently() -> None:
    model = VQVAE(
        vocab_size=4,
        context_length=8,
        latent_length=4,
        embed_dim=8,
        quantization_dim=4,
        num_heads=2,
        scale_lengths=[1, 2, 4],
        codebook_sizes=[4, 4, 16],
    ).eval()
    token_ids = torch.tensor([[0, 1, 2, 3, 3, 2, 1, 0]])

    with torch.no_grad():
        output = model(token_ids)
        scale_logits = model.decode_scales(output.indices_by_scale)

    assert len(scale_logits) == 3
    assert all(value.shape == output.logits.shape for value in scale_logits)
    torch.testing.assert_close(scale_logits[-1], output.logits)
    torch.testing.assert_close(
        model.decode_scale(output.indices_by_scale[0], 0), scale_logits[0]
    )


def test_validation_reports_independent_scale_reconstruction() -> None:
    model = VQVAE(
        vocab_size=4,
        context_length=8,
        latent_length=4,
        embed_dim=8,
        quantization_dim=4,
        num_heads=2,
        scale_lengths=[1, 2, 4],
        codebook_sizes=[4, 4, 16],
    )
    batch = {"input_ids": torch.tensor([[0, 1, 2, 3, 3, 2, 1, 0]])}

    metrics = evaluate(
        model,
        [batch],
        use_mixed_precision=False,
        hierarchy_prediction_weight=1.0,
        commitment_cost=0.25,
    )

    assert "accuracy_scale_1" in metrics
    assert "accuracy_scale_2" in metrics
    assert "accuracy_scale_4" in metrics
    assert "scale_latent_rms_scale_1" in metrics
    assert metrics["accuracy_scale_4"] == metrics["accuracy"]
    assert abs(
        metrics["reconstruction_loss_scale_4"] - metrics["full_reconstruction_loss"]
    ) < 1e-6
