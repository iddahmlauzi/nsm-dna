import torch

from nsm_dna.models.quantization import MultiscaleVectorQuantizer
from nsm_dna.models.vqvae import VQVAE
from scripts.training.train_vqvae import evaluate


def test_each_scale_quantizes_the_original_latent() -> None:
    quantizer = MultiscaleVectorQuantizer(
        scale_lengths=[1, 2],
        codebook_sizes=[2, 2],
        quantization_dim=1,
        latent_length=4,
    ).eval()
    with torch.no_grad():
        quantizer.codebooks[0].codebook.copy_(torch.tensor([[5.0], [100.0]]))
        for codebook in quantizer.codebooks[1:]:
            codebook.codebook.copy_(torch.tensor([[0.0], [10.0]]))

    latent = torch.tensor([[[0.0], [0.0], [10.0], [10.0]]])
    _, indices = quantizer(latent)
    scale_latents = quantizer.indices_to_scale_latents(indices)

    torch.testing.assert_close(indices[0], torch.tensor([[0]]))
    torch.testing.assert_close(indices[1], torch.tensor([[0, 1]]))
    torch.testing.assert_close(scale_latents[0], torch.full_like(latent, 5.0))
    torch.testing.assert_close(scale_latents[1], latent)


def test_partial_reconstruction_gradient_flows_through_pooling() -> None:
    quantizer = MultiscaleVectorQuantizer(
        scale_lengths=[1],
        codebook_sizes=[2],
        quantization_dim=1,
        latent_length=4,
    ).eval()
    latent = torch.randn(1, 4, 1, requires_grad=True)

    partial_latent, _ = quantizer(
        latent, include_partial_reconstruction=True
    )
    assert partial_latent is not None
    partial_latent.sum().backward()

    torch.testing.assert_close(latent.grad, torch.ones_like(latent))


def test_vqvae_decodes_each_scale_independently() -> None:
    model = VQVAE(
        vocab_size=4,
        context_length=8,
        latent_length=2,
        embed_dim=8,
        quantization_dim=4,
        num_heads=2,
        scale_lengths=[1, 2],
        codebook_sizes=[4, 256],
    ).eval()
    token_ids = torch.tensor([[0, 1, 2, 3, 3, 2, 1, 0]])

    with torch.no_grad():
        logits, _, indices = model(token_ids)
        scale_logits = model.decode_scales(indices)
        latent = model.encode(token_ids)
        final_vectors = model.final_codebook_vectors()[indices[-1]]

    assert len(scale_logits) == 2
    assert all(value.shape == logits.shape for value in scale_logits)
    torch.testing.assert_close(scale_logits[-1], logits)
    torch.testing.assert_close(final_vectors, latent)
    torch.testing.assert_close(model.decode_scale(indices[0], 0), scale_logits[0])


def test_validation_reports_independent_scale_reconstruction() -> None:
    model = VQVAE(
        vocab_size=4,
        context_length=8,
        latent_length=2,
        embed_dim=8,
        quantization_dim=4,
        num_heads=2,
        scale_lengths=[1, 2],
        codebook_sizes=[4, 256],
    )
    batch = {"input_ids": torch.tensor([[0, 1, 2, 3, 3, 2, 1, 0]])}

    metrics = evaluate(
        model,
        [batch],
        use_mixed_precision=False,
        partial_reconstruction_weight=0.25,
    )

    assert "accuracy_scale_1" in metrics
    assert "accuracy_scale_2" in metrics
    assert "scale_latent_rms_scale_1" in metrics
    assert metrics["accuracy_scale_2"] == metrics["accuracy"]
    assert abs(
        metrics["reconstruction_loss_scale_2"] - metrics["full_reconstruction_loss"]
    ) < 1e-6
