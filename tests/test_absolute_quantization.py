import torch

from nsm_dna.models.quantization import MultiscaleVectorQuantizer
from nsm_dna.models.vqvae import VQVAE
from scripts.training.train_vqvae import evaluate


def test_learned_downsampling_preserves_left_right_order() -> None:
    quantizer = MultiscaleVectorQuantizer(
        scale_lengths=[1, 2, 4],
        codebook_sizes=[2, 2, 2],
        quantization_dim=2,
        latent_length=4,
    ).eval()
    with torch.no_grad():
        first_downsampler = quantizer.downsamplers[0].convolution
        first_downsampler.weight.zero_()
        first_downsampler.weight[0, 0, 0] = 1.0
        first_downsampler.weight[1, 0, 1] = 1.0

    latent = torch.tensor([[[1.0, 0.0], [3.0, 0.0], [2.0, 0.0], [4.0, 0.0]]])
    swapped_latent = latent.clone()
    swapped_latent[:, :2] = latent[:, :2].flip(dims=[1])

    scale_two_latent = quantizer.downsample_to_scales(latent, latent)[1]
    swapped_scale_two_latent = quantizer.downsample_to_scales(
        swapped_latent,
        swapped_latent,
    )[1]

    assert not torch.allclose(
        scale_two_latent[:, 0],
        swapped_scale_two_latent[:, 0],
    )
    torch.testing.assert_close(
        scale_two_latent[:, 1],
        swapped_scale_two_latent[:, 1],
    )


def test_partial_reconstruction_gradient_flows_through_learned_downsampling(
    monkeypatch,
) -> None:
    quantizer = MultiscaleVectorQuantizer(
        scale_lengths=[1, 2, 4],
        codebook_sizes=[2, 2, 2],
        quantization_dim=4,
        latent_length=4,
    ).eval()
    latent = torch.randn(1, 4, 4, requires_grad=True)

    monkeypatch.setattr(torch, "randint", lambda *args, **kwargs: torch.tensor(0))
    _, partial_latent, _ = quantizer(
        latent,
        latent,
        include_partial_reconstruction=True,
    )
    assert partial_latent is not None
    partial_latent[..., 0].sum().backward()

    assert latent.grad is not None
    assert latent.grad.count_nonzero() > 0
    assert all(
        downsampler.convolution.weight.grad is not None
        and downsampler.convolution.weight.grad.count_nonzero() > 0
        for downsampler in quantizer.downsamplers
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
        logits, _, indices = model(token_ids)
        scale_logits = model.decode_scales(indices)

    assert len(scale_logits) == 3
    assert all(value.shape == logits.shape for value in scale_logits)
    torch.testing.assert_close(scale_logits[-1], logits)
    torch.testing.assert_close(model.decode_scale(indices[0], 0), scale_logits[0])


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

    metrics, _ = evaluate(
        model,
        [batch],
        use_mixed_precision=False,
        partial_reconstruction_weight=0.25,
    )

    assert "accuracy_scale_1" in metrics
    assert "accuracy_scale_2" in metrics
    assert "accuracy_scale_4" in metrics
    assert "scale_latent_rms_scale_1" in metrics
    assert metrics["accuracy_scale_4"] == metrics["accuracy"]
    assert (
        abs(
            metrics["reconstruction_loss_scale_4"] - metrics["full_reconstruction_loss"]
        )
        < 1e-6
    )
