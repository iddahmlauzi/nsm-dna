import torch

from nsm_dna.models.quantization import ResidualVectorQuantizer


def _build_quantizer(latent_length: int = 8) -> ResidualVectorQuantizer:
    scale_lengths = []
    scale_length = 1
    while scale_length <= latent_length:
        scale_lengths.append(scale_length)
        scale_length *= 2
    return ResidualVectorQuantizer(
        scale_lengths=scale_lengths,
        codebook_sizes=[4] * len(scale_lengths),
        quantization_dim=3,
        latent_length=latent_length,
    ).eval()


def test_mean_and_detail_decomposition_is_exact() -> None:
    quantizer = _build_quantizer()
    latent = torch.randn(2, quantizer.latent_length, quantizer.quantization_dim)

    coefficients = quantizer.decompose(latent)
    reconstruction = quantizer.reconstruct_coefficients(coefficients)[-1]

    torch.testing.assert_close(reconstruction, latent)
    assert [coefficient.shape[1] for coefficient in coefficients] == (
        quantizer.code_lengths
    )


def test_each_detail_preserves_its_parent_mean() -> None:
    quantizer = _build_quantizer()
    parents = torch.randn(2, 4, quantizer.quantization_dim)
    details = torch.randn_like(parents)

    children = quantizer._split_parents(parents, details)
    recovered_parents = children.reshape(2, 4, 2, -1).mean(dim=2)

    torch.testing.assert_close(recovered_parents, parents)


def test_partial_reconstruction_backpropagates_to_continuous_latent(
    monkeypatch,
) -> None:
    quantizer = _build_quantizer()
    latent = torch.randn(
        2,
        quantizer.latent_length,
        quantizer.quantization_dim,
        requires_grad=True,
    )
    monkeypatch.setattr(torch, "randint", lambda *args, **kwargs: torch.tensor(1))

    _, partial_latent, _ = quantizer(
        latent,
        include_partial_reconstruction=True,
    )
    assert partial_latent is not None
    partial_latent.square().mean().backward()

    assert latent.grad is not None
    assert latent.grad.count_nonzero() > 0
