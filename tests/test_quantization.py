import pytest
import torch

from nsm_dna.models.next_scale.quantization import (
    EMACodebook,
    MultiscaleResidualVectorQuantizer,
)


def test_codebook_uses_hard_nearest_codes_with_soft_gradients() -> None:
    codebook = EMACodebook(
        codebook_size=3,
        embed_dim=2,
        temperature=1.0,
    ).eval()
    with torch.no_grad():
        codebook.codebook.copy_(
            torch.tensor(
                [
                    [0.0, 0.0],
                    [2.0, 0.0],
                    [0.0, 2.0],
                ]
            )
        )

    latent = torch.tensor(
        [[[0.1, 0.2], [1.9, 0.1]]],
        requires_grad=True,
    )
    quantized, indices, probabilities, assignment_logits = codebook(latent)

    torch.testing.assert_close(indices, torch.tensor([[0, 1]]))
    torch.testing.assert_close(quantized, codebook.codebook[indices])
    torch.testing.assert_close(
        probabilities.sum(dim=-1),
        torch.ones(1, 2),
    )
    torch.testing.assert_close(assignment_logits.argmax(dim=-1), indices)

    weights = torch.tensor([[[1.0, -0.5], [0.25, 2.0]]])
    (quantized * weights).sum().backward()

    assert latent.grad is not None
    assert latent.grad.count_nonzero() > 0
    assert codebook.codebook.grad is None


def test_codebook_rejects_nonpositive_temperature() -> None:
    with pytest.raises(ValueError, match="temperature must be positive"):
        EMACodebook(codebook_size=3, embed_dim=2, temperature=0.0)


def test_quantizer_returns_differentiable_cumulative_hierarchy() -> None:
    quantizer = MultiscaleResidualVectorQuantizer(
        scale_lengths=[1, 2, 4],
        codebook_sizes=[8, 8, 8],
        embed_dim=4,
    ).eval()
    latent = torch.randn(2, 4, 4, requires_grad=True)

    output = quantizer(latent)

    assert len(output.cumulative_latents) == 3
    assert len(output.next_scale_inputs) == 2
    assert len(output.indices_by_scale) == 3
    assert len(output.assignment_probabilities_by_scale) == 3
    assert len(output.assignment_logits_by_scale) == 3
    expected_commitment = quantizer.commitment_cost * torch.nn.functional.mse_loss(
        output.final_latent.detach(),
        latent,
    )
    torch.testing.assert_close(output.encoder_commitment_loss, expected_commitment)
    torch.testing.assert_close(
        output.vq_loss,
        output.encoder_commitment_loss
        + torch.stack(output.quantization_losses_by_scale).mean(),
    )
    assert [
        probabilities.shape
        for probabilities in output.assignment_probabilities_by_scale
    ] == [
        (2, 1, 8),
        (2, 2, 8),
        (2, 4, 8),
    ]
    assert [logits.shape for logits in output.assignment_logits_by_scale] == [
        (2, 1, 8),
        (2, 2, 8),
        (2, 4, 8),
    ]
    torch.testing.assert_close(output.final_latent, output.cumulative_latents[-1])
    torch.testing.assert_close(
        output.final_latent,
        quantizer.indices_to_cumulative_latents(output.indices_by_scale)[-1],
    )

    output.final_latent.square().mean().backward()

    assert latent.grad is not None
    assert latent.grad.count_nonzero() > 0
    assert any(
        parameter.grad is not None and parameter.grad.count_nonzero() > 0
        for parameter in quantizer.first_scale_downsampler.parameters()
    )
    assert all(
        any(
            parameter.grad is not None and parameter.grad.count_nonzero() > 0
            for parameter in refiner.parameters()
        )
        for refiner in quantizer.refiners
    )
    assert all(codebook.codebook.grad is None for codebook in quantizer.codebooks)


def test_uncorrupted_inputs_match_hard_hierarchy_in_the_forward_pass() -> None:
    quantizer = MultiscaleResidualVectorQuantizer(
        scale_lengths=[1, 2, 4],
        codebook_sizes=[8, 8, 8],
        embed_dim=4,
    ).eval()
    latent = torch.randn(2, 4, 4)

    output = quantizer(latent, corruption_probability=0.0)
    expected_inputs = quantizer.indices_to_next_scale_inputs(output.indices_by_scale)

    for actual, expected in zip(
        output.next_scale_inputs,
        expected_inputs,
        strict=True,
    ):
        torch.testing.assert_close(actual, expected)


def test_prediction_logits_use_hard_codes_with_soft_gradients() -> None:
    quantizer = MultiscaleResidualVectorQuantizer(
        scale_lengths=[1, 2, 4],
        codebook_sizes=[8, 8, 8],
        embed_dim=4,
    ).eval()
    first_scale_indices = torch.randint(0, 8, (2, 1))
    logits_by_scale = [
        torch.randn(2, scale_length, 8, requires_grad=True) for scale_length in [2, 4]
    ]
    initial_latent = quantizer.indices_to_cumulative_latents([first_scale_indices])[0]

    predicted_latent = quantizer.teacher_forced_prediction_logits_to_final_latent(
        logits_by_scale,
        initial_latent=initial_latent,
    )
    expected_latent = quantizer.indices_to_cumulative_latents(
        [first_scale_indices] + [logits.argmax(dim=-1) for logits in logits_by_scale]
    )[-1]

    torch.testing.assert_close(predicted_latent, expected_latent)
    predicted_latent.square().mean().backward()
    assert all(logits.grad is not None for logits in logits_by_scale)
    assert all(logits.grad.count_nonzero() > 0 for logits in logits_by_scale)
    assert all(codebook.codebook.grad is None for codebook in quantizer.codebooks)


def test_quantizer_rejects_invalid_corruption_probability() -> None:
    quantizer = MultiscaleResidualVectorQuantizer(
        scale_lengths=[1, 2],
        codebook_sizes=[4, 4],
        embed_dim=2,
    )

    with pytest.raises(ValueError, match="between zero and one"):
        quantizer(torch.randn(1, 2, 2), corruption_probability=1.1)


def test_corruption_samples_from_nearest_codes_by_inverse_distance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quantizer = MultiscaleResidualVectorQuantizer(
        scale_lengths=[1, 2],
        codebook_sizes=[4, 4],
        embed_dim=1,
    ).eval()
    with torch.no_grad():
        quantizer.codebooks[0].codebook.copy_(
            torch.tensor([[0.0], [1.0], [2.0], [4.0]])
        )

    captured_weights = None

    def select_highest_weight(
        weights: torch.Tensor,
        num_samples: int,
    ) -> torch.Tensor:
        nonlocal captured_weights
        captured_weights = weights.clone()
        assert num_samples == 1
        return weights.argmax(dim=1, keepdim=True)

    monkeypatch.setattr(torch, "multinomial", select_highest_weight)
    corrupted = quantizer._corrupt_quantized_vectors(
        quantized=torch.tensor([[[0.0]]]),
        indices=torch.tensor([[0]]),
        scale_index=0,
        probability=1.0,
    )

    assert captured_weights is not None
    torch.testing.assert_close(
        captured_weights.sort(descending=True).values,
        torch.tensor([[1.0, 0.5, 0.25]]),
    )
    torch.testing.assert_close(corrupted, torch.tensor([[[1.0]]]))


def test_corruption_considers_only_the_twenty_nearest_alternatives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quantizer = MultiscaleResidualVectorQuantizer(
        scale_lengths=[1, 2],
        codebook_sizes=[32, 32],
        embed_dim=1,
    ).eval()
    with torch.no_grad():
        quantizer.codebooks[0].codebook.copy_(
            torch.arange(32, dtype=torch.float32).unsqueeze(1)
        )

    candidate_count = None

    def select_lowest_weight(
        weights: torch.Tensor,
        num_samples: int,
    ) -> torch.Tensor:
        nonlocal candidate_count
        candidate_count = weights.shape[1]
        assert num_samples == 1
        return weights.argmin(dim=1, keepdim=True)

    monkeypatch.setattr(torch, "multinomial", select_lowest_weight)
    corrupted = quantizer._corrupt_quantized_vectors(
        quantized=torch.tensor([[[0.0]]]),
        indices=torch.tensor([[0]]),
        scale_index=0,
        probability=1.0,
    )

    assert candidate_count == 20
    assert 1 <= corrupted.item() <= 20


def test_corruption_exempts_supplied_scale_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quantizer = MultiscaleResidualVectorQuantizer(
        scale_lengths=[1, 2, 4],
        codebook_sizes=[8, 8, 8],
        embed_dim=4,
    ).eval()
    corrupted_scale_indices = []
    corrupt_quantized_vectors = quantizer._corrupt_quantized_vectors

    def record_corrupted_scale(
        quantized: torch.Tensor,
        indices: torch.Tensor,
        scale_index: int,
        probability: float,
    ) -> torch.Tensor:
        corrupted_scale_indices.append(scale_index)
        return corrupt_quantized_vectors(
            quantized,
            indices,
            scale_index,
            probability,
        )

    monkeypatch.setattr(
        quantizer,
        "_corrupt_quantized_vectors",
        record_corrupted_scale,
    )
    output = quantizer(torch.randn(2, 4, 4), corruption_probability=1.0)
    clean_inputs = quantizer.indices_to_next_scale_inputs(output.indices_by_scale)

    assert corrupted_scale_indices == [1]
    torch.testing.assert_close(output.next_scale_inputs[0], clean_inputs[0])


def test_full_later_scale_corruption_preserves_scale_one_gradients_and_ema() -> None:
    clean_quantizer = MultiscaleResidualVectorQuantizer(
        scale_lengths=[1, 2, 4],
        codebook_sizes=[8, 8, 8],
        embed_dim=4,
    )
    corrupted_quantizer = MultiscaleResidualVectorQuantizer(
        scale_lengths=[1, 2, 4],
        codebook_sizes=[8, 8, 8],
        embed_dim=4,
    )
    corrupted_quantizer.load_state_dict(clean_quantizer.state_dict())
    latent = torch.randn(2, 4, 4)

    clean_output = clean_quantizer(
        latent.clone(),
        corruption_probability=0.0,
    )
    corrupted_latent = latent.clone().requires_grad_(True)
    corrupted_output = corrupted_quantizer(
        corrupted_latent,
        corruption_probability=1.0,
    )

    for clean_indices, corrupted_indices in zip(
        clean_output.indices_by_scale,
        corrupted_output.indices_by_scale,
        strict=True,
    ):
        torch.testing.assert_close(clean_indices, corrupted_indices)
    torch.testing.assert_close(
        clean_output.final_latent,
        corrupted_output.final_latent,
    )
    torch.testing.assert_close(clean_output.vq_loss, corrupted_output.vq_loss)
    for clean_codebook, corrupted_codebook in zip(
        clean_quantizer.codebooks,
        corrupted_quantizer.codebooks,
        strict=True,
    ):
        torch.testing.assert_close(
            clean_codebook.ema_counts,
            corrupted_codebook.ema_counts,
        )
        torch.testing.assert_close(
            clean_codebook.ema_vector_sums,
            corrupted_codebook.ema_vector_sums,
        )

    sum(
        scale_input.sum() for scale_input in corrupted_output.next_scale_inputs
    ).backward()
    assert corrupted_latent.grad is not None
    assert corrupted_latent.grad.count_nonzero() > 0
