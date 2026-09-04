import torch
import torch.nn as nn
import torch.nn.functional as F

from nsm_dna.models.next_scale import MultiscaleTokenizer
from nsm_dna.models.next_scale.quantization import (
    MultiscaleResidualVectorQuantizer,
)
from nsm_dna.models.next_scale.tokenizer import Decoder, Encoder


def test_encoder_uses_token_embeddings_without_absolute_positions() -> None:
    encoder = Encoder(vocab_size=4, embed_dim=4, dropout=0.0)
    with torch.no_grad():
        encoder.token_embedding.weight.copy_(torch.eye(4))

    first_sequence = torch.tensor([[3, 1, 0, 2]])
    shifted_sequence = torch.tensor([[3, 3, 1, 0]])

    first_embeddings = encoder(first_sequence)
    shifted_embeddings = encoder(shifted_sequence)

    torch.testing.assert_close(first_embeddings[:, :3], shifted_embeddings[:, 1:])


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
        embed_dim=8,
        num_heads=2,
        dropout=0.0,
    )
    recording_block = RecordingBlock()
    decoder.block = recording_block

    decoder(torch.randn(1, 4, 8))

    assert recording_block.rotary_embeddings is not None
    cosine, sine = recording_block.rotary_embeddings
    torch.testing.assert_close(cosine, decoder.rope_cosine)
    torch.testing.assert_close(sine, decoder.rope_sine)


def test_first_scale_sampler_uses_a_cascade() -> None:
    quantizer = MultiscaleResidualVectorQuantizer(
        scale_lengths=[4, 256],
        codebook_sizes=[4, 4],
        embed_dim=2,
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


def test_cumulative_decode_matches_full_reconstruction() -> None:
    model = MultiscaleTokenizer(
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


def test_encode_returns_continuous_prefix_latents() -> None:
    model = MultiscaleTokenizer(
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
    token_ids = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])

    prefix_latents = model.encode(token_ids)
    encoder_latents = model.encoder(token_ids)
    expected_latents = model.pre_quant_norm(encoder_latents.transpose(1, 2)).transpose(
        1, 2
    )

    assert prefix_latents.shape == (2, 4, 8)
    torch.testing.assert_close(prefix_latents, expected_latents)


def test_encode_pair_normalizes_prefix_and_target_independently() -> None:
    tokenizer = MultiscaleTokenizer(
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
    first_sequence = torch.tensor([[0, 1, 2, 0, 0, 0, 0]])
    changed_target = torch.tensor([[0, 1, 2, 3, 3, 3, 3]])

    prefix, target = tokenizer.encode_pair(first_sequence)
    unchanged_prefix, changed_target_latent = tokenizer.encode_pair(changed_target)

    torch.testing.assert_close(prefix, unchanged_prefix)
    assert prefix.shape[1] == 3
    torch.testing.assert_close(prefix, tokenizer.encode(first_sequence[:, :3]))
    torch.testing.assert_close(target, tokenizer.encode(first_sequence[:, 3:]))
    torch.testing.assert_close(
        changed_target_latent,
        tokenizer.encode(changed_target[:, 3:]),
    )


def test_next_scale_inputs_are_resized_cumulative_latents() -> None:
    quantizer = MultiscaleResidualVectorQuantizer(
        scale_lengths=[1, 2, 4],
        codebook_sizes=[8, 8, 8],
        embed_dim=8,
    ).eval()
    indices_by_scale = [
        torch.tensor([[0], [1]]),
        torch.tensor([[2, 3], [4, 5]]),
        torch.tensor([[6, 7, 0, 1], [2, 3, 4, 5]]),
    ]

    next_scale_inputs = quantizer.indices_to_next_scale_inputs(indices_by_scale)
    cumulative_latents = quantizer.indices_to_cumulative_latents(indices_by_scale)
    first_cumulative_latent = cumulative_latents[0].transpose(1, 2)
    expected_second_scale_input = F.interpolate(
        first_cumulative_latent,
        size=2,
        mode="area",
    ).transpose(1, 2)

    assert len(next_scale_inputs) == 2
    assert next_scale_inputs[0].shape == (2, 2, 8)
    assert next_scale_inputs[1].shape == (2, 4, 8)
    torch.testing.assert_close(next_scale_inputs[0], expected_second_scale_input)
    torch.testing.assert_close(next_scale_inputs[1], cumulative_latents[1])


def test_partial_reconstruction_uses_differentiable_assignments() -> None:
    model = MultiscaleTokenizer(
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
    assert any(
        parameter.grad is not None and parameter.grad.count_nonzero() > 0
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
    def build_model() -> MultiscaleTokenizer:
        return MultiscaleTokenizer(
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
        model: MultiscaleTokenizer,
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
    for baseline_gradient, split_gradient in zip(baseline_quantizer, split_quantizer):
        torch.testing.assert_close(split_gradient, 0.2 * baseline_gradient)
    for baseline_gradient, split_gradient in zip(baseline_decoder, split_decoder):
        torch.testing.assert_close(split_gradient, 0.01 * baseline_gradient)
