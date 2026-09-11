import torch
import torch.nn.functional as F

from nsm_dna.models.next_scale import NSM
from nsm_dna.models.next_token import NextTokenModel
from nsm_dna.models.vqvae import VQVAE
from scripts.evaluation.variant_effects_next_scale import (
    score_token_ids,
    target_block_window,
)
from scripts.evaluation.variant_effects_next_token import score_next_token_ids
from scripts.training.train_nsm import prepare_block_predictions


def _build_tokenizer() -> VQVAE:
    tokenizer = VQVAE(
        vocab_size=4,
        context_length=4,
        latent_length=4,
        embed_dim=8,
        quantization_dim=4,
        num_heads=2,
        scale_lengths=[1, 2, 4],
        codebook_sizes=[8, 8, 8],
        encoder_dropout=0.0,
        decoder_dropout=0.0,
        pre_quant_num_groups=2,
    ).eval()
    tokenizer.requires_grad_(False)
    return tokenizer


def test_target_block_window_uses_fixed_sequence_blocks() -> None:
    reference = "A" * 512
    expected_windows = (
        (10, None, None),
        (150, 0, 150),
        (290, 128, 162),
    )

    for mutation_position, expected_start, position_in_window in expected_windows:
        mutant = (
            reference[:mutation_position] + "C" + reference[mutation_position + 1 :]
        )
        window = target_block_window(reference, mutant, 128, 128)
        if expected_start is None:
            assert window is None
            continue
        assert window is not None
        assert window[2] == expected_start
        assert position_in_window is not None
        assert window[1][position_in_window] == "C"


def test_target_block_window_excludes_incomplete_target_blocks() -> None:
    assert (
        target_block_window(
            "A" * 200,
            "A" * 150 + "C" + "A" * 49,
            128,
            128,
        )
        is None
    )


def test_sequence_score_includes_hierarchy_and_decoder_probabilities() -> None:
    torch.manual_seed(0)
    tokenizer = _build_tokenizer()
    model = NSM(
        vq_embed_dim=4,
        model_dim=8,
        scale_lengths=[1, 2, 4],
        codebook_size=8,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        max_prefix_length=4,
    ).eval()
    input_ids = torch.arange(2 * 8).reshape(2, 8) % 4

    hierarchy_scores, decoder_scores = score_token_ids(model, tokenizer, input_ids)

    expected_hierarchy_scores = torch.zeros(2, 3, dtype=torch.float64)
    expected_decoder_scores = torch.zeros(2, dtype=torch.float64)
    for prediction in prepare_block_predictions(tokenizer, input_ids):
        logits = model(prediction.scale_inputs, prefix=prediction.prefix)
        logits_by_scale = torch.split(logits, tokenizer.scale_lengths, dim=1)
        for scale_index, (scale_logits, scale_targets) in enumerate(
            zip(logits_by_scale, prediction.targets_by_scale)
        ):
            scale_losses = F.cross_entropy(
                scale_logits.flatten(0, 1),
                scale_targets.flatten(),
                reduction="none",
            ).reshape(scale_targets.shape)
            expected_hierarchy_scores[:, scale_index] -= scale_losses.sum(1).double()

        decoder_logits = tokenizer.decode(prediction.targets_by_scale)
        decoder_losses = F.cross_entropy(
            decoder_logits.flatten(0, 1),
            prediction.target_ids.flatten(),
            reduction="none",
        ).reshape(prediction.target_ids.shape)
        expected_decoder_scores -= decoder_losses.sum(1).double()

    torch.testing.assert_close(hierarchy_scores, expected_hierarchy_scores)
    torch.testing.assert_close(decoder_scores, expected_decoder_scores)


def test_sequence_score_supports_target_without_prefix() -> None:
    torch.manual_seed(0)
    tokenizer = _build_tokenizer()
    model = NSM(
        vq_embed_dim=4,
        model_dim=8,
        scale_lengths=[1, 2, 4],
        codebook_size=8,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        max_prefix_length=4,
    ).eval()
    input_ids = torch.arange(2 * 4).reshape(2, 4) % 4

    hierarchy_scores, decoder_scores = score_token_ids(model, tokenizer, input_ids)

    assert hierarchy_scores.shape == (2, 3)
    assert decoder_scores.shape == (2,)
    assert torch.isfinite(hierarchy_scores).all()
    assert torch.isfinite(decoder_scores).all()


def test_next_token_sequence_score_sums_observed_base_log_probabilities() -> None:
    torch.manual_seed(0)
    model = NextTokenModel(
        vocab_size=4,
        model_dim=8,
        num_layers=1,
        num_heads=2,
        max_sequence_length=7,
        dropout=0.0,
    ).eval()
    input_ids = torch.arange(2 * 8).reshape(2, 8) % 4

    scores = score_next_token_ids(model, input_ids)

    targets = input_ids[:, 1:]
    losses = F.cross_entropy(
        model(input_ids[:, :-1]).flatten(0, 1),
        targets.flatten(),
        reduction="none",
    ).reshape(targets.shape)
    torch.testing.assert_close(scores, -losses.sum(1).double())
