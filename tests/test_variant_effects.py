from pathlib import Path

import torch
import torch.nn.functional as F

from nsm_dna.models.next_scale import MultiscaleTokenizer, NextScaleTransformer
from nsm_dna.models.next_token import NextTokenModel
from scripts.evaluation.variant_effects_next_scale import (
    prepare_block_predictions,
    read_assay_windows,
    score_token_ids,
    sequence_windows,
)
from scripts.evaluation.variant_effects_next_token import score_next_token_ids


def _build_tokenizer() -> MultiscaleTokenizer:
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
    tokenizer.requires_grad_(False)
    return tokenizer


def test_sequence_windows_tile_the_complete_sequence() -> None:
    windows = sequence_windows("AAAACCCCGGGG", window_length=4, stride=4)

    assert windows == ("AAAA", "CCCC", "GGGG")


def test_sequence_windows_add_an_end_aligned_remainder_window() -> None:
    windows = sequence_windows("AAAACCCCGG", window_length=4, stride=4)

    assert windows == ("AAAA", "CCCC", "CCGG")


def test_sequence_windows_support_reference_and_indel_lengths_independently() -> None:
    reference_windows = sequence_windows("AAAACCCC", window_length=4, stride=4)
    insertion_windows = sequence_windows("AAAACCCCC", window_length=4, stride=4)
    deletion_windows = sequence_windows("AAAACCC", window_length=4, stride=4)

    assert reference_windows == ("AAAA", "CCCC")
    assert insertion_windows == ("AAAA", "CCCC", "CCCC")
    assert deletion_windows == ("AAAA", "ACCC")


def test_read_assay_windows_keeps_indel_variants(tmp_path: Path) -> None:
    assay_path = tmp_path / "assay.csv"
    assay_path.write_text(
        "study_id,assay_id,wt_nt,mutant_nt\n"
        "study,assay,AAAACCCC,AAAACCCCC\n",
        encoding="utf-8",
    )

    variants, num_excluded = read_assay_windows(
        assay_path,
        prefix_length=0,
        target_length=4,
    )

    assert num_excluded == 0
    assert len(variants) == 1
    assert len(variants[0].reference_windows) == 2
    assert len(variants[0].mutant_windows) == 3


def test_sequence_score_includes_hierarchy_and_decoder_probabilities() -> None:
    torch.manual_seed(0)
    tokenizer = _build_tokenizer()
    model = NextScaleTransformer(
        input_dim=8,
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

    expected_hierarchy_scores = torch.zeros(2, 2, dtype=torch.float64)
    expected_decoder_scores = torch.zeros(2, dtype=torch.float64)
    for prediction in prepare_block_predictions(tokenizer, input_ids):
        logits = model(prediction.scale_inputs, prefix=prediction.prefix)
        logits_by_scale = torch.split(logits, tokenizer.scale_lengths[1:], dim=1)
        for scale_index, (scale_logits, scale_targets) in enumerate(
            zip(logits_by_scale, prediction.targets_by_scale[1:], strict=True)
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
    model = NextScaleTransformer(
        input_dim=8,
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

    assert hierarchy_scores.shape == (2, 2)
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
