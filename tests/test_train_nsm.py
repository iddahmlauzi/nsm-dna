from pathlib import Path

import torch
from omegaconf import OmegaConf

from nsm_dna.models.end_to_end import NSMDNA
from nsm_dna.models.next_scale import NSM
from nsm_dna.models.vqvae import VQVAE
from nsm_dna.training import calculate_training_steps
from nsm_dna.triplet_analysis import code_table_rows
from scripts.training.train_nsm import (
    build_scale_loss_weights,
    compute_losses,
    evaluate,
    split_blocks,
)


def _build_model() -> NSMDNA:
    tokenizer = VQVAE(
        vocab_size=4,
        context_length=8,
        latent_length=4,
        embed_dim=8,
        quantization_dim=4,
        num_heads=2,
        scale_lengths=[1, 2, 4],
        codebook_sizes=[4, 6, 16],
        decoder_num_layers=1,
    )
    nsm = NSM(
        prefix_dim=tokenizer.quantization_dim,
        model_dim=8,
        scale_lengths=tokenizer.scale_lengths,
        codebook_sizes=tokenizer.codebook_sizes,
        target_length=tokenizer.context_length,
        vocab_size=tokenizer.vocab_size,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        max_prefix_length=tokenizer.latent_length,
    )
    return NSMDNA(tokenizer, nsm)


def _input_ids(model: NSMDNA, batch_size: int = 2) -> torch.Tensor:
    sequence_length = 2 * model.tokenizer.context_length
    return torch.arange(batch_size * sequence_length).reshape(
        batch_size, sequence_length
    ) % model.tokenizer.vocab_size


def test_training_steps_are_derived_from_epochs(tmp_path: Path) -> None:
    (tmp_path / "subset_stats.json").write_text(
        '{"selection": {"chunk_length": 100}, "splits": {"train": {"chunks": 10}}}'
    )
    config = OmegaConf.create(
        {
            "data": {
                "subset_directory": str(tmp_path),
                "train_split": "train",
                "train_batch_size": 5,
            },
            "optimizer": {"gradient_accumulation_steps": 2},
            "training": {"num_epochs": 3, "max_steps": None},
        }
    )

    assert calculate_training_steps(config, world_size=2, sequence_length=16) == 9

    config.training.max_steps = 7
    assert calculate_training_steps(config, world_size=2, sequence_length=16) == 7


def test_scale_loss_weights_interpolate_between_scales_and_positions() -> None:
    scale_lengths = [1, 2, 4]
    equal_scales = build_scale_loss_weights(
        scale_lengths,
        scale_loss_alpha=1.0,
        device=torch.device("cpu"),
    )
    equal_positions = build_scale_loss_weights(
        scale_lengths,
        scale_loss_alpha=0.0,
        device=torch.device("cpu"),
    )

    torch.testing.assert_close(
        equal_scales,
        torch.full_like(equal_scales, 1 / len(scale_lengths)),
    )
    torch.testing.assert_close(
        equal_positions,
        torch.tensor(scale_lengths) / sum(scale_lengths),
    )


def test_triplet_table_groups_codes_with_amino_acids() -> None:
    rows = code_table_rows({"ATG": 1, "TTG": 1}, codebook_size=3)

    assert rows == [
        [0, "", ""],
        [1, "ATG, TTG", "Met, Leu"],
        [2, "", ""],
    ]


def test_end_to_end_forward_follows_the_tokenizer_shapes() -> None:
    model = _build_model()
    prefix_ids, target_ids = split_blocks(
        _input_ids(model),
        model.tokenizer.context_length,
    )

    output = model(prefix_ids, target_ids)

    expected_nucleotide_shape = (*target_ids.shape, model.tokenizer.vocab_size)
    assert output.reconstruction_logits.shape == expected_nucleotide_shape
    assert output.partial_reconstruction_logits is not None
    assert output.partial_reconstruction_logits.shape == expected_nucleotide_shape
    assert output.nucleotide_logits.shape == expected_nucleotide_shape
    for logits, targets, codebook_size in zip(
        output.hierarchy_logits,
        output.target_indices,
        model.tokenizer.codebook_sizes,
        strict=True,
    ):
        assert logits.shape == (*targets.shape, codebook_size)


def test_joint_loss_reaches_tokenizer_and_nsm() -> None:
    model = _build_model()
    prefix_ids, target_ids = split_blocks(
        _input_ids(model),
        model.tokenizer.context_length,
    )
    output = model(prefix_ids, target_ids)
    losses = compute_losses(
        output,
        target_ids,
        build_scale_loss_weights(
            model.tokenizer.scale_lengths,
            scale_loss_alpha=1.0,
            device=torch.device("cpu"),
        ),
        partial_reconstruction_weight=0.25,
    )

    losses.total.backward()

    parameter_groups = (
        model.tokenizer.encoder.parameters(),
        model.tokenizer.quantizer.parameters(),
        model.tokenizer.decoder.parameters(),
        model.nsm.parameters(),
    )
    for parameters in parameter_groups:
        assert any(
            parameter.grad is not None and parameter.grad.count_nonzero() > 0
            for parameter in parameters
        )


def test_evaluate_reports_each_objective_and_restores_training_mode() -> None:
    model = _build_model()
    metrics = evaluate(
        model,
        data_loader=[{"input_ids": _input_ids(model)}],
        scale_loss_weights=build_scale_loss_weights(
            model.tokenizer.scale_lengths,
            scale_loss_alpha=1.0,
            device=torch.device("cpu"),
        ),
        partial_reconstruction_weight=0.25,
        use_mixed_precision=False,
    )

    assert model.training
    for name in (
        "loss",
        "reconstruction_loss",
        "partial_reconstruction_loss",
        "hierarchy_loss",
        "nucleotide_loss",
    ):
        assert metrics[name] > 0
    for name in (
        "reconstruction_accuracy",
        "hierarchy_accuracy",
        "nucleotide_accuracy",
    ):
        assert 0 <= metrics[name] <= 1
    for scale_number, scale_length in enumerate(
        model.tokenizer.scale_lengths,
        start=1,
    ):
        section = f"scale_{scale_number:02d}_length_{scale_length}"
        assert 0 < metrics[f"{section}/codebook_usage"] <= 1
        assert 0 <= metrics[f"{section}/prediction_accuracy"] <= 1
