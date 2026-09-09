from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from nsm_dna.models.next_scale import NSM
from nsm_dna.models.vqvae import VQVAE
from nsm_dna.training import calculate_training_steps
from scripts.training.train_nsm import (
    build_scale_loss_weights,
    compute_next_scale_loss,
    corrupt_scale_indices,
    evaluate,
    prepare_block_predictions,
    rollout_scale_predictions,
)


def _build_tokenizer() -> VQVAE:
    return VQVAE(
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


def _build_nsm(max_prefix_length: int = 0) -> NSM:
    return NSM(
        vq_embed_dim=8,
        model_dim=8,
        scale_lengths=[1, 2, 4],
        codebook_size=8,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        max_prefix_length=max_prefix_length,
    )


def test_training_steps_are_derived_from_epochs(tmp_path: Path) -> None:
    (tmp_path / "subset_stats.json").write_text(
        '{"selection": {"chunk_length": 100}, '
        '"splits": {"train": {"chunks": 10}}}'
    )
    config = OmegaConf.create(
        {
            "data": {
                "subset_directory": str(tmp_path),
                "train_split": "train",
                "sequence_length": 16,
                "train_batch_size": 5,
            },
            "optimizer": {"gradient_accumulation_steps": 2},
            "training": {"num_epochs": 3, "max_steps": None},
        }
    )

    assert calculate_training_steps(config, world_size=2, sequence_length=16) == 12

    config.training.max_steps = 7
    assert calculate_training_steps(config, world_size=2, sequence_length=16) == 7


def test_scale_loss_alpha_controls_each_scale_share() -> None:
    equal_scale_weights = build_scale_loss_weights(
        [1, 2, 4],
        scale_loss_alpha=1.0,
        device=torch.device("cpu"),
    )
    equal_position_weights = build_scale_loss_weights(
        [1, 2, 4],
        scale_loss_alpha=0.0,
        device=torch.device("cpu"),
    )

    torch.testing.assert_close(
        equal_scale_weights,
        torch.full((3,), 1 / 3),
    )
    torch.testing.assert_close(
        equal_position_weights,
        torch.tensor([1 / 7, 2 / 7, 4 / 7]),
    )


def test_next_scale_loss_includes_and_aligns_every_scale() -> None:
    targets_by_scale = [
        torch.tensor([[0], [1]]),
        torch.tensor([[1, 2], [2, 3]]),
        torch.tensor([[3, 2, 1, 0], [0, 1, 2, 3]]),
    ]
    logits = torch.randn(2, 7, 4)
    scale_weights = torch.tensor([0.2, 0.3, 0.5])

    loss, losses_by_scale = compute_next_scale_loss(
        logits,
        targets_by_scale,
        scale_weights,
    )

    logits_by_scale = torch.split(logits, [1, 2, 4], dim=1)
    expected_losses = torch.stack(
        [
            F.cross_entropy(
                scale_logits.flatten(0, 1),
                scale_targets.flatten(),
            )
            for scale_logits, scale_targets in zip(
                logits_by_scale,
                targets_by_scale,
            )
        ]
    )

    torch.testing.assert_close(losses_by_scale, expected_losses)
    torch.testing.assert_close(loss, torch.sum(expected_losses * scale_weights))


def test_next_scale_loss_matches_original_position_weighting() -> None:
    scale_lengths = [1, 2, 4]
    scale_loss_alpha = 0.25
    targets_by_scale = [
        torch.tensor([[0], [1]]),
        torch.tensor([[1, 2], [2, 3]]),
        torch.tensor([[3, 2, 1, 0], [0, 1, 2, 3]]),
    ]
    targets = torch.cat(targets_by_scale, dim=1)
    logits = torch.randn(2, sum(scale_lengths), 4)
    scale_weights = build_scale_loss_weights(
        scale_lengths,
        scale_loss_alpha,
        device=torch.device("cpu"),
    )

    loss, _ = compute_next_scale_loss(
        logits,
        targets_by_scale,
        scale_weights,
    )

    normalization = sum(
        scale_length ** (1.0 - scale_loss_alpha) for scale_length in scale_lengths
    )
    position_weights = torch.cat(
        [
            torch.full(
                (scale_length,),
                scale_length**-scale_loss_alpha / normalization,
            )
            for scale_length in scale_lengths
        ]
    )
    loss_by_position = F.cross_entropy(
        logits.flatten(0, 1),
        targets.flatten(),
        reduction="none",
    ).reshape(targets.shape)
    expected_loss = (loss_by_position * position_weights).sum(dim=1).mean()

    torch.testing.assert_close(loss, expected_loss)


def test_tokenizer_checkpoint_is_restored_and_frozen(tmp_path: Path) -> None:
    tokenizer = _build_tokenizer()
    checkpoint_path = tmp_path / "tokenizer.pt"
    torch.save(
        {
            "config": {
                "model": {
                    "vocab_size": 4,
                    "context_length": 4,
                    "embed_dim": 8,
                    "num_heads": 2,
                    "scale_lengths": [1, 2, 4],
                    "codebook_sizes": [8, 8, 8],
                    "encoder_dropout": 0.0,
                    "decoder_dropout": 0.0,
                    "bias": False,
                    "pre_quant_num_groups": 2,
                    "commitment_cost": 0.25,
                    "decay": 0.99,
                    "eps": 1e-5,
                    "refinement_ratio": 0.5,
                    "refinement_kernel_size": 3,
                }
            },
            "model": tokenizer.state_dict(),
        },
        checkpoint_path,
    )

    restored_tokenizer = VQVAE.from_checkpoint(
        checkpoint_path,
        device=torch.device("cpu"),
        frozen=True,
    )

    assert restored_tokenizer.training is False
    assert all(
        parameter.requires_grad is False
        for parameter in restored_tokenizer.parameters()
    )
    for name, expected_value in tokenizer.state_dict().items():
        torch.testing.assert_close(
            restored_tokenizer.state_dict()[name],
            expected_value,
        )

    trainable_tokenizer = VQVAE.from_checkpoint(
        checkpoint_path,
        device=torch.device("cpu"),
    )
    assert trainable_tokenizer.training is True
    assert all(
        parameter.requires_grad for parameter in trainable_tokenizer.parameters()
    )


def test_stage_two_batch_stops_gradients_at_the_tokenizer() -> None:
    tokenizer = _build_tokenizer().eval()
    tokenizer.requires_grad_(False)
    model = _build_nsm()
    input_ids = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])
    scale_weights = build_scale_loss_weights(
        tokenizer.scale_lengths,
        scale_loss_alpha=0.25,
        device=torch.device("cpu"),
    )

    targets_by_scale = tokenizer.encode_indices(input_ids)
    next_scale_inputs = tokenizer.indices_to_next_scale_inputs(targets_by_scale)
    logits = model(next_scale_inputs)
    loss, _ = compute_next_scale_loss(logits, targets_by_scale, scale_weights)
    loss.backward()

    assert [scale_input.shape[1] for scale_input in next_scale_inputs] == [2, 4]
    assert [targets.shape[1] for targets in targets_by_scale] == [1, 2, 4]
    assert logits.shape == (2, 7, 8)
    assert all(parameter.grad is None for parameter in tokenizer.parameters())
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_block_prediction_uses_first_block_as_prefix_and_second_as_target() -> None:
    tokenizer = _build_tokenizer().eval()
    input_ids = torch.arange(2 * 8).reshape(2, 8) % 4
    prefix_ids = input_ids[:, :4]
    target_ids = input_ids[:, 4:]

    block_predictions = prepare_block_predictions(tokenizer, input_ids)
    assert len(block_predictions) == 1

    prediction = block_predictions[0]
    torch.testing.assert_close(prediction.target_ids, target_ids)
    torch.testing.assert_close(prediction.prefix, tokenizer.encode(prefix_ids))

    expected_targets = tokenizer.encode_indices(target_ids)
    expected_inputs = tokenizer.indices_to_next_scale_inputs(expected_targets)
    for actual, expected in zip(prediction.scale_inputs, expected_inputs):
        torch.testing.assert_close(actual, expected)
    for actual, expected in zip(prediction.targets_by_scale, expected_targets):
        torch.testing.assert_close(actual, expected)


def test_corruption_changes_inputs_without_changing_targets() -> None:
    targets_by_scale = [
        torch.tensor([[3]]),
        torch.tensor([[2, 4]]),
        torch.tensor([[1, 5, 6, 7]]),
    ]

    corrupted_indices = corrupt_scale_indices(
        targets_by_scale,
        codebook_sizes=[1, 1, 1],
        probability=1.0,
    )

    torch.testing.assert_close(
        corrupted_indices[0],
        torch.zeros_like(targets_by_scale[0]),
    )
    torch.testing.assert_close(
        corrupted_indices[1],
        torch.zeros_like(targets_by_scale[1]),
    )
    torch.testing.assert_close(corrupted_indices[2], targets_by_scale[2])
    torch.testing.assert_close(targets_by_scale[0], torch.tensor([[3]]))
    torch.testing.assert_close(targets_by_scale[1], torch.tensor([[2, 4]]))


def test_default_config_matches_long_context_recipe() -> None:
    config_path = Path(__file__).parents[1] / "configs" / "nsm.yaml"
    config = OmegaConf.load(config_path)

    assert config.run.resume_from is None
    assert config.tokenizer_checkpoint.endswith(
        "vqvae-512-8scale-500m-batch16/checkpoints/final.pt"
    )
    assert config.data.subset_directory.endswith("gtdb/500M_subset")
    assert config.data.sequence_length == 1024
    assert config.data.train_batch_size == 4
    assert config.data.validation_batch_size == 8
    assert config.model.model_dim == 640
    assert config.model.num_layers == 8
    assert config.model.num_heads == 10
    assert config.model.dropout == 0.1
    assert config.model.bias is False
    assert config.model.use_qk_norm is True
    assert config.optimizer.warmup_steps == 1907
    assert config.optimizer.beta_1 == 0.9
    assert config.optimizer.beta_2 == 0.95
    assert config.optimizer.weight_decay == 0.05
    assert config.optimizer.gradient_accumulation_steps == 8
    assert (
        config.data.sequence_length
        * config.data.train_batch_size
        * 4
        * config.optimizer.gradient_accumulation_steps
        == 131_072
    )
    assert config.training.num_epochs == 10
    assert config.training.input_code_corruption_probability == 0.1


def test_evaluate_reports_every_scale_and_restores_training_mode() -> None:
    tokenizer = _build_tokenizer().eval()
    tokenizer.requires_grad_(False)
    model = _build_nsm(max_prefix_length=4)
    input_ids = torch.arange(2 * 8).reshape(2, 8) % 4
    scale_weights = build_scale_loss_weights(
        tokenizer.scale_lengths,
        scale_loss_alpha=0.25,
        device=torch.device("cpu"),
    )

    metrics = evaluate(
        model,
        tokenizer,
        data_loader=[{"input_ids": input_ids}],
        scale_loss_weights=scale_weights,
        use_mixed_precision=False,
        rollout_max_batches=1,
    )

    assert model.training is True
    assert metrics["loss"] > 0
    assert 0 <= metrics["accuracy"] <= 1
    assert metrics["rollout_nucleotide_loss"] > 0
    assert 0 <= metrics["rollout_nucleotide_accuracy"] <= 1
    for scale_length in tokenizer.scale_lengths:
        assert 0 <= metrics[f"accuracy_scale_{scale_length}"] <= 1


def test_rollout_feeds_predictions_into_the_next_scale() -> None:
    tokenizer = _build_tokenizer().eval()
    model = _build_nsm()

    predicted_indices_by_scale = rollout_scale_predictions(
        model,
        tokenizer,
        batch_size=2,
        prefix=None,
    )

    assert [indices.shape for indices in predicted_indices_by_scale] == [
        (2, 1),
        (2, 2),
        (2, 4),
    ]

    teacher_forced_inputs = tokenizer.indices_to_next_scale_inputs(
        predicted_indices_by_scale
    )
    first_rollout_input = tokenizer.indices_to_next_scale_input(
        predicted_indices_by_scale[:1]
    )
    second_rollout_input = tokenizer.indices_to_next_scale_input(
        predicted_indices_by_scale[:2]
    )
    torch.testing.assert_close(first_rollout_input, teacher_forced_inputs[0])
    torch.testing.assert_close(second_rollout_input, teacher_forced_inputs[1])
