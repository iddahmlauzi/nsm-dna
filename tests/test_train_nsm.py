import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from nsm_dna.losses import (
    codebook_entropy_loss,
    next_scale_prediction_loss,
    nsm_dna_losses,
)
from nsm_dna.models.next_scale import NSMDNA, TokenizerStabilitySnapshot
from nsm_dna.training import (
    build_learning_rate_scheduler,
    load_model_checkpoint,
    load_training_checkpoint,
    save_training_checkpoint,
)
from scripts.training.train_nsm import (
    _validation_metrics_for_wandb,
    append_metrics,
    configure_training_phase,
    evaluate,
    target_ids_from_sequence,
    tokenizer_stability_metrics,
)


def _build_config():
    return OmegaConf.create(
        {
            "data": {"sequence_length": 8},
            "model": {
                "tokenizer": {
                    "vocab_size": 4,
                    "context_length": 4,
                    "embed_dim": 8,
                    "num_heads": 2,
                    "scale_lengths": [1, 2, 4],
                    "codebook_size": 8,
                    "encoder_dropout": 0.0,
                    "decoder_dropout": 0.0,
                    "bias": False,
                    "rope_base": 10_000.0,
                    "pre_quant_num_groups": 2,
                    "commitment_cost": 0.25,
                    "decay": 0.99,
                    "eps": 1e-5,
                    "temperature": 1.0,
                    "refinement_ratio": 0.5,
                    "refinement_kernel_size": 3,
                },
                "transformer": {
                    "model_dim": 8,
                    "num_layers": 1,
                    "num_heads": 2,
                    "dropout": 0.0,
                    "bias": False,
                    "use_qk_norm": True,
                    "rope_base": 10_000.0,
                    "head_num_blocks": 2,
                    "head_hidden_multiplier": 2.0,
                    "input_refinement_kernel_size": 3,
                },
            },
        }
    )


def _sequence_ids() -> torch.Tensor:
    return torch.tensor(
        [
            [0, 1, 2, 3, 3, 2, 1, 0],
            [3, 2, 1, 0, 0, 1, 2, 3],
        ]
    )


def test_next_scale_prediction_loss_weights_scales_equally() -> None:
    logits_by_scale = [
        torch.randn(2, 1, 8),
        torch.randn(2, 2, 8),
        torch.randn(2, 4, 8),
    ]
    targets_by_scale = [
        torch.tensor([[0], [1]]),
        torch.tensor([[1, 2], [2, 3]]),
        torch.tensor([[3, 2, 1, 0], [0, 1, 2, 3]]),
    ]

    loss, losses_by_scale = next_scale_prediction_loss(
        logits_by_scale,
        targets_by_scale,
    )

    expected_losses = [
        F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        for logits, targets in zip(logits_by_scale, targets_by_scale, strict=True)
    ]
    torch.testing.assert_close(
        torch.stack(losses_by_scale),
        torch.stack(expected_losses),
    )
    torch.testing.assert_close(loss, torch.stack(expected_losses).mean())


def test_entropy_loss_favors_confident_and_balanced_assignments() -> None:
    collapsed_logits = torch.zeros(1, 4, 4)
    collapsed_logits[..., 0] = 8.0
    balanced_logits = (torch.eye(4) * 8.0).unsqueeze(0).requires_grad_()
    uncertain_logits = torch.zeros(1, 4, 4)

    collapsed_loss = codebook_entropy_loss([collapsed_logits], temperature=1.0)
    balanced_loss = codebook_entropy_loss([balanced_logits], temperature=1.0)
    uncertain_loss = codebook_entropy_loss([uncertain_logits], temperature=1.0)

    torch.testing.assert_close(collapsed_loss, torch.tensor(0.0))
    torch.testing.assert_close(uncertain_loss, torch.tensor(0.0))
    assert balanced_loss < collapsed_loss
    assert balanced_loss > -math.log(4)
    balanced_loss.backward()
    assert balanced_logits.grad is not None
    assert balanced_logits.grad.count_nonzero() > 0

    two_scale_loss = codebook_entropy_loss(
        [collapsed_logits, balanced_logits.detach()],
        temperature=1.0,
    )
    torch.testing.assert_close(two_scale_loss, balanced_loss.detach() / 2)


def test_entropy_loss_rejects_nonpositive_temperature() -> None:
    with pytest.raises(ValueError, match="entropy temperature must be positive"):
        codebook_entropy_loss([torch.zeros(1, 1, 4)], temperature=0.0)


def test_total_loss_applies_generative_objective_weights() -> None:
    model = NSMDNA.from_config(_build_config())
    sequence_ids = _sequence_ids()
    target_ids = target_ids_from_sequence(sequence_ids, target_length=4)

    output = model(
        sequence_ids,
        corruption_probability=0.1,
        return_partial_reconstruction=True,
        return_teacher_forced_prediction_reconstruction=True,
        return_rollout=True,
    )
    losses = nsm_dna_losses(
        output,
        target_ids,
        partial_reconstruction_loss_weight=0.1,
        teacher_forced_prediction_reconstruction_loss_weight=1.0,
        next_scale_prediction_loss_weight=8.0,
        rollout_state_consistency_loss_weight=0.5,
        entropy_loss_weight=2.0,
    )

    expected_reconstruction = F.cross_entropy(
        output.reconstruction_logits.flatten(0, 1),
        target_ids.flatten(),
    )
    torch.testing.assert_close(
        losses.nucleotide_reconstruction,
        expected_reconstruction,
    )
    torch.testing.assert_close(losses.vq, output.quantizer.vq_loss)
    torch.testing.assert_close(
        losses.total,
        losses.nucleotide_reconstruction
        + 0.1 * losses.partial_reconstruction
        + losses.teacher_forced_prediction_reconstruction
        + losses.vq
        + 8.0 * losses.teacher_forced_prediction
        + 0.5 * losses.rollout_state_consistency
        + 2.0 * losses.entropy,
    )


def test_state_consistency_averages_scale_transitions_equally() -> None:
    model = NSMDNA.from_config(_build_config())
    sequence_ids = _sequence_ids()
    output = model(sequence_ids, return_rollout=True)

    losses = nsm_dna_losses(
        output,
        target_ids_from_sequence(sequence_ids, target_length=4),
        rollout_state_consistency_loss_weight=1.0,
        entropy_loss_weight=0.0,
    )

    torch.testing.assert_close(
        losses.rollout_state_consistency,
        torch.stack(losses.rollout_state_consistency_by_scale).mean(),
    )


def test_state_consistency_trains_only_the_transformer_without_a_prefix() -> None:
    config = _build_config()
    config.data.sequence_length = config.model.tokenizer.context_length
    model = NSMDNA.from_config(config)
    sequence_ids = _sequence_ids()[:, -4:]
    output = model(sequence_ids, return_rollout=True)
    assert output.rollout is not None
    final_rollout_logits = output.rollout.prediction_logits_by_scale[-1]
    final_rollout_logits.retain_grad()
    losses = nsm_dna_losses(
        output,
        target_ids_from_sequence(sequence_ids, target_length=4),
        rollout_state_consistency_loss_weight=1.0,
        entropy_loss_weight=0.0,
    )

    losses.rollout_state_consistency.backward()

    assert final_rollout_logits.grad is not None
    assert final_rollout_logits.grad.count_nonzero() > 0
    assert any(
        parameter.grad is not None and parameter.grad.count_nonzero() > 0
        for parameter in model.transformer.parameters()
    )
    assert all(
        parameter.grad is None or parameter.grad.count_nonzero() == 0
        for parameter in model.tokenizer.parameters()
    )


def test_state_consistency_requires_rollout_outputs() -> None:
    model = NSMDNA.from_config(_build_config())
    sequence_ids = _sequence_ids()
    output = model(sequence_ids)
    target_ids = target_ids_from_sequence(sequence_ids, target_length=4)

    with pytest.raises(ValueError, match="Rollout outputs are required"):
        nsm_dna_losses(
            output,
            target_ids,
            rollout_state_consistency_loss_weight=1.0,
        )


def test_validation_partial_loss_averages_nonfinal_scales() -> None:
    model = NSMDNA.from_config(_build_config()).eval()
    sequence_ids = _sequence_ids()
    target_ids = target_ids_from_sequence(sequence_ids, target_length=4)
    output = model(sequence_ids, return_cumulative_reconstructions=True)

    losses = nsm_dna_losses(
        output,
        target_ids,
        partial_reconstruction_loss_weight=0.1,
    )

    cumulative_logits = output.cumulative_reconstruction_logits_by_scale
    assert cumulative_logits is not None
    expected_partial_loss = torch.stack(
        [
            F.cross_entropy(scale_logits.flatten(0, 1), target_ids.flatten())
            for scale_logits in cumulative_logits[:-1]
        ]
    ).mean()
    torch.testing.assert_close(
        losses.partial_reconstruction,
        expected_partial_loss,
    )


def test_one_optimizer_step_updates_the_joint_model() -> None:
    model = NSMDNA.from_config(_build_config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer_parameter_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    trainable_parameter_ids = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    encoder_weight_before = (
        model.tokenizer.encoder.token_embedding.weight.detach().clone()
    )
    transformer_weight_before = (
        model.transformer.input_projection.weight.detach().clone()
    )

    sequence_ids = _sequence_ids()
    output = model(
        sequence_ids,
        corruption_probability=0.1,
        return_rollout=True,
    )
    losses = nsm_dna_losses(
        output,
        target_ids_from_sequence(sequence_ids, target_length=4),
        rollout_state_consistency_loss_weight=1.0,
    )
    losses.total.backward()
    optimizer.step()

    assert optimizer_parameter_ids == trainable_parameter_ids
    assert all(
        codebook.codebook.grad is None
        for codebook in model.tokenizer.quantizer.codebooks
    )
    assert not torch.equal(
        model.tokenizer.encoder.token_embedding.weight,
        encoder_weight_before,
    )
    assert not torch.equal(
        model.transformer.input_projection.weight,
        transformer_weight_before,
    )


def test_frozen_hierarchy_phase_trains_only_the_transformer() -> None:
    model = NSMDNA.from_config(_build_config())
    configure_training_phase(model, freeze_tokenizer=True)
    tokenizer_state_before = {
        name: value.detach().clone()
        for name, value in model.tokenizer.state_dict().items()
    }

    sequence_ids = _sequence_ids()
    output = model(
        sequence_ids,
        return_teacher_forced_prediction_reconstruction=True,
    )
    losses = nsm_dna_losses(
        output,
        target_ids_from_sequence(sequence_ids, target_length=4),
        teacher_forced_prediction_reconstruction_loss_weight=1.0,
        next_scale_prediction_loss_weight=8.0,
        entropy_loss_weight=0.0,
    )
    losses.total.backward()

    assert model.training is True
    assert model.tokenizer.training is False
    assert all(parameter.grad is None for parameter in model.tokenizer.parameters())
    assert any(
        parameter.grad is not None for parameter in model.transformer.parameters()
    )
    for name, expected_value in tokenizer_state_before.items():
        torch.testing.assert_close(model.tokenizer.state_dict()[name], expected_value)

    evaluate(
        model,
        data_loader=[{"input_ids": sequence_ids}],
        use_mixed_precision=False,
    )
    assert model.training is True
    assert model.tokenizer.training is False


def test_evaluate_reports_joint_and_per_scale_metrics() -> None:
    model = NSMDNA.from_config(_build_config())

    metrics = evaluate(
        model,
        data_loader=[{"input_ids": _sequence_ids()}],
        use_mixed_precision=False,
    )

    assert model.training is True
    assert metrics["total_loss"] > 0
    assert metrics["entropy_loss"] <= 0
    assert 0 <= metrics["nucleotide_reconstruction_accuracy"] <= 1
    assert 0 <= metrics["teacher_forced_prediction_accuracy"] <= 1
    assert 0 <= metrics["rollout_prediction_accuracy"] <= 1
    assert metrics["rollout_nucleotide_loss"] > 0
    assert 0 <= metrics["rollout_nucleotide_accuracy"] <= 1
    assert metrics["rollout_state_consistency_loss"] >= 0
    assert metrics["encoder_commitment_loss"] >= 0
    first_predicted_scale = model.tokenizer.scale_lengths[1]
    assert metrics[
        f"teacher_forced_rollout_agreement_scale_{first_predicted_scale}"
    ] == pytest.approx(1.0)
    for scale_length in model.tokenizer.scale_lengths:
        if scale_length != model.tokenizer.scale_lengths[0]:
            assert metrics[f"teacher_forced_prediction_loss_scale_{scale_length}"] > 0
            assert (
                0
                <= metrics[f"teacher_forced_prediction_accuracy_scale_{scale_length}"]
                <= 1
            )
            assert (
                0
                <= metrics[f"rollout_prediction_accuracy_scale_{scale_length}"]
                <= 1
            )
            assert (
                0
                <= metrics[f"teacher_forced_rollout_agreement_scale_{scale_length}"]
                <= 1
            )
        assert metrics[f"quantization_loss_scale_{scale_length}"] >= 0
        assert 0 <= metrics[f"cumulative_nucleotide_accuracy_scale_{scale_length}"] <= 1
        assert (
            0
            <= metrics[f"rollout_cumulative_nucleotide_accuracy_scale_{scale_length}"]
            <= 1
        )
        assert 0 <= metrics[f"code_usage_scale_{scale_length}"] <= 1
        assert metrics[f"code_perplexity_scale_{scale_length}"] >= 1
        assert 0 <= metrics[f"soft_confidence_scale_{scale_length}"] <= 1
        if scale_length != model.tokenizer.scale_lengths[0]:
            assert (
                metrics[
                    f"rollout_state_consistency_loss_after_scale_{scale_length}"
                ]
                >= 0
            )


def test_validation_wandb_metrics_are_grouped_by_scale() -> None:
    model = NSMDNA.from_config(_build_config())
    validation_metrics = evaluate(
        model,
        data_loader=[{"input_ids": _sequence_ids()}],
        use_mixed_precision=False,
    )

    wandb_metrics = _validation_metrics_for_wandb(
        validation_metrics,
        model.tokenizer.scale_lengths,
        best_validation_loss=1.5,
    )

    assert wandb_metrics["validation/objective/best_total"] == 1.5
    assert {name for name in wandb_metrics if name.startswith("validation/")} == {
        "validation/objective/total",
        "validation/objective/rollout_state_consistency",
        "validation/objective/best_total",
        "validation/accuracy/nucleotide_reconstruction",
        "validation/accuracy/teacher_forced_prediction_reconstruction",
        "validation/accuracy/teacher_forced_prediction",
        "validation/accuracy/rollout_prediction",
        "validation/accuracy/rollout_nucleotide",
    }
    for scale_number, scale_length in enumerate(
        model.tokenizer.scale_lengths,
        start=1,
    ):
        section = f"scale_{scale_number:02d}_length_{scale_length}"
        if scale_number == 1:
            assert f"{section}/teacher_forced_prediction_accuracy" not in wandb_metrics
        else:
            assert f"{section}/teacher_forced_prediction_accuracy" in wandb_metrics
            assert f"{section}/rollout_prediction_accuracy" in wandb_metrics
            assert f"{section}/teacher_forced_rollout_agreement" in wandb_metrics
        assert f"{section}/cumulative_nucleotide_accuracy" in wandb_metrics
        assert f"{section}/rollout_cumulative_nucleotide_accuracy" in wandb_metrics
        assert f"{section}/quantization_loss" not in wandb_metrics
        assert f"{section}/soft_assignment_confidence" not in wandb_metrics
        assert f"{section}/code_perplexity" in wandb_metrics


def test_tokenizer_stability_metrics_compare_the_same_anchor_states() -> None:
    previous = TokenizerStabilitySnapshot(
        encoder_latent=torch.zeros(1, 2, 1),
        indices_by_scale=[torch.tensor([[0]]), torch.tensor([[0, 1]])],
        codebooks_by_scale=[torch.zeros(2, 1), torch.zeros(2, 1)],
        clean_consistency_states_by_scale=[torch.zeros(1, 2, 1)],
        rollout_consistency_states_by_scale=[torch.zeros(1, 2, 1)],
    )
    current = TokenizerStabilitySnapshot(
        encoder_latent=torch.ones(1, 2, 1),
        indices_by_scale=[torch.tensor([[0]]), torch.tensor([[1, 1]])],
        codebooks_by_scale=[torch.ones(2, 1), torch.full((2, 1), 2.0)],
        clean_consistency_states_by_scale=[torch.full((1, 2, 1), 2.0)],
        rollout_consistency_states_by_scale=[torch.full((1, 2, 1), 5.0)],
    )

    metrics = tokenizer_stability_metrics(previous, current, [1, 2])

    assert metrics["encoder_latent_drift"] == pytest.approx(1.0)
    assert metrics["code_retention_scale_1"] == pytest.approx(1.0)
    assert metrics["code_retention_scale_2"] == pytest.approx(0.5)
    assert metrics["code_retention"] == pytest.approx(0.75)
    assert metrics["active_codebook_drift"] == pytest.approx(1.5)
    assert metrics["target_state_drift"] == pytest.approx(2.0)
    assert metrics["rollout_state_error"] == pytest.approx(3.0)
    assert metrics["target_drift_relative_to_rollout_error"] == pytest.approx(
        2.0 / 3.0
    )


def test_wandb_logs_only_aggregate_tokenizer_stability_metrics() -> None:
    validation_metrics = {
        "total_loss": 1.0,
        "rollout_state_consistency_loss": 0.5,
        "nucleotide_reconstruction_accuracy": 0.9,
        "teacher_forced_prediction_reconstruction_accuracy": 0.8,
        "teacher_forced_prediction_accuracy": 0.7,
        "rollout_prediction_accuracy": 0.6,
        "rollout_nucleotide_accuracy": 0.5,
        "code_retention": 0.75,
        "encoder_latent_drift": 0.1,
        "active_codebook_drift": 0.2,
        "target_state_drift": 0.3,
        "target_drift_relative_to_rollout_error": 0.4,
        "target_state_drift_after_scale_2": 0.35,
    }

    wandb_metrics = _validation_metrics_for_wandb(
        validation_metrics,
        scale_lengths=[1, 2],
        best_validation_loss=1.0,
    )

    assert wandb_metrics["validation/stability/code_retention"] == 0.75
    assert wandb_metrics["validation/stability/target_state_drift"] == 0.3
    assert not any("after_scale" in name for name in wandb_metrics)


def test_complete_metrics_are_appended_to_jsonl(tmp_path: Path) -> None:
    metrics_path = tmp_path / "validation_metrics.jsonl"

    append_metrics(metrics_path, 1000, {"total_loss": 1.5})
    append_metrics(metrics_path, 2000, {"total_loss": 1.0})

    records = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    assert records == [
        {"step": 1000, "total_loss": 1.5},
        {"step": 2000, "total_loss": 1.0},
    ]


def test_unified_checkpoint_restores_model_optimizer_and_scheduler(
    tmp_path: Path,
) -> None:
    config = _build_config()
    model = NSMDNA.from_config(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = build_learning_rate_scheduler(
        optimizer,
        warmup_steps=1,
        decay_end_step=10,
        learning_rate=1e-3,
        min_learning_rate=1e-4,
    )

    sequence_ids = _sequence_ids()
    losses = nsm_dna_losses(
        model(sequence_ids),
        target_ids_from_sequence(sequence_ids, target_length=4),
    )
    losses.total.backward()
    optimizer.step()
    scheduler.step()
    checkpoint_path = save_training_checkpoint(
        tmp_path,
        model,
        optimizer,
        scheduler,
        config,
        step=7,
        best_validation_loss=2.5,
    )

    restored_model = NSMDNA.from_config(config)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
    restored_scheduler = build_learning_rate_scheduler(
        restored_optimizer,
        warmup_steps=1,
        decay_end_step=10,
        learning_rate=1e-3,
        min_learning_rate=1e-4,
    )
    step, best_validation_loss = load_training_checkpoint(
        checkpoint_path,
        restored_model,
        restored_optimizer,
        restored_scheduler,
        torch.device("cpu"),
    )

    assert step == 7
    assert best_validation_loss == 2.5
    assert restored_scheduler.state_dict() == scheduler.state_dict()
    for name, expected_value in model.state_dict().items():
        torch.testing.assert_close(restored_model.state_dict()[name], expected_value)


def test_model_checkpoint_initialization_does_not_restore_training_state(
    tmp_path: Path,
) -> None:
    config = _build_config()
    model = NSMDNA.from_config(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = build_learning_rate_scheduler(
        optimizer,
        warmup_steps=1,
        decay_end_step=10,
        learning_rate=1e-3,
        min_learning_rate=1e-4,
    )
    checkpoint_path = save_training_checkpoint(
        tmp_path,
        model,
        optimizer,
        scheduler,
        config,
        step=7,
        best_validation_loss=2.5,
    )

    initialized_model = NSMDNA.from_config(config)
    source_step = load_model_checkpoint(
        checkpoint_path,
        initialized_model,
        torch.device("cpu"),
    )

    assert source_step == 7
    for name, expected_value in model.state_dict().items():
        torch.testing.assert_close(
            initialized_model.state_dict()[name],
            expected_value,
        )


def test_default_config_matches_the_joint_training_contract() -> None:
    config_path = Path(__file__).parents[1] / "configs" / "nsm.yaml"
    config = OmegaConf.load(config_path)

    assert config.run.resume_from is None
    assert config.run.initialize_from is None
    assert config.run.reset_best_validation_loss is False
    assert config.wandb.project == "nsm-dna-end-to-end"
    assert (
        config.wandb.name
        == "nsm-dna-end-to-end-contextual-no-prefix-supplied-scale4"
    )
    assert config.data.subset_directory.endswith("gtdb/500M_subset")
    assert config.data.sequence_length == 128
    assert config.model.tokenizer.context_length == 128
    assert config.model.tokenizer.embed_dim == 384
    assert config.model.tokenizer.encoder_num_layers == 1
    assert list(config.model.tokenizer.scale_lengths) == [
        4,
        9,
        16,
        25,
        36,
        49,
        64,
        128,
    ]
    assert config.model.tokenizer.codebook_size == 256
    assert config.model.transformer.model_dim == 640
    assert config.optimizer.learning_rate == 1e-4
    assert config.optimizer.min_learning_rate == 1e-5
    assert config.optimizer.warmup_fraction == 0.05
    assert config.optimizer.beta_1 == 0.9
    assert config.optimizer.beta_2 == 0.95
    assert config.optimizer.weight_decay == 0.05
    assert config.optimizer.max_gradient_norm == 1.0
    assert config.training.num_epochs == 2
    assert config.training.freeze_tokenizer is False
    assert config.training.partial_reconstruction_loss_weight == 0.1
    assert (
        config.training.teacher_forced_prediction_reconstruction_loss_weight == 1.0
    )
    assert config.training.rollout_state_consistency_loss_weight == 0.0
    assert config.training.entropy_temperature > 1.0
    assert config.training.next_scale_prediction_loss_weight == 8.0
    assert config.training.entropy_loss_weight == 2.0
    assert "loss_schedule" not in config.training
    assert "self_conditioning" not in config.training
    assert config.training.input_code_corruption_probability == 0.0
    assert config.evaluation.stability_anchor_size == 32
    assert config.checkpoint.recovery_interval == 1_000
    assert "huggingface" not in config.checkpoint
