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
from nsm_dna.models.next_scale import NSMDNA
from nsm_dna.training import (
    GenFirstLossSchedule,
    GenerativeLossWeights,
    build_learning_rate_scheduler,
    load_training_checkpoint,
    save_training_checkpoint,
)
from scripts.training.train_nsm import (
    _validation_metrics_for_wandb,
    evaluate,
    target_ids_from_sequence,
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
        return_autoregressive_reconstruction=True,
    )
    losses = nsm_dna_losses(
        output,
        target_ids,
        autoregressive_reconstruction_loss_weight=1.0,
        next_scale_prediction_loss_weight=8.0,
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
        + losses.autoregressive_reconstruction
        + losses.vq
        + 8.0 * losses.next_scale_prediction
        + 2.0 * losses.entropy,
    )


def test_genfirst_schedule_switches_to_reconstruction_refinement() -> None:
    generation_first_weights = GenerativeLossWeights(
        next_scale_prediction=8.0,
        entropy=2.0,
    )
    refinement_weights = GenerativeLossWeights(
        next_scale_prediction=2.0,
        entropy=0.5,
    )
    schedule = GenFirstLossSchedule(
        total_steps=10,
        generation_first_fraction=0.8,
        generation_first_weights=generation_first_weights,
        refinement_weights=refinement_weights,
    )

    assert schedule.generation_first_steps == 8
    assert schedule.weights_at_step(1) == generation_first_weights
    assert schedule.weights_at_step(8) == generation_first_weights
    assert schedule.weights_at_step(9) == refinement_weights
    assert schedule.phase_at_step(8) == "generation_first"
    assert schedule.phase_at_step(9) == "reconstruction_refinement"


def test_genfirst_schedule_rejects_invalid_configuration() -> None:
    weights = GenerativeLossWeights(next_scale_prediction=1.0, entropy=1.0)
    with pytest.raises(ValueError, match="between zero and one"):
        GenFirstLossSchedule(
            total_steps=10,
            generation_first_fraction=1.0,
            generation_first_weights=weights,
            refinement_weights=weights,
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
    output = model(sequence_ids, corruption_probability=0.1)
    losses = nsm_dna_losses(
        output,
        target_ids_from_sequence(sequence_ids, target_length=4),
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
    assert 0 <= metrics["next_scale_prediction_accuracy"] <= 1
    assert metrics["rollout_nucleotide_loss"] > 0
    assert 0 <= metrics["rollout_nucleotide_accuracy"] <= 1
    for scale_length in model.tokenizer.scale_lengths:
        assert metrics[f"next_scale_prediction_loss_scale_{scale_length}"] > 0
        assert 0 <= metrics[f"next_scale_prediction_accuracy_scale_{scale_length}"] <= 1
        assert metrics[f"quantization_loss_scale_{scale_length}"] >= 0
        assert metrics[f"vq_loss_scale_{scale_length}"] >= 0
        assert 0 <= metrics[f"cumulative_nucleotide_accuracy_scale_{scale_length}"] <= 1
        assert 0 <= metrics[f"code_usage_scale_{scale_length}"] <= 1
        assert metrics[f"code_perplexity_scale_{scale_length}"] >= 1
        assert 0 <= metrics[f"soft_confidence_scale_{scale_length}"] <= 1


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

    assert wandb_metrics["validation/best_total_loss"] == 1.5
    assert {
        name for name in wandb_metrics if name.startswith("validation/")
    } == {
        "validation/total_loss",
        "validation/nucleotide_reconstruction_loss",
        "validation/autoregressive_reconstruction_loss",
        "validation/vq_loss",
        "validation/next_scale_prediction_loss",
        "validation/entropy_loss",
        "validation/rollout_nucleotide_loss",
        "validation/nucleotide_reconstruction_accuracy",
        "validation/autoregressive_reconstruction_accuracy",
        "validation/next_scale_prediction_accuracy",
        "validation/rollout_nucleotide_accuracy",
        "validation/best_total_loss",
    }
    for scale_number, scale_length in enumerate(
        model.tokenizer.scale_lengths,
        start=1,
    ):
        section = f"scale_{scale_number:02d}_length_{scale_length}"
        assert f"{section}/prediction_loss" in wandb_metrics
        assert f"{section}/cumulative_nucleotide_accuracy" in wandb_metrics
        assert f"{section}/code_perplexity" in wandb_metrics


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


def test_default_config_matches_the_joint_training_contract() -> None:
    config_path = Path(__file__).parents[1] / "configs" / "nsm.yaml"
    config = OmegaConf.load(config_path)

    assert config.run.resume_from is None
    assert config.wandb.project == "nsm-dna-end-to-end"
    assert config.wandb.name == "nsm-dna-end-to-end-neighbor-corruption-apr"
    assert config.data.subset_directory.endswith("gtdb/500M_subset")
    assert config.data.sequence_length == 256
    assert config.model.tokenizer.context_length == 128
    assert config.model.tokenizer.embed_dim == 384
    assert len(config.model.tokenizer.scale_lengths) == 9
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
    assert config.training.autoregressive_reconstruction_loss_weight == 1.0
    assert config.training.entropy_temperature > 1.0
    assert config.training.loss_schedule.generation_first_fraction == 0.8
    assert (
        config.training.loss_schedule.generation_first.next_scale_prediction_loss_weight
        == 8.0
    )
    assert config.training.loss_schedule.generation_first.entropy_loss_weight == 2.0
    assert (
        config.training.loss_schedule.reconstruction_refinement.next_scale_prediction_loss_weight
        == 2.0
    )
    assert (
        config.training.loss_schedule.reconstruction_refinement.entropy_loss_weight
        == 0.5
    )
    assert config.training.input_code_corruption_probability == 0.1
    assert "huggingface" not in config.checkpoint
