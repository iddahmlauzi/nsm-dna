from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from nsm_dna.models.next_scale import NSM
from nsm_dna.models.vqvae import VQVAE
from nsm_dna.training import calculate_training_steps
from scripts.training.train_nsm import (
    build_codebook_neighbor_tables,
    build_scale_loss_weights,
    compute_losses,
    compute_student_consistency_loss,
    corrupt_context_indices,
    evaluate,
    prepare_block_predictions,
    rollout_hierarchy,
    student_consistency_is_active,
)


def _build_tokenizer() -> VQVAE:
    return VQVAE(
        vocab_size=4,
        context_length=8,
        latent_length=4,
        embed_dim=8,
        quantization_dim=4,
        num_heads=2,
        scale_lengths=[1, 2, 4],
        codebook_sizes=[4, 6, 16],
        decoder_num_layers=1,
        use_qk_norm=True,
    )


def _build_nsm() -> NSM:
    return NSM(
        prefix_dim=4,
        model_dim=8,
        scale_lengths=[1, 2, 4],
        codebook_sizes=[4, 6, 16],
        codebook_vectors=[torch.randn(4, 4), torch.randn(6, 4), torch.randn(16, 4)],
        num_layers=1,
        num_heads=2,
        dropout=0.0,
    )

def test_training_steps_are_derived_from_epochs(tmp_path: Path) -> None:
    (tmp_path / "subset_stats.json").write_text(
        '{"selection": {"chunk_length": 100}, "splits": {"train": {"chunks": 10}}}'
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

    assert calculate_training_steps(config, world_size=2, sequence_length=16) == 9

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


def test_context_corruption_uses_nearby_codes() -> None:
    neighbor_tables = build_codebook_neighbor_tables(
        [torch.tensor([[0.0], [1.0], [3.0]])],
        neighbor_count=1,
    )
    context = [torch.tensor([[0, 1, 2]])]

    corrupted = corrupt_context_indices(
        context,
        neighbor_tables,
        corruption_probabilities=[1.0],
    )

    torch.testing.assert_close(corrupted[0], torch.tensor([[1, 0, 1]]))
    torch.testing.assert_close(context[0], torch.tensor([[0, 1, 2]]))


def test_loss_weights_exact_classification_by_scale() -> None:
    targets_by_scale = [
        torch.tensor([[0], [1]]),
        torch.tensor([[1, 2], [2, 3]]),
        torch.tensor([[3, 2, 1, 0], [0, 1, 2, 3]]),
    ]
    logits_by_scale = [
        torch.randn(2, 1, 2),
        torch.randn(2, 2, 4),
        torch.randn(2, 4, 4),
    ]
    scale_weights = torch.tensor([0.2, 0.3, 0.5])
    losses = compute_losses(
        logits_by_scale,
        targets_by_scale,
        scale_weights,
    )

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

    expected_hierarchy = torch.sum(expected_losses * scale_weights)
    torch.testing.assert_close(losses.hierarchy_by_scale, expected_losses)
    torch.testing.assert_close(losses.total, expected_hierarchy)


def test_student_consistency_matches_teacher_without_updating_teacher() -> None:
    teacher_logits = [
        torch.randn(2, 1, 4, requires_grad=True),
        torch.randn(2, 2, 6, requires_grad=True),
        torch.randn(2, 4, 8, requires_grad=True),
    ]
    student_logits = [
        logits.detach().clone().requires_grad_() for logits in teacher_logits
    ]

    loss = compute_student_consistency_loss(
        student_logits,
        teacher_logits,
        torch.tensor([0.2, 0.3, 0.5]),
    )
    loss.backward()

    torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1e-6, rtol=0)
    assert all(logits.grad is None for logits in teacher_logits)
    assert all(logits.grad is not None for logits in student_logits)
    assert torch.count_nonzero(student_logits[0].grad) == 0


def test_student_consistency_starts_after_warmup_epoch() -> None:
    assert not student_consistency_is_active(20, 100, 5, start_epoch=1)
    assert student_consistency_is_active(21, 100, 5, start_epoch=1)


def test_tokenizer_checkpoint_is_restored_and_frozen(tmp_path: Path) -> None:
    tokenizer = _build_tokenizer()
    checkpoint_path = tmp_path / "tokenizer.pt"
    torch.save(
        {
            "config": {
                "model": {
                    "vocab_size": 4,
                    "context_length": 8,
                    "latent_length": 4,
                    "embed_dim": 8,
                    "quantization_dim": 4,
                    "num_heads": 2,
                    "third_base_scale": 1.0,
                    "scale_lengths": [1, 2, 4],
                    "codebook_sizes": [4, 6, 16],
                    "decoder_num_layers": 1,
                    "use_qk_norm": True,
                    "bias": False,
                    "rope_base": 10000.0,
                    "commitment_cost": 0.25,
                    "decay": 0.99,
                    "eps": 1e-5,
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
    input_ids = torch.arange(2 * 16).reshape(2, 16) % 4
    scale_weights = build_scale_loss_weights(
        tokenizer.scale_lengths,
        scale_loss_alpha=0.25,
        device=torch.device("cpu"),
    )

    prediction = prepare_block_predictions(
        tokenizer,
        input_ids,
        model.scale_lengths,
    )[0]
    indices_by_scale = prediction.targets_by_scale
    output = model(
        indices_by_scale[:-1],
        prefix_by_scale=prediction.prefix_by_scale,
    )
    losses = compute_losses(
        output.hierarchy_logits,
        indices_by_scale,
        scale_weights,
    )
    losses.total.backward()

    assert [targets.shape[1] for targets in indices_by_scale] == [1, 2, 4]
    assert [logits.shape for logits in output.hierarchy_logits] == [
        (2, 1, 4),
        (2, 2, 6),
        (2, 4, 16),
    ]
    assert all(parameter.grad is None for parameter in tokenizer.parameters())
    assert any(parameter.grad is not None for parameter in model.parameters())
def test_block_prediction_uses_first_block_as_prefix_and_second_as_target() -> None:
    tokenizer = _build_tokenizer().eval()
    input_ids = torch.arange(2 * 16).reshape(2, 16) % 4
    prefix_ids = input_ids[:, :8]
    target_ids = input_ids[:, 8:]
    selected_scale_lengths = [
        tokenizer.scale_lengths[0],
        tokenizer.scale_lengths[-1],
    ]

    block_predictions = prepare_block_predictions(
        tokenizer,
        input_ids,
        selected_scale_lengths,
    )
    assert len(block_predictions) == 1

    prediction = block_predictions[0]
    torch.testing.assert_close(prediction.target_ids, target_ids)
    tokenizer_prefix = tokenizer.encode_scales(prefix_ids)
    expected_prefix = [tokenizer_prefix[0], tokenizer_prefix[-1]]
    for actual, expected in zip(
        prediction.prefix_by_scale,
        expected_prefix,
        strict=True,
    ):
        torch.testing.assert_close(actual, expected)

    tokenizer_targets = tokenizer.encode_indices(target_ids)
    expected_targets = [tokenizer_targets[0], tokenizer_targets[-1]]
    for actual, expected in zip(prediction.targets_by_scale, expected_targets):
        torch.testing.assert_close(actual, expected)


def test_evaluate_reports_every_scale_and_restores_training_mode() -> None:
    tokenizer = _build_tokenizer().eval()
    tokenizer.requires_grad_(False)
    model = _build_nsm()
    input_ids = torch.arange(2 * 16).reshape(2, 16) % 4
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
    assert 0 <= metrics["hierarchy_accuracy"] <= 1
    assert 0 <= metrics["nucleotide_accuracy"] <= 1
    assert metrics["rollout_nucleotide_loss"] > 0
    assert 0 <= metrics["rollout_nucleotide_accuracy"] <= 1
    for scale_index, scale_length in enumerate(tokenizer.scale_lengths, start=1):
        section = f"scale_{scale_index:02d}_length_{scale_length}"
        assert 0 <= metrics[f"{section}/prediction_accuracy"] <= 1


def test_rollout_carries_hard_predictions_between_scales() -> None:
    tokenizer = _build_tokenizer().eval()
    codebook_sizes = tokenizer.codebook_sizes

    class StubModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.device_anchor = torch.nn.Parameter(torch.zeros(()))
            self.scale_lengths = tokenizer.scale_lengths
            self.vectors = [
                torch.arange(size, dtype=torch.float32).reshape(-1, 1).repeat(1, 4)
                for size in codebook_sizes
            ]
            self.inputs: list[torch.Tensor | None] = []
            self.logits: list[torch.Tensor] = []

        def codebook_vectors(self, scale_index: int) -> torch.Tensor:
            return self.vectors[scale_index]

        def predict_scale(
            self,
            prefix: torch.Tensor,
            scale_index: int,
            previous_scale_latent: torch.Tensor | None,
        ) -> torch.Tensor:
            self.inputs.append(
                None
                if previous_scale_latent is None
                else previous_scale_latent.clone()
            )
            call_index = len(self.inputs) - 1
            codebook_size = codebook_sizes[scale_index]
            scale_length = tokenizer.scale_lengths[scale_index]
            logits = torch.full((2, scale_length, codebook_size), -1.0)
            logits[:, :, (call_index + 1) % codebook_size] = 1.0
            self.logits.append(logits)
            return logits

    model = StubModel()
    prefix_by_scale = [
        torch.zeros(2, scale_length, 4)
        for scale_length in tokenizer.scale_lengths
    ]
    predicted_indices_by_scale = rollout_hierarchy(
        model,
        prefix_by_scale=prefix_by_scale,
    )

    assert [indices.shape for indices in predicted_indices_by_scale] == [
        (2, 1),
        (2, 2),
        (2, 4),
    ]
    torch.testing.assert_close(
        predicted_indices_by_scale[0],
        torch.ones(2, 1, dtype=torch.long),
    )
    assert model.inputs[0] is None
    expected_first_context = model.codebook_vectors(0)[
        predicted_indices_by_scale[0]
    ]
    torch.testing.assert_close(model.inputs[1], expected_first_context)
    expected_second_context = model.codebook_vectors(1)[
        predicted_indices_by_scale[1]
    ]
    torch.testing.assert_close(
        model.inputs[2],
        expected_second_context,
    )
