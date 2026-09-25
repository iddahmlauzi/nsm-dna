from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from nsm_dna.models.next_scale import NSM
from nsm_dna.models.vqvae import VQVAE
from nsm_dna.training import calculate_training_steps
from scripts.training.train_nsm import (
    build_codebook_distance_matrices,
    build_codebook_neighbor_tables,
    build_scale_loss_weights,
    compute_next_scale_loss,
    corrupt_context_indices,
    evaluate,
    prepare_block_predictions,
    rollout_hierarchy,
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


def _build_nsm(max_prefix_length: int = 4) -> NSM:
    return NSM(
        prefix_dim=4,
        model_dim=8,
        scale_lengths=[1, 2, 4],
        codebook_sizes=[4, 6, 16],
        codebook_vectors=[torch.randn(4, 4), torch.randn(6, 4), torch.randn(16, 4)],
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        max_prefix_length=max_prefix_length,
    )


def _codebook_distances(tokenizer: VQVAE) -> list[torch.Tensor]:
    return build_codebook_distance_matrices(
        [codebook.codebook for codebook in tokenizer.quantizer.codebooks]
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
    logits_by_scale = [
        torch.randn(2, 1, 2),
        torch.randn(2, 2, 4),
        torch.randn(2, 4, 4),
    ]
    scale_weights = torch.tensor([0.2, 0.3, 0.5])

    codebook_distances = [
        torch.ones(codebook_size, codebook_size)
        - torch.eye(codebook_size)
        for codebook_size in [2, 4, 4]
    ]
    loss, cross_entropy_by_scale, geometry_loss_by_scale = (
        compute_next_scale_loss(
            logits_by_scale,
            targets_by_scale,
            scale_weights,
            codebook_distances,
            geometry_loss_weight=0.1,
        )
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

    expected_combined_losses = expected_losses + 0.1 * geometry_loss_by_scale
    torch.testing.assert_close(cross_entropy_by_scale, expected_losses)
    torch.testing.assert_close(
        loss,
        torch.sum(expected_combined_losses * scale_weights),
    )


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
    logits_by_scale = list(torch.split(logits, scale_lengths, dim=1))
    scale_weights = build_scale_loss_weights(
        scale_lengths,
        scale_loss_alpha,
        device=torch.device("cpu"),
    )

    loss, _, _ = compute_next_scale_loss(
        logits_by_scale,
        targets_by_scale,
        scale_weights,
        [torch.zeros(4, 4) for _ in scale_lengths],
        geometry_loss_weight=0.0,
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


def test_geometry_loss_prefers_probability_on_nearby_codes() -> None:
    target = [torch.tensor([[0]])]
    distances = [
        torch.tensor(
            [
                [0.0, 1.0, 3.0],
                [1.0, 0.0, 2.0],
                [3.0, 2.0, 0.0],
            ]
        )
    ]
    near_logits = [torch.tensor([[[0.0, 2.0, 0.0]]])]
    far_logits = [torch.tensor([[[0.0, 0.0, 2.0]]])]

    near_loss, near_cross_entropy, near_geometry = compute_next_scale_loss(
        near_logits,
        target,
        torch.ones(1),
        distances,
        geometry_loss_weight=1.0,
    )
    far_loss, far_cross_entropy, far_geometry = compute_next_scale_loss(
        far_logits,
        target,
        torch.ones(1),
        distances,
        geometry_loss_weight=1.0,
    )

    torch.testing.assert_close(near_cross_entropy, far_cross_entropy)
    assert near_geometry.item() < far_geometry.item()
    assert near_loss.item() < far_loss.item()


def test_noisy_context_replaces_codes_with_nearest_neighbors() -> None:
    distance_matrices = [
        torch.tensor(
            [
                [0.0, 1.0, 3.0],
                [1.0, 0.0, 2.0],
                [3.0, 2.0, 0.0],
            ]
        ),
        torch.tensor(
            [
                [0.0, 1.0, 3.0],
                [1.0, 0.0, 2.0],
                [3.0, 2.0, 0.0],
            ]
        ),
    ]
    neighbor_tables = build_codebook_neighbor_tables(
        distance_matrices,
        neighbor_count=1,
    )
    clean_context = [
        torch.tensor([[0, 1, 2]]),
        torch.tensor([[2, 1, 0]]),
    ]

    corrupted_context = corrupt_context_indices(
        clean_context,
        neighbor_tables,
        corruption_probabilities=[1.0, 0.0],
    )

    torch.testing.assert_close(
        neighbor_tables[0],
        torch.tensor([[1], [0], [1]]),
    )
    torch.testing.assert_close(
        corrupted_context[0],
        torch.tensor([[1, 0, 1]]),
    )
    torch.testing.assert_close(corrupted_context[1], clean_context[1])
    torch.testing.assert_close(clean_context[0], torch.tensor([[0, 1, 2]]))


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

    prediction = prepare_block_predictions(tokenizer, input_ids)[0]
    indices_by_scale = prediction.targets_by_scale
    logits_by_scale = model(
        indices_by_scale[:-1],
        prefix=prediction.prefix,
    )
    loss, _, _ = compute_next_scale_loss(
        logits_by_scale,
        indices_by_scale,
        scale_weights,
        _codebook_distances(tokenizer),
        geometry_loss_weight=0.1,
    )
    loss.backward()

    assert [targets.shape[1] for targets in indices_by_scale] == [1, 2, 4]
    assert [logits.shape for logits in logits_by_scale] == [
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

    block_predictions = prepare_block_predictions(tokenizer, input_ids)
    assert len(block_predictions) == 1

    prediction = block_predictions[0]
    torch.testing.assert_close(prediction.target_ids, target_ids)
    torch.testing.assert_close(prediction.prefix, tokenizer.encode(prefix_ids))

    expected_targets = tokenizer.encode_indices(target_ids)
    for actual, expected in zip(prediction.targets_by_scale, expected_targets):
        torch.testing.assert_close(actual, expected)


def test_default_config_matches_fixed_hierarchy_recipe() -> None:
    config_path = Path(__file__).parents[1] / "configs" / "nsm.yaml"
    config = OmegaConf.load(config_path)

    assert config.run.resume_from is None
    assert config.tokenizer_checkpoint.endswith(
        "vqvae-256-dinucleotide/checkpoints/best.pt"
    )
    assert config.wandb.name == "nsm-256-packed-next-scale-learned-hierarchy"
    assert config.data.subset_directory.endswith("gtdb/500M_subset")
    assert config.data.sequence_length == 512
    assert config.data.train_batch_size == 64
    assert config.data.validation_batch_size == 8
    assert config.model.model_dim == 768
    assert config.model.num_layers == 12
    assert config.model.num_heads == 12
    assert config.model.dropout == 0.1
    assert config.model.bias is False
    assert config.model.use_qk_norm is True
    assert config.optimizer.warmup_steps == 1907
    assert config.optimizer.beta_1 == 0.9
    assert config.optimizer.beta_2 == 0.95
    assert config.optimizer.weight_decay == 0.05
    assert config.optimizer.gradient_accumulation_steps == 1
    assert config.optimizer.geometry_loss_weight == 0.1
    assert (
        config.data.sequence_length
        * config.data.train_batch_size
        * 4
        * config.optimizer.gradient_accumulation_steps
        == 131_072
    )
    assert config.training.num_epochs == 10
    assert list(config.training.context_corruption_probabilities) == [
        0.30,
        0.25,
        0.20,
        0.15,
        0.10,
        0.075,
        0.05,
    ]
    assert config.training.context_neighbor_count == 5
    assert config.checkpoint.interval == 5000


def test_evaluate_reports_every_scale_and_restores_training_mode() -> None:
    tokenizer = _build_tokenizer().eval()
    tokenizer.requires_grad_(False)
    model = _build_nsm(max_prefix_length=4)
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
        codebook_distances_by_scale=_codebook_distances(tokenizer),
        geometry_loss_weight=0.1,
        use_mixed_precision=False,
        rollout_max_batches=1,
    )

    assert model.training is True
    assert metrics["loss"] > 0
    assert metrics["cross_entropy_loss"] > 0
    assert metrics["geometry_loss"] > 0
    assert 0 <= metrics["accuracy"] <= 1
    assert metrics["rollout_nucleotide_loss"] > 0
    assert 0 <= metrics["rollout_nucleotide_accuracy"] <= 1
    for scale_length in tokenizer.scale_lengths:
        assert 0 <= metrics[f"accuracy_scale_{scale_length}"] <= 1


def test_rollout_feeds_each_prediction_back_into_the_hierarchy() -> None:
    tokenizer = _build_tokenizer().eval()
    codebook_sizes = tokenizer.codebook_sizes

    class StubModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.device_anchor = torch.nn.Parameter(torch.zeros(()))
            self.inputs: list[tuple[torch.Tensor, list[torch.Tensor]]] = []

        def predict_scale(
            self,
            prefix: torch.Tensor,
            completed_scales: list[torch.Tensor],
        ) -> torch.Tensor:
            self.inputs.append(
                (
                    prefix.clone(),
                    [indices.clone() for indices in completed_scales],
                )
            )
            call_index = len(self.inputs) - 1
            scale_index = len(completed_scales)
            codebook_size = codebook_sizes[scale_index]
            scale_length = tokenizer.scale_lengths[scale_index]
            logits = torch.full((2, scale_length, codebook_size), -1.0)
            logits[:, :, (call_index + 1) % codebook_size] = 1.0
            return logits

    model = StubModel()
    predicted_indices_by_scale = rollout_hierarchy(
        model,
        tokenizer,
        prefix=torch.zeros(2, 4, 4),
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
    torch.testing.assert_close(model.inputs[0][0], torch.zeros(2, 4, 4))
    torch.testing.assert_close(model.inputs[1][1][0], predicted_indices_by_scale[0])
    torch.testing.assert_close(
        model.inputs[2][1][1],
        predicted_indices_by_scale[1],
    )
