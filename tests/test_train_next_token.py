from pathlib import Path

import torch
from omegaconf import OmegaConf

from nsm_dna.models.next_token import NextTokenModel
from nsm_dna.training import save_training_checkpoint
from scripts.training.train_next_token import (
    compute_next_token_loss,
    evaluate,
    prepare_next_token_batch,
)


def _build_model() -> NextTokenModel:
    return NextTokenModel(
        vocab_size=4,
        model_dim=8,
        num_layers=1,
        num_heads=2,
        max_sequence_length=7,
        dropout=0.0,
    )


def test_next_token_batch_aligns_each_context_with_the_following_token() -> None:
    input_ids = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])

    context_ids, targets = prepare_next_token_batch(input_ids)

    torch.testing.assert_close(context_ids, input_ids[:, :-1])
    torch.testing.assert_close(targets, input_ids[:, 1:])


def test_next_token_loss_trains_the_model() -> None:
    model = _build_model()
    input_ids = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])
    context_ids, targets = prepare_next_token_batch(input_ids)

    logits = model(context_ids)
    loss = compute_next_token_loss(logits, targets)
    loss.backward()

    assert loss.item() > 0
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_evaluate_reports_metrics_and_restores_training_mode() -> None:
    model = _build_model()
    input_ids = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])

    metrics = evaluate(
        model,
        data_loader=[{"input_ids": input_ids}],
        use_mixed_precision=False,
    )

    assert model.training is True
    assert metrics["loss"] > 0
    assert 0 <= metrics["accuracy"] <= 1


def test_checkpoint_is_saved_locally(tmp_path: Path) -> None:
    model = _build_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    config = OmegaConf.create({})

    checkpoint_path = save_training_checkpoint(
        tmp_path,
        model,
        optimizer,
        scheduler,
        config,
        step=1000,
        best_validation_loss=1.0,
        checkpoint_name="latest.pt",
    )

    assert checkpoint_path.exists()
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    assert checkpoint["step"] == 1000
