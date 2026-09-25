import math

import pytest
import torch

from scripts.evaluation.analyze_nsm_hierarchy import (
    empty_condition_metrics,
    make_target_controls,
    rollout_with_oracle_scale,
    summarize_condition,
    update_condition_metrics,
)


def test_target_controls_keep_prefix_and_shuffle_target_composition() -> None:
    input_ids = torch.tensor(
        [
            [0, 1, 2, 3, 0, 0, 1, 2],
            [3, 2, 1, 0, 3, 3, 2, 1],
        ]
    )
    controls = make_target_controls(
        input_ids,
        prefix_length=4,
        generator=torch.Generator().manual_seed(4),
    )

    for control in controls.values():
        torch.testing.assert_close(control[:, :4], input_ids[:, :4])
    for real_target, shuffled_target in zip(
        input_ids[:, 4:],
        controls["composition_shuffled"][:, 4:],
    ):
        torch.testing.assert_close(
            real_target.sort().values,
            shuffled_target.sort().values,
        )


def test_condition_summary_reports_per_scale_and_sequence_nll() -> None:
    targets = [torch.tensor([[0, 1]]), torch.tensor([[0, 1, 2, 3]])]
    logits = [
        torch.zeros(1, 2, 2),
        torch.zeros(1, 4, 4),
    ]
    metrics = empty_condition_metrics(num_scales=2)
    update_condition_metrics(metrics, logits, targets)

    summary = summarize_condition(
        metrics,
        scale_lengths=[2, 4],
        codebook_sizes=[2, 4],
        target_length=8,
    )

    assert summary["scales"]["2"]["accuracy"] == 0.5
    assert summary["scales"]["2"]["nll_nats_per_code"] == pytest.approx(
        math.log(2)
    )
    assert summary["scales"]["4"]["accuracy"] == 0.25
    assert summary["scales"]["4"]["nll_nats_per_code"] == pytest.approx(
        math.log(4)
    )
    assert summary["bits_per_target_base"] == pytest.approx(1.25)


def test_oracle_rollout_replaces_one_scale_and_uses_it_downstream() -> None:
    class StubTokenizer:
        scale_lengths = [1, 2, 4]

    class StubModel:
        def __init__(self) -> None:
            self.completed_scales: list[list[torch.Tensor]] = []

        def predict_scale(
            self,
            prefix_by_scale: list[torch.Tensor],
            completed_scales: list[torch.Tensor],
            *,
            prefix_token_ids: torch.Tensor | None = None,
        ) -> torch.Tensor:
            del prefix_by_scale, prefix_token_ids
            self.completed_scales.append(
                [indices.clone() for indices in completed_scales]
            )
            scale_length = StubTokenizer.scale_lengths[len(completed_scales)]
            logits = torch.zeros(1, scale_length, 4)
            logits[..., 1] = 1.0
            return logits

    model = StubModel()
    true_indices = [
        torch.tensor([[0]]),
        torch.tensor([[2, 2]]),
        torch.tensor([[3, 3, 3, 3]]),
    ]

    rollout = rollout_with_oracle_scale(
        model,
        StubTokenizer(),
        prefix_by_scale=[torch.zeros(1, 4, 2)],
        prefix_token_ids=torch.zeros(1, 8, dtype=torch.long),
        true_indices_by_scale=true_indices,
        oracle_scale_index=1,
    )

    torch.testing.assert_close(rollout[0], torch.tensor([[1]]))
    torch.testing.assert_close(rollout[1], true_indices[1])
    torch.testing.assert_close(model.completed_scales[2][1], true_indices[1])
    torch.testing.assert_close(rollout[2], torch.tensor([[1, 1, 1, 1]]))
