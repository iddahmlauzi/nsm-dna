import torch

from scripts.evaluation.analyze_rollout_geometry import replace_generated_scale


def test_oracle_replaces_complete_generated_scale() -> None:
    generated = torch.tensor([[0, 1, 2, 3]])
    true = torch.tensor([[0, 2, 1, 3]])
    neighbors = torch.tensor(
        [[1, 2], [0, 2], [1, 3], [2, 1]]
    )

    replaced = replace_generated_scale(
        generated,
        true,
        "oracle_true",
        neighbors,
        torch.Generator().manual_seed(0),
    )

    torch.testing.assert_close(replaced, true)


def test_nearest_replacement_preserves_the_generated_correctness_mask() -> None:
    generated = torch.tensor([[0, 1, 2, 3]])
    true = torch.tensor([[0, 2, 1, 3]])
    neighbors = torch.tensor(
        [[1, 2], [0, 2], [3, 0], [2, 1]]
    )

    replaced = replace_generated_scale(
        generated,
        true,
        "same_errors_nearest_1",
        neighbors,
        torch.Generator().manual_seed(0),
    )

    torch.testing.assert_close(replaced, torch.tensor([[0, 3, 0, 3]]))
    torch.testing.assert_close(replaced == true, generated == true)


def test_random_replacement_never_accidentally_uses_true_code() -> None:
    generated = torch.tensor([[0, 1, 2, 3]]).repeat(32, 1)
    true = torch.tensor([[0, 2, 1, 3]]).repeat(32, 1)
    neighbors = torch.tensor(
        [[1, 2], [0, 2], [3, 0], [2, 1]]
    )

    replaced = replace_generated_scale(
        generated,
        true,
        "same_errors_random",
        neighbors,
        torch.Generator().manual_seed(0),
    )

    torch.testing.assert_close(replaced == true, generated == true)
