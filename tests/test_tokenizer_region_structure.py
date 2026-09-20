import numpy as np

from scripts.evaluation.tokenizer_region_structure import (
    code_profiles,
    longest_run,
    score_children,
)


def test_region_profiles_measure_local_code_preferences() -> None:
    real = np.array([[[0], [0], [0], [0]], [[1], [1], [1], [1]]])
    mixed = np.array([[[0], [1], [0], [1]], [[1], [0], [1], [0]]])

    _, profiles, real_summary = code_profiles(real, codebook_size=2)
    _, _, mixed_summary = code_profiles(mixed, codebook_size=2)

    assert real_summary["median_top_eighth_coverage"] == 1.0
    assert mixed_summary["median_top_eighth_coverage"] == 0.5
    assert profiles[0]["top_codes"] == [[0, 4]]
    assert longest_run(real[0].ravel()) == 4


def test_heldout_child_prediction_uses_parent_code() -> None:
    parents = np.array(
        [
            [[0], [1]],
            [[1], [0]],
            [[0], [1]],
            [[1], [0]],
            [[0], [1]],
            [[1], [0]],
        ]
    )
    children = np.repeat(parents, 2, axis=2)

    result, counts = score_children(
        parents,
        children,
        parent_size=2,
        child_size=2,
        train_regions=np.arange(4),
        test_regions=np.arange(4, 6),
    )

    assert counts.shape == (2, 2, 2)
    assert result["left"]["parent_top1_accuracy"] == 1.0
    assert result["right"]["parent_top1_accuracy"] == 1.0
    assert result["mean_bits_gained_from_parent"] > 0
