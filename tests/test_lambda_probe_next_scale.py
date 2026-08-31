from scripts.evaluation.lambda_probe_next_scale import segment_window_starts


def test_segment_windows_advance_by_one_target_block() -> None:
    assert segment_window_starts(2_000, 256, 128) == [
        *range(0, 1_665, 128),
        1_744,
    ]


def test_segment_windows_do_not_duplicate_an_aligned_final_window() -> None:
    assert segment_window_starts(512, 256, 128) == [0, 128, 256]
