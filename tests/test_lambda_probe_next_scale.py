import torch
import torch.nn as nn

from scripts.evaluation.lambda_probe_next_scale import (
    NSMWindowEncoder,
    segment_window_starts,
)


def test_window_encoder_pools_prefix_and_finest_scale_states() -> None:
    class StubTokenizer(nn.Module):
        context_length = 2
        scale_lengths = [1]

        def encode_scales(self, token_ids: torch.Tensor) -> list[torch.Tensor]:
            return [token_ids[:, :1].unsqueeze(-1).float()]

        def encode_indices(self, token_ids: torch.Tensor) -> list[torch.Tensor]:
            return [token_ids]

    class StubModel(nn.Module):
        scale_lengths = [1]
        code_lengths = [1]
        prefix_length = 1

        def encode(
            self,
            indices_by_scale: list[torch.Tensor],
            *,
            prefix_by_scale: list[torch.Tensor],
        ) -> torch.Tensor:
            del indices_by_scale, prefix_by_scale
            return torch.tensor(
                [
                    [
                        [1.0, 2.0],
                        [7.0, 8.0],
                    ]
                ]
            )

    encoder = NSMWindowEncoder(StubModel(), StubTokenizer())

    embedding = encoder(torch.tensor([[0, 1, 2, 3]]))

    torch.testing.assert_close(
        embedding,
        torch.tensor([[(1.0 + 7.0) / 2, (2.0 + 8.0) / 2]]),
    )


def test_segment_windows_advance_by_one_target_block() -> None:
    assert segment_window_starts(2_000, 256, 128) == [
        *range(0, 1_665, 128),
        1_744,
    ]


def test_segment_windows_do_not_duplicate_an_aligned_final_window() -> None:
    assert segment_window_starts(512, 256, 128) == [0, 128, 256]
