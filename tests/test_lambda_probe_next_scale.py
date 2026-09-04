import torch
import torch.nn as nn

from scripts.evaluation.lambda_probe_next_scale import (
    NSMWindowEncoder,
    segment_window_starts,
)


def test_window_encoder_pools_positions_then_scales_equally() -> None:
    class StubTokenizer(nn.Module):
        context_length = 2
        scale_lengths = [1, 2]

        def encode(self, token_ids: torch.Tensor) -> torch.Tensor:
            return token_ids.unsqueeze(-1).float()

        def encode_indices(self, token_ids: torch.Tensor) -> list[torch.Tensor]:
            return [token_ids[:, :1], token_ids]

        def indices_to_next_scale_inputs(
            self,
            indices_by_scale: list[torch.Tensor],
        ) -> list[torch.Tensor]:
            return [indices_by_scale[0].expand(-1, 2).unsqueeze(-1).float()]

    class StubModel(nn.Module):
        def encode(
            self,
            scale_inputs: list[torch.Tensor],
            *,
            prefix: torch.Tensor,
        ) -> torch.Tensor:
            return torch.tensor(
                [
                    [
                        [1.0, 2.0],
                        [3.0, 4.0],
                        [5.0, 6.0],
                        [7.0, 8.0],
                        [9.0, 10.0],
                    ]
                ]
            )

    encoder = NSMWindowEncoder(StubModel(), StubTokenizer())

    embedding = encoder(torch.tensor([[0, 1, 2, 3]]))

    # Prefix states are excluded. The length-1 scale pools to [5, 6], the
    # length-2 scale pools to [8, 9], and the two scales are weighted equally.
    torch.testing.assert_close(embedding, torch.tensor([[6.5, 7.5]]))


def test_segment_windows_advance_by_one_target_block() -> None:
    assert segment_window_starts(2_000, 256, 128) == [
        *range(0, 1_665, 128),
        1_744,
    ]


def test_segment_windows_do_not_duplicate_an_aligned_final_window() -> None:
    assert segment_window_starts(512, 256, 128) == [0, 128, 256]
