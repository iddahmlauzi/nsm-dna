import torch
import torch.nn as nn

from scripts.evaluation.lambda_probe_next_scale import (
    NSMWindowEncoder,
    segment_window_starts,
)


def test_window_encoder_pools_prefix_and_finest_scale_states() -> None:
    class StubQuantizer:
        def indices_to_vectors(
            self,
            scale_indices: torch.Tensor,
            scale_index: int,
        ) -> torch.Tensor:
            del scale_index
            return scale_indices.unsqueeze(-1).float()

    class StubTokenizer(nn.Module):
        context_length = 2

        def __init__(self) -> None:
            super().__init__()
            self.quantizer = StubQuantizer()

        def encode(self, token_ids: torch.Tensor) -> torch.Tensor:
            return token_ids.unsqueeze(-1).float()

        def encode_indices(self, token_ids: torch.Tensor) -> list[torch.Tensor]:
            return [token_ids]

    class StubModel(nn.Module):
        scale_lengths = [1, 2]

        def encode(
            self,
            indices_by_scale: list[torch.Tensor],
            *,
            prefix: torch.Tensor,
        ) -> torch.Tensor:
            del indices_by_scale, prefix
            return torch.tensor(
                [
                    [
                        [1.0, 2.0],
                        [3.0, 4.0],
                        [5.0, 6.0],
                        [7.0, 8.0],
                    ]
                ]
            )

    encoder = NSMWindowEncoder(StubModel(), StubTokenizer())

    embedding = encoder(torch.tensor([[0, 1, 2, 3]]))

    torch.testing.assert_close(embedding, torch.tensor([[4.0, 5.0]]))


def test_segment_windows_advance_by_one_target_block() -> None:
    assert segment_window_starts(2_000, 256, 128) == [
        *range(0, 1_665, 128),
        1_744,
    ]


def test_segment_windows_do_not_duplicate_an_aligned_final_window() -> None:
    assert segment_window_starts(512, 256, 128) == [0, 128, 256]
