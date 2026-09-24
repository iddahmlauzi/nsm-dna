import torch
import torch.nn as nn

from scripts.evaluation.lambda_probe_tokenizer import (
    TokenizerWindowEncoder,
    frame_window_starts,
)


class _Quantizer(nn.Module):
    def forward(
        self,
        fine_latent: torch.Tensor,
        hierarchy_latent: torch.Tensor,
    ) -> tuple:
        return fine_latent + 1, None, None


class _Decoder(nn.Module):
    def encode(self, latent: torch.Tensor) -> torch.Tensor:
        return 2 * latent


class _Tokenizer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.quantizer = _Quantizer()
        self.decoder = _Decoder()

    def encode_latents(
        self,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent = input_ids.float().unsqueeze(-1)
        return latent, latent


def test_tokenizer_encoder_returns_mean_decoder_hidden_state() -> None:
    encoder = TokenizerWindowEncoder(_Tokenizer())

    embeddings = encoder(torch.tensor([[0, 1, 2, 3]]))

    torch.testing.assert_close(embeddings, torch.tensor([[5.0]]))


def test_frame_windows_preserve_triplet_alignment() -> None:
    assert frame_window_starts(2_000, 384, 0) == [0, 384, 768, 1152, 1536]
    assert frame_window_starts(2_000, 384, 1) == [1, 385, 769, 1153, 1537]
    assert frame_window_starts(2_000, 384, 2) == [2, 386, 770, 1154, 1538]
