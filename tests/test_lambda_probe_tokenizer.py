import torch
import torch.nn as nn

from scripts.evaluation.lambda_probe_tokenizer import TokenizerWindowEncoder


class _Quantizer(nn.Module):
    def forward(self, latent: torch.Tensor) -> tuple:
        return latent + 1, None, None, None


class _Tokenizer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.quantizer = _Quantizer()

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        return input_ids.float().unsqueeze(-1)


def test_tokenizer_encoder_returns_pre_quant_and_pre_decode_means() -> None:
    encoder = TokenizerWindowEncoder(_Tokenizer())

    embeddings = encoder(torch.tensor([[0, 1, 2, 3]]))

    torch.testing.assert_close(embeddings, torch.tensor([[1.5, 2.5]]))
