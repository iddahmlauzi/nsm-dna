import torch
from scripts.evaluation.lambda_probe_tokenizer import TokenizerWindowEncoder


class _Tokenizer(torch.nn.Module):
    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        return input_ids.float().unsqueeze(-1)


def test_tokenizer_encoder_returns_mean_dinucleotide_vector() -> None:
    encoder = TokenizerWindowEncoder(_Tokenizer())

    embeddings = encoder(torch.tensor([[0, 1, 2, 3]]))

    torch.testing.assert_close(embeddings, torch.tensor([[1.5]]))
