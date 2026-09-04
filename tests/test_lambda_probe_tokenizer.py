import torch
import torch.nn as nn

from nsm_dna.models.next_scale.quantization import QuantizerOutput
from scripts.evaluation.lambda_probe_tokenizer import TokenizerWindowEncoder


class _Quantizer(nn.Module):
    def forward(self, latent: torch.Tensor) -> QuantizerOutput:
        zero = latent.new_zeros(())
        return QuantizerOutput(
            final_latent=latent + 1,
            cumulative_latents=[latent + 1],
            next_scale_inputs=[],
            indices_by_scale=[torch.zeros_like(latent[..., 0], dtype=torch.long)],
            assignment_probabilities_by_scale=[latent.new_ones(*latent.shape[:2], 1)],
            assignment_logits_by_scale=[latent.new_zeros(*latent.shape[:2], 1)],
            commitment_losses_by_scale=[zero],
            quantization_losses_by_scale=[zero],
        )


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
