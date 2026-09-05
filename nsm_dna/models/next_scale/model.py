from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from .quantization import QuantizerOutput
from .tokenizer import MultiscaleTokenizer
from .transformer import NextScaleTransformer


@dataclass(frozen=True)
class NSMDNAOutput:
    """Outputs from one joint tokenizer and next-scale forward pass."""

    reconstruction_logits: Float[Tensor, "batch target_length vocab_size"]
    partial_reconstruction_logits: (
        Float[Tensor, "batch target_length vocab_size"] | None
    )
    autoregressive_reconstruction_logits: (
        Float[Tensor, "batch target_length vocab_size"] | None
    )
    cumulative_reconstruction_logits_by_scale: (
        list[Float[Tensor, "batch target_length vocab_size"]] | None
    )
    next_scale_logits_by_scale: list[Float[Tensor, "batch scale_length codebook_size"]]
    quantizer: QuantizerOutput


@dataclass(frozen=True)
class NSMDNAGeneration:
    """A hard predicted hierarchy and its decoded nucleotide distribution."""

    nucleotide_logits: Float[Tensor, "batch target_length vocab_size"]
    indices_by_scale: list[Int[Tensor, "batch scale_length"]]


class NSMDNA(nn.Module):
    """Joint multiscale tokenizer and next-scale DNA model."""

    def __init__(
        self,
        tokenizer: MultiscaleTokenizer,
        transformer: NextScaleTransformer,
    ) -> None:
        super().__init__()

        if len(set(tokenizer.codebook_sizes)) != 1:
            raise ValueError("NSM-DNA requires equal codebook sizes across scales.")
        if transformer.scale_lengths != tokenizer.scale_lengths:
            raise ValueError("Tokenizer and transformer scale lengths must match.")
        if transformer.codebook_size != tokenizer.codebook_sizes[0]:
            raise ValueError("Tokenizer and transformer codebook sizes must match.")

        self.tokenizer = tokenizer
        self.transformer = transformer

    @classmethod
    def from_config(cls, config: DictConfig) -> "NSMDNA":
        """Construct the complete model from one end-to-end configuration."""
        tokenizer_config = config.model.tokenizer
        scale_lengths = list(tokenizer_config.scale_lengths)
        tokenizer = MultiscaleTokenizer(
            vocab_size=tokenizer_config.vocab_size,
            context_length=tokenizer_config.context_length,
            embed_dim=tokenizer_config.embed_dim,
            num_heads=tokenizer_config.num_heads,
            scale_lengths=scale_lengths,
            codebook_sizes=[tokenizer_config.codebook_size] * len(scale_lengths),
            encoder_num_layers=getattr(tokenizer_config, "encoder_num_layers", 0),
            encoder_dropout=tokenizer_config.encoder_dropout,
            decoder_dropout=tokenizer_config.decoder_dropout,
            bias=tokenizer_config.bias,
            rope_base=getattr(tokenizer_config, "rope_base", 10000.0),
            pre_quant_num_groups=tokenizer_config.pre_quant_num_groups,
            commitment_cost=tokenizer_config.commitment_cost,
            decay=tokenizer_config.decay,
            eps=tokenizer_config.eps,
            temperature=getattr(tokenizer_config, "temperature", 1.0),
            refinement_ratio=tokenizer_config.refinement_ratio,
            refinement_kernel_size=tokenizer_config.refinement_kernel_size,
        )

        transformer_config = config.model.transformer
        transformer = NextScaleTransformer(
            input_dim=tokenizer.embed_dim,
            model_dim=transformer_config.model_dim,
            scale_lengths=tokenizer.scale_lengths,
            codebook_size=tokenizer.codebook_sizes[0],
            num_layers=transformer_config.num_layers,
            num_heads=transformer_config.num_heads,
            dropout=transformer_config.dropout,
            bias=transformer_config.bias,
            use_qk_norm=transformer_config.use_qk_norm,
            rope_base=transformer_config.rope_base,
            head_num_blocks=transformer_config.head_num_blocks,
            head_hidden_multiplier=transformer_config.head_hidden_multiplier,
            input_refinement_kernel_size=(
                transformer_config.input_refinement_kernel_size
            ),
            max_prefix_length=(config.data.sequence_length - tokenizer.context_length),
        )
        return cls(tokenizer, transformer)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        device: torch.device,
        *,
        frozen: bool = False,
    ) -> tuple["NSMDNA", int]:
        """Restore the complete tokenizer-transformer model and training step."""
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        config = OmegaConf.create(checkpoint["config"])
        model = cls.from_config(config)
        model.load_state_dict(checkpoint["model"])
        model = model.to(device)

        if frozen:
            model.eval()
            model.requires_grad_(False)

        return model, int(checkpoint["step"])

    def forward(
        self,
        sequence_ids: Int[Tensor, "batch sequence_length"],
        corruption_probability: float = 0.0,
        *,
        return_partial_reconstruction: bool = False,
        return_cumulative_reconstructions: bool = False,
        return_autoregressive_reconstruction: bool = False,
    ) -> NSMDNAOutput:
        prefix_latent, target_latent = self.tokenizer.encode_pair(sequence_ids)
        quantizer_output = self.tokenizer.quantize(
            target_latent,
            include_partial_reconstruction=return_partial_reconstruction,
            corruption_probability=corruption_probability,
        )

        scale_inputs = quantizer_output.next_scale_inputs
        next_scale_logits = self.transformer(
            scale_inputs,
            prefix=prefix_latent,
        )
        next_scale_logits_by_scale = list(
            torch.split(
                next_scale_logits,
                self.tokenizer.scale_lengths[1:],
                dim=1,
            )
        )
        reconstruction_logits = self.tokenizer.decode_latent(
            quantizer_output.final_latent
        )
        partial_reconstruction_logits = None
        if return_partial_reconstruction:
            partial_latent = quantizer_output.partial_latent
            if partial_latent is None:
                raise RuntimeError("Quantizer did not return a partial latent.")
            partial_reconstruction_logits = self.tokenizer.decode_latent(partial_latent)
        autoregressive_reconstruction_logits = None
        if return_autoregressive_reconstruction:
            predicted_latent = (
                self.tokenizer.quantizer.prediction_logits_to_final_latent(
                    next_scale_logits_by_scale,
                    initial_latent=quantizer_output.cumulative_latents[0],
                )
            )
            autoregressive_reconstruction_logits = self.tokenizer.decode_latent(
                predicted_latent
            )
        cumulative_reconstruction_logits_by_scale = None
        if return_cumulative_reconstructions:
            cumulative_reconstruction_logits_by_scale = [
                self.tokenizer.decode_latent(cumulative_latent)
                for cumulative_latent in quantizer_output.cumulative_latents[:-1]
            ]
            cumulative_reconstruction_logits_by_scale.append(reconstruction_logits)

        return NSMDNAOutput(
            reconstruction_logits=reconstruction_logits,
            partial_reconstruction_logits=partial_reconstruction_logits,
            autoregressive_reconstruction_logits=(autoregressive_reconstruction_logits),
            cumulative_reconstruction_logits_by_scale=(
                cumulative_reconstruction_logits_by_scale
            ),
            next_scale_logits_by_scale=next_scale_logits_by_scale,
            quantizer=quantizer_output,
        )

    @torch.no_grad()
    def generate(
        self,
        prefix_ids: Int[Tensor, "batch prefix_length"],
        first_scale_indices: Int[Tensor, "batch first_scale_length"],
    ) -> NSMDNAGeneration:
        """Greedily predict scales 4 onward from a supplied first-scale code."""
        if self.training:
            raise RuntimeError("Call model.eval() before generating sequences.")
        if prefix_ids.shape[1] > self.transformer.max_prefix_length:
            raise ValueError(
                f"Prefix length {prefix_ids.shape[1]} exceeds the configured maximum "
                f"of {self.transformer.max_prefix_length}."
            )

        prefix_latent = self.tokenizer.encode(prefix_ids)
        batch_size = prefix_ids.shape[0]
        if first_scale_indices.shape != (
            batch_size,
            self.tokenizer.scale_lengths[0],
        ):
            raise ValueError(
                "First-scale indices must have shape "
                f"({batch_size}, {self.tokenizer.scale_lengths[0]})."
            )
        scale_inputs = [
            prefix_latent.new_zeros(batch_size, scale_length, self.tokenizer.embed_dim)
            for scale_length in self.tokenizer.scale_lengths[1:]
        ]
        predicted_indices_by_scale = [first_scale_indices]
        scale_inputs[0] = self.tokenizer.quantizer.indices_to_next_scale_input(
            predicted_indices_by_scale
        )

        for scale_index in range(1, len(self.tokenizer.scale_lengths)):
            logits = self.transformer(scale_inputs, prefix=prefix_latent)
            scale_logits = torch.split(
                logits,
                self.tokenizer.scale_lengths[1:],
                dim=1,
            )[scale_index - 1]
            predicted_indices_by_scale.append(scale_logits.argmax(dim=-1))
            if scale_index < len(self.tokenizer.scale_lengths) - 1:
                scale_inputs[scale_index] = (
                    self.tokenizer.quantizer.indices_to_next_scale_input(
                        predicted_indices_by_scale
                    )
                )

        final_latent = self.tokenizer.quantizer.indices_to_cumulative_latents(
            predicted_indices_by_scale
        )[-1]
        nucleotide_logits = self.tokenizer.decode_latent(final_latent)
        return NSMDNAGeneration(
            nucleotide_logits=nucleotide_logits,
            indices_by_scale=predicted_indices_by_scale,
        )
