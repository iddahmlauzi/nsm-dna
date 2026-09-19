import math
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from .common import (
    RMSNorm,
    RotaryEmbeddings,
    TransformerBlock,
    precompute_rope_cosine_and_sine,
)

if TYPE_CHECKING:
    from .vqvae import VQVAE


class _ResidualOutputHeadBlock(nn.Module):
    """Learn a widened squared-ReLU correction to model-space hidden states."""

    def __init__(
        self,
        model_dim: int,
        hidden_dim: int,
        dropout: float,
        bias: bool,
    ) -> None:
        super().__init__()

        self.up_projection = nn.Linear(model_dim, hidden_dim, bias=bias)
        self.down_projection = nn.Linear(hidden_dim, model_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

        # Begin with a zero correction so the shared head is initially the
        # plain linear codebook classifier. The FFN learns a residual correction.
        nn.init.zeros_(self.down_projection.weight)
        if self.down_projection.bias is not None:
            nn.init.zeros_(self.down_projection.bias)

    def forward(
        self,
        x: Float[Tensor, "batch length model_dim"],
    ) -> Float[Tensor, "batch length model_dim"]:
        residual = x
        x = torch.relu(self.up_projection(x)).square()
        x = self.dropout(self.down_projection(x))
        return residual + x


class MultiscaleOutputHead(nn.Module):
    """Apply shared residual corrections before scale-specific classifiers.

    Each block expands to a wider hidden dimension and projects back to
    model_dim. Each scale then projects into its own codebook because codebook
    sizes and code identities are specific to that scale.
    """

    def __init__(
        self,
        model_dim: int,
        scale_lengths: list[int],
        codebook_sizes: list[int],
        num_blocks: int = 2,
        hidden_multiplier: float = 2.0,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()

        self.scale_lengths = list(scale_lengths)
        hidden_dim = int(round(hidden_multiplier * model_dim))
        self.blocks = nn.ModuleList(
            [
                _ResidualOutputHeadBlock(
                    model_dim,
                    hidden_dim,
                    dropout,
                    bias,
                )
                for _ in range(num_blocks)
            ]
        )
        self.codebook_projections = nn.ModuleList(
            nn.Linear(model_dim, codebook_size, bias=False)
            for codebook_size in codebook_sizes
        )

    def forward(
        self,
        x: Float[Tensor, "batch length model_dim"],
    ) -> list[Float[Tensor, "batch scale_length codebook_size"]]:
        for block in self.blocks:
            x = block(x)

        hidden_states_by_scale = torch.split(x, self.scale_lengths, dim=1)
        return [
            projection(hidden_states)
            for projection, hidden_states in zip(
                self.codebook_projections,
                hidden_states_by_scale,
                strict=True,
            )
        ]


class NSM(nn.Module):
    """Predict each scale from preceding scales and earlier codes at that scale."""

    def __init__(
        self,
        prefix_dim: int,
        model_dim: int,
        scale_lengths: list[int],
        codebook_sizes: list[int],
        codebook_vectors: list[Tensor],
        num_layers: int,
        num_heads: int,
        *,
        dropout: float = 0.1,
        bias: bool = False,
        use_qk_norm: bool = True,
        rope_base: float = 10000.0,
        head_num_blocks: int = 2,
        head_hidden_multiplier: float = 2.0,
        max_prefix_length: int = 0,
    ) -> None:
        super().__init__()

        self.prefix_dim = prefix_dim
        self.model_dim = model_dim
        self.scale_lengths = list(scale_lengths)
        self.codebook_sizes = list(codebook_sizes)
        self.latent_length = self.scale_lengths[-1]
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.use_qk_norm = use_qk_norm
        self.rope_base = rope_base
        self.max_prefix_length = max_prefix_length

        self._codebook_buffer_names: list[str] = []
        for scale_index, vectors in enumerate(codebook_vectors):
            buffer_name = f"codebook_vectors_{scale_index}"
            self.register_buffer(
                buffer_name,
                vectors.detach().float().clone(),
                persistent=False,
            )
            self._codebook_buffer_names.append(buffer_name)

        self.input_projection = nn.Linear(
            self.prefix_dim,
            self.model_dim,
            bias=bias,
        )
        self.scale_input_convs = nn.ModuleList(
            nn.Conv1d(
                self.prefix_dim,
                self.prefix_dim,
                kernel_size=3,
                padding=1,
                bias=bias,
            )
            for _ in self.scale_lengths[1:]
        )
        for conv in self.scale_input_convs:
            nn.init.zeros_(conv.weight)
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)

        # A learned scale vector marks the codebook predicted at each position.
        self.scale_embedding = nn.Embedding(len(self.scale_lengths), self.model_dim)
        nn.init.normal_(self.scale_embedding.weight, mean=0.0, std=0.02)

        scale_ids = torch.cat(
            [
                torch.full((scale_length,), scale_index)
                for scale_index, scale_length in enumerate(self.scale_lengths)
            ]
        )
        self.register_buffer("scale_ids", scale_ids, persistent=False)

        hierarchy_attention_mask = (
            (scale_ids[:, None] == scale_ids[None, :])
            & torch.ones(len(scale_ids), len(scale_ids), dtype=torch.bool).tril()
        )
        self.register_buffer(
            "hierarchy_attention_mask",
            hierarchy_attention_mask.reshape(1, 1, len(scale_ids), len(scale_ids)),
            persistent=False,
        )

        # Each scale is its own causal section; cumulative inputs carry earlier scales.
        rope_positions = torch.cat(
            [torch.arange(scale_length) for scale_length in self.scale_lengths]
        )
        head_dim = self.model_dim // self.num_heads
        rope_cosine, rope_sine = precompute_rope_cosine_and_sine(
            rope_positions,
            head_dim,
            self.rope_base,
        )
        self.register_buffer("rope_cosine", rope_cosine, persistent=False)
        self.register_buffer("rope_sine", rope_sine, persistent=False)

        # Prefix positions run sequentially across all preceding DNA blocks.
        prefix_positions = torch.arange(self.max_prefix_length)
        prefix_rope_cosine, prefix_rope_sine = precompute_rope_cosine_and_sine(
            prefix_positions,
            head_dim,
            self.rope_base,
        )
        self.register_buffer(
            "prefix_rope_cosine",
            prefix_rope_cosine,
            persistent=False,
        )
        self.register_buffer(
            "prefix_rope_sine",
            prefix_rope_sine,
            persistent=False,
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=self.model_dim,
                    num_heads=self.num_heads,
                    dropout=dropout,
                    bias=bias,
                    use_qk_norm=self.use_qk_norm,
                    use_rms_norm=True,
                    rms_norm_eps=1e-5,
                )
                for _ in range(self.num_layers)
            ]
        )
        self._initialize_residual_projections()
        self.final_norm = RMSNorm(self.model_dim, eps=1e-5)
        self.output_head = MultiscaleOutputHead(
            self.model_dim,
            self.scale_lengths,
            self.codebook_sizes,
            num_blocks=head_num_blocks,
            hidden_multiplier=head_hidden_multiplier,
            dropout=dropout,
            bias=bias,
        )

    def _initialize_residual_projections(self) -> None:
        """Scale residual branches so their variance does not grow with depth."""
        if not self.blocks:
            return

        residual_standard_deviation = 0.02 / math.sqrt(2 * self.num_layers)
        for block in self.blocks:
            nn.init.normal_(
                block.attn.out_proj.weight,
                mean=0.0,
                std=residual_standard_deviation,
            )
            nn.init.normal_(
                block.mlp.down_proj.weight,
                mean=0.0,
                std=residual_standard_deviation,
            )

    @classmethod
    def from_config(cls, config: DictConfig, tokenizer: "VQVAE") -> "NSM":
        """Build NSM-DNA from an experiment configuration and its tokenizer."""
        return cls(
            prefix_dim=tokenizer.quantization_dim,
            model_dim=config.model.model_dim,
            scale_lengths=tokenizer.scale_lengths,
            codebook_sizes=tokenizer.codebook_sizes,
            codebook_vectors=[
                codebook.codebook for codebook in tokenizer.quantizer.codebooks
            ],
            num_layers=config.model.num_layers,
            num_heads=config.model.num_heads,
            dropout=config.model.dropout,
            bias=config.model.bias,
            use_qk_norm=config.model.use_qk_norm,
            rope_base=config.model.rope_base,
            head_num_blocks=config.model.head_num_blocks,
            head_hidden_multiplier=config.model.head_hidden_multiplier,
            max_prefix_length=(config.data.sequence_length - tokenizer.context_length),
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        tokenizer: "VQVAE",
        device: torch.device,
        *,
        frozen: bool = False,
    ) -> tuple["NSM", int]:
        """Rebuild NSM-DNA and return it with its saved training step."""
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        config = OmegaConf.create(checkpoint["config"])

        model = cls.from_config(config, tokenizer)
        model.load_state_dict(checkpoint["model"])
        model = model.to(device)

        if frozen:
            model.eval()
            model.requires_grad_(False)

        return model, int(checkpoint["step"])

    def _embed_hierarchy_inputs(
        self,
        indices_by_scale: list[Int[Tensor, "batch scale_length"]],
    ) -> Float[Tensor, "batch hierarchy_length model_dim"]:
        """Combine cumulative prior-scale inputs with shifted same-scale codes."""
        codebook_vectors = [
            getattr(self, buffer_name) for buffer_name in self._codebook_buffer_names
        ]
        batch_size = indices_by_scale[0].shape[0]
        cumulative_latent = codebook_vectors[0].new_zeros(
            batch_size, self.latent_length, self.prefix_dim
        )
        scale_inputs = []

        for scale_index, (indices, vectors) in enumerate(
            zip(indices_by_scale, codebook_vectors, strict=True)
        ):
            scale_length = self.scale_lengths[scale_index]
            if scale_index == 0:
                prior_scale_input = cumulative_latent[:, :scale_length]
            else:
                prior_scale_input = F.interpolate(
                    cumulative_latent.transpose(1, 2),
                    size=scale_length,
                    mode="area",
                ).transpose(1, 2)
                correction = self.scale_input_convs[scale_index - 1](
                    prior_scale_input.transpose(1, 2)
                ).transpose(1, 2)
                prior_scale_input = prior_scale_input + correction

            current_codes = vectors[indices]
            shifted_codes = torch.cat(
                [torch.zeros_like(current_codes[:, :1]), current_codes[:, :-1]],
                dim=1,
            )
            scale_inputs.append(prior_scale_input + shifted_codes)

            # Only completed earlier scales contribute to the next scale's input.
            cumulative_latent = cumulative_latent + current_codes.repeat_interleave(
                self.latent_length // scale_length,
                dim=1,
            )

        return self.input_projection(torch.cat(scale_inputs, dim=1))

    def _build_attention_mask(
        self,
        prefix_length: int,
    ) -> Bool[Tensor, "1 1 length length"]:
        """Let each scale read the prefix and its own earlier positions."""
        if prefix_length == 0:
            return self.hierarchy_attention_mask

        hierarchy_length = self.hierarchy_attention_mask.shape[-1]
        attention_mask = torch.zeros(
            prefix_length + hierarchy_length,
            prefix_length + hierarchy_length,
            dtype=torch.bool,
            device=self.hierarchy_attention_mask.device,
        )
        attention_mask[:prefix_length, :prefix_length] = True
        attention_mask[prefix_length:, :prefix_length] = True
        attention_mask[prefix_length:, prefix_length:] = self.hierarchy_attention_mask[
            0, 0
        ]
        return attention_mask.reshape(
            1,
            1,
            prefix_length + hierarchy_length,
            prefix_length + hierarchy_length,
        )

    def _get_rotary_embeddings(self, prefix_length: int) -> RotaryEmbeddings:
        """Combine prefix positions with the scale-local hierarchy positions."""
        cosine = torch.cat(
            [self.prefix_rope_cosine[:prefix_length], self.rope_cosine],
            dim=0,
        )
        sine = torch.cat(
            [self.prefix_rope_sine[:prefix_length], self.rope_sine],
            dim=0,
        )
        return cosine, sine

    def encode(
        self,
        indices_by_scale: list[Int[Tensor, "batch scale_length"]],
        *,
        prefix: Float[Tensor, "batch prefix_length prefix_dim"] | None = None,
    ) -> Float[Tensor, "batch length model_dim"]:
        """Return causal hierarchy states after an optional encoded DNA prefix."""
        hierarchy_hidden_states = self._embed_hierarchy_inputs(indices_by_scale)
        hierarchy_hidden_states = hierarchy_hidden_states + self.scale_embedding(
            self.scale_ids
        )

        prefix_length = 0 if prefix is None else prefix.shape[1]
        if prefix_length > self.max_prefix_length:
            raise ValueError(
                f"Prefix length {prefix_length} exceeds the configured maximum "
                f"of {self.max_prefix_length}."
            )

        if prefix is None:
            hidden_states = hierarchy_hidden_states
        else:
            prefix_hidden_states = self.input_projection(prefix)
            hidden_states = torch.cat(
                [prefix_hidden_states, hierarchy_hidden_states],
                dim=1,
            )

        attention_mask = self._build_attention_mask(prefix_length)
        rotary_embeddings = self._get_rotary_embeddings(prefix_length)
        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                attention_mask=attention_mask,
                rotary_embeddings=rotary_embeddings,
            )

        return self.final_norm(hidden_states)

    def forward(
        self,
        indices_by_scale: list[Int[Tensor, "batch scale_length"]],
        *,
        prefix: Float[Tensor, "batch prefix_length prefix_dim"] | None = None,
    ) -> list[Float[Tensor, "batch scale_length codebook_size"]]:
        hidden_states = self.encode(indices_by_scale, prefix=prefix)
        prefix_length = 0 if prefix is None else prefix.shape[1]
        hierarchy_hidden_states = hidden_states[:, prefix_length:]
        return self.output_head(hierarchy_hidden_states)
