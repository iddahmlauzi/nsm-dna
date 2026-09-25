import math
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
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

    def predict_scale(
        self,
        hidden_states: Float[Tensor, "batch length model_dim"],
        scale_index: int,
    ) -> Float[Tensor, "batch length codebook_size"]:
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return self.codebook_projections[scale_index](hidden_states)


class NSM(nn.Module):
    """Predict absolute-scale codes from a latent prefix and earlier target codes."""

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
        max_prefix_length: int = 64,
    ) -> None:
        super().__init__()

        self.prefix_dim = prefix_dim
        self.model_dim = model_dim
        self.scale_lengths = list(scale_lengths)
        self.codebook_sizes = list(codebook_sizes)
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.use_qk_norm = use_qk_norm
        self.rope_base = rope_base
        self.max_prefix_length = max_prefix_length

        # The model receives codebook indices as inputs, but needs the actual
        # codebook vectors to condition the next scale. Store the tokenizer's
        # codebooks directly on NSM for easy lookup, and register them as buffers
        # so they move with the model without becoming trainable parameters.
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
        self.bos = nn.Parameter(torch.empty(1, 1, self.model_dim))
        nn.init.normal_(self.bos, mean=0.0, std=0.02)

        # Each input block is labeled by the scale it is trying to predict.
        self.scale_embedding = nn.Embedding(len(self.scale_lengths), self.model_dim)
        nn.init.normal_(self.scale_embedding.weight, mean=0.0, std=0.02)

        head_dim = self.model_dim // self.num_heads
        # Place the prefix immediately before the target, then express every
        # target scale in the finest scale's coordinates. For a finest length
        # of 128, scale 2 uses [31.5, 95.5], the centers of 0–63 and 64–127.
        prefix_positions = torch.arange(
            -self.max_prefix_length,
            0,
            dtype=torch.float32,
        )
        hierarchy_positions = torch.cat(
            [
                (torch.arange(scale_length, dtype=torch.float32) + 0.5)
                * (self.scale_lengths[-1] / scale_length)
                - 0.5
                for scale_length in self.scale_lengths
            ]
        )
        prefix_rope_cosine, prefix_rope_sine = precompute_rope_cosine_and_sine(
            prefix_positions,
            head_dim,
            self.rope_base,
        )
        hierarchy_rope_cosine, hierarchy_rope_sine = (
            precompute_rope_cosine_and_sine(
                hierarchy_positions,
                head_dim,
                self.rope_base,
            )
        )
        self.register_buffer(
            "prefix_positions", prefix_positions, persistent=False
        )
        self.register_buffer(
            "hierarchy_positions", hierarchy_positions, persistent=False
        )
        self.register_buffer(
            "prefix_rope_cosine", prefix_rope_cosine, persistent=False
        )
        self.register_buffer(
            "prefix_rope_sine", prefix_rope_sine, persistent=False
        )
        self.register_buffer(
            "hierarchy_rope_cosine", hierarchy_rope_cosine, persistent=False
        )
        self.register_buffer(
            "hierarchy_rope_sine", hierarchy_rope_sine, persistent=False
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
            max_prefix_length=tokenizer.latent_length,
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

    def _build_scale_inputs(
        self,
        completed_scales: list[Int[Tensor, "batch scale_length"]],
        batch_size: int,
    ) -> Float[Tensor, "batch length model_dim"]:
        """Build BOS and repeated prior-scale blocks for parallel prediction."""
        scale_inputs = [
            self.bos.expand(batch_size, self.scale_lengths[0], -1)
            + self.scale_embedding.weight[0]
        ]

        for target_scale_index, source_indices in enumerate(
            completed_scales,
            start=1,
        ):
            source_scale_index = target_scale_index - 1
            vectors = getattr(
                self,
                self._codebook_buffer_names[source_scale_index],
            )
            source_inputs = self.input_projection(vectors[source_indices])
            target_scale_length = self.scale_lengths[target_scale_index]
            repeats_per_code = target_scale_length // source_indices.shape[1]
            scale_inputs.append(
                source_inputs.repeat_interleave(repeats_per_code, dim=1)
                + self.scale_embedding.weight[target_scale_index]
            )

        return torch.cat(scale_inputs, dim=1)

    def _build_attention_mask(
        self,
        prefix_length: int,
        num_scale_blocks: int,
    ) -> Bool[Tensor, "1 1 length length"]:
        """Allow full attention within each block and to every earlier block."""
        input_length = prefix_length + sum(
            self.scale_lengths[:num_scale_blocks]
        )
        attention_mask = torch.zeros(
            input_length,
            input_length,
            dtype=torch.bool,
            device=self.prefix_positions.device,
        )
        attention_mask[:prefix_length, :prefix_length] = True

        block_start = prefix_length
        for scale_length in self.scale_lengths[:num_scale_blocks]:
            block_end = block_start + scale_length
            attention_mask[block_start:block_end, :block_end] = True
            block_start = block_end

        return attention_mask.reshape(1, 1, input_length, input_length)

    def _get_rotary_embeddings(
        self,
        prefix_length: int,
        num_scale_blocks: int,
    ) -> RotaryEmbeddings:
        hierarchy_length = sum(self.scale_lengths[:num_scale_blocks])
        prefix_start = self.max_prefix_length - prefix_length
        cosine = torch.cat(
            [
                self.prefix_rope_cosine[prefix_start:],
                self.hierarchy_rope_cosine[:hierarchy_length],
            ],
            dim=0,
        )
        sine = torch.cat(
            [
                self.prefix_rope_sine[prefix_start:],
                self.hierarchy_rope_sine[:hierarchy_length],
            ],
            dim=0,
        )
        return cosine, sine

    def _encode_packed(
        self,
        prefix: Float[Tensor, "batch prefix_length prefix_dim"],
        completed_scales: list[Int[Tensor, "batch scale_length"]],
    ) -> Float[Tensor, "batch length model_dim"]:
        """Run the shared packed transformer path used by training and rollout.

        Training supplies one possibly corrupted context for every scale after
        scale 1, creating all prediction blocks in one pass. Rollout supplies only
        the scales generated so far, creating the next-scale block it currently
        needs.
        """
        prefix_length = prefix.shape[1]
        num_scale_blocks = len(completed_scales) + 1
        hidden_states = torch.cat(
            [
                self.input_projection(prefix),
                self._build_scale_inputs(
                    completed_scales,
                    batch_size=prefix.shape[0],
                ),
            ],
            dim=1,
        )
        attention_mask = self._build_attention_mask(
            prefix_length,
            num_scale_blocks,
        )
        rotary_embeddings = self._get_rotary_embeddings(
            prefix_length,
            num_scale_blocks,
        )
        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                attention_mask=attention_mask,
                rotary_embeddings=rotary_embeddings,
            )
        return self.final_norm(hidden_states)

    def encode(
        self,
        context_indices_by_scale: list[Int[Tensor, "batch scale_length"]],
        *,
        prefix: Float[Tensor, "batch prefix_length prefix_dim"],
    ) -> Float[Tensor, "batch length model_dim"]:
        """Return prefix states and parallel prediction states for every scale."""
        if prefix.shape[1] > self.max_prefix_length:
            raise ValueError(
                f"Prefix length {prefix.shape[1]} exceeds the configured maximum "
                f"of {self.max_prefix_length}."
            )
        if len(context_indices_by_scale) != len(self.scale_lengths) - 1:
            raise ValueError(
                "One context scale is required for every prediction scale "
                "after scale 1."
            )
        return self._encode_packed(prefix, context_indices_by_scale)

    def forward(
        self,
        context_indices_by_scale: list[Int[Tensor, "batch scale_length"]],
        *,
        prefix: Float[Tensor, "batch prefix_length prefix_dim"],
    ) -> list[Float[Tensor, "batch scale_length codebook_size"]]:
        hidden_states = self.encode(context_indices_by_scale, prefix=prefix)
        return self.output_head(hidden_states[:, prefix.shape[1] :])

    def predict_scale(
        self,
        prefix: Float[Tensor, "batch prefix_length prefix_dim"],
        completed_scales: list[Int[Tensor, "batch scale_length"]],
    ) -> Float[Tensor, "batch scale_length codebook_size"]:
        """Predict every code at the next scale in parallel."""
        scale_index = len(completed_scales)
        hidden_states = self._encode_packed(prefix, completed_scales)
        scale_length = self.scale_lengths[scale_index]
        return self.output_head.predict_scale(
            hidden_states[:, -scale_length:], scale_index
        )
