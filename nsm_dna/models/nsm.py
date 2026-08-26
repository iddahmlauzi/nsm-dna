import einx
import torch
import torch.nn as nn
from jaxtyping import Bool, Float
from torch import Tensor

from .common import (
    LayerNorm,
    RotaryEmbeddings,
    TransformerBlock,
    precompute_rope_cosine_and_sine,
)


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


class SharedOutputHead(nn.Module):
    """Apply residual FFN corrections before one shared linear readout.

    Each block expands to a wider hidden dimension and projects back to
    model_dim. This preserves a direct path from the Transformer states to the
    codebook classifier while adding nonlinear capacity around that path.
    """

    def __init__(
        self,
        model_dim: int,
        codebook_size: int,
        num_blocks: int = 2,
        hidden_multiplier: float = 2.0,
        dropout: float = 0.1,
        bias: bool = False,
    ) -> None:
        super().__init__()

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
        self.codebook_projection = nn.Linear(
            model_dim,
            codebook_size,
            bias=False,
        )

    def forward(
        self,
        x: Float[Tensor, "batch length model_dim"],
    ) -> Float[Tensor, "batch length codebook_size"]:
        for block in self.blocks:
            x = block(x)

        return self.codebook_projection(x)


class NSM(nn.Module):
    """Predict a discrete VQ-VAE hierarchy from coarse to fine."""

    def __init__(
        self,
        vq_embed_dim: int,
        model_dim: int,
        scale_lengths: list[int],
        codebook_size: int,
        num_layers: int,
        num_heads: int,
        *,
        dropout: float = 0.1,
        bias: bool = False,
        use_qk_norm: bool = True,
        rope_base: float = 10000.0,
        head_num_blocks: int = 2,
        head_hidden_multiplier: float = 2.0,
        input_refinement_kernel_size: int = 3,
        max_prefix_length: int = 0,
    ) -> None:
        super().__init__()

        if input_refinement_kernel_size % 2 == 0:
            raise ValueError("input_refinement_kernel_size must be odd.")

        self.vq_embed_dim = vq_embed_dim
        self.model_dim = model_dim
        self.scale_lengths = list(scale_lengths)
        self.codebook_size = codebook_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.use_qk_norm = use_qk_norm
        self.rope_base = rope_base
        self.input_refinement_kernel_size = input_refinement_kernel_size
        self.max_prefix_length = max_prefix_length

        # Each resized scale input receives its own local residual correction
        # before all scales share the model-space input projection.
        self.scale_input_convs = nn.ModuleList(
            [
                nn.Conv1d(
                    self.vq_embed_dim,
                    self.vq_embed_dim,
                    kernel_size=self.input_refinement_kernel_size,
                    padding=self.input_refinement_kernel_size // 2,
                    bias=bias,
                )
                for _ in self.scale_lengths[1:]
            ]
        )

        # Start with no residual correction so refinement initially leaves the
        # resized scale inputs unchanged.
        for conv in self.scale_input_convs:
            nn.init.zeros_(conv.weight)
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)

        self.input_projection = nn.Linear(
            self.vq_embed_dim,
            self.model_dim,
            bias=bias,
        )

        # The first scale has no preceding reconstruction to use as input, so
        # it receives learned BOS embeddings in model space.
        self.bos = nn.Parameter(
            torch.empty(1, self.scale_lengths[0], self.model_dim)
        )
        nn.init.normal_(self.bos, mean=0.0, std=0.02)

        # Reset RoPE positions do not identify the scale, so add one learned
        # model-space vector per scale for the shared output head.
        self.scale_embedding = nn.Embedding(len(self.scale_lengths), self.model_dim)
        nn.init.normal_(self.scale_embedding.weight, mean=0.0, std=0.02)

        # Each scale input already contains the cumulative reconstruction from
        # preceding scales, so attention remains within each scale section.
        scale_ids = torch.cat(
            [
                torch.full((scale_length,), scale_index)
                for scale_index, scale_length in enumerate(self.scale_lengths)
            ]
        )
        self.register_buffer("scale_ids", scale_ids, persistent=False)

        row_scale_ids = einx.id("row -> row 1", scale_ids)
        column_scale_ids = einx.id("column -> 1 column", scale_ids)
        scale_attention_mask = einx.id(
            "row column -> 1 1 row column",
            row_scale_ids == column_scale_ids,
        )
        self.register_buffer(
            "scale_attention_mask",
            scale_attention_mask,
            persistent=False,
        )

        # Block-diagonal scale sections use independent position ranges, so
        # RoPE restarts from position zero at every scale.
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
                )
                for _ in range(self.num_layers)
            ]
        )
        self.final_norm = LayerNorm(self.model_dim, bias=bias)
        self.output_head = SharedOutputHead(
            self.model_dim,
            self.codebook_size,
            num_blocks=head_num_blocks,
            hidden_multiplier=head_hidden_multiplier,
            dropout=dropout,
            bias=bias,
        )

    def _refine_scale_inputs(
        self,
        x: list[Float[Tensor, "batch scale_length vq_dim"]],
    ) -> list[Float[Tensor, "batch scale_length vq_dim"]]:
        """Adapt the tokenizer's resized reconstructions for NSM prediction.

        Each target scale receives a cumulative reconstruction that the tokenizer
        has resized from the preceding scales. A separate Conv1d lets NSM learn a
        local correction for each resolution. The correction is added residually,
        and zero initialization makes this operation an identity at the start of
        training.
        """
        refined_scale_inputs = []

        for scale_input, conv in zip(x, self.scale_input_convs):
            scale_input_channels_first = einx.id("b l d -> b d l", scale_input)
            correction = conv(scale_input_channels_first)
            correction = einx.id("b d l -> b l d", correction)
            refined_scale_inputs.append(scale_input + correction)

        return refined_scale_inputs

    def _build_attention_mask(
        self,
        prefix_length: int,
    ) -> Bool[Tensor, "1 1 length length"]:
        """Let target scales read the prefix while keeping scales isolated."""
        if prefix_length == 0:
            return self.scale_attention_mask

        prefix_section_ids = torch.full(
            (prefix_length,),
            -1,
            device=self.scale_ids.device,
        )
        section_ids = torch.cat([prefix_section_ids, self.scale_ids])
        row_section_ids = einx.id("row -> row 1", section_ids)
        column_section_ids = einx.id("column -> 1 column", section_ids)

        same_section = row_section_ids == column_section_ids
        target_reads_prefix = (row_section_ids >= 0) & (column_section_ids == -1)
        return einx.id(
            "row column -> 1 1 row column",
            same_section | target_reads_prefix,
        )

    def _get_rotary_embeddings(self, prefix_length: int) -> RotaryEmbeddings:
        """Combine sequential prefix positions with reset per-scale positions."""
        cosine = torch.cat(
            [self.prefix_rope_cosine[:prefix_length], self.rope_cosine],
            dim=0,
        )
        sine = torch.cat(
            [self.prefix_rope_sine[:prefix_length], self.rope_sine],
            dim=0,
        )
        return cosine, sine

    def forward(
        self,
        x: list[Float[Tensor, "batch scale_length vq_dim"]],
        *,
        prefix: Float[Tensor, "batch prefix_length vq_dim"] | None = None,
    ) -> Float[Tensor, "batch hierarchy_length codebook_size"]:
        x = self._refine_scale_inputs(x)
        x = torch.cat(x, dim=1)

        prefix_length = 0 if prefix is None else prefix.shape[1]
        if prefix_length > self.max_prefix_length:
            raise ValueError(
                f"Prefix length {prefix_length} exceeds the configured maximum "
                f"of {self.max_prefix_length}."
            )
        if prefix is not None:
            x = torch.cat([prefix, x], dim=1)

        x = self.input_projection(x)
        prefix_hidden_states = x[:, :prefix_length]
        x = x[:, prefix_length:]

        bos = self.bos.expand(x.shape[0], -1, -1)
        x = torch.cat([bos, x], dim=1)
        x = x + self.scale_embedding(self.scale_ids)

        # The prefix comes first in the Transformer sequence; BOS begins the
        # target hierarchy and is followed by the remaining scale inputs.
        x = torch.cat([prefix_hidden_states, x], dim=1)

        attention_mask = self._build_attention_mask(prefix_length)
        rotary_embeddings = self._get_rotary_embeddings(prefix_length)
        for block in self.blocks:
            x = block(
                x,
                attention_mask=attention_mask,
                rotary_embeddings=rotary_embeddings,
            )

        x = self.final_norm(x)
        x = x[:, prefix_length:]
        return self.output_head(x)
