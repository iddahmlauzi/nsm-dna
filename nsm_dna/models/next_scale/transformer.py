import math

import einx
import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Int
from torch import Tensor

from ..common import (
    RMSNorm,
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
        dropout: float = 0.0,
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


class NextScaleTransformer(nn.Module):
    """Predict a discrete multiscale hierarchy from coarse to fine."""

    def __init__(
        self,
        input_dim: int,
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

        self.input_dim = input_dim
        self.model_dim = model_dim
        self.scale_lengths = list(scale_lengths)
        self.predicted_scale_lengths = self.scale_lengths[1:]
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
                    self.input_dim,
                    self.input_dim,
                    kernel_size=self.input_refinement_kernel_size,
                    padding=self.input_refinement_kernel_size // 2,
                    bias=bias,
                )
                for _ in self.predicted_scale_lengths
            ]
        )

        # Start with no residual correction so refinement initially leaves the
        # resized scale inputs unchanged.
        for conv in self.scale_input_convs:
            nn.init.zeros_(conv.weight)
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)

        self.input_projection = nn.Linear(
            self.input_dim,
            self.model_dim,
            bias=bias,
        )

        # The first scale is supplied as the cumulative input used to predict the
        # next scale. The transformer therefore has sections only for predicted
        # scales rather than a learned BOS section.
        self.scale_embedding = nn.Embedding(
            len(self.predicted_scale_lengths),
            self.model_dim,
        )
        nn.init.normal_(self.scale_embedding.weight, mean=0.0, std=0.02)

        # Each scale input already contains the cumulative reconstruction from
        # preceding scales, so attention remains within each scale section.
        scale_ids = torch.cat(
            [
                torch.full((scale_length,), scale_index)
                for scale_index, scale_length in enumerate(self.predicted_scale_lengths)
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
            [
                torch.arange(scale_length)
                for scale_length in self.predicted_scale_lengths
            ]
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
        self.output_head = SharedOutputHead(
            self.model_dim,
            self.codebook_size,
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

    def _refine_scale_inputs(
        self,
        scale_inputs: list[Float[Tensor, "batch scale_length input_dim"]],
    ) -> list[Float[Tensor, "batch scale_length input_dim"]]:
        """Adapt the tokenizer's resized reconstructions for NSM prediction.

        Each target scale after the first receives a cumulative reconstruction
        that the tokenizer has resized from the preceding scales. A separate
        Conv1d lets NSM learn a local correction for each resolution. The
        correction is added residually, and zero initialization makes this
        operation an identity at the start of training.
        """
        refined_scale_inputs = []

        if len(scale_inputs) != len(self.scale_input_convs):
            raise ValueError("Expected one input for every predicted scale.")
        for prediction_index, scale_input in enumerate(scale_inputs):
            refined_scale_inputs.append(
                self._refine_scale_input(scale_input, prediction_index)
            )

        return refined_scale_inputs

    def _refine_scale_input(
        self,
        scale_input: Float[Tensor, "batch scale_length input_dim"],
        prediction_index: int,
    ) -> Float[Tensor, "batch scale_length input_dim"]:
        """Apply the input correction belonging to one predicted scale."""
        scale_input_channels_first = einx.id("b l d -> b d l", scale_input)
        correction = self.scale_input_convs[prediction_index](
            scale_input_channels_first
        )
        correction = einx.id("b d l -> b l d", correction)
        return scale_input + correction

    def _build_attention_mask(
        self,
        prefix_length: int,
        scale_ids: Tensor | None = None,
    ) -> Bool[Tensor, "1 1 length length"]:
        """Let every scale read the prefix while keeping scale sections isolated."""
        if scale_ids is None and prefix_length == 0:
            return self.scale_attention_mask

        if scale_ids is None:
            scale_ids = self.scale_ids

        prefix_section_ids = torch.full(
            (prefix_length,),
            -1,
            device=scale_ids.device,
        )
        section_ids = torch.cat([prefix_section_ids, scale_ids])
        row_section_ids = einx.id("row -> row 1", section_ids)
        column_section_ids = einx.id("column -> 1 column", section_ids)

        same_section = row_section_ids == column_section_ids
        target_reads_prefix = (row_section_ids >= 0) & (column_section_ids == -1)
        return einx.id(
            "row column -> 1 1 row column",
            same_section | target_reads_prefix,
        )

    def _get_rotary_embeddings(
        self,
        prefix_length: int,
        scale_cosine: Tensor | None = None,
        scale_sine: Tensor | None = None,
    ) -> RotaryEmbeddings:
        """Combine sequential prefix positions with reset per-scale positions."""
        if scale_cosine is None:
            scale_cosine = self.rope_cosine
        if scale_sine is None:
            scale_sine = self.rope_sine
        cosine = torch.cat(
            [self.prefix_rope_cosine[:prefix_length], scale_cosine],
            dim=0,
        )
        sine = torch.cat(
            [self.prefix_rope_sine[:prefix_length], scale_sine],
            dim=0,
        )
        return cosine, sine

    def _encode_hierarchy(
        self,
        hierarchy_inputs: Float[Tensor, "batch hierarchy_length input_dim"],
        hierarchy_scale_ids: Int[Tensor, "hierarchy_length"],
        hierarchy_rope_cosine: Float[Tensor, "hierarchy_length head_dim"],
        hierarchy_rope_sine: Float[Tensor, "hierarchy_length head_dim"],
        *,
        prefix: Float[Tensor, "batch prefix_length input_dim"] | None,
    ) -> Float[Tensor, "batch length model_dim"]:
        """Encode an arbitrary set of isolated hierarchy sections."""
        prefix_length = 0 if prefix is None else prefix.shape[1]
        if prefix_length > self.max_prefix_length:
            raise ValueError(
                f"Prefix length {prefix_length} exceeds the configured maximum "
                f"of {self.max_prefix_length}."
            )

        if prefix is None:
            combined_inputs = hierarchy_inputs
        else:
            combined_inputs = torch.cat([prefix, hierarchy_inputs], dim=1)

        projected_inputs = self.input_projection(combined_inputs)
        prefix_hidden_states = projected_inputs[:, :prefix_length]
        hierarchy_hidden_states = projected_inputs[:, prefix_length:]
        hierarchy_hidden_states = hierarchy_hidden_states + self.scale_embedding(
            hierarchy_scale_ids
        )
        hidden_states = torch.cat(
            [prefix_hidden_states, hierarchy_hidden_states],
            dim=1,
        )

        attention_mask = self._build_attention_mask(
            prefix_length,
            hierarchy_scale_ids,
        )
        rotary_embeddings = self._get_rotary_embeddings(
            prefix_length,
            hierarchy_rope_cosine,
            hierarchy_rope_sine,
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
        scale_inputs: list[Float[Tensor, "batch scale_length input_dim"]],
        *,
        prefix: Float[Tensor, "batch prefix_length input_dim"] | None = None,
    ) -> Float[Tensor, "batch length model_dim"]:
        """Return final normalized states for the prefix and target hierarchy."""
        refined_scale_inputs = self._refine_scale_inputs(scale_inputs)
        hierarchy_inputs = torch.cat(refined_scale_inputs, dim=1)
        return self._encode_hierarchy(
            hierarchy_inputs,
            self.scale_ids,
            self.rope_cosine,
            self.rope_sine,
            prefix=prefix,
        )

    def predict_scale(
        self,
        scale_input: Float[Tensor, "batch scale_length input_dim"],
        prediction_index: int,
        *,
        prefix: Float[Tensor, "batch prefix_length input_dim"] | None = None,
    ) -> Float[Tensor, "batch scale_length codebook_size"]:
        """Predict one scale without evaluating the other isolated sections."""
        if not 0 <= prediction_index < len(self.predicted_scale_lengths):
            raise IndexError("prediction_index is outside the predicted scales.")

        expected_length = self.predicted_scale_lengths[prediction_index]
        if scale_input.shape[1] != expected_length:
            raise ValueError(
                f"Prediction index {prediction_index} expects scale length "
                f"{expected_length}, received {scale_input.shape[1]}."
            )

        start = sum(self.predicted_scale_lengths[:prediction_index])
        end = start + expected_length
        refined_scale_input = self._refine_scale_input(
            scale_input,
            prediction_index,
        )
        hidden_states = self._encode_hierarchy(
            refined_scale_input,
            self.scale_ids[start:end],
            self.rope_cosine[start:end],
            self.rope_sine[start:end],
            prefix=prefix,
        )
        prefix_length = 0 if prefix is None else prefix.shape[1]
        return self.output_head(hidden_states[:, prefix_length:])

    def forward(
        self,
        scale_inputs: list[Float[Tensor, "batch scale_length input_dim"]],
        *,
        prefix: Float[Tensor, "batch prefix_length input_dim"] | None = None,
    ) -> Float[Tensor, "batch hierarchy_length codebook_size"]:
        hidden_states = self.encode(scale_inputs, prefix=prefix)
        prefix_length = 0 if prefix is None else prefix.shape[1]
        hierarchy_hidden_states = hidden_states[:, prefix_length:]
        return self.output_head(hierarchy_hidden_states)
