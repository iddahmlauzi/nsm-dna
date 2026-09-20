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
        # Code embeddings identify which absolute codebook each input came from.
        self.scale_embedding = nn.Embedding(len(self.scale_lengths), self.model_dim)
        nn.init.normal_(self.scale_embedding.weight, mean=0.0, std=0.02)

        # One prefix code precedes the target hierarchy in every pass.
        max_input_length = self.max_prefix_length + sum(self.scale_lengths)
        head_dim = self.model_dim // self.num_heads
        rope_cosine, rope_sine = precompute_rope_cosine_and_sine(
            torch.arange(max_input_length),
            head_dim,
            self.rope_base,
        )
        self.register_buffer("rope_cosine", rope_cosine, persistent=False)
        self.register_buffer("rope_sine", rope_sine, persistent=False)

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
                *[codebook.codebook for codebook in tokenizer.quantizer.codebooks],
                tokenizer.final_codebook_vectors(),
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

    def _embed_code_inputs(
        self,
        indices_by_scale: list[Int[Tensor, "batch length"]],
    ) -> Float[Tensor, "batch length model_dim"]:
        code_inputs = []
        for scale_index, indices in enumerate(indices_by_scale):
            if indices.shape[1] == 0:
                continue
            vectors = getattr(self, self._codebook_buffer_names[scale_index])
            code_inputs.append(
                self.input_projection(vectors[indices])
                + self.scale_embedding.weight[scale_index]
            )
        return torch.cat(code_inputs, dim=1)

    def _build_attention_mask(
        self,
        prefix_length: int,
        input_length: int,
    ) -> Bool[Tensor, "1 1 length length"]:
        """Let target codes read the known prefix and preceding target codes."""
        attention_mask = torch.ones(
            input_length,
            input_length,
            dtype=torch.bool,
            device=self.rope_cosine.device,
        ).tril()
        attention_mask[:prefix_length, :prefix_length] = True
        return attention_mask.reshape(1, 1, input_length, input_length)

    def _encode_scale(
        self,
        prefix: Float[Tensor, "batch prefix_length prefix_dim"],
        prefix_code: Int[Tensor, "batch 1"],
        code_indices: list[Int[Tensor, "batch length"]],
    ) -> Float[Tensor, "batch length model_dim"]:
        prefix_length = prefix.shape[1] + 1
        hidden_states = torch.cat(
            [self.input_projection(prefix), self._embed_code_inputs([prefix_code])],
            dim=1,
        )
        if code_indices and any(indices.shape[1] for indices in code_indices):
            hidden_states = torch.cat(
                [hidden_states, self._embed_code_inputs(code_indices)], dim=1
            )

        input_length = hidden_states.shape[1]
        attention_mask = self._build_attention_mask(prefix_length, input_length)
        rotary_embeddings: RotaryEmbeddings = (
            self.rope_cosine[:input_length],
            self.rope_sine[:input_length],
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
        indices_by_scale: list[Int[Tensor, "batch scale_length"]],
        *,
        prefix: Float[Tensor, "batch prefix_length prefix_dim"],
        prefix_code: Int[Tensor, "batch 1"],
    ) -> Float[Tensor, "batch length model_dim"]:
        """Return prefix states and causal prediction states for every scale."""
        prefix_length = prefix.shape[1] + 1
        if prefix.shape[1] > self.max_prefix_length:
            raise ValueError(
                f"Prefix length {prefix.shape[1]} exceeds the configured maximum "
                f"of {self.max_prefix_length}."
            )
        # All but the final target code are input tokens. The state immediately
        # before each code predicts it, so no target sees its own identity.
        code_indices = [*indices_by_scale[:-1], indices_by_scale[-1][:, :-1]]
        hidden_states = self._encode_scale(prefix, prefix_code, code_indices)
        prediction_states = hidden_states[:, prefix_length - 1 :]
        return torch.cat(
            [hidden_states[:, :prefix_length], prediction_states], dim=1
        )

    def forward(
        self,
        indices_by_scale: list[Int[Tensor, "batch scale_length"]],
        *,
        prefix: Float[Tensor, "batch prefix_length prefix_dim"],
        prefix_code: Int[Tensor, "batch 1"],
    ) -> list[Float[Tensor, "batch scale_length codebook_size"]]:
        hidden_states = self.encode(
            indices_by_scale, prefix=prefix, prefix_code=prefix_code
        )
        return self.output_head(hidden_states[:, prefix.shape[1] + 1 :])

    def predict_scale(
        self,
        prefix: Float[Tensor, "batch prefix_length prefix_dim"],
        prefix_code: Int[Tensor, "batch 1"],
        completed_scales: list[Int[Tensor, "batch scale_length"]],
        current_codes: Int[Tensor, "batch generated_length"],
    ) -> Float[Tensor, "batch codebook_size"]:
        """Predict the next code using only already generated target codes."""
        scale_index = len(completed_scales)
        hidden_states = self._encode_scale(
            prefix, prefix_code, [*completed_scales, current_codes]
        )
        return self.output_head.predict_scale(
            hidden_states[:, -1:], scale_index
        )[:, 0]
