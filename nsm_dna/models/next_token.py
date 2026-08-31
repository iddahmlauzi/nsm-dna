import math
from pathlib import Path

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from .common import RMSNorm, TransformerBlock, precompute_rope_cosine_and_sine


class NextTokenModel(nn.Module):
    """Predict the nucleotide following each position in a DNA sequence."""

    def __init__(
        self,
        vocab_size: int,
        model_dim: int,
        num_layers: int,
        num_heads: int,
        max_sequence_length: int,
        *,
        dropout: float = 0.0,
        bias: bool = False,
        use_qk_norm: bool = True,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_sequence_length = max_sequence_length
        self.use_qk_norm = use_qk_norm
        self.rope_base = rope_base

        self.token_embedding = nn.Embedding(self.vocab_size, self.model_dim)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)

        positions = torch.arange(self.max_sequence_length)
        head_dim = self.model_dim // self.num_heads
        rope_cosine, rope_sine = precompute_rope_cosine_and_sine(
            positions,
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
        self.output_projection = nn.Linear(
            self.model_dim,
            self.vocab_size,
            bias=False,
        )
        self.output_projection.weight = self.token_embedding.weight

    def _initialize_residual_projections(self) -> None:
        """Scale residual branches so their variance does not grow with depth."""
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
    def from_config(cls, config: DictConfig) -> "NextTokenModel":
        """Build a next-token model from the experiment configuration."""
        return cls(
            vocab_size=config.model.vocab_size,
            model_dim=config.model.model_dim,
            num_layers=config.model.num_layers,
            num_heads=config.model.num_heads,
            max_sequence_length=config.data.sequence_length - 1,
            dropout=config.model.dropout,
            bias=config.model.bias,
            use_qk_norm=config.model.use_qk_norm,
            rope_base=config.model.rope_base,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        device: torch.device,
        *,
        frozen: bool = False,
    ) -> tuple["NextTokenModel", int]:
        """Rebuild a next-token model and return its saved training step."""
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

    def encode(
        self,
        input_ids: Int[Tensor, "batch sequence_length"],
    ) -> Float[Tensor, "batch sequence_length model_dim"]:
        """Return final normalized hidden states for each nucleotide."""
        sequence_length = input_ids.shape[1]
        if sequence_length > self.max_sequence_length:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds the configured maximum "
                f"of {self.max_sequence_length}."
            )

        hidden_states = self.token_embedding(input_ids)
        rotary_embeddings = (
            self.rope_cosine[:sequence_length],
            self.rope_sine[:sequence_length],
        )

        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                rotary_embeddings=rotary_embeddings,
                is_causal=True,
            )

        return self.final_norm(hidden_states)

    def forward(
        self,
        input_ids: Int[Tensor, "batch sequence_length"],
    ) -> Float[Tensor, "batch sequence_length vocab_size"]:
        return self.output_projection(self.encode(input_ids))
