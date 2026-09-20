import einx
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor


class EMACodebook(nn.Module):
    """Vector-quantization codebook updated with exponential moving averages."""

    def __init__(
        self,
        codebook_size: int,
        quantization_dim: int,
        decay: float = 0.99,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()

        self.codebook_size = codebook_size
        self.quantization_dim = quantization_dim
        self.base_decay = decay
        self.eps = eps

        codebook = torch.randn(codebook_size, quantization_dim)
        self.register_buffer("codebook", codebook)
        self.register_buffer("ema_counts", torch.ones(codebook_size))
        self.register_buffer("ema_vector_sums", codebook.clone())
        self.register_buffer(
            "codebook_hits", torch.zeros(codebook_size, dtype=torch.bool)
        )

    def _get_ema_decay(self) -> float:
        """Adjust EMA update strength for DDP's summed batch statistics."""
        world_size = (
            dist.get_world_size()
            if dist.is_available() and dist.is_initialized()
            else 1
        )
        return 1.0 - (1.0 - self.base_decay) / world_size

    def forward(
        self,
        x: Float[Tensor, "batch length quantization_dim"],
    ) -> tuple[
        Float[Tensor, "batch length quantization_dim"],
        Int[Tensor, "batch length"],
    ]:
        flat_input = einx.id("b l d -> (b l) d", x.detach().float())

        # Compute the distance from each input to every codebook vector.
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self.codebook**2, dim=1)
            - 2 * einx.dot("n d, k d -> n k", flat_input, self.codebook)
        )
        flat_indices = distances.argmin(dim=-1)
        indices = einx.id("(b l) -> b l", flat_indices, b=x.shape[0])

        if self.training:
            batch_counts = torch.bincount(
                flat_indices,
                minlength=self.codebook_size,
            ).to(self.ema_counts.dtype)

            batch_vector_sums = torch.zeros_like(self.ema_vector_sums)
            batch_vector_sums.index_add_(0, flat_indices, flat_input)

            # Combine batch statistics so every DDP worker applies the same update.
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(batch_counts, op=dist.ReduceOp.SUM)
                dist.all_reduce(batch_vector_sums, op=dist.ReduceOp.SUM)

            decay = self._get_ema_decay()
            with torch.no_grad():
                self.ema_counts.mul_(decay).add_(
                    batch_counts,
                    alpha=1 - decay,
                )
                self.ema_vector_sums.mul_(decay).add_(
                    batch_vector_sums,
                    alpha=1 - decay,
                )

                # Smooth the counts before calculating each code's running mean.
                total_count = self.ema_counts.sum()
                smoothed_counts = (
                    (self.ema_counts + self.eps)
                    / (total_count + self.codebook_size * self.eps)
                    * total_count
                )
                smoothed_counts = einx.id("k -> k 1", smoothed_counts)
                self.codebook.copy_(self.ema_vector_sums / smoothed_counts)
                self.codebook_hits.logical_or_(batch_counts > 0)

        quantized = self.codebook[indices]

        return quantized.to(x.dtype), indices

    @property
    def utilization(self) -> Float[Tensor, ""]:
        return self.codebook_hits.float().mean()


class MultiscaleVectorQuantizer(nn.Module):
    """Quantize independent pooled views of one encoder latent."""

    def __init__(
        self,
        scale_lengths: list[int],
        codebook_sizes: list[int],
        quantization_dim: int,
        *,
        latent_length: int,
        decay: float = 0.99,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()

        self.scale_lengths = list(scale_lengths)
        self.codebook_sizes = list(codebook_sizes)
        self.quantization_dim = quantization_dim
        self.latent_length = latent_length

        self.codebooks = nn.ModuleList(
            EMACodebook(
                codebook_size,
                quantization_dim,
                decay=decay,
                eps=eps,
            )
            for codebook_size in codebook_sizes
        )

    def _resize_to_scale(
        self,
        latent: Float[Tensor, "batch length quantization_dim"],
        scale_index: int,
    ) -> Float[Tensor, "batch scale_length quantization_dim"]:
        """Average equal, non-overlapping latent blocks into one scale."""
        scale_length = self.scale_lengths[scale_index]
        if scale_length == self.latent_length:
            return latent

        # Area interpolation computes one mean for each contiguous latent block.
        latent = einx.id("b l d -> b d l", latent)
        resized_latent = F.interpolate(
            latent,
            size=scale_length,
            mode="area",
        )

        return einx.id("b d l -> b l d", resized_latent)

    def _upsample_to_full_length(
        self,
        quantized: Float[Tensor, "batch scale_length quantization_dim"],
        scale_index: int,
    ) -> Float[Tensor, "batch length quantization_dim"]:
        """Repeat each scale vector across its corresponding latent block."""
        scale_length = self.scale_lengths[scale_index]
        if scale_length == self.latent_length:
            return quantized

        repeats_per_position = self.latent_length // scale_length
        return quantized.repeat_interleave(repeats_per_position, dim=1)

    def forward(
        self,
        x: Float[Tensor, "batch length quantization_dim"],
        *,
        include_partial_reconstruction: bool = False,
    ) -> tuple[
        Float[Tensor, "batch length quantization_dim"] | None,
        list[Int[Tensor, "batch scale_length"]],
    ]:
        """Quantize pooled views of the continuous latent at coarse scales.

        When partial reconstruction is enabled, return one randomly selected
        scale latent for auxiliary reconstruction.
        """
        x = x.float()

        partial_scale_index = None
        if include_partial_reconstruction:
            partial_scale_index = torch.randint(
                low=0,
                high=len(self.scale_lengths),
                size=(),
            ).item()

        indices_by_scale: list[Int[Tensor, "batch scale_length"]] = []
        partial_quantized_latent: Tensor | None = None

        for scale_index, codebook in enumerate(self.codebooks):
            pooled_latent = self._resize_to_scale(x, scale_index)
            quantized_at_scale, scale_indices = codebook(pooled_latent)
            indices_by_scale.append(scale_indices)

            # Backpropagate reconstruction through the same pooling operation
            # that produced the codebook input.
            quantized_with_gradient = pooled_latent + (
                quantized_at_scale - pooled_latent
            ).detach()
            scale_latent = self._upsample_to_full_length(
                quantized_with_gradient, scale_index
            )

            if scale_index == partial_scale_index:
                partial_quantized_latent = scale_latent

        return partial_quantized_latent, indices_by_scale

    @torch.no_grad()
    def indices_to_scale_latent(
        self,
        scale_indices: Int[Tensor, "batch scale_length"],
        scale_index: int,
    ) -> Float[Tensor, "batch length quantization_dim"]:
        """Expand one scale's code vectors to the full latent length."""
        codebook = self.codebooks[scale_index]
        return self._upsample_to_full_length(
            codebook.codebook[scale_indices], scale_index
        )

    @torch.no_grad()
    def indices_to_scale_latents(
        self,
        indices_by_scale: list[Int[Tensor, "batch scale_length"]],
    ) -> list[Float[Tensor, "batch length quantization_dim"]]:
        """Expand each scale's codes without combining them with other scales."""
        scale_latents: list[Tensor] = []

        for scale_index, scale_indices in enumerate(indices_by_scale):
            scale_latents.append(
                self.indices_to_scale_latent(scale_indices, scale_index)
            )

        return scale_latents

    @property
    def utilization_by_scale(self) -> list[Float[Tensor, ""]]:
        """Fraction of each scale's codes that have been used."""
        return [codebook.utilization for codebook in self.codebooks]

    @property
    def global_utilization(self) -> Float[Tensor, ""]:
        """Fraction of codebook entries used across all scales."""
        used_codes = torch.stack(
            [codebook.codebook_hits.sum() for codebook in self.codebooks]
        ).sum()
        return used_codes / sum(self.codebook_sizes)

    @property
    def num_codebook_parameters(self) -> int:
        """Number of learned codebook values included in model-size reporting."""
        return sum(codebook.codebook.numel() for codebook in self.codebooks)
