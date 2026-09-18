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


class MultiscaleResidualVectorQuantizer(nn.Module):
    """Multiscale residual vector quantizer."""

    def __init__(
        self,
        scale_lengths: list[int],
        codebook_sizes: list[int],
        quantization_dim: int,
        *,
        latent_length: int,
        # Codebook updates and quantization loss
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()

        if len(scale_lengths) != len(codebook_sizes):
            raise ValueError("Each scale length must have one codebook size.")
        if scale_lengths[-1] != latent_length:
            raise ValueError("The final scale length must equal the latent length.")

        self.scale_lengths = list(scale_lengths)
        self.codebook_sizes = list(codebook_sizes)
        self.quantization_dim = quantization_dim
        self.latent_length = latent_length
        self.commitment_cost = commitment_cost

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
        quantized: Float[Tensor, "batch quantization_dim scale_length"],
        scale_index: int,
    ) -> Float[Tensor, "batch quantization_dim length"]:
        """Repeat each scale vector across its corresponding latent block."""
        scale_length = self.scale_lengths[scale_index]
        if scale_length == self.latent_length:
            return quantized

        repeats_per_position = self.latent_length // scale_length
        return quantized.repeat_interleave(repeats_per_position, dim=-1)

    def _prepare_scale_contribution(
        self,
        quantized_at_scale: Float[
            Tensor,
            "batch scale_length quantization_dim",
        ],
        scale_index: int,
    ) -> Float[Tensor, "batch length quantization_dim"]:
        """Expand one scale's quantized vectors to the full latent length."""
        quantized_at_scale = einx.id("b l d -> b d l", quantized_at_scale)
        scale_contribution = self._upsample_to_full_length(
            quantized_at_scale,
            scale_index,
        )
        return einx.id("b d l -> b l d", scale_contribution)

    def forward(
        self,
        x: Float[Tensor, "batch length quantization_dim"],
        *,
        include_partial_reconstruction: bool = False,
    ) -> tuple[
        Float[Tensor, "batch length quantization_dim"],
        Float[Tensor, "batch length quantization_dim"] | None,
        Float[Tensor, ""],
        list[Int[Tensor, "batch scale_length"]],
    ]:
        """Quantize an encoder latent into cumulative multiscale contributions.

        When partial reconstruction is enabled, the second return value is one
        randomly selected non-final cumulative latent. The caller decodes that
        latent and computes the auxiliary reconstruction loss against the input
        tokens.
        """
        x = x.float()

        # Quantize the encoder output without backpropagating through the residual
        # hierarchy. The commitment loss and final STE provide encoder gradients.
        detached_x = x.detach()
        residual = detached_x
        reconstruction = torch.zeros_like(residual)

        partial_scale_index = None
        if include_partial_reconstruction:
            partial_scale_index = torch.randint(
                low=0,
                high=len(self.scale_lengths) - 1,
                size=(),
            ).item()

        indices_by_scale: list[Int[Tensor, "batch scale_length"]] = []
        partial_quantized_latent: Tensor | None = None

        for scale_index, codebook in enumerate(self.codebooks):
            scaled_residual = self._resize_to_scale(residual, scale_index)
            quantized_at_scale, scale_indices = codebook(scaled_residual)
            indices_by_scale.append(scale_indices)

            scale_contribution = self._prepare_scale_contribution(
                quantized_at_scale,
                scale_index,
            )
            reconstruction = reconstruction + scale_contribution
            residual = residual - scale_contribution

            if scale_index == partial_scale_index:
                partial_quantized_latent = x + (reconstruction - x).detach()

        commitment_loss = self.commitment_cost * F.mse_loss(
            x,
            reconstruction.detach(),
        )
        quantized_latent = x + (reconstruction - x).detach()

        return (
            quantized_latent,
            partial_quantized_latent,
            commitment_loss,
            indices_by_scale,
        )

    @torch.no_grad()
    def indices_to_cumulative_latents(
        self,
        indices_by_scale: list[Int[Tensor, "batch scale_length"]],
    ) -> list[Float[Tensor, "batch length quantization_dim"]]:
        """Reconstruct the latent after successively adding each scale."""
        batch_size = indices_by_scale[0].shape[0]
        reconstruction = self.codebooks[0].codebook.new_zeros(
            batch_size,
            self.latent_length,
            self.quantization_dim,
        )
        cumulative_latents: list[Tensor] = []

        for scale_index, (codebook, scale_indices) in enumerate(
            zip(self.codebooks, indices_by_scale)
        ):
            quantized_at_scale = codebook.codebook[scale_indices]
            scale_contribution = self._prepare_scale_contribution(
                quantized_at_scale,
                scale_index,
            )
            reconstruction = reconstruction + scale_contribution
            cumulative_latents.append(reconstruction)

        return cumulative_latents

    @torch.no_grad()
    def indices_to_next_scale_inputs(
        self,
        indices_by_scale: list[Int[Tensor, "batch scale_length"]],
    ) -> list[Float[Tensor, "batch scale_length quantization_dim"]]:
        """Construct the teacher-forced scale inputs for NSM-DNA.

        Each scale input is the cumulative reconstruction through the
        preceding scale, resized to the length of the scale to be predicted.
        """
        cumulative_latents = self.indices_to_cumulative_latents(
            indices_by_scale[:-1]
        )
        return [
            self._resize_to_scale(
                cumulative_latent,
                scale_index=scale_index + 1,
            )
            for scale_index, cumulative_latent in enumerate(cumulative_latents)
        ]

    @torch.no_grad()
    def indices_to_next_scale_input(
        self,
        preceding_indices_by_scale: list[Int[Tensor, "batch scale_length"]],
    ) -> Float[Tensor, "batch next_scale_length quantization_dim"]:
        """Construct the next input from an autoregressively predicted prefix."""
        cumulative_latent = self.indices_to_cumulative_latents(
            preceding_indices_by_scale
        )[-1]
        return self._resize_to_scale(
            cumulative_latent,
            scale_index=len(preceding_indices_by_scale),
        )

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
