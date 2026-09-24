import einx
import torch
import torch.distributed as dist
import torch.nn as nn
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

    @torch.no_grad()
    def initialize(self, vectors: Float[Tensor, "codebook_size quantization_dim"]):
        """Replace the codebook and reset its EMA state."""
        if vectors.shape != self.codebook.shape:
            raise ValueError("Initial vectors must match the codebook shape.")

        self.codebook.copy_(vectors)
        self.ema_counts.fill_(1)
        self.ema_vector_sums.copy_(vectors)
        self.codebook_hits.fill_(True)

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


class ResidualVectorQuantizer(nn.Module):
    """Quantize a block-local mean-and-detail hierarchy."""

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

        expected_scale_lengths = []
        scale_length = 1
        while scale_length <= self.latent_length:
            expected_scale_lengths.append(scale_length)
            scale_length *= 2
        if self.scale_lengths != expected_scale_lengths:
            raise ValueError(
                "scale_lengths must double from 1 through latent_length."
            )
        if len(self.codebook_sizes) != len(self.scale_lengths):
            raise ValueError("Each hierarchy scale requires one codebook size.")

        # Scale 1 stores one regional mean. Every later scale stores one detail
        # for each parent, so one code reconstructs both of the parent's children.
        self.code_lengths = [1, *self.scale_lengths[:-1]]
        self.codebooks = nn.ModuleList(
            EMACodebook(
                codebook_size,
                quantization_dim,
                decay=decay,
                eps=eps,
            )
            for codebook_size in codebook_sizes
        )

    def mean_pool_to_scales(
        self,
        latent: Float[Tensor, "batch latent_length quantization_dim"],
    ) -> list[Float[Tensor, "batch scale_length quantization_dim"]]:
        """Return deterministic block means from coarse to fine."""
        latents_by_length = {self.latent_length: latent}
        current_latent = latent
        while current_latent.shape[1] > 1:
            current_latent = current_latent.reshape(
                current_latent.shape[0],
                current_latent.shape[1] // 2,
                2,
                current_latent.shape[2],
            ).mean(dim=2)
            latents_by_length[current_latent.shape[1]] = current_latent

        return [latents_by_length[length] for length in self.scale_lengths]

    def decompose(
        self,
        latent: Float[Tensor, "batch latent_length quantization_dim"],
    ) -> list[Float[Tensor, "batch code_length quantization_dim"]]:
        """Return the regional mean followed by left-child detail vectors."""
        means_by_scale = self.mean_pool_to_scales(latent)
        coefficients = [means_by_scale[0]]
        for parent_mean, child_means in zip(
            means_by_scale,
            means_by_scale[1:],
            strict=False,
        ):
            coefficients.append(child_means[:, 0::2] - parent_mean)
        return coefficients

    @staticmethod
    def _split_parents(
        parents: Float[Tensor, "batch parent_length quantization_dim"],
        details: Float[Tensor, "batch parent_length quantization_dim"],
    ) -> Float[Tensor, "batch child_length quantization_dim"]:
        """Reconstruct ordered children while preserving every parent mean."""
        children = torch.stack((parents + details, parents - details), dim=2)
        return children.flatten(1, 2)

    def _expand_to_latent_length(
        self,
        latent: Float[Tensor, "batch scale_length quantization_dim"],
    ) -> Float[Tensor, "batch latent_length quantization_dim"]:
        repeats_per_position = self.latent_length // latent.shape[1]
        return latent.repeat_interleave(repeats_per_position, dim=1)

    def reconstruct_coefficients(
        self,
        coefficients: list[Float[Tensor, "batch code_length quantization_dim"]],
    ) -> list[Float[Tensor, "batch latent_length quantization_dim"]]:
        """Return the cumulative full-length reconstruction at every scale."""
        reconstruction = coefficients[0]
        cumulative_latents = [self._expand_to_latent_length(reconstruction)]
        for details in coefficients[1:]:
            reconstruction = self._split_parents(reconstruction, details)
            cumulative_latents.append(
                self._expand_to_latent_length(reconstruction)
            )
        return cumulative_latents

    def forward(
        self,
        latent: Float[Tensor, "batch latent_length quantization_dim"],
        *,
        include_partial_reconstruction: bool = False,
    ) -> tuple[
        Float[Tensor, "batch length quantization_dim"],
        Float[Tensor, "batch length quantization_dim"] | None,
        list[Int[Tensor, "batch scale_length"]],
    ]:
        """Quantize the regional mean and every spatial detail scale.

        When partial reconstruction is enabled and a non-final scale exists,
        return one randomly selected cumulative latent for auxiliary reconstruction.
        """
        latent = latent.float()

        partial_scale_index = None
        if include_partial_reconstruction and len(self.scale_lengths) > 1:
            partial_scale_index = torch.randint(
                low=0,
                high=len(self.scale_lengths) - 1,
                size=(),
            ).item()

        indices_by_scale: list[Int[Tensor, "batch scale_length"]] = []
        quantized_coefficients: list[Tensor] = []
        coefficients = self.decompose(latent)

        for coefficient, codebook in zip(
            coefficients,
            self.codebooks,
            strict=True,
        ):
            quantized, scale_indices = codebook(coefficient)
            indices_by_scale.append(scale_indices)
            quantized_coefficients.append(
                coefficient + (quantized - coefficient).detach()
            )

        cumulative_latents = self.reconstruct_coefficients(quantized_coefficients)
        partial_quantized_latent = (
            cumulative_latents[partial_scale_index]
            if partial_scale_index is not None
            else None
        )

        return (
            cumulative_latents[-1],
            partial_quantized_latent,
            indices_by_scale,
        )

    @torch.no_grad()
    def indices_to_cumulative_latents(
        self,
        indices_by_scale: list[Int[Tensor, "batch scale_length"]],
    ) -> list[Float[Tensor, "batch length quantization_dim"]]:
        """Reconstruct the latent after each successive code scale."""
        coefficients = [
            codebook.codebook[scale_indices]
            for codebook, scale_indices in zip(
                self.codebooks,
                indices_by_scale,
                strict=True,
            )
        ]
        return self.reconstruct_coefficients(coefficients)

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
