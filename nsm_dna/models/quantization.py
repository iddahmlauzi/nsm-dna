from dataclasses import dataclass

import einx
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor


@dataclass
class QuantizationResult:
    """Intermediate values produced by multiscale quantization."""

    quantized_latent: Tensor
    indices_by_scale: list[Tensor]
    latents_by_scale: list[Tensor]
    quantized_latents_by_scale: list[Tensor]
    commitment_losses_by_scale: Tensor


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
        *,
        group_indices: Int[Tensor, "batch length"] | None = None,
        num_groups: int | None = None,
    ) -> tuple[
        Float[Tensor, "batch length quantization_dim"],
        Int[Tensor, "batch length"],
    ]:
        flat_input = einx.id("b l d -> (b l) d", x.detach().float())

        if group_indices is None:
            # Compute the distance from each input to every codebook vector.
            distances = (
                torch.sum(flat_input**2, dim=1, keepdim=True)
                + torch.sum(self.codebook**2, dim=1)
                - 2 * einx.dot("n d, k d -> n k", flat_input, self.codebook)
            )
            flat_indices = distances.argmin(dim=-1)
        else:
            if num_groups is None:
                raise ValueError(
                    "num_groups is required for grouped codebook lookup."
                )
            if group_indices.shape != x.shape[:2]:
                raise ValueError(
                    "group_indices must match the batch and length dimensions "
                    "of the codebook input."
                )
            if self.codebook_size % num_groups != 0:
                raise ValueError(
                    f"Codebook size {self.codebook_size} must be divisible by "
                    f"the {num_groups} lookup groups."
                )

            codes_per_group = self.codebook_size // num_groups
            flat_groups = group_indices.detach().flatten()

            if codes_per_group == 1:
                flat_indices = flat_groups
            else:
                grouped_codebook = self.codebook.reshape(
                    num_groups,
                    codes_per_group,
                    self.quantization_dim,
                )
                candidate_vectors = grouped_codebook[flat_groups]
                distances = (
                    torch.sum(flat_input**2, dim=1, keepdim=True)
                    + torch.sum(candidate_vectors**2, dim=2)
                    - 2
                    * torch.sum(
                        flat_input.unsqueeze(1) * candidate_vectors,
                        dim=2,
                    )
                )
                within_group_indices = distances.argmin(dim=-1)
                flat_indices = (
                    flat_groups * codes_per_group + within_group_indices
                )
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


class DeterministicCodebook(nn.Module):
    """Trainable vectors selected by externally determined discrete IDs."""

    def __init__(self, codebook_size: int, quantization_dim: int) -> None:
        super().__init__()

        self.codebook_size = codebook_size
        self.quantization_dim = quantization_dim
        self.codebook = nn.Parameter(
            torch.randn(codebook_size, quantization_dim)
        )
        self.register_buffer(
            "codebook_hits", torch.zeros(codebook_size, dtype=torch.bool)
        )

    def forward(
        self,
        indices: Int[Tensor, "batch length"],
    ) -> Float[Tensor, "batch length quantization_dim"]:
        if self.training:
            batch_counts = torch.bincount(
                indices.flatten(),
                minlength=self.codebook_size,
            )
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(batch_counts, op=dist.ReduceOp.SUM)
            with torch.no_grad():
                self.codebook_hits.logical_or_(batch_counts > 0)

        return self.codebook[indices]

    @property
    def utilization(self) -> Float[Tensor, ""]:
        return self.codebook_hits.float().mean()


class _LearnedDownsamplingBlock(nn.Module):
    """Combine each ordered pair of latent positions into one parent."""

    def __init__(self, quantization_dim: int) -> None:
        super().__init__()

        self.convolution = nn.Conv1d(
            in_channels=quantization_dim,
            out_channels=quantization_dim,
            kernel_size=2,
            stride=2,
            bias=False,
        )
        self.norm = nn.LayerNorm(
            quantization_dim,
            elementwise_affine=False,
        )

    def forward(
        self,
        latent: Float[Tensor, "batch length quantization_dim"],
    ) -> Float[Tensor, "batch reduced_length quantization_dim"]:
        latent = einx.id("b l d -> b d l", latent)
        latent = self.convolution(latent)
        latent = einx.id("b d l -> b l d", latent)
        return self.norm(latent)


class MultiscaleVectorQuantizer(nn.Module):
    """Build coarser VQ scales from exact dinucleotide vectors."""

    def __init__(
        self,
        scale_lengths: list[int],
        codebook_sizes: list[int],
        quantization_dim: int,
        *,
        latent_length: int,
        decay: float = 0.99,
        eps: float = 1e-5,
        group_codes_by_left_child: bool = False,
    ) -> None:
        super().__init__()

        self.scale_lengths = list(scale_lengths)
        self.codebook_sizes = list(codebook_sizes)
        self.quantization_dim = quantization_dim
        self.latent_length = latent_length
        self.group_codes_by_left_child = group_codes_by_left_child

        if self.group_codes_by_left_child:
            for parent_size, child_size in zip(
                self.codebook_sizes[:-1],
                self.codebook_sizes[1:],
                strict=True,
            ):
                if parent_size < child_size or parent_size % child_size != 0:
                    raise ValueError(
                        "Each structured parent codebook must contain an "
                        "integer number of codes for every possible left-child "
                        f"code, but received parent size {parent_size} and "
                        f"child size {child_size}."
                    )

        # Exact dinucleotide vectors supply scale 128. Learned reductions then
        # build the remaining hierarchy: 128 → 64 → 32 → 16 → 8 → 4 → 2 → 1.
        self.downsampled_lengths = list(reversed(self.scale_lengths[:-1]))

        self.downsamplers = nn.ModuleList(
            _LearnedDownsamplingBlock(self.quantization_dim)
            for _ in self.downsampled_lengths
        )
        self.codebooks = nn.ModuleList(
            [
                EMACodebook(
                    codebook_size,
                    quantization_dim,
                    decay=decay,
                    eps=eps,
                )
                for codebook_size in codebook_sizes[:-1]
            ]
            + [
                DeterministicCodebook(
                    codebook_sizes[-1],
                    quantization_dim,
                )
            ]
        )

    def _quantize_scale(
        self,
        latent: Float[Tensor, "batch length quantization_dim"],
        scale_index: int,
        left_child_indices: Int[Tensor, "batch length"],
    ) -> tuple[Tensor, Tensor, Tensor]:
        codebook = self.codebooks[scale_index]
        if not isinstance(codebook, EMACodebook):
            raise RuntimeError("Deterministic codebooks require explicit IDs.")
        if self.group_codes_by_left_child:
            quantized, indices = codebook(
                latent,
                group_indices=left_child_indices,
                num_groups=self.codebook_sizes[scale_index + 1],
            )
        else:
            quantized, indices = codebook(latent)
        commitment_loss = F.mse_loss(latent.float(), quantized.detach().float())
        quantized_with_gradient = latent + (quantized - latent).detach()
        return quantized_with_gradient, indices, commitment_loss

    def _build_discrete_hierarchy(
        self,
        finest_scale_indices: Int[Tensor, "batch latent_length"],
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor], list[Tensor]]:
        """Build each parent from pairs of quantized child vectors."""
        latents_by_length: dict[int, Tensor] = {}
        quantized_by_length: dict[int, Tensor] = {}
        indices_by_length: dict[int, Tensor] = {}
        commitment_by_length: dict[int, Tensor] = {}
        finest_scale_index = len(self.scale_lengths) - 1
        finest_scale_length = self.scale_lengths[finest_scale_index]
        finest_codebook = self.codebooks[finest_scale_index]
        if not isinstance(finest_codebook, DeterministicCodebook):
            raise RuntimeError("Expected a deterministic finest codebook.")
        current_quantized = finest_codebook(finest_scale_indices)
        current_indices = finest_scale_indices
        latents_by_length[finest_scale_length] = current_quantized
        quantized_by_length[finest_scale_length] = current_quantized
        indices_by_length[finest_scale_length] = finest_scale_indices

        for downsampler_index, scale_index in enumerate(
            reversed(range(finest_scale_index))
        ):
            scale_length = self.scale_lengths[scale_index]
            left_child_indices = current_indices[:, 0::2]
            current_latent = self.downsamplers[downsampler_index](
                current_quantized.detach()
            )
            latents_by_length[scale_length] = current_latent
            (
                current_quantized,
                scale_indices,
                commitment_loss,
            ) = self._quantize_scale(
                current_latent,
                scale_index,
                left_child_indices,
            )
            commitment_by_length[scale_length] = commitment_loss
            quantized_by_length[scale_length] = current_quantized
            indices_by_length[scale_length] = scale_indices
            current_indices = scale_indices

        return (
            [latents_by_length[length] for length in self.scale_lengths],
            [quantized_by_length[length] for length in self.scale_lengths],
            [indices_by_length[length] for length in self.scale_lengths],
            [commitment_by_length[length] for length in self.scale_lengths[:-1]],
        )

    def _downsample_to_scales(
        self,
        finest_scale_indices: Int[Tensor, "batch latent_length"],
    ) -> list[Float[Tensor, "batch scale_length quantization_dim"]]:
        """Return the pre-quantization state at every configured scale."""
        latents_by_scale, _, _, _ = self._build_discrete_hierarchy(
            finest_scale_indices,
        )
        return latents_by_scale

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

    def quantize(
        self,
        finest_scale_indices: Int[Tensor, "batch latent_length"],
    ) -> QuantizationResult:
        """Quantize every scale and retain values needed by training losses."""
        (
            latents_by_scale,
            quantized_latents_by_scale,
            indices_by_scale,
            commitment_losses_by_scale,
        ) = self._build_discrete_hierarchy(finest_scale_indices)

        return QuantizationResult(
            quantized_latent=quantized_latents_by_scale[-1],
            indices_by_scale=indices_by_scale,
            latents_by_scale=latents_by_scale,
            quantized_latents_by_scale=quantized_latents_by_scale,
            commitment_losses_by_scale=torch.stack(commitment_losses_by_scale),
        )

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
        """Number of EMA codebook values omitted from model parameters."""
        return sum(
            codebook.codebook.numel()
            for codebook in self.codebooks
            if isinstance(codebook, EMACodebook)
        )
