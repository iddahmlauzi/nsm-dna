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
        embed_dim: int,
        decay: float = 0.99,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()

        self.codebook_size = codebook_size
        self.embed_dim = embed_dim
        self.base_decay = decay
        self.eps = eps

        codebook = torch.randn(codebook_size, embed_dim)
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
        x: Float[Tensor, "batch length embed_dim"],
    ) -> tuple[
        Float[Tensor, "batch length embed_dim"],
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


def _make_learned_downsampler(
    embed_dim: int,
    stride: int,
) -> nn.Conv1d:
    """Create a learned downsampler initialized as average pooling."""
    downsampler = nn.Conv1d(
        embed_dim,
        embed_dim,
        kernel_size=stride,
        stride=stride,
    )
    average_weight = 1.0 / stride

    with torch.no_grad():
        downsampler.weight.zero_()
        channel_indices = torch.arange(embed_dim, device=downsampler.weight.device)
        downsampler.weight[channel_indices, channel_indices, :] = average_weight
        downsampler.bias.zero_()

    return downsampler


def _make_learned_upsampler(
    embed_dim: int,
    stride: int,
) -> nn.ConvTranspose1d:
    """Create a learned upsampler initialized as nearest-neighbor repetition.

    Because the kernel size equals the stride, each coarse position is expanded
    independently. BlendedConv1d later mixes information between adjacent positions.
    """
    upsampler = nn.ConvTranspose1d(
        embed_dim,
        embed_dim,
        kernel_size=stride,
        stride=stride,
    )

    with torch.no_grad():
        upsampler.weight.zero_()
        channel_indices = torch.arange(embed_dim, device=upsampler.weight.device)
        upsampler.weight[channel_indices, channel_indices, :] = 1.0
        upsampler.bias.zero_()

    return upsampler


class ChannelsFirstLayerNorm(nn.Module):
    """Apply non-affine LayerNorm to channels-first sequence features."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(embed_dim, elementwise_affine=False)

    def forward(
        self,
        x: Float[Tensor, "batch embed_dim length"],
    ) -> Float[Tensor, "batch embed_dim length"]:
        x = einx.id("b d l -> b l d", x)
        x = self.normalization(x)
        return einx.id("b l d -> b d l", x)


def _plan_cascade_strides(
    total_stride: int,
    base_stride: int = 4,
) -> list[int]:
    """Factor a large sampling stride into a sequence of smaller strides."""
    remaining_stride = total_stride
    cascade_strides = []

    while remaining_stride > 1:
        stage_stride = min(base_stride, remaining_stride)
        while stage_stride > 1 and remaining_stride % stage_stride != 0:
            stage_stride -= 1

        if stage_stride == 1:
            stage_stride = remaining_stride

        cascade_strides.append(stage_stride)
        remaining_stride //= stage_stride

    return cascade_strides or [1]


def _make_cascaded_downsampler(
    embed_dim: int,
    total_stride: int,
) -> nn.Sequential:
    """Create normalized small-stride stages for a large downsampling operation."""
    strides = _plan_cascade_strides(total_stride)
    modules: list[nn.Module] = []

    for stage_index, stride in enumerate(strides):
        modules.append(_make_learned_downsampler(embed_dim, stride))

        # Normalize before the next learned stage so numerical gain cannot compound
        # through the cascade. The final stage is normalized by first_scale_norm
        # immediately before codebook lookup.
        if stage_index < len(strides) - 1:
            modules.append(ChannelsFirstLayerNorm(embed_dim))

    return nn.Sequential(*modules)


def _make_cascaded_upsampler(
    embed_dim: int,
    total_stride: int,
) -> nn.Sequential:
    """Reverse a cascaded downsampler with learned transposed convolutions."""
    return nn.Sequential(
        *(
            _make_learned_upsampler(embed_dim, stride)
            for stride in reversed(_plan_cascade_strides(total_stride))
        )
    )


# ----------------------------------------------------------------------
# Per-scale blended convolution
#
# Each codebook produces quantized vectors at its scale. These vectors are
# upsampled to the full latent length when necessary; the final scale is already
# full length.
#
# BlendedConv1d lets neighboring quantized positions interact before this scale's
# contribution is added to the reconstruction. This can correct local artifacts
# introduced by quantization and upsampling.
#
# refined = (1 - ratio) * quantized + ratio * Conv1d(quantized)
#
# Each scale has its own trainable convolution. The fixed refinement ratio
# defaults to 0.5, the empirically best configuration in the NCM ablations.
# ----------------------------------------------------------------------
class BlendedConv1d(nn.Module):
    """Blend an input with a learned one-dimensional convolution."""

    def __init__(
        self,
        embed_dim: int,
        refinement_ratio: float = 0.5,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve sequence length.")

        self.refinement_ratio = refinement_ratio
        self.conv = nn.Conv1d(
            embed_dim,
            embed_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

    def forward(
        self,
        x: Float[Tensor, "batch embed_dim length"],
    ) -> Float[Tensor, "batch embed_dim length"]:
        # Blend the original contribution with its locally refined form.
        convolved = self.conv(x)
        refined = x * (1 - self.refinement_ratio) + convolved * self.refinement_ratio
        return refined


class MultiscaleResidualVectorQuantizer(nn.Module):
    """Multiscale residual vector quantizer."""

    def __init__(
        self,
        scale_lengths: list[int],
        codebook_sizes: list[int],
        embed_dim: int,
        *,
        # Codebook updates and quantization loss
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        eps: float = 1e-5,
        # Per-scale post-quantization refinement
        refinement_ratio: float = 0.5,
        refinement_kernel_size: int = 3,
    ) -> None:
        super().__init__()

        if len(scale_lengths) != len(codebook_sizes):
            raise ValueError("Each scale length must have one codebook size.")

        self.scale_lengths = scale_lengths
        self.codebook_sizes = codebook_sizes
        self.embed_dim = embed_dim
        self.commitment_cost = commitment_cost
        self.refinement_ratio = refinement_ratio
        self.refinement_kernel_size = refinement_kernel_size

        full_scale_length = scale_lengths[-1]
        first_scale_length = scale_lengths[0]
        if full_scale_length % first_scale_length != 0:
            raise ValueError(
                "The full scale length must be divisible by the first scale length."
            )

        # Factor the large first-scale stride into small learned stages so its
        # parameter count grows with the number of stages, not the full kernel.
        first_scale_stride = full_scale_length // first_scale_length
        self.first_scale_downsampler = _make_cascaded_downsampler(
            embed_dim,
            total_stride=first_scale_stride,
        )
        self.first_scale_norm = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.first_scale_upsampler = _make_cascaded_upsampler(
            embed_dim,
            total_stride=first_scale_stride,
        )

        self.codebooks = nn.ModuleList(
            EMACodebook(codebook_size, embed_dim, decay=decay, eps=eps)
            for codebook_size in codebook_sizes
        )
        self.refiners = nn.ModuleList(
            BlendedConv1d(
                embed_dim,
                refinement_ratio=refinement_ratio,
                kernel_size=refinement_kernel_size,
            )
            for _ in scale_lengths
        )

    def _downsample_to_scale(
        self,
        residual: Float[Tensor, "batch length embed_dim"],
        scale_index: int,
    ) -> Float[Tensor, "batch scale_length embed_dim"]:
        """Downsample the residual to the selected scale's sequence length.

        The first scale uses cascaded learned strided convolutions. Intermediate
        scales use area interpolation, and the final scale is already full length.
        """
        scale_length = self.scale_lengths[scale_index]
        if scale_index == len(self.scale_lengths) - 1:
            return residual

        # Conv1d and one-dimensional interpolation expect channels first.
        residual = einx.id("b l d -> b d l", residual)
        if scale_index == 0:
            scaled_residual = self.first_scale_downsampler(residual)
            scaled_residual = einx.id("b d l -> b l d", scaled_residual)

            # The encoder was normalized before quantization, but the learned
            # convolution can increase its magnitude again. Normalize each coarse
            # vector before codebook lookup so the codebook and refiner do not chase
            # and then cancel increasingly large values. Disabling LayerNorm's
            # affine transform prevents it from learning that scale back.
            return self.first_scale_norm(scaled_residual)
        else:
            scaled_residual = F.interpolate(
                residual,
                size=scale_length,
                mode="area",
            )

        return einx.id("b d l -> b l d", scaled_residual)

    def _upsample_to_full_length(
        self,
        quantized: Float[Tensor, "batch embed_dim scale_length"],
        scale_index: int,
    ) -> Float[Tensor, "batch embed_dim length"]:
        """Upsample a quantized contribution to the full latent length.

        The first scale uses a learned transposed convolution. Intermediate scales
        use linear interpolation, and the final scale is already full length.
        The channels-first layout can pass directly into BlendedConv1d afterward.
        """
        if scale_index == len(self.scale_lengths) - 1:
            return quantized

        if scale_index == 0:
            return self.first_scale_upsampler(quantized)

        return F.interpolate(
            quantized,
            size=self.scale_lengths[-1],
            mode="linear",
            align_corners=False,
        )

    def _prepare_scale_contribution(
        self,
        quantized_at_scale: Float[Tensor, "batch scale_length embed_dim"],
        scale_index: int,
    ) -> Float[Tensor, "batch length embed_dim"]:
        """Upsample and refine one scale's quantized vectors."""
        quantized_at_scale = einx.id("b l d -> b d l", quantized_at_scale)
        scale_contribution = self._upsample_to_full_length(
            quantized_at_scale,
            scale_index,
        )
        scale_contribution = self.refiners[scale_index](scale_contribution)
        return einx.id("b d l -> b l d", scale_contribution)

    def forward(
        self,
        x: Float[Tensor, "batch length embed_dim"],
        *,
        include_partial_reconstruction: bool = False,
    ) -> tuple[
        Float[Tensor, "batch length embed_dim"],
        Float[Tensor, "batch length embed_dim"] | None,
        Float[Tensor, ""],
        list[Int[Tensor, "batch scale_length"]],
    ]:
        x = x.float()

        # Quantize the encoder output without backpropagating through the residual
        # hierarchy. The commitment loss and final STE provide encoder gradients.
        detached_x = x.detach()
        residual = detached_x.clone()
        reconstruction = torch.zeros_like(residual)

        partial_scale_index = None
        if include_partial_reconstruction:
            if len(self.scale_lengths) < 2:
                raise ValueError(
                    "Partial reconstruction requires at least two quantization scales."
                )
            partial_scale_index = torch.randint(
                low=0,
                high=len(self.scale_lengths) - 1,
                size=(),
            ).item()

        vq_loss = x.new_zeros(())
        indices_by_scale: list[Int[Tensor, "batch scale_length"]] = []
        partial_reconstruction: Tensor | None = None

        for scale_index, codebook in enumerate(self.codebooks):
            scaled_residual = self._downsample_to_scale(residual, scale_index)
            quantized_at_scale, scale_indices = codebook(scaled_residual)
            indices_by_scale.append(scale_indices)

            # Nearest-code selection blocks gradients to the learned downsampler.
            # This preserves the selected code in the forward pass while treating
            # the lookup as an identity when calculating downsampler gradients.
            if scale_index == 0 and len(self.scale_lengths) > 1:
                quantized_at_scale = (
                    scaled_residual + (quantized_at_scale - scaled_residual).detach()
                )

            scale_contribution = self._prepare_scale_contribution(
                quantized_at_scale,
                scale_index,
            )

            reconstruction = reconstruction + scale_contribution
            residual = residual - scale_contribution

            # Pull the encoder toward the current quantized reconstruction.
            encoder_commitment_loss = self.commitment_cost * F.mse_loss(
                reconstruction.detach(),
                x,
            )

            # Train the learned samplers and refiners against a fixed encoder target.
            # The codebooks themselves are updated separately through EMA.
            quantizer_reconstruction_loss = F.mse_loss(
                reconstruction,
                detached_x,
            )
            vq_loss = vq_loss + encoder_commitment_loss + quantizer_reconstruction_loss

            if scale_index == partial_scale_index:
                partial_reconstruction = reconstruction

        vq_loss = vq_loss / len(self.scale_lengths)

        # Give the decoder quantized values while passing its gradients to the encoder.
        quantized_latent = x + (reconstruction - x).detach()

        # The auxiliary partial loss follows the quantized reconstruction directly.
        # Its gradients train the learned samplers and refiners without moving the
        # shared encoder target. The main full-reconstruction path above retains the
        # straight-through encoder gradient.
        return quantized_latent, partial_reconstruction, vq_loss, indices_by_scale

    @torch.no_grad()
    def indices_to_cumulative_latents(
        self,
        indices_by_scale: list[Int[Tensor, "batch scale_length"]],
    ) -> list[Float[Tensor, "batch length embed_dim"]]:
        """Reconstruct the latent after successively adding each scale."""
        batch_size = indices_by_scale[0].shape[0]
        full_length = self.scale_lengths[-1]
        reconstruction = self.codebooks[0].codebook.new_zeros(
            batch_size,
            full_length,
            self.embed_dim,
        )
        cumulative_latents = []

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
