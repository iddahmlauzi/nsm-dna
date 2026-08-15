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
        self.register_buffer("codebook_hits", torch.zeros(codebook_size, dtype=torch.bool))

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

        # Fine-scale dropout
        fine_dropout_prob: float = 0.5,
        min_scales_to_keep: int = 1,
    ) -> None:
        super().__init__()

        if len(scale_lengths) != len(codebook_sizes):
            raise ValueError("Each scale length must have one codebook size.")
        if not 1 <= min_scales_to_keep <= len(scale_lengths):
            raise ValueError(
                "min_scales_to_keep must be between 1 and the number of scales."
            )

        self.scale_lengths = scale_lengths
        self.codebook_sizes = codebook_sizes
        self.embed_dim = embed_dim
        self.commitment_cost = commitment_cost
        self.refinement_ratio = refinement_ratio
        self.refinement_kernel_size = refinement_kernel_size

        # During training, fine-scale dropout sometimes truncates the reconstruction
        # at a random scale after min_scales_to_keep. This forces the decoder to use
        # information carried by the coarse and intermediate scales.
        self.fine_dropout_prob = fine_dropout_prob
        self.min_scales_to_keep = min_scales_to_keep

        # Sampling at the first scale is learned.
        # A cascade can be added later for large strides.
        full_scale_length = scale_lengths[-1]
        first_scale_length = scale_lengths[0]
        if full_scale_length % first_scale_length != 0:
            raise ValueError("The full scale length must be divisible by the first scale length.")

        self.first_scale_stride = full_scale_length // first_scale_length
        self.first_scale_downsampler = _make_learned_downsampler(
            embed_dim,
            stride=self.first_scale_stride,
        )
        self.first_scale_upsampler = _make_learned_upsampler(
            embed_dim,
            stride=self.first_scale_stride,
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

        The first scale uses the learned strided convolution, intermediate scales
        use area interpolation, and the final scale is already full length.
        """
        scale_length = self.scale_lengths[scale_index]
        if scale_index == len(self.scale_lengths) - 1:
            return residual

        # Conv1d and one-dimensional interpolation expect channels first.
        residual = einx.id("b l d -> b d l", residual)
        if scale_index == 0:
            scaled_residual = self.first_scale_downsampler(residual)
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

        The first scale uses the learned transposed convolution, intermediate
        scales use linear interpolation, and the final scale is already full length.
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

    def forward(
        self,
        x: Float[Tensor, "batch length embed_dim"],
    ) -> tuple[
        Float[Tensor, "batch length embed_dim"],
        Float[Tensor, ""],
        list[Int[Tensor, "batch scale_length"]],
    ]:
        x = x.float()

        # Quantize the encoder output without backpropagating through the residual
        # hierarchy. The commitment loss and final STE provide encoder gradients.
        detached_x = x.detach()
        residual = detached_x.clone()
        reconstruction = torch.zeros_like(residual)

        # Decide whether this training step will drop fine-scale contributions.
        can_drop_fine_scales = self.min_scales_to_keep < len(self.scale_lengths)
        apply_fine_dropout = (
            self.training
            and can_drop_fine_scales
            and self.fine_dropout_prob > 0.0
            and torch.rand(()).item() < self.fine_dropout_prob
        )
        num_scales_to_keep = len(self.scale_lengths)
        if apply_fine_dropout:
            num_scales_to_keep = torch.randint(
                low=self.min_scales_to_keep,
                high=len(self.scale_lengths),
                size=(),
            ).item()

        vq_loss = x.new_zeros(())
        indices_by_scale: list[Int[Tensor, "batch scale_length"]] = []
        decoder_reconstruction: Tensor | None = None

        for scale_index, codebook in enumerate(self.codebooks):
            scaled_residual = self._downsample_to_scale(residual, scale_index)
            quantized_at_scale, scale_indices = codebook(scaled_residual)
            indices_by_scale.append(scale_indices)

            # Nearest-code selection blocks gradients to the learned downsampler.
            # At the first scale, use an STE whose forward value is the selected
            # code but whose backward path treats the lookup as an identity. The
            # codebook's EMA update is unaffected because it uses detached inputs.
            if scale_index == 0 and len(self.scale_lengths) > 1:
                quantized_at_scale = scaled_residual + (
                    quantized_at_scale - scaled_residual
                ).detach()

            # Upsample and refine this scale's quantized contribution.
            quantized_at_scale = einx.id("b l d -> b d l", quantized_at_scale)
            scale_contribution = self._upsample_to_full_length(
                quantized_at_scale,
                scale_index,
            )
            scale_contribution = self.refiners[scale_index](scale_contribution)
            scale_contribution = einx.id("b d l -> b l d", scale_contribution)

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

            # Save the reconstruction after the selected number of scales.
            if scale_index + 1 == num_scales_to_keep:
                decoder_reconstruction = reconstruction

        vq_loss = vq_loss / len(self.scale_lengths)
        assert decoder_reconstruction is not None

        # Give the decoder quantized values while passing its gradients to the encoder.
        quantized_latent = x + (decoder_reconstruction - x).detach()
        return quantized_latent, vq_loss, indices_by_scale

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
