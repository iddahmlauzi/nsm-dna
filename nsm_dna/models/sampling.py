import einx
import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor


def _make_learned_downsampler(
    embed_dim: int,
    stride: int,
    bias: bool,
) -> nn.Conv1d:
    """Create a learned downsampler initialized as average pooling."""
    downsampler = nn.Conv1d(
        embed_dim,
        embed_dim,
        kernel_size=stride,
        stride=stride,
        bias=bias,
    )
    average_weight = 1.0 / stride

    with torch.no_grad():
        downsampler.weight.zero_()
        channel_indices = torch.arange(embed_dim, device=downsampler.weight.device)
        downsampler.weight[channel_indices, channel_indices, :] = average_weight
        if downsampler.bias is not None:
            downsampler.bias.zero_()

    return downsampler


def _make_learned_upsampler(
    embed_dim: int,
    stride: int,
    bias: bool,
) -> nn.ConvTranspose1d:
    """Create a learned upsampler initialized as nearest-neighbor repetition."""
    upsampler = nn.ConvTranspose1d(
        embed_dim,
        embed_dim,
        kernel_size=stride,
        stride=stride,
        bias=bias,
    )

    with torch.no_grad():
        upsampler.weight.zero_()
        channel_indices = torch.arange(embed_dim, device=upsampler.weight.device)
        upsampler.weight[channel_indices, channel_indices, :] = 1.0
        if upsampler.bias is not None:
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
    base_stride: int,
) -> list[int]:
    """Factor a large sampling stride into a sequence of smaller strides."""
    remaining_stride = total_stride
    cascade_strides = []

    while remaining_stride > 1:
        stage_stride = min(base_stride, remaining_stride)

        # Choose a factor of the remaining stride so the cascade reduces the
        # sequence to exactly the requested length without a partial final stage.
        while stage_stride > 1 and remaining_stride % stage_stride != 0:
            stage_stride -= 1

        if stage_stride == 1:
            stage_stride = remaining_stride

        cascade_strides.append(stage_stride)
        remaining_stride //= stage_stride

    return cascade_strides


def make_cascaded_downsampler(
    embed_dim: int,
    total_stride: int,
    *,
    base_stride: int = 4,
    bias: bool = True,
) -> nn.Sequential:
    """Create a large downsampler from normalized small-stride stages."""
    strides = _plan_cascade_strides(total_stride, base_stride)
    modules: list[nn.Module] = []

    for stride in strides:
        modules.append(_make_learned_downsampler(embed_dim, stride, bias))
        # Interstage norms prevent numerical gain from compounding. The final norm
        # fixes the scale presented to the next component.
        modules.append(ChannelsFirstLayerNorm(embed_dim))

    return nn.Sequential(*modules)


def make_cascaded_upsampler(
    embed_dim: int,
    total_stride: int,
    *,
    base_stride: int = 4,
    bias: bool = True,
) -> nn.Sequential:
    """Reverse a cascaded downsampler with learned transposed convolutions."""
    return nn.Sequential(
        *(
            _make_learned_upsampler(embed_dim, stride, bias)
            for stride in reversed(_plan_cascade_strides(total_stride, base_stride))
        )
    )
