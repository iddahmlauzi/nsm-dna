from dataclasses import dataclass

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor

from .models.next_scale import NSMDNAOutput


@dataclass(frozen=True)
class NSMDNALosses:
    """Joint training objectives and per-scale prediction values."""

    total: Float[Tensor, ""]
    nucleotide_reconstruction: Float[Tensor, ""]
    partial_reconstruction: Float[Tensor, ""]
    teacher_forced_prediction_reconstruction: Float[Tensor, ""]
    vq: Float[Tensor, ""]
    teacher_forced_prediction: Float[Tensor, ""]
    teacher_forced_prediction_by_scale: list[Float[Tensor, ""]]
    entropy: Float[Tensor, ""]


def nucleotide_reconstruction_loss(
    logits: Float[Tensor, "batch target_length vocab_size"],
    target_ids: Int[Tensor, "batch target_length"],
) -> Float[Tensor, ""]:
    """Average nucleotide cross-entropy over the target block."""
    return F.cross_entropy(logits.flatten(0, 1), target_ids.flatten())


def next_scale_prediction_loss(
    logits_by_scale: list[Float[Tensor, "batch scale_length codebook_size"]],
    targets_by_scale: list[Int[Tensor, "batch scale_length"]],
) -> tuple[Float[Tensor, ""], list[Float[Tensor, ""]]]:
    """Average token cross-entropy within each scale, then across scales."""
    losses_by_scale = [
        F.cross_entropy(scale_logits.flatten(0, 1), scale_targets.flatten())
        for scale_logits, scale_targets in zip(
            logits_by_scale,
            targets_by_scale,
            strict=True,
        )
    ]
    return torch.stack(losses_by_scale).mean(), losses_by_scale


def codebook_entropy_loss(
    assignment_logits_by_scale: list[Float[Tensor, "batch scale_length codebook_size"]],
    temperature: float,
) -> Float[Tensor, ""]:
    """Favor confident assignments and diverse aggregate code use at every scale."""
    if temperature <= 0:
        raise ValueError("entropy temperature must be positive.")

    losses_by_scale = []
    for assignment_logits in assignment_logits_by_scale:
        # The entropy objective gets its own softer distribution because squared
        # distances across a wide latent can saturate the tokenizer's temperature-1
        # probabilities. This does not change hard codes or their EMA updates.
        probabilities = torch.softmax(
            assignment_logits.float() / temperature,
            dim=-1,
        )
        flat_probabilities = probabilities.flatten(0, 1)
        log_probabilities = torch.log(flat_probabilities.clamp_min(1e-10))
        mean_assignment_entropy = (
            -(flat_probabilities * log_probabilities).sum(dim=-1).mean()
        )

        marginal_probabilities = flat_probabilities.mean(dim=0)
        marginal_entropy = -(
            marginal_probabilities * torch.log(marginal_probabilities.clamp_min(1e-10))
        ).sum()

        # Minimizing the difference keeps each individual assignment sharp while
        # maximizing the entropy of the assignments aggregated across the batch.
        losses_by_scale.append(mean_assignment_entropy - marginal_entropy)

    return torch.stack(losses_by_scale).mean()


def nsm_dna_losses(
    output: NSMDNAOutput,
    target_ids: Int[Tensor, "batch target_length"],
    *,
    partial_reconstruction_loss_weight: float = 0.0,
    teacher_forced_prediction_reconstruction_loss_weight: float = 0.0,
    next_scale_prediction_loss_weight: float = 1.0,
    entropy_loss_weight: float = 1.0,
    entropy_temperature: float = 1.0,
) -> NSMDNALosses:
    """Calculate reconstruction, VQ, prediction, and entropy objectives."""
    nucleotide_reconstruction = nucleotide_reconstruction_loss(
        output.reconstruction_logits,
        target_ids,
    )
    if partial_reconstruction_loss_weight == 0:
        partial_reconstruction = nucleotide_reconstruction.new_zeros(())
    elif output.partial_reconstruction_logits is not None:
        # Training samples one non-final cumulative latent so this auxiliary
        # objective costs only one additional decoder pass per batch.
        partial_reconstruction = nucleotide_reconstruction_loss(
            output.partial_reconstruction_logits,
            target_ids,
        )
    else:
        cumulative_logits = output.cumulative_reconstruction_logits_by_scale
        if cumulative_logits is None or len(cumulative_logits) < 2:
            raise ValueError(
                "Partial or cumulative reconstruction logits are required when "
                "partial reconstruction has nonzero weight."
            )
        # Validation already decodes every cumulative latent. Their non-final
        # mean is the exact expectation of uniform scale sampling in training.
        partial_reconstruction = torch.stack(
            [
                nucleotide_reconstruction_loss(scale_logits, target_ids)
                for scale_logits in cumulative_logits[:-1]
            ]
        ).mean()
    if teacher_forced_prediction_reconstruction_loss_weight == 0:
        teacher_forced_prediction_reconstruction = nucleotide_reconstruction.new_zeros(
            ()
        )
    else:
        if output.teacher_forced_prediction_reconstruction_logits is None:
            raise ValueError(
                "Teacher-forced prediction reconstruction logits are required when "
                "their loss has nonzero weight."
            )
        teacher_forced_prediction_reconstruction = nucleotide_reconstruction_loss(
            output.teacher_forced_prediction_reconstruction_logits,
            target_ids,
        )
    teacher_forced_prediction, teacher_forced_prediction_by_scale = (
        next_scale_prediction_loss(
            output.next_scale_logits_by_scale,
            output.quantizer.indices_by_scale[1:],
        )
    )
    entropy = codebook_entropy_loss(
        output.quantizer.assignment_logits_by_scale,
        temperature=entropy_temperature,
    )
    vq = output.quantizer.vq_loss
    return NSMDNALosses(
        total=(
            nucleotide_reconstruction
            + partial_reconstruction_loss_weight * partial_reconstruction
            + teacher_forced_prediction_reconstruction_loss_weight
            * teacher_forced_prediction_reconstruction
            + vq
            + next_scale_prediction_loss_weight * teacher_forced_prediction
            + entropy_loss_weight * entropy
        ),
        nucleotide_reconstruction=nucleotide_reconstruction,
        partial_reconstruction=partial_reconstruction,
        teacher_forced_prediction_reconstruction=(
            teacher_forced_prediction_reconstruction
        ),
        vq=vq,
        teacher_forced_prediction=teacher_forced_prediction,
        teacher_forced_prediction_by_scale=teacher_forced_prediction_by_scale,
        entropy=entropy,
    )
