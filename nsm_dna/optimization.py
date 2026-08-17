import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def build_learning_rate_scheduler(
    optimizer: Optimizer,
    warmup_steps: int,
    decay_end_step: int,
    learning_rate: float,
    min_learning_rate: float,
) -> LambdaLR:
    """Create a linear-warmup, cosine-decay learning-rate schedule.

    The learning rate increases to `learning_rate` over the warmup, follows a
    cosine curve down to `min_learning_rate`, and remains there after
    `decay_end_step`.
    """
    min_learning_rate_factor = min_learning_rate / learning_rate

    def learning_rate_factor(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / (warmup_steps + 1)

        if step >= decay_end_step:
            return min_learning_rate_factor

        decay_progress = (step - warmup_steps) / max(
            1,
            decay_end_step - warmup_steps,
        )
        cosine_factor = 0.5 * (1 + math.cos(math.pi * decay_progress))
        return min_learning_rate_factor + cosine_factor * (
            1 - min_learning_rate_factor
        )

    return LambdaLR(optimizer, learning_rate_factor)
