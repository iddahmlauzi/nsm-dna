import einx
import torch
import torch.distributed as dist
import torch.nn as nn


class EMACodebook(nn.Module):
    """Vector-quantization codebook updated with exponential moving averages."""

    def __init__(self, codebook_size, embed_dim, decay=0.99, eps=1e-5):
        super().__init__()

        self.codebook_size = codebook_size
        self.embed_dim = embed_dim
        self.decay = decay
        self.eps = eps

        codebook = torch.randn(codebook_size, embed_dim)
        self.register_buffer("codebook", codebook)
        self.register_buffer("ema_counts", torch.ones(codebook_size))
        self.register_buffer("ema_vector_sums", codebook.clone())
        self.register_buffer("codebook_hits", torch.zeros(codebook_size, dtype=torch.bool))

    def forward(self, x, valid_mask):
        flat_input = einx.id("b l d -> (b l) d", x.detach().float())
        flat_valid_mask = einx.id("b l -> (b l)", valid_mask).bool()

        # Exclude padding from codebook assignment and EMA updates.
        valid_input = flat_input[flat_valid_mask]

        # Compute the distance from each valid input to every codebook vector.
        distances = (
            torch.sum(valid_input**2, dim=1, keepdim=True)
            + torch.sum(self.codebook**2, dim=1)
            - 2 * einx.dot("n d, k d -> n k", valid_input, self.codebook)
        )
        valid_indices = distances.argmin(dim=-1)

        # Restore the original layout, leaving zero placeholders at padded positions.
        flat_indices = torch.zeros_like(flat_valid_mask, dtype=torch.long)
        flat_indices[flat_valid_mask] = valid_indices
        indices = einx.id("(b l) -> b l", flat_indices, b=x.shape[0])

        if self.training:
            batch_counts = torch.bincount(
                valid_indices,
                minlength=self.codebook_size,
            ).to(self.ema_counts.dtype)

            batch_vector_sums = torch.zeros_like(self.ema_vector_sums)
            batch_vector_sums.index_add_(0, valid_indices, valid_input)

            # Combine batch statistics so every DDP worker applies the same update.
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(batch_counts, op=dist.ReduceOp.SUM)
                dist.all_reduce(batch_vector_sums, op=dist.ReduceOp.SUM)

            with torch.no_grad():
                self.ema_counts.mul_(self.decay).add_(
                    batch_counts,
                    alpha=1 - self.decay,
                )
                self.ema_vector_sums.mul_(self.decay).add_(
                    batch_vector_sums,
                    alpha=1 - self.decay,
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

        # Look up selected codes and restore the original batch and sequence layout.
        flat_quantized = torch.zeros_like(flat_input)
        flat_quantized[flat_valid_mask] = self.codebook[valid_indices]
        quantized = einx.id("(b l) d -> b l d", flat_quantized, b=x.shape[0])

        return quantized.to(x.dtype), indices

    @property
    def utilization(self):
        return self.codebook_hits.float().mean()
