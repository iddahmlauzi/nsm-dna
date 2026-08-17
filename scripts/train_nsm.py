# This script is reserved for second-stage NSM training.
#
# Planned responsibilities:
# 1. Load one plain YAML run configuration.
# 2. Load the selected VQ-VAE checkpoint and reconstruct its architecture from
#    the configuration stored inside that checkpoint.
# 3. Freeze the VQ-VAE, including its EMA codebooks and scale refiners.
# 4. Encode DNA batches into one target tensor per scale during training.
# 5. Train the NSM transformer with scale-balanced, teacher-forced loss.
# 6. Reuse the established checkpoint and logging behavior from VQ-VAE training
#    once that shared behavior exists in both scripts.
#
# This file should remain a separate runnable workflow. Shared helpers should
# not be extracted merely in anticipation of duplication.

