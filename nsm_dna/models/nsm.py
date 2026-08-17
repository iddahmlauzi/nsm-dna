# This file is reserved for the second-stage Next-Scale Model.
#
# It will contain the transformer that predicts the discrete hierarchy learned
# by the VQ-VAE from coarse to fine. The first implementation will use teacher
# forcing and average cross-entropy within each scale so longer, finer scales do
# not dominate training.
#
# The model will receive a frozen VQ-VAE loaded from a checkpoint. The VQ-VAE
# architecture, codebook sizes, and level schedule must be recovered from that
# checkpoint rather than repeated in the NSM run configuration.
#
# Code corruption, scheduled sampling, generation caches, and length curricula
# are intentionally deferred until the basic coarse-to-fine model works.

