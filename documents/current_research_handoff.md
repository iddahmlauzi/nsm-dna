# NSM-DNA research handoff

## Current objective

Improve the middle layers of the dinucleotide tokenizer hierarchy. The goal is
not merely good DNA reconstruction. A useful hierarchy should satisfy both of
these properties:

1. The coarsest code is a global summary of the 256-base region.
2. Knowing a parent code substantially narrows the plausible codes for its two
   children, while the children still add finer information.

The original dinucleotide tokenizer has this property weakly and asymmetrically
in its middle layers. This is the current problem to solve.

## Current repository state

- Local branch: `nsm-hierarchy-redesign`
- Restored base commit: `e4968b0014c8`
- The branch deliberately excludes the later NSM attention ablations, student
  forcing, tokenizer child-prediction heads, and scale-64 initialization work.
- The retained NSM was launched from the restored code with uncommitted
  configuration changes. `configs/nsm.yaml` now records the exact resolved
  values captured by that W&B run.
- The Mac repository is the source of truth. `/workspace/nsm-dna` on Vast is an
  rsynced working copy without Git metadata and may contain newer experimental
  source than the local checkout.

## Original dinucleotide tokenizer

- Run: `vqvae-256-dinucleotide`
- W&B: [nsm-dna/vqvae/k9xwpbs4](https://wandb.ai/nsm-dna/vqvae/runs/k9xwpbs4)
- Source: `nsm-hierarchy-redesign` at `6b2e2d7`
- Checkpoint used by the NSM experiments:
  `/workspace/runs/vqvae-256-dinucleotide/checkpoints/best.pt`
- Scales: `[1, 2, 4, 8, 16, 32, 64, 128]`
- Codebook sizes: `[32, 32, 64, 64, 128, 128, 128, 16]`
- Scale 128 represents exact dinucleotides.
- The hierarchy uses learned kernel-2, stride-2 downsampling and independently
  quantized absolute scales, with partial reconstruction weight `0.25`.

## Original NSM trained on this tokenizer

- Run: `nsm-256-dinucleotide-attn-prefix-all-prev-scales`
- W&B: [nsm-dna/nsm/fmm4dqed](https://wandb.ai/nsm-dna/nsm/runs/fmm4dqed)
- Conditioning: full prefix plus all earlier generated scales
- Geometry loss weight: `0.5`
- Nearest-neighbor context corruption probabilities:
  `[0.25, 0.25, 0.25, 0.30, 0.20, 0.075, 0.05]`
- Validation at step 19,000 reached 55.85% aggregate teacher-forced code
  accuracy and 33.84% sequential-rollout nucleotide accuracy.

The detailed configuration, per-scale accuracies, and oracle-scale rollout
results are in [00_09_24_26.md](progress/00_09_24_26.md).

## What has been learned

Several tokenizer redesigns tried to make child codes easier to predict:

- Child-prediction losses made prediction easier partly by collapsing code
  usage. Detaching the child targets prevented one direct shortcut but did not
  solve the hierarchy design problem.
- Initializing scale 64 to group 4-mers by their first three bases increased its
  use of the codebook but made the following scale harder to predict. The
  initialization was removed.
- A left-grouped hierarchy made each parent preserve one child and produced
  roughly 50% child accuracy, but its coarse scales stopped behaving as useful
  global summaries. It was abandoned.
- These failures showed that predictability alone is insufficient. The parent
  must summarize both children while genuinely restricting their possibilities.

An analysis of the original tokenizer measured how much knowing a parent reduces
held-out uncertainty about one child. It used 462 validation genomes to estimate
the frequency tables and 154 disjoint genomes for evaluation.

| Parent to child | Uncertainty resolved | Effective child choices after parent | Parent-only top-1 accuracy |
|---|---:|---:|---:|
| 1 to 2 | 35.8% | 9.2 | 35.19% |
| 2 to 4 | 21.6% | 26.0 | 13.78% |
| 4 to 8 | 19.9% | 27.9 | 13.90% |
| 8 to 16 | 22.0% | 43.8 | 9.52% |
| 16 to 32 | 29.5% | 30.5 | 12.34% |
| 32 to 64 | 40.1% | 13.0 | 20.83% |
| 64 to 128 | 76.5% | 1.9 | 65.47% |

The middle hierarchy is therefore a real weakness. The parent resolves only
about 20% to 30% of child uncertainty from scales 2 through 32. The 8 to 16
transition is the hardest after conditioning, with about 44 effective child
choices remaining. These transitions are also asymmetric: the right child is
substantially more constrained than the left child. A plausible cause is the
unconstrained stride-2 convolution learning a side-specific shortcut, but that
mechanism has not yet been directly established.

The durable analysis result is:

```text
/workspace/results/vqvae-256-dinucleotide/hierarchy-dependence/results-step-38000.json
```

The next investigation should focus on why the middle downsamplers preserve the
right child more strongly and how to require balanced, useful summaries of both
children without encouraging codebook collapse.

## Vast navigation

Read the instance guide before acting:

```bash
ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=120 vast \
  'cat /etc/vast-agents-guide.md'
```

Important locations:

```text
/workspace/nsm-dna       rsynced source tree, not a Git checkout
/workspace/runs          training runs and checkpoints
/workspace/results       durable evaluation and analysis outputs
/workspace/run-launchers detached training logs and launcher PIDs
```

Use SSH keepalives for every connection. Put one-off analysis scripts under
`/tmp`, place durable outputs under `/workspace/results/<run-name>/`, and remove
the temporary script afterward. Before launching a run, inspect the resolved
configuration and verify that no training process is using the GPUs. Full runs
must use `nohup` and `setsid`; do not use `tmux`.

As of this handoff, the three run directories intentionally retained on Vast are:

```text
/workspace/runs/next-token-512-gpt2-small-500m-5epochs
/workspace/runs/vqvae-256-dinucleotide
/workspace/runs/nsm-256-dinucleotide-attn-prefix-all-prev-scales
```
