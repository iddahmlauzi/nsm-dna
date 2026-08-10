# Tsuboyama et al. 2023 — Multiprotein folding stability

## Study

Tsuboyama et al. used cDNA-display proteolysis to measure folding stability for hundreds of natural and designed protein domains. The Evo prokaryotic panel uses 19 ProteinGym assays derived from this study.

## Benchmark mapping

- Evo study: multiprotein thermostability
- ProteinGym assays: the 19 files matching `*_Tsuboyama_2023_*.csv` in this folder
- Targets: 19 prokaryotic protein domains
- Original score: change in modeled folding stability (`ddG_ML`)
- ProteinGym raw directionality: `1`

## Files

| File | Role and provenance |
|---|---|
| `Processed_K50_dG_datasets.zip` | Original processed author dataset from [Zenodo record 7844779](https://zenodo.org/records/7844779). It contains the nucleotide sequences, target amino-acid sequences, assay metadata, and modeled folding stabilities. |
| `*_Tsuboyama_2023_*.csv` | Nineteen ProteinGym v1.3 prokaryotic substitution assays extracted from [Zenodo record 15293562](https://zenodo.org/records/15293562). |

## Sequence provenance

The author archive contains `dna_seq`, `aa_seq_full`, and `aa_seq`. The full construct translation can contain residues outside the target domain. The converter locates `aa_seq` within `aa_seq_full` and slices the corresponding codons from `dna_seq`.

For each assay, the first author WT construct is the common WT reference. ProteinGym defines the protein-variant set, while the author archive supplies the exact nucleotide sequences and their corresponding `ddG_ML` scores. The 30,553 ProteinGym protein variants map to 31,761 unique, scored author nucleotide sequences because some protein variants were created with more than one codon background. The converter retains every background as a separate row and does not average their scores.

## References

- Tsuboyama K et al. [Mega-scale experimental analysis of protein folding stability in biology and design](https://doi.org/10.1038/s41586-023-06328-6). *Nature* (2023).
- Original processed data: [Zenodo record 7844779](https://zenodo.org/records/7844779).
- ProteinGym v1.3: [Zenodo record 15293562](https://zenodo.org/records/15293562).
