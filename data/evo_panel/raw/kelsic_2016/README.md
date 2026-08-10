# Kelsic et al. 2016 — Translation initiation factor IF-1

## Study

Kelsic et al. used MAGE-seq to measure the fitness effects of codon substitutions in the essential *Escherichia coli* `infA` gene, which encodes translation initiation factor IF-1. The study examined how codon identity and local messenger-RNA structure influence fitness.

## Benchmark mapping

- Evo study: IF-1
- ProteinGym assay: `IF1_ECOLI_Kelsic_2016`
- Target: translation initiation factor IF-1 (`infA`)
- Organism: *Escherichia coli*
- ProteinGym phenotype: `fitness_rich`
- ProteinGym raw directionality: `1`

## Files

| File | Role and provenance |
|---|---|
| `IF1_ECOLI_Kelsic_2016.csv` | ProteinGym v1.3 substitution assay, extracted from [Zenodo record 15293562](https://zenodo.org/records/15293562). |
| `NIHMS836960-supplement-1.csv` | Original author Data S1 from [PubMed Central](https://pmc.ncbi.nlm.nih.gov/articles/instance/5234859/bin/NIHMS836960-supplement-1.csv). It reports all 64 codons at each of 73 `infA` positions, the WT-codon indicator, and fitness in minimal and rich media. |

## Sequence provenance

Data S1 contains exactly one row with `is_wt = 1` at each of the 73 codon positions. Concatenating those codons in position order reconstructs the 219-nucleotide experimental WT `infA` coding sequence directly from the author data. Each other row identifies a complete mutant codon and its measured fitness, so no external nucleotide reference is required.

## References

- Kelsic ED et al. [RNA structural determinants of optimal codons revealed by MAGE-seq](https://doi.org/10.1016/j.cels.2016.11.004). *Cell Systems* (2016).
- [Open-access article](https://pmc.ncbi.nlm.nih.gov/articles/PMC5234859/).
- ProteinGym v1.3: [Zenodo record 15293562](https://zenodo.org/records/15293562).
