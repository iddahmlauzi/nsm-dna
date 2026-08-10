# Firnberg et al. 2014 — TEM-1 beta-lactamase

## Study

Firnberg et al. measured the fitness effects of point and codon mutations in the *Escherichia coli* TEM-1 beta-lactamase gene. Variant fitness was inferred from bacterial growth across ampicillin concentrations.

## Benchmark mapping

- Evo study: Firnberg beta-lactamase
- ProteinGym assay: `BLAT_ECOLX_Firnberg_2014`
- Target: TEM-1 beta-lactamase
- Organism: *Escherichia coli*
- ProteinGym raw directionality: `1`

## Files

| File | Role and provenance |
|---|---|
| `BLAT_ECOLX_Firnberg_2014.csv` | ProteinGym v1.3 substitution assay, extracted from [Zenodo record 15293562](https://zenodo.org/records/15293562). |
| `supp_msu081_Data_S1-S4.xlsx` | Original supplementary workbook associated with the study. Sheet `S1 Codon fitnesses` contains the nucleotide-level measurements. |

## Sequence provenance

The `WT_codon` column in `S1 Codon fitnesses` explicitly identifies the WT codon at every position. The study converter will concatenate these values in position order to recover the 861-nucleotide WT coding sequence, including the terminal stop codon. The recovered sequence exactly matches the TEM-1 coding sequence in NCBI accession [J01749.1](https://www.ncbi.nlm.nih.gov/nuccore/J01749.1). ProteinGym uses the complete 286-amino-acid precursor, including the 23-residue signal peptide.

## References

- Firnberg E et al. [A comprehensive, high-resolution map of a gene's fitness landscape](https://doi.org/10.1093/molbev/msu081). *Molecular Biology and Evolution* (2014).
- ProteinGym v1.3: [Zenodo record 15293562](https://zenodo.org/records/15293562).
