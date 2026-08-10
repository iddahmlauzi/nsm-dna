# Adkar et al. 2012 — CcdB

## Study

Adkar et al. measured the effects of single-residue substitutions in the 101-amino-acid *Escherichia coli* CcdB toxin. Mutant abundance after expression at several induction levels was used to identify residues required for CcdB folding and toxin activity.

## Benchmark mapping

- Evo study: CcdB
- ProteinGym assay: `CCDB_ECOLI_Adkar_2012`
- Target: CcdB
- Organism: *Escherichia coli*
- Original score: `RankScore`; lower values indicate greater CcdB activity
- ProteinGym raw directionality: `-1`

## Files

| File | Role and provenance |
|---|---|
| `CCDB_ECOLI_Adkar_2012.csv` | ProteinGym v1.3 substitution assay, extracted from [Zenodo record 15293562](https://zenodo.org/records/15293562). |
| `Adkar_2012_CcdB_original_results_mmc2.xlsx` | Original supplementary table from the publisher's [`mmc2.xls`](https://ars.els-cdn.com/content/image/1-s2.0-S0969212612000068-mmc2.xls), converted locally to `.xlsx` without intentional changes to the tabular values. |
| `NC_002483.1_ccdB_CDS.fasta` | CcdB coding sequence from NCBI accession [NC_002483.1](https://www.ncbi.nlm.nih.gov/nuccore/NC_002483.1), coordinates 46560–46865. |

## Sequence provenance

The publication and supplement do not identify the exact nucleotide sequence of the laboratory pBAD24-CcdB construct. The FASTA is therefore a reference-derived WT sequence, not an author-supplied construct sequence. It translates to the expected CcdB protein, and the WT residues and mutant codons reported in all 1,208 spreadsheet rows agree with this reference. Silent codons may still differ from the experimental construct.

## References

- Adkar BV et al. [Protein model discrimination using mutational sensitivity derived from deep sequencing](https://doi.org/10.1016/j.str.2011.11.021). *Structure* (2012).
- ProteinGym v1.3: [Zenodo record 15293562](https://zenodo.org/records/15293562).
