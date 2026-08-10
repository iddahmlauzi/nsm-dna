# Weeks et al. 2023 — RNase III

## Study

Weeks et al. measured fitness and RNA-cleavage phenotypes for variants of the *Escherichia coli* `rnc` gene, which encodes RNase III. ProteinGym uses the study's aggregate functional score.

## Benchmark mapping

- Evo study: Rnc/RNase III
- ProteinGym assay: `RNC_ECOLI_Weeks_2023`
- Target: RNase III (`rnc`)
- Organism: *Escherichia coli*
- ProteinGym phenotype: `Functional Score Weighted Mean`
- ProteinGym raw directionality: `1`

## Files

| File | Role and provenance |
|---|---|
| `RNC_ECOLI_Weeks_2023.csv` | ProteinGym v1.3 substitution assay, extracted from [Zenodo record 15293562](https://zenodo.org/records/15293562). |
| `Supplementary Data S8.csv` | Original authors' codon-level functional-score table from the [journal supplementary archive](https://oup.silverchair-cdn.com/oup/backfile/Content_public/Journal/mbe/40/3/10.1093_molbev_msad047/1/msad047_supplementary_data.zip). It reports both experimental replicas and their weighted mean for every codon substitution. |

## Sequence provenance

`Supplementary Data S8.csv` covers all 64 codons at each of the 226 RNase III positions. The table gives the WT amino acid at every position and leaves the WT codon's weighted-mean score blank. At each position, exactly one synonymous codon has this pattern, so the 678-nt WT coding sequence can be recovered without choosing among multiple candidates. Its translation exactly matches the WT amino-acid sequence in the authors' table. The table does not include a terminal stop codon.

The same table provides the nucleotide identity, amino-acid consequence, and experimental score for each single-codon variant, including stop-codon variants. The WT coding sequence will be derived from this primary table by the study converter rather than stored as a separate reconstructed source file.

## References

- Weeks KM et al. [Fitness and functional landscapes of the *E. coli* RNase III gene `rnc`](https://doi.org/10.1093/molbev/msad047). *Molecular Biology and Evolution* (2023).
- Authors' deposited experimental plasmid: [pRnc2-gGFP, Addgene #189563](https://www.addgene.org/189563/).
- ProteinGym v1.3: [Zenodo record 15293562](https://zenodo.org/records/15293562).
