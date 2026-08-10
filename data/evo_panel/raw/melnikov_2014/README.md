# Melnikov et al. 2014 — APH(3′)II

## Study

Melnikov et al. measured single-amino-acid substitutions in APH(3′)II under six aminoglycoside antibiotics and multiple concentrations. The ProteinGym assay represents selection with kanamycin and measures antibiotic-resistance fitness.

## Benchmark mapping

- Evo study: APH(3′)II
- ProteinGym assay: `KKA2_KLEPN_Melnikov_2014`
- Target: aminoglycoside phosphotransferase APH(3′)II
- Organism: *Klebsiella pneumoniae*
- ProteinGym phenotype: `Kan18_avg`
- ProteinGym raw directionality: `1`

## Files

| File | Role and provenance |
|---|---|
| `KKA2_KLEPN_Melnikov_2014.csv` | ProteinGym v1.3 substitution assay, extracted from [Zenodo record 15293562](https://zenodo.org/records/15293562). |
| `supp_gku511_nar-01049-met-h-2014-File006.xlsx` | Author Supplementary Data 1 from the [publication](https://academic.oup.com/nar/article/42/14/e112/1266940). Contains the synthetic WT coding sequence, primers, and exact oligonucleotide sequences used to create the substitution library. |
| `supp_gku511_nar-01049-met-h-2014-File007.zip` | Author Supplementary Data 2 from the [publication](https://academic.oup.com/nar/article/42/14/e112/1266940). Contains processed amino-acid counts and selection measurements for each antibiotic, concentration, and replicate. |

## Sequence provenance

Supplementary Data 1 reports the 795-nt synthetic `KKA2_KLEPN_opt` ORF used in the experiment, including its terminal stop codon. The `OLS_A` and `OLS_B` worksheets contain 4,997 explicitly designed nucleotide variants covering every non-WT amino-acid substitution at protein positions 2 through 264. Each oligonucleotide differs from its WT tile at one codon; the complete mutant ORF can therefore be reconstructed without selecting or inferring a codon.

## Score provenance

`Kan18` is selection with kanamycin at one-eighth of the WT minimum inhibitory concentration. For all overlapping rows in the ProteinGym file, `DMS_score` equals the arithmetic mean of the author measurements in `KKA2_S1_Kan18_L1.aadiff.txt` and `KKA2_S3_Kan18_L1.aadiff.txt`. This averages two measurements of the same mutation, not different sequences. The author README marks the other `Kan18` replicate, `S2`, as a failed sequencing library.

The standardized assay contains all 4,997 author-designed variants at positions 2 through 264. Of the ProteinGym rows, 4,942 have an exact author-designed nucleotide oligonucleotide. The 18 unsupported position-1 substitutions are excluded, while the 55 author-designed nucleotide variants absent from ProteinGym are retained.

## References

- Melnikov A et al. [Comprehensive mutational scanning of a kinase in vivo reveals substrate-dependent fitness landscapes](https://doi.org/10.1093/nar/gku511). *Nucleic Acids Research* (2014).
- [Open-access article](https://pmc.ncbi.nlm.nih.gov/articles/PMC4132701/).
- ProteinGym v1.3: [Zenodo record 15293562](https://zenodo.org/records/15293562).
