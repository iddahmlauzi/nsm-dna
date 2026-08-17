# Evo 1 non-coding RNA DMS panel

## Benchmark

Seven deep mutational scanning studies measuring the effects of nucleotide variants on non-coding RNA function. The sequences use the DNA alphabet with `T` rather than `U`.

## Source

`arvind_dms.tar.gz` is the Evo 1 evaluation artifact provided by the Evo authors ([download](https://www.dropbox.com/scl/fi/bi3z1ov4hepak42z3vizc/arvind_dms.tar.gz?rlkey=m1v7vmsrghwju3ti3a3dvk4p1&dl=0)). The converter reads these seven members directly:

| Study | Source member | Variants |
|---|---|---:|
| Andreasson 2020 | `processed_dms_nt_andreasson_2020.tsv` | 4,497 |
| Domingo 2018 | `processed_dms_nt_domingo_2018.tsv` | 4,175 |
| Guy 2014 | `processed_dms_nt_guy_2014.tsv` | 25,491 |
| Hayden 2011 | `processed_dms_nt_hayden_2011.tsv` | 208 |
| Kobori 2016 | `processed_dms_nt_kobori_2016.tsv` | 10,296 |
| Pitt 2010 | `processed_dms_nt_pitt_2010.tsv` | 135 |
| Zhang 2009 | `processed_dms_nt_zhang_2009.tsv` | 24 |
| **Total** | | **44,826** |

Each source row provides the WT sequence, mutant sequence, and experimental fitness. The converter preserves these values and derives only the nucleotide-edit description.
