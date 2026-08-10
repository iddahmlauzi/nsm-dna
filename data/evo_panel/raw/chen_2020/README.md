# Chen et al. 2020 — VIM-2 beta-lactamase

## Study

Chen et al. measured the effects of single-amino-acid substitutions in VIM-2 metallo-beta-lactamase under multiple beta-lactam antibiotics and temperatures. The experiments characterized protein translocation, stability, catalytic activity, and substrate specificity.

## Benchmark mapping

- Evo study: VIM-2
- ProteinGym assay: `A4GRB6_PSEAI_Chen_2020`
- Target: VIM-2 metallo-beta-lactamase
- Organism: *Pseudomonas aeruginosa*
- ProteinGym phenotype: `0.031ug/mL_MEM_37C`
- ProteinGym raw directionality: `1`

## Files

| File | Role and provenance |
|---|---|
| `A4GRB6_PSEAI_Chen_2020.csv` | ProteinGym v1.3 substitution assay, extracted from [Zenodo record 15293562](https://zenodo.org/records/15293562). |
| `elife-56707-supp2-v2.xlsx` | Original authors' [Supplementary file 2](https://cdn.elifesciences.org/articles/56707/elife-56707-supp2-v2.xlsx). Sheet `SF2A Fitness Scores` contains amino-acid fitness under every selection condition; `SF2D Codon Fitness scores` contains WT and mutant codons with codon-level fitness under 128 µg/mL AMP. |

## Sequence provenance

`SF2D Codon Fitness scores` explicitly reports one WT codon for every VIM-2 position from 1 through 266. The study converter will concatenate these codons to recover the 798-nucleotide WT coding sequence rather than store a reconstructed FASTA. Its translation exactly matches ProteinGym's 266-residue target. Entries labelled `G2` describe an inserted glycine present only in an in-house sequence and are not part of the ProteinGym target.

## Standardized selection

The converter retains the 15,909 scored, position-numbered codon variants in SF2D and uses their 128 µg/mL ampicillin fitness values. It excludes the 62 separate `G2` insertion records. The ProteinGym CSV remains the benchmark reference, but its amino-acid-level meropenem scores are not substituted for the author codon-level measurements.

## References

- Chen JZ et al. [Comprehensive exploration of the translocation, stability and substrate recognition requirements in VIM-2 lactamase](https://doi.org/10.7554/eLife.56707). *eLife* (2020).
- [Open-access article and supplementary materials](https://pmc.ncbi.nlm.nih.gov/articles/PMC7308095/).
- ProteinGym v1.3: [Zenodo record 15293562](https://zenodo.org/records/15293562).
