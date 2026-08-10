# Rockah-Shmuel et al. 2015 — HaeIII methyltransferase

## Study

Rockah-Shmuel et al. followed the accumulation of mutations during prolonged genetic drift to characterize the mutational tolerance of the HaeIII DNA methyltransferase. The measurements describe relative activity or fitness after successive drift rounds.

## Benchmark mapping

- Evo study: HaeIII
- ProteinGym assay: `MTH3_HAEAE_RockahShmuel_2015`
- Target: HaeIII DNA methyltransferase
- Organism: *Haemophilus aegyptius*
- ProteinGym phenotype: `Wrel_G17_filtered`
- ProteinGym raw directionality: `1`

## Files

| File | Role and provenance |
|---|---|
| `MTH3_HAEAE_RockahShmuel_2015.csv` | ProteinGym v1.3 substitution assay, extracted from [Zenodo record 15293562](https://zenodo.org/records/15293562). |
| `pcbi.1004421.s002.xlsx` | Original authors' [S2 workbook](https://journals.plos.org/ploscompbiol/article/file?id=10.1371%2Fjournal.pcbi.1004421.s002&type=supplementary), containing the exact reference codon and raw mutation frequencies for G0, G3, G7, and G17. |
| `pcbi.1004421.s003.xlsx` | Original authors' [S3 workbook](https://journals.plos.org/ploscompbiol/article/file?id=10.1371%2Fjournal.pcbi.1004421.s003&type=supplementary), containing processed frequencies and G3, G7, and G17 relative-fitness values for missense, synonymous, and nonsense mutations. |

## Sequence provenance

S2 explicitly reports a reference amino acid and codon at every position. Positions `-20` through `-1` are the unmutated His-tag and thrombin-cleavage region; positions `2` through `330` are the mutagenized M.HaeIII ORF; position `331` is the terminal stop codon. The study converter will recover the WT nucleotide sequence from these source columns rather than store a reconstructed FASTA.

The standardized construct contains author positions `0` through `331`: the initial methionine, the unmutated alanine at position `1`, the mutagenized ORF, and the terminal stop. It excludes the upstream tag and cleavage region. Because the author tables report amino-acid-level `Wrel G17` scores, the converter emits every single-nucleotide codon that produces each reported missense, synonymous, or nonsense consequence. Codons sharing an author-reported effect retain that same documented score.

## References

- Rockah-Shmuel L et al. [Systematic mapping of protein mutational space by prolonged drift reveals the deleterious effects of seemingly neutral mutations](https://doi.org/10.1371/journal.pcbi.1004421). *PLOS Computational Biology* (2015).
- ProteinGym v1.3: [Zenodo record 15293562](https://zenodo.org/records/15293562).
