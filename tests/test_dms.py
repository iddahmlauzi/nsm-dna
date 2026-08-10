import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from dms.mavedb import apply_nucleotide_edit  # noqa: E402
from dms.jacquier import apply_substitutions  # noqa: E402
from dms.melnikov import find_mutant_codon, reverse_complement  # noqa: E402
from dms.shared import (  # noqa: E402
    describe_coding_edit,
    read_fasta,
    replace_codon,
    translate_dna,
)
from dms.tsuboyama import iter_assay_variants, target_coding_sequence  # noqa: E402


class MaveDBSequenceTest(unittest.TestCase):
    """Small sequence examples used by the MaveDB converter."""

    def test_wild_type(self) -> None:
        self.assertEqual(apply_nucleotide_edit("ATGGCT", "c.="), "ATGGCT")

    def test_single_substitution(self) -> None:
        self.assertEqual(
            apply_nucleotide_edit("ATGGCT", "c.4G>T"),
            "ATGTCT",
        )

    def test_multiple_substitutions(self) -> None:
        self.assertEqual(
            apply_nucleotide_edit("ATGGCT", "c.[4G>T;6T>C]"),
            "ATGTCC",
        )

    def test_insertion_deletion_and_delins(self) -> None:
        self.assertEqual(apply_nucleotide_edit("ATGGCT", "c.3_4insAAA"), "ATGAAAGCT")
        self.assertEqual(apply_nucleotide_edit("ATGAAAGCT", "c.4_6del"), "ATGGCT")
        self.assertEqual(apply_nucleotide_edit("ATGGCT", "c.4_6delinsTAA"), "ATGTAA")

    def test_stop_codon_is_preserved(self) -> None:
        self.assertEqual(translate_dna("ATGTAA"), "M*")


class SharedSequenceTest(unittest.TestCase):
    """Small sequence examples shared by the Evo converters."""

    def test_read_fasta(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            fasta_path = Path(temp_dir) / "target.fasta"
            fasta_path.write_text(">target\nATG\nGCT\n", encoding="utf-8")
            self.assertEqual(read_fasta(fasta_path), "ATGGCT")

    def test_replace_codon(self) -> None:
        self.assertEqual(replace_codon("ATGGCT", 2, "TAA"), "ATGTAA")

    def test_describe_synonymous_edit(self) -> None:
        mutant_nt = replace_codon("ATGGCT", 2, "GCC")
        self.assertEqual(describe_coding_edit("ATGGCT", mutant_nt), "c.6T>C")
        self.assertEqual(translate_dna(mutant_nt), "MA")

    def test_describe_multiple_substitutions(self) -> None:
        self.assertEqual(
            describe_coding_edit("ATGGCT", "ATGTCG"),
            "c.[4G>T;6T>G]",
        )


class EvoSourceTest(unittest.TestCase):
    """Small source-format examples used by the Evo converters."""

    def test_jacquier_multiple_substitutions(self) -> None:
        self.assertEqual(
            apply_substitutions("ATGGCT", "A1G_T6C"),
            "GTGGCC",
        )

    def test_melnikov_forward_oligo(self) -> None:
        self.assertEqual(
            find_mutant_codon("AAAATTCCC", "AAATATCCC", "ATT", "Y"),
            "TAT",
        )

    def test_melnikov_reverse_oligo(self) -> None:
        wt_tile = reverse_complement("AAACTGCCC")
        mutant_tile = reverse_complement("AAATTCCCC")
        self.assertEqual(
            find_mutant_codon(wt_tile, mutant_tile, "CTG", "F"),
            "TTC",
        )

    def test_tsuboyama_target_slice(self) -> None:
        source_row = {
            "name": "example",
            "aa_seq": "MA",
            "aa_seq_full": "QMAK",
            "dna_seq": "CAAATGGCTAAA",
        }
        self.assertEqual(target_coding_sequence(source_row), "ATGGCT")

    def test_tsuboyama_multiple_substitutions(self) -> None:
        source_rows = [
            {
                "name": "example",
                "aa_change": "M1I:A2V",
                "experimental_score": "-0.5",
                "mutant_nt": "ATAGTT",
            }
        ]
        variants = list(
            iter_assay_variants(
                "1AOY.pdb",
                "ATGGCT",
                source_rows,
            )
        )
        self.assertEqual(variants[0]["mutant_aa"], "IV")

    def test_tsuboyama_disambiguates_reused_source_names(self) -> None:
        source_rows = [
            {
                "name": "reused",
                "aa_change": "M1I:A2V",
                "experimental_score": "-0.5",
                "mutant_nt": "ATAGTT",
            },
            {
                "name": "reused",
                "aa_change": "M1I:A2V",
                "experimental_score": "-0.4",
                "mutant_nt": "ATCGTA",
            },
        ]
        variants = list(
            iter_assay_variants(
                "1AOY.pdb",
                "ATGGCT",
                source_rows,
            )
        )
        variant_ids = {variant["variant_id"] for variant in variants}
        self.assertEqual(len(variant_ids), 2)


if __name__ == "__main__":
    unittest.main()
