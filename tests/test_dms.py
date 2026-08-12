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
    describe_amino_acid_changes,
    describe_coding_edit,
    read_fasta,
    replace_codon,
    translate_dna,
)


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
            self.assertEqual(read_fasta(fasta_path), {"target": "ATGGCT"})

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

    def test_amino_acid_changes(self) -> None:
        self.assertEqual(describe_amino_acid_changes("MA", "IV"), "M1I:A2V")
        self.assertEqual(describe_amino_acid_changes("MA", "MA"), "p.=")
        self.assertEqual(describe_amino_acid_changes("MA", "QIVK"), "insertion")

    def test_nucleotide_insertion(self) -> None:
        self.assertEqual(
            describe_coding_edit("ATGGCT", "CAAATAGTTAAA"),
            "insertion",
        )

    def test_nucleotide_deletion(self) -> None:
        self.assertEqual(describe_coding_edit("ATGGCT", "ATG"), "deletion")


if __name__ == "__main__":
    unittest.main()
