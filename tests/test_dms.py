import csv
import io
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from dms.jacquier import apply_substitutions  # noqa: E402
from dms.mavedb import apply_nucleotide_edit  # noqa: E402
from dms.melnikov import find_mutant_codon, reverse_complement  # noqa: E402
from dms.ncrna import standardize_ncrna  # noqa: E402
from dms.shared import (  # noqa: E402
    describe_amino_acid_changes,
    describe_coding_edit,
    read_fasta,
    replace_codon,
    translate_dna,
)


class SharedSequenceTest(unittest.TestCase):
    """Sequence operations shared by the benchmark converters."""

    def test_read_fasta(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            fasta_path = Path(temp_dir) / "target.fasta"
            fasta_path.write_text(">target\nATG\nGCT\n", encoding="utf-8")
            self.assertEqual(read_fasta(fasta_path), {"target": "ATGGCT"})

    def test_replace_codon(self) -> None:
        self.assertEqual(replace_codon("ATGGCT", 2, "TAA"), "ATGTAA")

    def test_stop_codon_is_preserved(self) -> None:
        self.assertEqual(translate_dna("ATGTAA"), "M*")

    def test_describe_synonymous_edit(self) -> None:
        mutant_nt = replace_codon("ATGGCT", 2, "GCC")
        self.assertEqual(describe_coding_edit("ATGGCT", mutant_nt), "c.6T>C")
        self.assertEqual(translate_dna(mutant_nt), "MA")

    def test_describe_multiple_substitutions(self) -> None:
        self.assertEqual(
            describe_coding_edit("ATGGCT", "ATGTCG"),
            "c.[4G>T;6T>G]",
        )

    def test_describe_nucleotide_insertion(self) -> None:
        self.assertEqual(
            describe_coding_edit("ATGGCT", "CAAATAGTTAAA"),
            "insertion",
        )

    def test_describe_nucleotide_deletion(self) -> None:
        self.assertEqual(describe_coding_edit("ATGGCT", "ATG"), "deletion")

    def test_describe_amino_acid_changes(self) -> None:
        self.assertEqual(describe_amino_acid_changes("MA", "IV"), "M1I:A2V")
        self.assertEqual(describe_amino_acid_changes("MA", "MA"), "p.=")
        self.assertEqual(describe_amino_acid_changes("MA", "QIVK"), "insertion")


class MaveDBEditTest(unittest.TestCase):
    """HGVS edits used by the MaveDB converter."""

    def test_equal_hgvs_preserves_wt(self) -> None:
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


class JacquierEditTest(unittest.TestCase):
    """Mutation notation used by the Jacquier source table."""

    def test_multiple_substitutions(self) -> None:
        self.assertEqual(
            apply_substitutions("ATGGCT", "A1G_T6C"),
            "GTGGCC",
        )


class MelnikovOligoTest(unittest.TestCase):
    """Codon recovery from the Melnikov oligonucleotides."""

    def test_forward_oligo(self) -> None:
        self.assertEqual(
            find_mutant_codon("AAAATTCCC", "AAATATCCC", "ATT", "Y"),
            "TAT",
        )

    def test_reverse_oligo(self) -> None:
        wt_tile = reverse_complement("AAACTGCCC")
        mutant_tile = reverse_complement("AAATTCCCC")
        self.assertEqual(
            find_mutant_codon(wt_tile, mutant_tile, "CTG", "F"),
            "TTC",
        )


class NcRNATest(unittest.TestCase):
    """Conversion of the Evo 1 ncRNA source files."""

    def test_standardize_ncrna(self) -> None:
        from dms.ncrna import SOURCE_MEMBERS

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            archive_path = temp_path / "ncrna.tar.gz"
            source_text = (
                "wt_seq_nt\tmt_seq_nt\tfitness\n"
                "ACGT\tATGT\t0.5\n"
            ).encode()

            with tarfile.open(archive_path, "w:gz") as archive:
                for member_name in SOURCE_MEMBERS:
                    member = tarfile.TarInfo(member_name)
                    member.size = len(source_text)
                    archive.addfile(member, io.BytesIO(source_text))

            output_dir = temp_path / "output"
            row_counts = standardize_ncrna(archive_path, output_dir)

            self.assertEqual(set(row_counts.values()), {1})

            with (output_dir / "andreasson_2020.csv").open() as output_file:
                row = next(csv.DictReader(output_file))

            self.assertEqual(row["mutant_nt"], "ATGT")
            self.assertEqual(row["nt_edit"], "c.2C>T")
            self.assertEqual(row["experimental_score"], "0.5")


if __name__ == "__main__":
    unittest.main()
