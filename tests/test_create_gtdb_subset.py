import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

SCRIPTS_DIR = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from create_gtdb_subset import (  # noqa: E402
    PARENT_CHUNK_LENGTH,
    GenomeAllocation,
    Taxonomy,
    allocate_balanced_chunks,
    create_gtdb_subset,
)


def taxonomy(genus: str, species: str) -> Taxonomy:
    return Taxonomy(
        domain="Bacteria",
        phylum="Pseudomonadota",
        taxonomic_class="Gammaproteobacteria",
        order="Enterobacterales",
        family="Enterobacteriaceae",
        genus=genus,
        species=species,
    )


class GTDBSubsetTest(unittest.TestCase):
    """Subset selection over small synthetic GTDB Parquet shards."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.dataset_dir = self.root / "gtdb"
        self.input_dir = self.dataset_dir / "processed"
        self.raw_dir = self.dataset_dir / "raw"
        self.input_dir.mkdir(parents=True)
        self.raw_dir.mkdir()
        self.bacterial_taxonomy_path = (
            self.raw_dir / "bac120_taxonomy_r232.tsv.gz"
        )
        self.archaeal_taxonomy_path = self.raw_dir / "ar53_taxonomy_r232.tsv.gz"

        self.genomes = {
            "RS_GCF_000001.1": ("Escherichia", 4),
            "RS_GCF_000002.1": ("Escherichia_A", 3),
            "RS_GCF_000003.1": ("Klebsiella", 1),
            "RS_GCF_000004.1": ("Pseudomonas", 4),
            "RS_GCF_000005.1": ("Bacillus", 5),
            "RS_GCF_000006.1": ("Streptomyces", 6),
            "RS_GCF_000007.1": ("Vibrio", 7),
            "RS_GCF_000008.1": ("Sulfolobus", 8),
            "RS_GCF_000009.1": ("Tinygenus", 0),
        }
        self._write_taxonomy(self.genomes)
        with gzip.open(self.archaeal_taxonomy_path, "wt", encoding="utf-8"):
            pass
        self._write_source_shards(self.genomes)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write_taxonomy(self, genomes: dict[str, tuple[str, int]]) -> None:
        with gzip.open(
            self.bacterial_taxonomy_path,
            "wt",
            encoding="utf-8",
        ) as handle:
            for accession, (genus, _) in genomes.items():
                lineage = (
                    "d__Bacteria;p__Pseudomonadota;c__Gammaproteobacteria;"
                    "o__Enterobacterales;f__Enterobacteriaceae;"
                    f"g__{genus};s__{genus} species"
                )
                handle.write(f"{accession}\t{lineage}\n")

    def _chunk_row(
        self,
        accession: str,
        chunk_index: int,
        chunk_length: int = PARENT_CHUNK_LENGTH,
    ) -> dict[str, object]:
        start = chunk_index * PARENT_CHUNK_LENGTH
        end = start + chunk_length
        return {
            "chunk_id": f"{accession}|record|{start}-{end}",
            "ncbi_accession": accession.removeprefix("RS_"),
            "gtdb_accession": accession,
            "archive_path": f"archive/{accession}.fna.gz",
            "record_id": "record",
            "record_length": PARENT_CHUNK_LENGTH * 20,
            "chunk_start": start,
            "chunk_end": end,
            "chunk_length": chunk_length,
            "sequence": "A" * chunk_length,
        }

    def _write_source_shards(self, genomes: dict[str, tuple[str, int]]) -> None:
        rows = []
        for accession, (_, full_chunks) in genomes.items():
            rows.extend(
                self._chunk_row(accession, chunk_index)
                for chunk_index in range(full_chunks)
            )
            rows.append(self._chunk_row(accession, full_chunks, chunk_length=100))

        # Split through the middle of one genome to test grouping across shards.
        boundary = next(
            index
            for index, row in enumerate(rows)
            if row["gtdb_accession"] == "RS_GCF_000005.1"
        ) + 2
        pq.write_table(
            pa.Table.from_pylist(rows[:boundary]),
            self.input_dir / "chunks-00000.parquet",
            compression="zstd",
        )
        pq.write_table(
            pa.Table.from_pylist(rows[boundary:]),
            self.input_dir / "chunks-00001.parquet",
            compression="zstd",
        )

    def _create_subset(self, output_name: str, seed: int = 17) -> Path:
        train_bases = 8 * PARENT_CHUNK_LENGTH
        create_gtdb_subset(
            dataset_dir=self.dataset_dir,
            train_bases=train_bases,
            validation_fraction=0.25,
            seed=seed,
        )
        output_dir = self.root / output_name
        budget_billions = train_bases / 1_000_000_000
        generated_output = self.dataset_dir / f"{budget_billions:g}B_subset"
        generated_output.rename(output_dir)
        return output_dir

    def _read_split_rows(self, output_dir: Path, split: str) -> list[dict]:
        paths = sorted((output_dir / split).glob("chunks-*.parquet"))
        return pq.read_table(paths).to_pylist() if paths else []

    def test_creates_disjoint_reproducible_splits(self) -> None:
        first_output = self._create_subset("subset-a")
        second_output = self._create_subset("subset-b")

        first_manifest = pq.read_table(
            first_output / "genome_manifest.parquet"
        ).to_pylist()
        second_manifest = pq.read_table(
            second_output / "genome_manifest.parquet"
        ).to_pylist()
        self.assertEqual(first_manifest, second_manifest)

        rows_by_split = {
            split: self._read_split_rows(first_output, split)
            for split in ("train", "validation", "escherichia_holdout")
        }
        accessions_by_split = {
            split: {row["gtdb_accession"] for row in rows}
            for split, rows in rows_by_split.items()
        }
        self.assertTrue(
            accessions_by_split["train"].isdisjoint(
                accessions_by_split["validation"]
            )
        )
        self.assertTrue(
            accessions_by_split["escherichia_holdout"].isdisjoint(
                accessions_by_split["train"] | accessions_by_split["validation"]
            )
        )

        manifest_by_accession = {
            row["gtdb_accession"]: row for row in first_manifest
        }
        self.assertEqual(
            manifest_by_accession["RS_GCF_000001.1"]["split"],
            "escherichia_holdout",
        )
        self.assertEqual(
            manifest_by_accession["RS_GCF_000002.1"]["split"],
            "escherichia_holdout",
        )
        self.assertEqual(
            manifest_by_accession["RS_GCF_000009.1"]["split"],
            None,
        )
        self.assertTrue(
            all(
                row["chunk_length"] == PARENT_CHUNK_LENGTH
                for rows in rows_by_split.values()
                for row in rows
            )
        )

        first_chunk_ids = {
            split: [row["chunk_id"] for row in rows]
            for split, rows in rows_by_split.items()
        }
        second_chunk_ids = {
            split: [
                row["chunk_id"]
                for row in self._read_split_rows(second_output, split)
            ]
            for split in rows_by_split
        }
        self.assertEqual(first_chunk_ids, second_chunk_ids)

        subset_stats = json.loads(
            (first_output / "subset_stats.json").read_text(encoding="utf-8")
        )
        for split, rows in rows_by_split.items():
            manifest_count = sum(
                row["selected_chunks"]
                for row in first_manifest
                if row["split"] == split
            )
            self.assertEqual(len(rows), manifest_count)
            self.assertEqual(
                len(rows),
                subset_stats["splits"][split]["chunks"],
            )

    def test_different_seed_changes_the_selection(self) -> None:
        first_output = self._create_subset("subset-a", seed=17)
        second_output = self._create_subset("subset-b", seed=29)

        first_ids = {
            row["chunk_id"]
            for split in ("train", "validation", "escherichia_holdout")
            for row in self._read_split_rows(first_output, split)
        }
        second_ids = {
            row["chunk_id"]
            for split in ("train", "validation", "escherichia_holdout")
            for row in self._read_split_rows(second_output, split)
        }
        self.assertNotEqual(first_ids, second_ids)

    def test_balanced_allocation_redistributes_limited_capacity(self) -> None:
        allocations = [
            GenomeAllocation("limited", taxonomy("Genus1", "species1"), 1),
            GenomeAllocation("large-a", taxonomy("Genus2", "species2"), 5),
            GenomeAllocation("large-b", taxonomy("Genus3", "species3"), 5),
        ]

        allocate_balanced_chunks(
            allocations,
            target_chunks=7,
            seed=3,
            purpose="test",
        )

        selected = {
            allocation.gtdb_accession: allocation.selected_chunks
            for allocation in allocations
        }
        self.assertEqual(selected["limited"], 1)
        self.assertEqual(sum(selected.values()), 7)
        self.assertLessEqual(abs(selected["large-a"] - selected["large-b"]), 1)

    def test_missing_taxonomy_fails_before_writing(self) -> None:
        with gzip.open(
            self.bacterial_taxonomy_path,
            "wt",
            encoding="utf-8",
        ) as handle:
            handle.write(
                "RS_GCF_000001.1\t"
                "d__Bacteria;p__Pseudomonadota;c__Gammaproteobacteria;"
                "o__Enterobacterales;f__Enterobacteriaceae;"
                "g__Escherichia;s__Escherichia coli\n"
            )

        with self.assertRaisesRegex(ValueError, "Missing taxonomy"):
            create_gtdb_subset(
                dataset_dir=self.dataset_dir,
                train_bases=8 * PARENT_CHUNK_LENGTH,
                validation_fraction=0.25,
                seed=17,
            )
        self.assertFalse(any(self.dataset_dir.glob("*B_subset")))

    def test_insufficient_capacity_and_existing_output_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "contain only .* full chunks"):
            create_gtdb_subset(
                dataset_dir=self.dataset_dir,
                train_bases=1_000 * PARENT_CHUNK_LENGTH,
                validation_fraction=0.25,
                seed=17,
            )

        budget_billions = 8 * PARENT_CHUNK_LENGTH / 1_000_000_000
        existing_output = self.dataset_dir / f"{budget_billions:g}B_subset"
        existing_output.mkdir()
        with self.assertRaises(FileExistsError):
            create_gtdb_subset(
                dataset_dir=self.dataset_dir,
                train_bases=8 * PARENT_CHUNK_LENGTH,
                validation_fraction=0.25,
                seed=17,
            )

    def test_output_loads_as_hugging_face_streaming_splits(self) -> None:
        output_dir = self._create_subset("subset")
        data_files = {
            split: str(output_dir / split / "chunks-*.parquet")
            for split in ("train", "validation", "escherichia_holdout")
        }
        # This sandbox blocks the shared-memory helper that Hugging Face uses
        # only to communicate epochs to persistent DataLoader workers.
        with patch(
            "datasets.iterable_dataset._maybe_share_with_torch_persistent_workers",
            side_effect=lambda value: value,
        ):
            dataset = load_dataset(
                "parquet",
                data_files=data_files,
                streaming=True,
                cache_dir=self.root / "huggingface-cache",
            )

            for split in data_files:
                first_row = next(iter(dataset[split]))
                self.assertEqual(first_row["chunk_length"], PARENT_CHUNK_LENGTH)


if __name__ == "__main__":
    unittest.main()
