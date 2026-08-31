import csv
import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

from scripts.data.create_gtdb_subset import (
    PARENT_CHUNK_LENGTH,
    GenomeAllocation,
    Taxonomy,
    allocate_balanced_chunks,
    create_gtdb_subset,
    selected_chunk_ordinals,
)
from scripts.data.plot_data_composition import plot_composition


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
        self.assay_dir = self.root / "assays"
        self.input_dir.mkdir(parents=True)
        self.raw_dir.mkdir()
        self.assay_dir.mkdir()
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
        self._write_assay("CCCCGGGG", "Escherichia coli")

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

    def _write_assay(self, wild_type: str, organism: str) -> None:
        path = self.assay_dir / "evaluation.csv"
        with path.open("w", encoding="utf-8", newline="") as assay_file:
            writer = csv.DictWriter(
                assay_file,
                fieldnames=["assay_id", "organism", "wt_nt"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "assay_id": "evaluation",
                    "organism": organism,
                    "wt_nt": wild_type,
                }
            )

    def _create_subset(self, output_name: str, seed: int = 17) -> Path:
        train_bases = 8 * PARENT_CHUNK_LENGTH
        create_gtdb_subset(
            dataset_dir=self.dataset_dir,
            assay_dir=self.assay_dir,
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
            for split in ("train", "validation")
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

        manifest_by_accession = {
            row["gtdb_accession"]: row for row in first_manifest
        }
        self.assertIn(
            manifest_by_accession["RS_GCF_000001.1"]["split"],
            {"train", "validation"},
        )
        self.assertIn(
            manifest_by_accession["RS_GCF_000002.1"]["split"],
            {"train", "validation"},
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
            for rank in ("domain", "phylum"):
                taxon_counts = subset_stats["composition"][split][rank].values()
                for count_name in ("genomes", "chunks"):
                    self.assertEqual(
                        sum(counts[count_name] for counts in taxon_counts),
                        subset_stats["splits"][split][count_name],
                    )

    def test_different_seed_changes_the_selection(self) -> None:
        first_output = self._create_subset("subset-a", seed=17)
        second_output = self._create_subset("subset-b", seed=29)

        first_ids = {
            row["chunk_id"]
            for split in ("train", "validation")
            for row in self._read_split_rows(first_output, split)
        }
        second_ids = {
            row["chunk_id"]
            for split in ("train", "validation")
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

    def test_smaller_chunk_selections_are_nested(self) -> None:
        small = GenomeAllocation(
            "genome",
            taxonomy("Genus", "species"),
            available_full_chunks=20,
            selected_chunks=3,
        )
        large = GenomeAllocation(
            "genome",
            taxonomy("Genus", "species"),
            available_full_chunks=20,
            selected_chunks=8,
        )

        small_ordinals = selected_chunk_ordinals(small, seed=17)
        large_ordinals = selected_chunk_ordinals(large, seed=17)

        self.assertTrue(small_ordinals < large_ordinals)

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
                assay_dir=self.assay_dir,
                train_bases=8 * PARENT_CHUNK_LENGTH,
                validation_fraction=0.25,
                seed=17,
            )
        self.assertFalse(any(self.dataset_dir.glob("*B_subset")))

    def test_insufficient_capacity_and_existing_output_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "contain only .* full chunks"):
            create_gtdb_subset(
                dataset_dir=self.dataset_dir,
                assay_dir=self.assay_dir,
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
                assay_dir=self.assay_dir,
                train_bases=8 * PARENT_CHUNK_LENGTH,
                validation_fraction=0.25,
                seed=17,
            )

    def test_replaces_matching_training_chunk(self) -> None:
        baseline = self._create_subset("baseline")
        baseline_rows = self._read_split_rows(baseline, "train")
        manifest = pq.read_table(baseline / "genome_manifest.parquet").to_pylist()
        target_accession = next(
            row["gtdb_accession"]
            for row in manifest
            if row["split"] == "train"
            and row["available_full_chunks"] >= row["selected_chunks"] + 2
        )
        reference = "ACGTCCGTTGCA"
        reverse_complement = reference.translate(
            str.maketrans("ACGT", "TGCA")
        )[::-1]

        source_rows = [
            row
            for path in sorted(self.input_dir.glob("chunks-*.parquet"))
            for row in pq.read_table(path).to_pylist()
            if row["gtdb_accession"] == target_accession
            and row["chunk_length"] == PARENT_CHUNK_LENGTH
        ]
        selected_ids = {row["chunk_id"] for row in baseline_rows}
        left_row, right_row = next(
            (left, right)
            for left, right in zip(source_rows, source_rows[1:])
            if left["record_id"] == right["record_id"]
            and left["chunk_end"] == right["chunk_start"]
            and {left["chunk_id"], right["chunk_id"]} & selected_ids
        )
        split = len(reverse_complement) // 2

        for path in sorted(self.input_dir.glob("chunks-*.parquet")):
            table = pq.read_table(path)
            rows = table.to_pylist()
            for row in rows:
                if row["chunk_id"] == left_row["chunk_id"]:
                    row["sequence"] = (
                        row["sequence"][:-split] + reverse_complement[:split]
                    )
                elif row["chunk_id"] == right_row["chunk_id"]:
                    row["sequence"] = (
                        reverse_complement[split:] + row["sequence"][split:]
                    )
            pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)

        genus = self.genomes[target_accession][0].split("_", maxsplit=1)[0]
        self._write_assay(reference, f"{genus} species")
        filtered = self._create_subset("filtered")
        filtered_rows = self._read_split_rows(filtered, "train")
        self.assertEqual(len(filtered_rows), len(baseline_rows))
        filtered_ids = {row["chunk_id"] for row in filtered_rows}
        self.assertTrue(
            {left_row["chunk_id"], right_row["chunk_id"]}.isdisjoint(filtered_ids)
        )

    def test_output_loads_as_hugging_face_streaming_splits(self) -> None:
        output_dir = self._create_subset("subset")
        data_files = {
            split: str(output_dir / split / "chunks-*.parquet")
            for split in ("train", "validation")
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

    def test_plots_composition_from_subset_statistics(self) -> None:
        output_dir = self._create_subset("subset")
        statistics = json.loads(
            (output_dir / "subset_stats.json").read_text(encoding="utf-8")
        )
        output_path = output_dir / "composition.png"

        plot_composition(statistics, output_path)

        self.assertGreater(output_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
