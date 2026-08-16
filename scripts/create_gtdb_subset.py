import argparse
import gzip
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from tqdm import tqdm

PARENT_CHUNK_LENGTH = 8_192
DEFAULT_TRAIN_BASES = 20_000_000_000
DEFAULT_VALIDATION_FRACTION = 0.01
CHUNKS_PER_SHARD = 10_000
SEQUENCE_BATCH_SIZE = 4_096

SPLITS = (
    "train",
    "validation",
    "escherichia_holdout",
)


@dataclass(frozen=True)
class Taxonomy:
    """GTDB taxonomy assigned to one representative genome."""

    domain: str
    phylum: str
    taxonomic_class: str
    order: str
    family: str
    genus: str
    species: str


@dataclass
class GenomeAllocation:
    """Chunk allocation for one source genome.

    Each eligible genome belongs entirely to one split, so its chunks never
    cross splits. Genomes without full-length chunks have no split.
    """

    gtdb_accession: str
    taxonomy: Taxonomy
    available_full_chunks: int
    split: str | None = None
    selected_chunks: int = 0


class ParquetShardWriter:
    """Write one dataset split as bounded-size Parquet shards."""

    def __init__(
        self,
        output_dir: Path,
        schema: pa.Schema,
        chunks_per_shard: int,
    ) -> None:
        self.output_dir = output_dir
        self.schema = schema
        self.chunks_per_shard = chunks_per_shard
        self.buffer: list[dict[str, object]] = []
        self.shard_index = 0

        self.output_dir.mkdir(parents=True)

    def add(self, row: dict[str, object]) -> None:
        self.buffer.append(row)

        if len(self.buffer) >= self.chunks_per_shard:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return

        output_path = self.output_dir / f"chunks-{self.shard_index:05d}.parquet"
        table = pa.Table.from_pylist(self.buffer, schema=self.schema)
        pq.write_table(table, output_path, compression="zstd")

        self.buffer.clear()
        self.shard_index += 1

    def close(self) -> None:
        self.flush()


def stable_score(seed: int, purpose: str, value: str) -> bytes:
    """Create a stable random-looking score for deterministic ordering."""
    text = f"{seed}:{purpose}:{value}"
    return hashlib.sha256(text.encode("utf-8")).digest()


def get_taxonomy_by_accession(taxonomy_files: list[Path]) -> dict[str, Taxonomy]:
    """Load GTDB accession-to-taxonomy mappings from compressed TSV files."""
    rank_names = {
        "d": "domain",
        "p": "phylum",
        "c": "taxonomic_class",
        "o": "order",
        "f": "family",
        "g": "genus",
        "s": "species",
    }
    taxonomy_by_accession: dict[str, Taxonomy] = {}

    for path in taxonomy_files:
        with gzip.open(path, mode="rt", encoding="utf-8") as f:
            for line in f:
                accession, lineage = line.rstrip("\n").split("\t", maxsplit=1)

                ranks = {}
                for taxon in lineage.split(";"):
                    prefix, name = taxon.split("__", maxsplit=1)
                    ranks[rank_names[prefix]] = name

                taxonomy_by_accession[accession] = Taxonomy(**ranks)

    return taxonomy_by_accession


def get_source_shards(input_dir: Path) -> tuple[list[Path], pa.Schema]:
    """Find the processed source shards and their shared schema."""
    shard_paths = sorted(input_dir.glob("chunks-*.parquet"))
    if not shard_paths:
        raise FileNotFoundError(f"No chunks-*.parquet files found in {input_dir}")

    source_schema = pq.ParquetFile(shard_paths[0]).schema_arrow
    return shard_paths, source_schema


def count_full_chunks_by_genome(shard_paths: list[Path]) -> dict[str, int]:
    """Count the available full-length chunks for each genome."""
    full_chunks_by_genome: dict[str, int] = {}

    for path in tqdm(shard_paths, desc="Scanning source metadata", unit="shard"):
        table = pq.read_table(
            path,
            columns=["gtdb_accession", "chunk_length"],
        )
        for accession in pc.unique(table.column("gtdb_accession")).to_pylist():
            full_chunks_by_genome.setdefault(accession, 0)

        full_chunks = table.filter(
            pc.equal(table.column("chunk_length"), PARENT_CHUNK_LENGTH)
        )
        if full_chunks.num_rows == 0:
            continue

        grouped_counts = full_chunks.group_by("gtdb_accession").aggregate(
            [("chunk_length", "count")]
        )
        for accession, count in zip(
            grouped_counts.column("gtdb_accession").to_pylist(),
            grouped_counts.column("chunk_length_count").to_pylist(),
        ):
            full_chunks_by_genome[accession] += count

    return full_chunks_by_genome


def is_escherichia(genus: str) -> bool:
    """Match the GTDB Escherichia genus and any suffixed subdivisions."""
    return genus == "Escherichia" or genus.startswith("Escherichia_")


def assign_genome_splits(
    full_chunks_by_genome: dict[str, int],
    taxonomy_by_accession: dict[str, Taxonomy],
    validation_fraction: float,
    seed: int,
) -> dict[str, GenomeAllocation]:
    """Assign whole genomes to training, validation, or holdout."""
    allocations_by_accession: dict[str, GenomeAllocation] = {}
    for accession, available_chunks in full_chunks_by_genome.items():
        taxonomy = taxonomy_by_accession.get(accession)
        if taxonomy is None:
            raise ValueError(f"Missing taxonomy for source genome {accession}")
        allocations_by_accession[accession] = GenomeAllocation(
            gtdb_accession=accession,
            taxonomy=taxonomy,
            available_full_chunks=available_chunks,
        )

    eligible_allocations = [
        allocation
        for allocation in allocations_by_accession.values()
        if allocation.available_full_chunks > 0
    ]
    escherichia_allocations = [
        allocation
        for allocation in eligible_allocations
        if is_escherichia(allocation.taxonomy.genus)
    ]

    for allocation in escherichia_allocations:
        allocation.split = "escherichia_holdout"

    non_holdout_allocations = [
        allocation
        for allocation in eligible_allocations
        if allocation.split != "escherichia_holdout"
    ]
    validation_count = math.floor(
        len(non_holdout_allocations) * validation_fraction + 0.5
    )
    if validation_count < 1 or validation_count >= len(non_holdout_allocations):
        raise ValueError(
            "validation_fraction must select at least one validation genome "
            "and leave at least one training genome"
        )

    ordered_for_validation = sorted(
        non_holdout_allocations,
        key=lambda allocation: stable_score(
            seed,
            "validation_split",
            allocation.gtdb_accession,
        ),
    )
    validation_accessions = {
        allocation.gtdb_accession
        for allocation in ordered_for_validation[:validation_count]
    }

    for allocation in non_holdout_allocations:
        allocation.split = (
            "validation"
            if allocation.gtdb_accession in validation_accessions
            else "train"
        )

    return allocations_by_accession


def allocate_balanced_chunks(
    allocations: list[GenomeAllocation],
    target_chunks: int,
    seed: int,
    purpose: str,
) -> None:
    """Allocate a target across genomes one chunk per genome per round."""
    available_chunks = sum(
        allocation.available_full_chunks for allocation in allocations
    )
    if target_chunks > available_chunks:
        raise ValueError(
            f"The {purpose} split requests {target_chunks:,} chunks, but its "
            f"genomes contain only {available_chunks:,} full chunks"
        )

    ordered_allocations = sorted(
        allocations,
        key=lambda allocation: stable_score(
            seed,
            f"{purpose}_quota",
            allocation.gtdb_accession,
        ),
    )
    remaining_chunks = target_chunks

    while remaining_chunks:
        # Every genome receives one chunk before any eligible genome receives
        # another. Genomes that run out are skipped in later rounds.
        for allocation in ordered_allocations:
            if allocation.selected_chunks >= allocation.available_full_chunks:
                continue

            allocation.selected_chunks += 1
            remaining_chunks -= 1

            if remaining_chunks == 0:
                break


def allocate_split_quotas(
    allocations_by_accession: dict[str, GenomeAllocation],
    train_bases: int,
    seed: int,
) -> None:
    """Allocate training, validation, and holdout chunks at one sampling rate."""
    target_train_chunks = train_bases // PARENT_CHUNK_LENGTH

    allocations_by_split = {
        split: [
            allocation
            for allocation in allocations_by_accession.values()
            if allocation.split == split
        ]
        for split in SPLITS
    }
    train_allocations = allocations_by_split["train"]
    if not train_allocations:
        raise ValueError("No training genomes remain after split assignment")

    allocate_balanced_chunks(
        train_allocations,
        target_train_chunks,
        seed,
        "train",
    )
    chunks_per_training_genome = target_train_chunks / len(train_allocations)

    for split in ("validation", "escherichia_holdout"):
        split_allocations = allocations_by_split[split]
        # Match the training split's average chunks per genome.
        target_chunks = math.floor(
            chunks_per_training_genome * len(split_allocations) + 0.5
        )
        allocate_balanced_chunks(
            split_allocations,
            target_chunks,
            seed,
            split,
        )


def selected_chunk_ordinals(allocation: GenomeAllocation, seed: int) -> set[int]:
    """Select random full-chunk positions for one genome."""
    if allocation.selected_chunks == 0:
        return set()
    if allocation.selected_chunks == allocation.available_full_chunks:
        return set(range(allocation.available_full_chunks))

    genome_seed = int.from_bytes(
        stable_score(seed, "chunk_selection", allocation.gtdb_accession)[:8],
        byteorder="big",
    )
    generator = np.random.default_rng(genome_seed)
    # Taking a prefix of one permutation makes smaller data budgets exact
    # subsets of larger budgets created with the same seed.
    ordinals = generator.permutation(allocation.available_full_chunks)
    ordinals = ordinals[: allocation.selected_chunks]
    return {int(ordinal) for ordinal in ordinals}


def write_selected_chunks(
    shard_paths: list[Path],
    source_schema: pa.Schema,
    allocations_by_accession: dict[str, GenomeAllocation],
    temporary_dir: Path,
    seed: int,
) -> None:
    """Stream the source and materialize only each genome's selected chunks."""
    writers = {
        split: ParquetShardWriter(
            temporary_dir / split,
            source_schema,
            CHUNKS_PER_SHARD,
        )
        for split in SPLITS
    }
    writers_by_accession = {
        accession: writers[allocation.split]
        for accession, allocation in allocations_by_accession.items()
        if allocation.split is not None
    }

    current_accession: str | None = None
    current_allocation: GenomeAllocation | None = None
    current_full_chunk = 0
    selected_ordinals: set[int] = set()

    for path in tqdm(shard_paths, desc="Writing subset", unit="shard"):
        parquet_file = pq.ParquetFile(path)

        for batch in parquet_file.iter_batches(batch_size=SEQUENCE_BATCH_SIZE):
            accession_column = batch.column(
                batch.schema.get_field_index("gtdb_accession")
            )
            chunk_length_column = batch.column(
                batch.schema.get_field_index("chunk_length")
            )
            accession_runs = pc.run_end_encode(accession_column)
            selected_rows = np.zeros(batch.num_rows, dtype=bool)
            run_start = 0

            # Build an Arrow filter for the selected full-chunk ordinals. This
            # converts only selected sequence rows into Python dictionaries.
            for run_end, accession in zip(
                accession_runs.run_ends.to_pylist(),
                accession_runs.values.to_pylist(),
            ):
                if accession != current_accession:
                    current_accession = accession
                    current_allocation = allocations_by_accession[accession]
                    current_full_chunk = 0
                    selected_ordinals = selected_chunk_ordinals(
                        current_allocation,
                        seed,
                    )

                run_lengths = chunk_length_column.slice(
                    run_start,
                    run_end - run_start,
                ).to_numpy(zero_copy_only=False)
                full_chunk_offsets = np.flatnonzero(
                    run_lengths == PARENT_CHUNK_LENGTH
                )
                run_full_chunk_end = current_full_chunk + len(full_chunk_offsets)

                # A genome can span multiple batches. Convert each selected
                # chunk in this part of the genome to its row in this batch.
                for ordinal in selected_ordinals:
                    if current_full_chunk <= ordinal < run_full_chunk_end:
                        offset_index = ordinal - current_full_chunk
                        selected_rows[
                            run_start + int(full_chunk_offsets[offset_index])
                        ] = True

                current_full_chunk = run_full_chunk_end
                run_start = run_end

            selected_batch = batch.filter(pa.array(selected_rows))
            for row in selected_batch.to_pylist():
                writers_by_accession[row["gtdb_accession"]].add(row)

    for writer in writers.values():
        writer.close()


def write_genome_manifest(
    allocations_by_accession: dict[str, GenomeAllocation],
    output_path: Path,
) -> None:
    """Record the taxonomy, split, and quota for every source genome."""
    rows = []

    for accession in sorted(allocations_by_accession):
        allocation = allocations_by_accession[accession]
        rows.append(
            {
                "gtdb_accession": accession,
                "domain": allocation.taxonomy.domain,
                "phylum": allocation.taxonomy.phylum,
                "class": allocation.taxonomy.taxonomic_class,
                "order": allocation.taxonomy.order,
                "family": allocation.taxonomy.family,
                "genus": allocation.taxonomy.genus,
                "species": allocation.taxonomy.species,
                "split": allocation.split,
                "available_full_chunks": allocation.available_full_chunks,
                "selected_chunks": allocation.selected_chunks,
            }
        )

    pq.write_table(pa.Table.from_pylist(rows), output_path, compression="zstd")


def split_statistics(
    allocations_by_accession: dict[str, GenomeAllocation],
) -> dict[str, dict[str, int]]:
    """Summarize genome, chunk, and base counts for each split."""
    statistics: dict[str, dict[str, int]] = {}

    for split in SPLITS:
        split_allocations = [
            allocation
            for allocation in allocations_by_accession.values()
            if allocation.split == split
        ]
        selected_chunks = sum(
            allocation.selected_chunks for allocation in split_allocations
        )
        statistics[split] = {
            "genomes": len(split_allocations),
            "chunks": selected_chunks,
            "bases": selected_chunks * PARENT_CHUNK_LENGTH,
        }

    return statistics


def write_subset_stats(
    output_path: Path,
    allocations_by_accession: dict[str, GenomeAllocation],
    validation_fraction: float,
    seed: int,
) -> None:
    """Write the selection parameters and resulting dataset counts."""
    holdout_genera = sorted(
        {
            allocation.taxonomy.genus
            for allocation in allocations_by_accession.values()
            if allocation.split == "escherichia_holdout"
        }
    )
    split_stats = split_statistics(allocations_by_accession)
    summary = {
        "selection": {
            "seed": seed,
            "chunk_length": PARENT_CHUNK_LENGTH,
            "validation_fraction": validation_fraction,
            "holdout_genera": holdout_genera,
        },
        "splits": split_stats,
        "genomes_without_full_chunks": sum(
            allocation.available_full_chunks == 0
            for allocation in allocations_by_accession.values()
        ),
    }
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def create_gtdb_subset(
    dataset_dir: Path,
    train_bases: int,
    validation_fraction: float,
    seed: int,
) -> dict[str, dict[str, int]]:
    """Create a balanced GTDB subset with genome-level data splits."""
    dataset_dir = dataset_dir.expanduser().resolve()
    input_dir = dataset_dir / "processed"
    budget_billions = train_bases / 1_000_000_000
    output_dir = dataset_dir / f"{budget_billions:g}B_subset"
    taxonomy_files = [
        dataset_dir / "raw" / "bac120_taxonomy_r232.tsv.gz",
        dataset_dir / "raw" / "ar53_taxonomy_r232.tsv.gz",
    ]
    temporary_dir = output_dir.with_name(f".{output_dir.name}.incomplete")

    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    if temporary_dir.exists():
        raise FileExistsError(
            f"Incomplete output directory already exists: {temporary_dir}"
        )

    shard_paths, source_schema = get_source_shards(input_dir)
    taxonomy_by_accession = get_taxonomy_by_accession(taxonomy_files)
    full_chunks_by_genome = count_full_chunks_by_genome(shard_paths)
    allocations_by_accession = assign_genome_splits(
        full_chunks_by_genome,
        taxonomy_by_accession,
        validation_fraction,
        seed,
    )
    allocate_split_quotas(
        allocations_by_accession,
        train_bases,
        seed,
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir.mkdir()

    write_selected_chunks(
        shard_paths,
        source_schema,
        allocations_by_accession,
        temporary_dir,
        seed,
    )
    write_genome_manifest(
        allocations_by_accession,
        temporary_dir / "genome_manifest.parquet",
    )
    write_subset_stats(
        temporary_dir / "subset_stats.json",
        allocations_by_accession,
        validation_fraction,
        seed,
    )

    temporary_dir.rename(output_dir)
    return split_statistics(allocations_by_accession)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a balanced, genome-level subset of processed GTDB R232.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help=(
            "GTDB dataset root containing processed/, "
            "raw/bac120_taxonomy_r232.tsv.gz, and "
            "raw/ar53_taxonomy_r232.tsv.gz. Writes the subset directly under "
            "this directory."
        ),
    )
    parser.add_argument(
        "--train-bases",
        type=int,
        default=DEFAULT_TRAIN_BASES,
        help="Maximum number of full-chunk bases in the training split.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=DEFAULT_VALIDATION_FRACTION,
        help="Fraction of eligible non-Escherichia genomes used for validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used for genome and chunk selection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    statistics = create_gtdb_subset(
        dataset_dir=args.dataset_dir,
        train_bases=args.train_bases,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    print(json.dumps(statistics, indent=2))


if __name__ == "__main__":
    main()
