import argparse
import csv
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


@dataclass(frozen=True)
class EvaluationReference:
    """Wild-type DNA sequence and source genus for one evaluation assay."""

    assay_id: str
    genus: str
    sequence: str


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


def load_evaluation_references(assay_dir: Path) -> list[EvaluationReference]:
    """Load one wild-type DNA sequence and source genus per assay."""
    assay_paths = sorted(assay_dir.glob("*.csv"))
    if not assay_paths:
        raise FileNotFoundError(f"No CSV assay files found in {assay_dir}")

    references = []
    for path in assay_paths:
        assay_ids = set()
        organisms = set()
        sequences = set()
        with path.open(encoding="utf-8", newline="") as assay_file:
            for row in csv.DictReader(assay_file):
                assay_ids.add(row["assay_id"])
                organisms.add(row["organism"])
                sequences.add(row["wt_nt"].upper())

        if len(assay_ids) != 1 or len(organisms) != 1 or len(sequences) != 1:
            raise ValueError(
                f"Expected one assay, organism, and wild-type sequence in {path}"
            )

        sequence = sequences.pop()
        if len(sequence) > PARENT_CHUNK_LENGTH:
            raise ValueError(
                f"Evaluation sequence in {path} exceeds the stored chunk length"
            )

        references.append(
            EvaluationReference(
                assay_id=assay_ids.pop(),
                genus=organisms.pop().split(maxsplit=1)[0],
                sequence=sequence,
            )
        )

    return references


def evaluation_patterns_by_genus(
    references: list[EvaluationReference],
) -> dict[str, tuple[str, ...]]:
    """Group evaluation sequences and reverse complements by source genus."""
    patterns_by_genus: dict[str, set[str]] = {}
    complement = str.maketrans("ACGT", "TGCA")

    for reference in references:
        reverse_complement = reference.sequence.translate(complement)[::-1]
        patterns_by_genus.setdefault(reference.genus, set()).update(
            {reference.sequence, reverse_complement}
        )

    return {
        genus: tuple(sorted(patterns))
        for genus, patterns in patterns_by_genus.items()
    }


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


def assign_genome_splits(
    full_chunks_by_genome: dict[str, int],
    taxonomy_by_accession: dict[str, Taxonomy],
    validation_fraction: float,
    seed: int,
) -> dict[str, GenomeAllocation]:
    """Assign each eligible genome wholly to training or validation."""
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
    validation_count = math.floor(
        len(eligible_allocations) * validation_fraction + 0.5
    )
    if validation_count < 1 or validation_count >= len(eligible_allocations):
        raise ValueError(
            "validation_fraction must select at least one validation genome "
            "and leave at least one training genome"
        )

    ordered_for_validation = sorted(
        eligible_allocations,
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

    for allocation in eligible_allocations:
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
    """Allocate training and validation chunks at one sampling rate."""
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

    validation_allocations = allocations_by_split["validation"]
    # Match the training split's average chunks per genome.
    target_validation_chunks = math.floor(
        chunks_per_training_genome * len(validation_allocations) + 0.5
    )
    allocate_balanced_chunks(
        validation_allocations,
        target_validation_chunks,
        seed,
        "validation",
    )


def ordered_chunk_ordinals(
    allocation: GenomeAllocation,
    seed: int,
) -> list[int]:
    """Order one genome's full chunks deterministically."""
    genome_seed = int.from_bytes(
        stable_score(seed, "chunk_selection", allocation.gtdb_accession)[:8],
        byteorder="big",
    )
    generator = np.random.default_rng(genome_seed)
    return [
        int(ordinal)
        for ordinal in generator.permutation(allocation.available_full_chunks)
    ]


def selected_chunk_ordinals(allocation: GenomeAllocation, seed: int) -> set[int]:
    """Select random full-chunk positions for one genome."""
    # Taking a prefix of one permutation makes smaller data budgets exact
    # subsets of larger budgets created with the same seed.
    return set(ordered_chunk_ordinals(allocation, seed)[: allocation.selected_chunks])


def matching_chunk_ordinals(
    rows: list[dict[str, object]],
    evaluation_patterns: tuple[str, ...],
) -> set[int]:
    """Find chunks containing or jointly spanning an evaluation sequence."""
    # First catch references contained entirely within one stored chunk.
    matching_ordinals = {
        ordinal
        for ordinal, row in enumerate(rows)
        if any(pattern in row["sequence"] for pattern in evaluation_patterns)
    }

    # Every reference is shorter than a stored chunk, so a chunked reference
    # can cross at most one boundary. Coordinates prevent joining across
    # different records or an ambiguous-base gap removed during processing.
    for left_ordinal, (left, right) in enumerate(zip(rows, rows[1:])):
        if (
            left["record_id"] != right["record_id"]
            or left["chunk_end"] != right["chunk_start"]
        ):
            continue

        left_sequence = left["sequence"]
        combined_sequence = left_sequence + right["sequence"]
        boundary = len(left_sequence)
        for pattern in evaluation_patterns:
            start = combined_sequence.find(
                pattern,
                max(0, boundary - len(pattern) + 1),
            )
            while 0 <= start < boundary:
                if start + len(pattern) > boundary:
                    matching_ordinals.update({left_ordinal, left_ordinal + 1})
                    break
                start = combined_sequence.find(pattern, start + 1)

    return matching_ordinals


def select_nonleaking_chunks(
    rows: list[dict[str, object]],
    allocation: GenomeAllocation,
    seed: int,
    evaluation_patterns: tuple[str, ...],
) -> tuple[list[dict[str, object]], int]:
    """Replace matching chunks using the genome's deterministic chunk order."""
    matching_ordinals = matching_chunk_ordinals(rows, evaluation_patterns)
    safe_rows = {
        ordinal: row
        for ordinal, row in enumerate(rows)
        if ordinal not in matching_ordinals
    }
    # Filter the existing deterministic order instead of resampling. This keeps
    # smaller data budgets nested within larger budgets after replacements.
    ordered_ordinals = ordered_chunk_ordinals(allocation, seed)
    selected_ordinals = [
        ordinal for ordinal in ordered_ordinals if ordinal in safe_rows
    ][: allocation.selected_chunks]

    if len(selected_ordinals) != allocation.selected_chunks:
        raise ValueError(
            f"Genome {allocation.gtdb_accession} does not contain enough "
            "non-evaluation chunks to satisfy its allocation"
        )

    selected_rows = [safe_rows[ordinal] for ordinal in sorted(selected_ordinals)]
    return selected_rows, len(matching_ordinals)


def write_selected_chunks(
    shard_paths: list[Path],
    source_schema: pa.Schema,
    allocations_by_accession: dict[str, GenomeAllocation],
    temporary_dir: Path,
    seed: int,
    evaluation_references: list[EvaluationReference],
) -> dict[str, object]:
    """Stream the source and materialize only each genome's selected chunks."""
    patterns_by_genus = evaluation_patterns_by_genus(evaluation_references)
    patterns_by_accession = {}
    for accession, allocation in allocations_by_accession.items():
        # GTDB appends suffixes such as Escherichia_A when it subdivides a
        # genus. Compare only genomes belonging to an evaluation source genus.
        source_genus = allocation.taxonomy.genus.split("_", maxsplit=1)[0]
        if (
            allocation.split == "train"
            and allocation.selected_chunks > 0
            and source_genus in patterns_by_genus
        ):
            patterns_by_accession[accession] = patterns_by_genus[source_genus]

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
    # One genome can span batches and Parquet shards. Buffer only genomes that
    # need boundary-aware checking, then choose their safe rows as a unit.
    buffered_accession: str | None = None
    buffered_rows: list[dict[str, object]] = []
    checked_genera = set()
    chunks_checked = 0
    matching_chunks_excluded = 0

    def flush_checked_genome() -> None:
        nonlocal buffered_accession
        nonlocal buffered_rows
        nonlocal chunks_checked
        nonlocal matching_chunks_excluded

        if buffered_accession is None:
            return

        allocation = allocations_by_accession[buffered_accession]
        selected_rows, matches = select_nonleaking_chunks(
            buffered_rows,
            allocation,
            seed,
            patterns_by_accession[buffered_accession],
        )
        for row in selected_rows:
            writers_by_accession[buffered_accession].add(row)

        checked_genera.add(allocation.taxonomy.genus.split("_", maxsplit=1)[0])
        chunks_checked += len(buffered_rows)
        matching_chunks_excluded += matches
        buffered_accession = None
        buffered_rows = []

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
            materialized_rows = np.zeros(batch.num_rows, dtype=bool)
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

                if accession in patterns_by_accession:
                    # Materialize all full chunks from relevant genomes so an
                    # unsafe selected chunk can be replaced in this same pass.
                    materialized_rows[run_start + full_chunk_offsets] = True
                else:
                    # A genome can span multiple batches. Convert each selected
                    # chunk in this part of the genome to its row in this batch.
                    for ordinal in selected_ordinals:
                        if current_full_chunk <= ordinal < run_full_chunk_end:
                            offset_index = ordinal - current_full_chunk
                            materialized_rows[
                                run_start + int(full_chunk_offsets[offset_index])
                            ] = True

                current_full_chunk = run_full_chunk_end
                run_start = run_end

            candidate_batch = batch.filter(pa.array(materialized_rows))
            for row in candidate_batch.to_pylist():
                accession = row["gtdb_accession"]
                if accession in patterns_by_accession:
                    if buffered_accession != accession:
                        flush_checked_genome()
                        buffered_accession = accession
                    buffered_rows.append(row)
                else:
                    flush_checked_genome()
                    writers_by_accession[accession].add(row)

    flush_checked_genome()
    for writer in writers.values():
        writer.close()

    return {
        "genera_checked": sorted(checked_genera),
        "genomes_checked": len(patterns_by_accession),
        "chunks_checked": chunks_checked,
        "matching_chunks_excluded": matching_chunks_excluded,
    }


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
    """Summarize genome and chunk counts for each split."""
    statistics: dict[str, dict[str, int]] = {}

    for split in SPLITS:
        split_allocations = [
            allocation
            for allocation in allocations_by_accession.values()
            if allocation.split == split and allocation.selected_chunks > 0
        ]
        selected_chunks = sum(
            allocation.selected_chunks for allocation in split_allocations
        )
        statistics[split] = {
            "genomes": len(split_allocations),
            "chunks": selected_chunks,
        }

    return statistics


def composition_statistics(
    allocations_by_accession: dict[str, GenomeAllocation],
) -> dict[str, dict[str, dict[str, dict[str, int]]]]:
    """Summarize selected genomes and chunks by domain and phylum."""
    composition = {}

    for split in SPLITS:
        composition[split] = {}
        selected_allocations = [
            allocation
            for allocation in allocations_by_accession.values()
            if allocation.split == split and allocation.selected_chunks > 0
        ]

        for rank in ("domain", "phylum"):
            counts_by_taxon: dict[str, dict[str, int]] = {}
            for allocation in selected_allocations:
                taxon = getattr(allocation.taxonomy, rank)
                counts = counts_by_taxon.setdefault(
                    taxon,
                    {"genomes": 0, "chunks": 0},
                )
                counts["genomes"] += 1
                counts["chunks"] += allocation.selected_chunks

            composition[split][rank] = dict(sorted(counts_by_taxon.items()))

    return composition


def write_subset_stats(
    output_path: Path,
    allocations_by_accession: dict[str, GenomeAllocation],
    validation_fraction: float,
    seed: int,
    evaluation_references: list[EvaluationReference],
    evaluation_sequence_check: dict[str, object],
) -> None:
    """Write the selection parameters and resulting dataset counts."""
    split_stats = split_statistics(allocations_by_accession)
    summary = {
        "selection": {
            "seed": seed,
            "chunk_length": PARENT_CHUNK_LENGTH,
            "validation_fraction": validation_fraction,
        },
        "splits": split_stats,
        "composition": composition_statistics(allocations_by_accession),
        "evaluation_sequence_check": {
            "assay_ids": sorted(
                reference.assay_id for reference in evaluation_references
            ),
            "unique_wild_type_sequences": len(
                {reference.sequence for reference in evaluation_references}
            ),
            "reverse_complements_checked": True,
            **evaluation_sequence_check,
        },
        "genomes_without_full_chunks": sum(
            allocation.available_full_chunks == 0
            for allocation in allocations_by_accession.values()
        ),
    }
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def create_gtdb_subset(
    dataset_dir: Path,
    assay_dir: Path,
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
    evaluation_references = load_evaluation_references(assay_dir)
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
    evaluation_sequence_check = write_selected_chunks(
        shard_paths,
        source_schema,
        allocations_by_accession,
        temporary_dir,
        seed,
        evaluation_references,
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
        evaluation_references,
        evaluation_sequence_check,
    )

    # The temporary directory is the dataset under construction. Renaming it
    # exposes the single completed dataset without creating a second copy.
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
        "--assay-dir",
        type=Path,
        required=True,
        help="Directory containing standardized evaluation assay CSV files.",
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
        help="Fraction of eligible genomes used for validation.",
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
        assay_dir=args.assay_dir,
        train_bases=args.train_bases,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    print(json.dumps(statistics, indent=2))


if __name__ == "__main__":
    main()
