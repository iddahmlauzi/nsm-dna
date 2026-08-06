import argparse
import gzip
import io
import json
import re
import tarfile
from collections import deque
from collections.abc import Iterator
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path
from typing import TypedDict

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

CHUNK_LENGTH = 8_192
CHUNKS_PER_SHARD = 100_000
ACGT_SEGMENT_PATTERN = re.compile(rb"[ACGT]+")

GTDB_ARCHIVE_PATH = Path("/home/iddah/datasets/gtdb/raw/gtdb_genomes_reps.tar.gz")
OUTPUT_DIR = Path("/home/iddah/datasets/gtdb/processed")
NUM_WORKERS = 8
MAX_GENOMES = 199_923
GTDB_RELEASE = "R232"


class ChunkRecord(TypedDict):
    """Metadata and sequence for one processed chunk."""

    chunk_id: str
    ncbi_accession: str
    gtdb_accession: str
    archive_path: str
    record_id: str
    record_length: int
    chunk_start: int
    chunk_end: int
    chunk_length: int
    sequence: str


@dataclass
class ProcessingStats:
    """Counts describing the processed corpus."""

    genomes_processed: int = 0
    fasta_records_processed: int = 0
    input_bases: int = 0
    ambiguous_bases: int = 0
    retained_bases: int = 0
    chunks: int = 0
    full_length_chunks: int = 0
    short_chunks: int = 0
    short_chunk_bases: int = 0

    def add(self, other: "ProcessingStats") -> None:
        """Add counts from another processing result."""
        self.genomes_processed += other.genomes_processed
        self.fasta_records_processed += other.fasta_records_processed
        self.input_bases += other.input_bases
        self.ambiguous_bases += other.ambiguous_bases
        self.retained_bases += other.retained_bases
        self.chunks += other.chunks
        self.full_length_chunks += other.full_length_chunks
        self.short_chunks += other.short_chunks
        self.short_chunk_bases += other.short_chunk_bases


Genome = tuple[str, bytes]
ProcessedGenome = tuple[list[ChunkRecord], ProcessingStats]


def parse_genome_accessions(archive_path: str) -> tuple[str, str]:
    """Extract the NCBI and GTDB accessions from a genome archive path.

    Args:
        archive_path: Genome's file path inside the GTDB archive.

    Returns:
        A tuple containing the NCBI accession and corresponding GTDB accession.

    Raises:
        ValueError: If the filename does not contain a supported NCBI accession.
    """
    filename = Path(archive_path).name
    genome_suffix = "_genomic.fna.gz"

    if not filename.endswith(genome_suffix):
        raise ValueError(f"Unsupported genome filename: {filename}")

    ncbi_accession = filename.removesuffix(genome_suffix)

    if ncbi_accession.startswith("GCF_"):
        gtdb_accession = f"RS_{ncbi_accession}"
    elif ncbi_accession.startswith("GCA_"):
        gtdb_accession = f"GB_{ncbi_accession}"
    else:
        raise ValueError(f"Unsupported genome filename: {filename}")

    return ncbi_accession, gtdb_accession


def iter_genomes(gtdb_archive_path: Path) -> Iterator[tuple[str, bytes]]:
    """Iterate over compressed genome FASTA files in a GTDB archive.

    Args:
        gtdb_archive_path: Path to the GTDB `.tar.gz` archive.

    Yields:
        A tuple containing the file's path inside the archive and its
        compressed `.fna.gz` contents.
    """
    with tarfile.open(gtdb_archive_path, mode="r|gz") as archive:
        for member in archive:
            # Skip directories and non-FASTA files
            if not member.isfile() or not member.name.endswith(".fna.gz"):
                continue

            genome_file = archive.extractfile(member)
            if genome_file is None:
                continue

            with genome_file:
                compressed_contents = genome_file.read()

            yield member.name, compressed_contents


def iter_fasta_records(compressed_contents: bytes) -> Iterator[tuple[str, bytes]]:
    """Iterate over the FASTA records in a compressed genome file.

    Args:
        compressed_contents: Contents of a `.fna.gz` genome file.

    Yields:
        A tuple containing the FASTA record identifier and its nucleotide
        sequence.
    """
    current_record_id: str | None = None
    current_sequence_parts: list[bytes] = []

    with gzip.open(io.BytesIO(compressed_contents), mode="rb") as fasta_file:
        for line in fasta_file:
            line = line.strip()

            if not line:
                continue

            if line.startswith(b">"):
                if current_record_id is not None:
                    yield current_record_id, b"".join(current_sequence_parts)

                header = line[1:].decode("utf-8")
                current_record_id = header.split(maxsplit=1)[0]
                current_sequence_parts = []
            else:
                current_sequence_parts.append(line)

    if current_record_id is not None:
        yield current_record_id, b"".join(current_sequence_parts)


def iter_unambiguous_segments(sequence: bytes) -> Iterator[tuple[int, bytes]]:
    """Split a nucleotide sequence at non-ACGT characters.

    Args:
        sequence: Nucleotide sequence from one FASTA record.

    Yields:
        A tuple containing the segment's zero-based start coordinate and
        ACGT-only sequence.
    """
    normalized_sequence = sequence.upper()

    for match in ACGT_SEGMENT_PATTERN.finditer(normalized_sequence):
        yield match.start(), match.group()


def iter_chunks(
    segment: bytes,
    segment_start: int,
) -> Iterator[tuple[int, int, bytes]]:
    """Divide an unambiguous segment into chunks.

    Args:
        segment: ACGT-only segment from a FASTA record.
        segment_start: Segment's start coordinate in the original FASTA record.

    Yields:
        A tuple containing the chunk's zero-based start coordinate, exclusive
        end coordinate, and nucleotide sequence.
    """
    for offset in range(0, len(segment), CHUNK_LENGTH):
        chunk = segment[offset : offset + CHUNK_LENGTH]

        chunk_start = segment_start + offset
        chunk_end = chunk_start + len(chunk)

        yield chunk_start, chunk_end, chunk


def process_genome(
    archive_path: str,
    compressed_contents: bytes,
) -> tuple[list[ChunkRecord], ProcessingStats]:
    """Process every FASTA record in one compressed genome.

    Args:
        archive_path: Genome's file path inside the GTDB archive.
        compressed_contents: Contents of the compressed genome FASTA.

    Returns:
        A tuple containing the processed chunks and their processing counts.
    """
    ncbi_accession, gtdb_accession = parse_genome_accessions(archive_path)
    processed_chunks: list[ChunkRecord] = []
    processing_stats = ProcessingStats(genomes_processed=1)

    for record_id, sequence in iter_fasta_records(compressed_contents):
        record_length = len(sequence)
        retained_record_bases = 0

        processing_stats.fasta_records_processed += 1
        processing_stats.input_bases += record_length

        for segment_start, segment in iter_unambiguous_segments(sequence):
            retained_record_bases += len(segment)

            for chunk_start, chunk_end, chunk in iter_chunks(
                segment,
                segment_start,
            ):
                chunk_length = len(chunk)
                chunk_id = (
                    f"{gtdb_accession}|{record_id}|{chunk_start}-{chunk_end}"
                )
                chunk_record = {
                    "chunk_id": chunk_id,
                    "ncbi_accession": ncbi_accession,
                    "gtdb_accession": gtdb_accession,
                    "archive_path": archive_path,
                    "record_id": record_id,
                    "record_length": record_length,
                    "chunk_start": chunk_start,
                    "chunk_end": chunk_end,
                    "chunk_length": chunk_length,
                    "sequence": chunk.decode("ascii"),
                }
                processed_chunks.append(chunk_record)

                processing_stats.chunks += 1
                processing_stats.retained_bases += chunk_length

                if chunk_length == CHUNK_LENGTH:
                    processing_stats.full_length_chunks += 1
                else:
                    processing_stats.short_chunks += 1
                    processing_stats.short_chunk_bases += chunk_length

        processing_stats.ambiguous_bases += (
            record_length - retained_record_bases
        )

    return processed_chunks, processing_stats


def process_genomes_in_parallel(
    genomes: Iterator[Genome],
    num_workers: int,
) -> Iterator[ProcessedGenome]:
    """Process genomes concurrently while preserving their archive order.

    Args:
        genomes: Compressed genome files and their paths inside the archive.
        num_workers: Number of worker processes.

    Yields:
        The processed chunks and processing counts for each genome.

    Raises:
        ValueError: If fewer than one worker is requested.
    """
    if num_workers < 1:
        raise ValueError("num_workers must be at least 1")

    if num_workers == 1:
        for archive_path, compressed_contents in genomes:
            yield process_genome(archive_path, compressed_contents)
        return

    max_pending_genomes = num_workers * 2

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        pending_results: deque[Future[ProcessedGenome]] = deque()

        for archive_path, compressed_contents in genomes:
            pending_results.append(
                executor.submit(
                    process_genome,
                    archive_path,
                    compressed_contents,
                )
            )

            if len(pending_results) >= max_pending_genomes:
                yield pending_results.popleft().result()

        while pending_results:
            yield pending_results.popleft().result()


def write_chunks(
    processed_chunks: list[ChunkRecord],
    output_path: Path,
) -> None:
    """Write processed chunks to a Parquet file.

    Args:
        processed_chunks: Chunks produced from one genome.
        output_path: Path of the Parquet file to write.
    """
    table = pa.Table.from_pylist(processed_chunks)
    pq.write_table(table, output_path, compression="zstd")


def write_processing_stats(
    processing_stats: ProcessingStats,
    output_dir: Path,
) -> None:
    """Write aggregate corpus statistics to a JSON file.

    Args:
        processing_stats: Counts aggregated across all processed genomes.
        output_dir: Directory containing the processed dataset.
    """
    summary = {
        "gtdb_release": GTDB_RELEASE,
        "chunk_length": CHUNK_LENGTH,
        **asdict(processing_stats),
    }

    output_path = output_dir / "processing_stats.json"
    output_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )


def main(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)

    genomes = islice(
        iter_genomes(args.gtdb_archive),
        args.max_genomes,
    )

    chunk_buffer: list[ChunkRecord] = []
    processing_stats = ProcessingStats()
    shard_index = 0
    processed_genomes = process_genomes_in_parallel(
        genomes,
        args.num_workers,
    )

    for processed_chunks, genome_stats in tqdm(
        processed_genomes,
        total=args.max_genomes,
        unit="genome",
    ):
        chunk_buffer.extend(processed_chunks)
        processing_stats.add(genome_stats)

        while len(chunk_buffer) >= CHUNKS_PER_SHARD:
            output_path = args.output_dir / f"chunks-{shard_index:05d}.parquet"
            write_chunks(chunk_buffer[:CHUNKS_PER_SHARD], output_path)
            del chunk_buffer[:CHUNKS_PER_SHARD]
            shard_index += 1

    if chunk_buffer:
        output_path = args.output_dir / f"chunks-{shard_index:05d}.parquet"
        write_chunks(chunk_buffer, output_path)

    write_processing_stats(processing_stats, args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--gtdb-archive", type=Path, default=GTDB_ARCHIVE_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--max-genomes", type=int, default=MAX_GENOMES)

    args = parser.parse_args()
    main(args)
