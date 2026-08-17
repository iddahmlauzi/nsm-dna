import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from nsm_dna.data import (
    collate_dna_sequences,
    load_gtdb_dataset,
    split_sequences,
)


class DNADataTest(unittest.TestCase):
    def test_splits_stored_sequences_at_the_context_length(self) -> None:
        examples = {
            "sequence": ["AAAACCCC", "GGGGTTTT"],
        }

        result = split_sequences(examples, context_length=4)

        self.assertEqual(result["sequence"], ["AAAA", "CCCC", "GGGG", "TTTT"])

    def test_collates_equal_length_sequences_without_padding(self) -> None:
        batch = collate_dna_sequences(
            [
                {"sequence": "ACGT"},
                {"sequence": "TGCA"},
            ]
        )

        self.assertEqual(batch["input_ids"].tolist(), [[0, 1, 2, 3], [3, 2, 1, 0]])
        self.assertEqual(set(batch), {"input_ids"})

    def test_streams_the_requested_physical_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            subset_directory = Path(temporary_directory)
            validation_directory = subset_directory / "validation"
            validation_directory.mkdir()
            table = pa.table({"sequence": ["AAAACCCC", "GGGGTTTT"]})
            pq.write_table(
                table,
                validation_directory / "chunks-00000.parquet",
            )

            with patch(
                "datasets.iterable_dataset."
                "_maybe_share_with_torch_persistent_workers",
                side_effect=lambda value: value,
            ):
                dataset = load_gtdb_dataset(
                    subset_directory,
                    split="validation",
                    context_length=4,
                )

                self.assertEqual(
                    [example["sequence"] for example in dataset],
                    ["AAAA", "CCCC", "GGGG", "TTTT"],
                )

    def test_distributed_ranks_receive_different_sequences(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            subset_directory = Path(temporary_directory)
            validation_directory = subset_directory / "validation"
            validation_directory.mkdir()
            table = pa.table({"sequence": ["AAAACCCC", "GGGGTTTT"]})
            pq.write_table(
                table,
                validation_directory / "chunks-00000.parquet",
            )

            sequences_by_rank = []
            with patch(
                "datasets.iterable_dataset."
                "_maybe_share_with_torch_persistent_workers",
                side_effect=lambda value: value,
            ):
                for rank in range(2):
                    dataset = load_gtdb_dataset(
                        subset_directory,
                        split="validation",
                        context_length=4,
                        rank=rank,
                        world_size=2,
                    )
                    sequences_by_rank.append(
                        {example["sequence"] for example in dataset}
                    )

            self.assertTrue(sequences_by_rank[0].isdisjoint(sequences_by_rank[1]))
            self.assertEqual(
                sequences_by_rank[0] | sequences_by_rank[1],
                {"AAAA", "CCCC", "GGGG", "TTTT"},
            )


if __name__ == "__main__":
    unittest.main()
