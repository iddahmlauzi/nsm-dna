import csv
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from evaluate_evo1 import (  # noqa: E402
    REPORTED_CORRELATIONS,
    STUDY_NAMES,
    calculate_spearman,
    correlation_rows,
    prediction_is_complete,
    select_assay_files,
    write_predictions,
)

SOURCE_COLUMNS = (
    "study_id",
    "assay_id",
    "variant_id",
    "mutant_nt",
    "mutant_aa",
    "experimental_score",
    "directionality",
)


def source_row(
    study_id: str,
    assay_id: str,
    variant_id: str,
) -> dict[str, str]:
    """Build one small standardized row for evaluation tests."""
    return {
        "study_id": study_id,
        "assay_id": assay_id,
        "variant_id": variant_id,
        "mutant_nt": "ATGGCT",
        "mutant_aa": "MA",
        "experimental_score": "1.0",
        "directionality": "1",
    }


def write_source(path: Path, rows: list[dict[str, str]]) -> None:
    """Write a minimal standardized assay CSV."""
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=SOURCE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


class EvoEvaluationTest(unittest.TestCase):
    """Test the data handling around Evo inference."""

    def test_selects_every_assay_for_a_study(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir)
            write_source(
                input_dir / "target_a.csv",
                [source_row("tsuboyama_2023", "target_a", "variant_a")],
            )
            write_source(
                input_dir / "target_b.csv",
                [source_row("tsuboyama_2023", "target_b", "variant_b")],
            )
            write_source(
                input_dir / "other.csv",
                [source_row("adkar_2012", "other", "variant_c")],
            )

            selected = select_assay_files(input_dir, ("tsuboyama_2023",))

            self.assertEqual(
                [path.name for path in selected],
                ["target_a.csv", "target_b.csv"],
            )

    def test_directionality_changes_adkar_to_higher_is_better(self) -> None:
        rows = [
            {
                "experimental_score": experimental_score,
                "directionality": "-1",
                "evo_log_likelihood": model_score,
            }
            for experimental_score, model_score in (
                ("3", "1"),
                ("2", "2"),
                ("1", "3"),
            )
        ]

        self.assertAlmostEqual(calculate_spearman(rows), 1.0)

    def test_macro_average_uses_the_nine_studies_equally(self) -> None:
        results = {
            study: {
                "num_variants": index + 1,
                "spearman": REPORTED_CORRELATIONS[study],
            }
            for index, study in enumerate(STUDY_NAMES)
        }

        rows = correlation_rows(results, STUDY_NAMES)

        self.assertEqual(rows[-1]["study_id"], "macro_average")
        self.assertAlmostEqual(float(rows[-1]["spearman"]), 4.13 / 9)
        self.assertEqual(rows[-1]["num_variants"], sum(range(1, 10)))

    def test_completed_prediction_must_match_source_identifiers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "predictions.csv"
            source_rows = [
                source_row("adkar_2012", "assay", "variant_a"),
                source_row("adkar_2012", "assay", "variant_b"),
            ]
            write_predictions(source_rows, [0.1, 0.2], output_path)

            self.assertTrue(prediction_is_complete(source_rows, output_path))

            changed_rows = [
                source_rows[0],
                source_row("adkar_2012", "assay", "variant_c"),
            ]
            self.assertFalse(prediction_is_complete(changed_rows, output_path))


if __name__ == "__main__":
    unittest.main()
