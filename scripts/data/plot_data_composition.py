import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def top_categories(counts: dict[str, int], limit: int) -> dict[str, int]:
    """Keep the largest categories and combine the remainder as Other."""
    ordered = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    shown = dict(ordered[:limit])
    other = sum(count for _, count in ordered[limit:])
    if other:
        shown["Other"] = other
    return shown


def plot_composition(
    statistics: dict[str, object],
    output_path: Path,
    split: str = "train",
) -> None:
    """Plot genome counts by domain and phylum for one selected split."""
    split_composition = statistics["composition"][split]
    rank_counts = {
        rank: {
            taxon or "Unclassified": counts["genomes"]
            for taxon, counts in split_composition[rank].items()
        }
        for rank in ("domain", "phylum")
    }
    rank_counts["phylum"] = top_categories(rank_counts["phylum"], limit=10)

    figure, axes = plt.subplots(1, 2, figsize=(13, 6))
    for axis, rank in zip(axes, ("domain", "phylum"), strict=True):
        counts = rank_counts[rank]
        wedges, _ = axis.pie(
            counts.values(),
            colors=plt.get_cmap("tab20").colors[: len(counts)],
            startangle=90,
            wedgeprops={"width": 0.65, "edgecolor": "white"},
        )
        axis.set_title(rank.capitalize())
        axis.legend(
            wedges,
            [
                f"{taxon} (N = {count:,})"
                for taxon, count in counts.items()
            ],
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            frameon=False,
        )

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot domain and phylum composition from subset_stats.json."
    )
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("train", "validation"), default="train"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    statistics = json.loads(args.stats.read_text(encoding="utf-8"))
    plot_composition(statistics, args.output, args.split)


if __name__ == "__main__":
    main()
