"""Draw the search chart from a leaderboard CSV. No hardcoded scores.

python scripts/make_search_chart.py results/baby-food/leaderboard.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from raglab.search_report import write_search_chart


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw a search chart from leaderboard.csv")
    parser.add_argument("csv", type=Path, help="Leaderboard CSV written by raglab search")
    parser.add_argument(
        "--output", type=Path, help="PNG path. Defaults to leaderboard.png beside the CSV"
    )
    args = parser.parse_args()
    output = args.output or args.csv.with_suffix(".png")
    write_search_chart(args.csv, output)
    print(output)


if __name__ == "__main__":
    main()
