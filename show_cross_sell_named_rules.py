from argparse import ArgumentParser
from pathlib import Path

import pandas as pd


DEFAULT_RULES_PATH = (
    Path("artifacts")
    / "recommendation_outputs"
    / "cross_sell_association_rules_best_with_product_names.parquet"
)


def main() -> None:
    parser = ArgumentParser(
        description="Show cross-sell association rules with product names."
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=20,
        help="Number of rows to print. Default: 20",
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_RULES_PATH,
        help=f"Rules parquet path. Default: {DEFAULT_RULES_PATH}",
    )
    args = parser.parse_args()

    if not args.path.exists():
        raise FileNotFoundError(f"Rules file not found: {args.path}")

    rules = pd.read_parquet(args.path)
    print(rules.head(args.rows).to_string(index=False))


if __name__ == "__main__":
    main()
