"""Command-line entry point for running structured pruning experiments."""

import argparse
import sys
from pathlib import Path

CWD = Path(__file__).resolve().parent
SRC_DIR = CWD / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rlp.cli.run import run_experiment


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Structured pruning controller")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_experiment(args.config)


if __name__ == "__main__":
    main()
