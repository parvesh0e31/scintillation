"""Train a detector from a labeled CSV.

This module is importable for notebooks and runnable as a CLI:
``python src/train_model.py data/dataset.csv --artifacts models``.
"""

import argparse
import json

from phishing_detector.model import train_and_save


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the URL phishing detector")
    parser.add_argument("dataset", help="CSV containing url,label columns")
    parser.add_argument("--artifacts", default="models", help="Artifact output directory")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()
    print(json.dumps(train_and_save(args.dataset, args.artifacts, args.test_size), indent=2))


if __name__ == "__main__":
    main()
