"""Validate a labeled URL CSV and write its canonical feature-ready form."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phishing_detector.dataset import load_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and canonicalize a URL dataset")
    parser.add_argument("input_csv")
    parser.add_argument("output_csv")
    args = parser.parse_args()
    frame = load_dataset(args.input_csv)
    frame.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(frame)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
