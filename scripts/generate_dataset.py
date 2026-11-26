"""Validate/normalize a labeled CSV; never downloads or invents URLs."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from phishing_detector.dataset import load_dataset

parser = argparse.ArgumentParser(description="Validate a URL,label CSV and write canonical CSV")
parser.add_argument("input_csv")
parser.add_argument("output_csv")
args = parser.parse_args()
frame = load_dataset(args.input_csv)
frame.to_csv(args.output_csv, index=False)
print(f"Wrote {len(frame)} rows to {args.output_csv}")
