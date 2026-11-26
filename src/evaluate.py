"""Evaluate a saved detector against a labeled CSV."""

import argparse
import json

import joblib

from phishing_detector.dataset import load_dataset
from phishing_detector.model import evaluate_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the URL phishing detector")
    parser.add_argument("dataset")
    parser.add_argument("--model", default="models/model.joblib")
    args = parser.parse_args()
    print(json.dumps(evaluate_model(joblib.load(args.model), load_dataset(args.dataset)), indent=2))


if __name__ == "__main__":
    main()
