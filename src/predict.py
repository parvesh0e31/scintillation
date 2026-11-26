"""Predict one URL with a saved detector."""

import argparse
import json

from phishing_detector.predict import load_model, predict_url


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify one URL without visiting it")
    parser.add_argument("url")
    parser.add_argument("--model", default="models/model.joblib")
    args = parser.parse_args()
    print(json.dumps(predict_url(load_model(args.model), args.url), indent=2))


if __name__ == "__main__":
    main()
