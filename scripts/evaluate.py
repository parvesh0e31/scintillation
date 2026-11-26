import argparse, json
from pathlib import Path
import sys, joblib
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from phishing_detector.dataset import load_dataset
from phishing_detector.model import evaluate_model

parser = argparse.ArgumentParser()
parser.add_argument("dataset")
parser.add_argument("--model", default="artifacts/model.joblib")
args = parser.parse_args()
print(json.dumps(evaluate_model(joblib.load(args.model), load_dataset(args.dataset)), indent=2))
