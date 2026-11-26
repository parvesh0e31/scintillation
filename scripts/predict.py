import argparse, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from phishing_detector.predict import load_model, predict_url

parser = argparse.ArgumentParser()
parser.add_argument("url")
parser.add_argument("--model", default="models/model.joblib")
args = parser.parse_args()
print(json.dumps(predict_url(load_model(args.model), args.url), indent=2))
