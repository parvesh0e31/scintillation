import argparse, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from phishing_detector.model import train_and_save

parser = argparse.ArgumentParser()
parser.add_argument("dataset")
parser.add_argument("--artifacts", default="artifacts")
args = parser.parse_args()
print(json.dumps(train_and_save(args.dataset, args.artifacts), indent=2))
