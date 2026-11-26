"""Train with the bundled demo data if needed, then launch the Streamlit UI."""

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "data" / "phishing_dataset_v1.csv"
DEFAULT_MODEL = ROOT / "models" / "model.joblib"


def main() -> None:
    configured_model = os.environ.get("PHISHING_MODEL")
    if configured_model:
        model_path = Path(configured_model).expanduser()
        if not model_path.is_absolute():
            model_path = ROOT / model_path
        if not model_path.is_file():
            raise SystemExit(
                f"PHISHING_MODEL points to a missing file: {model_path}\n"
                "Unset PHISHING_MODEL to use the bundled demo model, or train "
                "a model at the configured path."
            )
    else:
        model_path = DEFAULT_MODEL
        if not model_path.is_file():
            if not DATASET.is_file():
                raise SystemExit(f"Demo dataset not found: {DATASET}")
            print("No trained model found; creating one from the bundled demo data.")
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "src" / "train_model.py"),
                    str(DATASET),
                    "--artifacts",
                    str(model_path.parent),
                ],
                cwd=ROOT,
                check=True,
            )

    os.environ["PHISHING_MODEL"] = str(model_path)
    print("Starting the URL phishing detector. Open the Streamlit URL shown below.")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(ROOT / "app" / "streamlit_app.py"),
        ],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
