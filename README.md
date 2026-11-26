# URL-only phishing detector baseline

This repository implements a reproducible, deliberately conservative baseline
for classifying URLs. It extracts lexical/structural features locally; **it
never resolves DNS, opens sockets, follows redirects, or fetches URL content**.
It is a research baseline, not a security verdict.

## Dataset schema

Input is CSV with the required header `url,label`. `url` must be an absolute
`http://` or `https://` URL without credentials and at most 2048 characters.
`label` is `0`/`1`, or `legitimate`/`phishing` (also `benign`). Extra columns
are ignored. No dataset or model binary is included. To canonicalize a
user-supplied dataset:

```bash
python utils/build_features.py raw.csv data/phishing_dataset_v1.csv
```

For a quick demonstration, prepare a small, **manually labeled** CSV with
multiple unrelated domains in both classes (the project does not generate
sample labels or download a dataset):

```csv
url,label
https://your-labeled-benign-domain.example/,legitimate
https://another-labeled-benign-domain.example/,legitimate
https://your-labeled-phishing-domain.example/login,phishing
https://another-labeled-phishing-domain.example/verify,phishing
```

Use a sufficiently large, representative dataset for meaningful evaluation;
four rows only illustrate the expected format and may not produce a valid
grouped train/test split.

## Quick start

```bash
git clone https://github.com/parvesh0e31/scintillation.git
cd scintillation
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

Prepare a labeled CSV as described above, then train, evaluate, and classify:

```bash
python src/train_model.py data/phishing_dataset_v1.csv --artifacts models
python src/evaluate.py data/phishing_dataset_v1.csv --model models/model.joblib
python src/predict.py https://example.com/login --model models/model.joblib
```

Training uses a grouped holdout split by registrable-domain approximation so
related subdomains cannot leak between train and test. Artifacts are
`model.joblib` (the complete scaler-plus-classifier pipeline), `best_model.pkl`
(a compatibility alias), `scaler.pkl`, and human-readable `metadata.json`.
They are generated locally and intentionally not committed.

## API and UI

After training:

```bash
uvicorn app.api:app --reload
streamlit run app/streamlit_app.py
```

Open the Streamlit URL printed in the terminal (usually
`http://localhost:8501`) and enter a URL to analyze. For the API, use a second
terminal:

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"url":"https://example.com/login"}'
```

`GET /health` reports service status and `POST /predict` accepts
`{"url":"https://example.com"}`. Inputs are length-bounded and validated.
The API returns HTTP 503 until a model artifact exists and HTTP 422 for invalid
URLs. Set `PHISHING_MODEL` to use a model artifact at another path.

## Development

```bash
pytest -q
```

The tests cover deterministic feature extraction, URL validation, and an
end-to-end train/save/load/predict flow. Treat model scores as baseline outputs,
not calibrated probabilities; collect a representative, independently labeled
dataset for any real use.
