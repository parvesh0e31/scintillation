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

## Train and evaluate

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python src/train_model.py data/phishing_dataset_v1.csv --artifacts models
python src/evaluate.py data/phishing_dataset_v1.csv --model models/best_model.pkl
python src/predict.py https://example.com/login --model models/best_model.pkl
```

Training uses a grouped holdout split by registrable-domain approximation so
related subdomains cannot leak between train and test. Artifacts are
`best_model.pkl` (the complete scaler-plus-classifier pipeline), `scaler.pkl`,
and human-readable `metadata.json`; they are generated locally and
intentionally not committed. `model.joblib` is also written as a compatibility
alias.

## API and UI

After training:

```bash
uvicorn app.api:app --reload
streamlit run app/streamlit_app.py
```

`GET /health` reports service status and `POST /predict` accepts
`{"url":"https://example.com"}`. Inputs are length-bounded and validated.
The API returns HTTP 503 until a model artifact exists and HTTP 422 for invalid
URLs.

## Development

```bash
pytest -q
```

The tests cover deterministic feature extraction and validation. Do not treat
the generated sample outputs as calibrated probabilities; collect a
representative, independently labeled dataset for any real use.
