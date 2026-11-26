"""CSV schema and loading helpers.

Required columns: ``url`` (absolute http/https URL), ``label`` (0/1 or
``legitimate``/``phishing``). Additional columns are ignored.
"""
from __future__ import annotations
import pandas as pd
from .features import validate_url

SCHEMA = "url,label"


def load_dataset(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"url", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {', '.join(sorted(missing))}")
    frame = frame[["url", "label"]].dropna().copy()
    frame["url"] = frame["url"].map(validate_url)
    def normalize(label):
        if isinstance(label, str):
            text = label.strip().lower()
            if text in {"phishing", "1", "true", "yes"}: return 1
            if text in {"legitimate", "benign", "0", "false", "no"}: return 0
        if label in (0, 1): return int(label)
        raise ValueError("label must be 0/1 or legitimate/phishing")
    frame["label"] = frame["label"].map(normalize).astype("int8")
    if frame.empty or frame["label"].nunique() < 2:
        raise ValueError("dataset must contain both label classes")
    return frame.drop_duplicates(subset=["url"]).reset_index(drop=True)
