from __future__ import annotations
import json
from pathlib import Path
import joblib
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from .dataset import load_dataset
from .features import FEATURE_NAMES, feature_matrix


def _group(url: str) -> str:
    from urllib.parse import urlsplit
    host = urlsplit(url).hostname or ""
    parts = host.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


def split_dataset(frame: pd.DataFrame, test_size=0.2, random_state=42):
    if not 0 < test_size < 1:
        raise ValueError("test_size must be greater than 0 and less than 1")
    if frame["label"].nunique() != 2:
        raise ValueError("dataset must contain both label classes")

    groups = frame["url"].map(_group)
    if groups.nunique() < 4:
        raise ValueError(
            "at least four distinct domain groups are required for a grouped "
            "train/test split with both classes in each set"
        )
    groups_per_class = pd.DataFrame({"label": frame["label"], "group": groups}).groupby("label")[
        "group"
    ].nunique()
    if (groups_per_class < 2).any():
        raise ValueError("each label class must appear in at least two distinct domain groups")

    # Grouping by registrable-domain approximation prevents related subdomains
    # from crossing the evaluation boundary (a common URL leakage).
    splitter = GroupShuffleSplit(n_splits=100, test_size=test_size, random_state=random_state)
    for train_idx, test_idx in splitter.split(frame, frame["label"], groups):
        train, test = frame.iloc[train_idx], frame.iloc[test_idx]
        if train["label"].nunique() == 2 and test["label"].nunique() == 2:
            return train, test
    raise ValueError(
        "unable to create a grouped split with both classes in train and test; "
        "add more distinct domains for each class or adjust test_size"
    )


def train_model(train: pd.DataFrame) -> Pipeline:
    pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))])
    pipeline.fit(feature_matrix(train["url"].tolist()), train["label"])
    return pipeline


def evaluate_model(model, frame: pd.DataFrame) -> dict:
    x = feature_matrix(frame["url"].tolist())
    y = frame["label"]
    pred = model.predict(x)
    probabilities = model.predict_proba(x)[:, 1]
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    result = {"accuracy": float(accuracy_score(y, pred)), "precision": float(precision), "recall": float(recall), "f1": float(f1)}
    if len(set(y)) == 2:
        result["roc_auc"] = float(roc_auc_score(y, probabilities))
    return result


def train_and_save(csv_path: str, artifact_dir: str, test_size=0.2) -> dict:
    frame = load_dataset(csv_path)
    train, test = split_dataset(frame, test_size=test_size)
    model = train_model(train)
    metrics = evaluate_model(model, test)
    output = Path(artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output / "model.joblib")
    # Keep the complete pipeline as the serving artifact, plus conventional
    # names for consumers that expect separate model/scaler files.
    joblib.dump(model, output / "best_model.pkl")
    joblib.dump(model.named_steps["scaler"], output / "scaler.pkl")
    (output / "metadata.json").write_text(
        json.dumps(
            {
                "feature_names": FEATURE_NAMES,
                "metrics": metrics,
                "train_rows": len(train),
                "test_rows": len(test),
                "scikit_learn_version": sklearn.__version__,
                "artifacts": ["model.joblib", "best_model.pkl", "scaler.pkl", "metadata.json"],
            },
            indent=2,
        )
    )
    return metrics
