import json
import pandas as pd
import pytest
import sklearn
from fastapi.testclient import TestClient

from phishing_detector import api
from phishing_detector.dataset import load_dataset
from phishing_detector.features import extract_features, validate_url
from phishing_detector.model import _group, split_dataset, train_and_save
from phishing_detector.predict import load_model, predict_url

def test_features_are_deterministic_and_numeric():
    first = extract_features("https://example.com/login?a=1")
    assert first == extract_features("https://example.com/login?a=1")
    assert first["has_https"] == 1

@pytest.mark.parametrize("url", ["example.com", "ftp://example.com", "https://user:pass@example.com/x", "x" * 2049])
def test_rejects_unsafe_urls(url):
    with pytest.raises(ValueError):
        validate_url(url)

def test_group_split_does_not_crash():
    frame = pd.DataFrame({"url": ["https://a.example.com/x", "https://b.sample.net/y", "https://x.other.org/a", "https://y.site.io/b"], "label": [0, 0, 1, 1]})
    train, test = split_dataset(frame, test_size=.5)
    assert set(train.index).isdisjoint(test.index)
    assert train["label"].nunique() == 2
    assert test["label"].nunique() == 2

    assert {_group(url) for url in train["url"]}.isdisjoint(
        {_group(url) for url in test["url"]}
    )

def test_group_split_rejects_too_few_domains():
    frame = pd.DataFrame(
        {"url": ["https://a.example.com/x", "https://b.example.com/y"], "label": [0, 1]}
    )
    with pytest.raises(ValueError, match="four distinct domain groups"):
        split_dataset(frame)

def test_dataset_rejects_conflicting_duplicate_labels(tmp_path):
    dataset_path = tmp_path / "conflicting.csv"
    pd.DataFrame(
        {
            "url": ["https://example.com/", "https://example.com/"],
            "label": ["legitimate", "phishing"],
        }
    ).to_csv(dataset_path, index=False)
    with pytest.raises(ValueError, match="conflicting labels"):
        load_dataset(str(dataset_path))

def test_training_saves_a_model_that_can_predict(tmp_path, monkeypatch):
    rows = [
        (f"https://benign-{index}.example/", 0)
        for index in range(5)
    ] + [
        (f"https://phishing-{index}.example/login-verify", 1)
        for index in range(5)
    ]
    dataset_path = tmp_path / "dataset.csv"
    pd.DataFrame(rows, columns=["url", "label"]).to_csv(dataset_path, index=False)
    artifact_dir = tmp_path / "models"

    metrics = train_and_save(str(dataset_path), str(artifact_dir))
    model = load_model(str(artifact_dir / "model.joblib"))
    result = predict_url(model, "https://example.test/account")

    assert "accuracy" in metrics
    assert (artifact_dir / "best_model.pkl").exists()
    metadata = json.loads((artifact_dir / "metadata.json").read_text())
    assert metadata["scikit_learn_version"] == sklearn.__version__
    assert result["label"] in {"legitimate", "phishing"}
    assert 0.0 <= result["phishing_probability"] <= 1.0

    monkeypatch.setattr(api, "MODEL_PATH", str(artifact_dir / "model.joblib"))
    monkeypatch.setattr(api, "_model", None)
    client = TestClient(api.app)
    response = client.post("/predict", json={"url": "https://example.test/account"})
    assert response.status_code == 200
    assert response.json()["label"] in {"legitimate", "phishing"}
    assert client.post("/predict", json={"url": "ftp://example.test"}).status_code == 422
