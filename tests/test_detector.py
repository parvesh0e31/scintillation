import pytest
from phishing_detector.features import extract_features, validate_url
from phishing_detector.model import split_dataset
import pandas as pd

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
