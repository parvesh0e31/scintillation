"""Compatibility entry point for the URL feature extractor."""

from phishing_detector.features import (
    FEATURE_NAMES,
    MAX_URL_LENGTH,
    extract_features,
    feature_matrix,
    validate_url,
)

__all__ = [
    "FEATURE_NAMES",
    "MAX_URL_LENGTH",
    "extract_features",
    "feature_matrix",
    "validate_url",
]
