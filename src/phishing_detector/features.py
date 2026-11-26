"""Safe, URL-only feature extraction. No network access is performed."""
from __future__ import annotations

import ipaddress
import math
import re
from urllib.parse import urlsplit

MAX_URL_LENGTH = 2048
SUPPORTED_SCHEMES = {"http", "https"}
FEATURE_NAMES = [
    "url_length", "hostname_length", "path_length", "query_length",
    "fragment_length", "num_dots", "num_hyphens", "num_digits",
    "num_special_chars", "num_subdomains", "has_ip", "has_at",
    "has_punycode", "has_port", "has_https", "has_suspicious_word",
    "entropy", "path_depth", "query_params",
]
_SUSPICIOUS = re.compile(r"(login|verify|secure|account|update|signin|bank|confirm|password)", re.I)


def validate_url(url: str) -> str:
    if not isinstance(url, str):
        raise ValueError("url must be a string")
    value = url.strip()
    if not value or len(value) > MAX_URL_LENGTH:
        raise ValueError(f"url must be between 1 and {MAX_URL_LENGTH} characters")
    try:
        parsed = urlsplit(value)
        if parsed.scheme.lower() not in SUPPORTED_SCHEMES or not parsed.netloc:
            raise ValueError("only absolute http(s) URLs are supported")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("URLs containing credentials are not accepted")
        if not parsed.hostname:
            raise ValueError("URL hostname is required")
        parsed.port  # force validation of malformed ports
    except (ValueError, UnicodeError) as exc:
        raise ValueError(f"invalid URL: {exc}") from exc
    return value


def _entropy(value: str) -> float:
    if not value:
        return 0.0
    counts = {char: value.count(char) for char in set(value)}
    return -sum((n / len(value)) * math.log2(n / len(value)) for n in counts.values())


def _is_ip(hostname: str) -> bool:
    try:
        ipaddress.ip_address(hostname)
        return True
    except ValueError:
        return False


def extract_features(url: str) -> dict[str, float]:
    """Return deterministic numeric features for one validated URL."""
    value = validate_url(url)
    parsed = urlsplit(value)
    host = parsed.hostname or ""
    lower = value.lower()
    labels = host.split(".")
    return {
        "url_length": float(len(value)),
        "hostname_length": float(len(host)),
        "path_length": float(len(parsed.path)),
        "query_length": float(len(parsed.query)),
        "fragment_length": float(len(parsed.fragment)),
        "num_dots": float(value.count(".")),
        "num_hyphens": float(value.count("-")),
        "num_digits": float(sum(c.isdigit() for c in value)),
        "num_special_chars": float(sum(not c.isalnum() for c in value)),
        "num_subdomains": float(max(0, len(labels) - 2)),
        "has_ip": float(_is_ip(host)),
        "has_at": float("@" in value),
        "has_punycode": float(any(label.startswith("xn--") for label in labels)),
        "has_port": float(parsed.port is not None),
        "has_https": float(parsed.scheme.lower() == "https"),
        "has_suspicious_word": float(bool(_SUSPICIOUS.search(lower))),
        "entropy": _entropy(value),
        "path_depth": float(len([part for part in parsed.path.split("/") if part])),
        "query_params": float(len([part for part in parsed.query.split("&") if part])),
    }


def feature_matrix(urls: list[str]):
    import pandas as pd
    return pd.DataFrame([extract_features(url) for url in urls], columns=FEATURE_NAMES)
