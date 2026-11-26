"""Application entry point for the FastAPI service."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phishing_detector.api import app  # noqa: E402

__all__ = ["app"]
