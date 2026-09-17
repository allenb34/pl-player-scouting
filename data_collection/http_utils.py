"""Shared HTTP helpers: rate limiting, retry/backoff, and error logging.

Used by the FBref fetch scripts so they handle throttling and transient
failures the same way. Adapted from pl-transfer-value-predictor's
data_collection/http_utils.py.
"""
import time
from datetime import datetime, timezone
from pathlib import Path

import truststore

truststore.inject_into_ssl()  # fixes CERTIFICATE_VERIFY_FAILED behind this machine's TLS-inspecting proxy

import requests

ERROR_LOG_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "collection_errors.log"

# FBref (Cloudflare) is UA-sensitive - a realistic browser UA is required just to get
# past the initial request, though it is not on its own sufficient to bypass the
# Cloudflare JS challenge (confirmed during development - see fetch_fbref_direct.py).
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
}


def log_error(context: str, reason: str) -> None:
    ERROR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    with open(ERROR_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {context} | {reason}\n")


class RateLimiter:
    """Ensures at least `min_interval` seconds pass between calls."""

    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._last_call = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last_call
        remaining = self.min_interval - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self._last_call = time.monotonic()


def get_html(
    session: requests.Session,
    url: str,
    rate_limiter: RateLimiter,
    context: str,
    max_retries: int = 3,
    timeout: int = 20,
):
    """GET a URL and return response text, or None on unrecoverable failure.

    Retries on network errors, 429 (respecting Retry-After), and 5xx with
    exponential backoff. Logs every failure (transient or final) via
    log_error and never raises - callers rely on a None return to fall
    through to the next data-source attempt.
    """
    for attempt in range(1, max_retries + 1):
        rate_limiter.wait()
        try:
            resp = session.get(url, headers=BROWSER_HEADERS, timeout=timeout)
        except requests.RequestException as e:
            log_error(context, f"network error on attempt {attempt}/{max_retries}: {e}")
            time.sleep(min(2 ** attempt, 30))
            continue

        if resp.status_code == 200:
            return resp.text

        if resp.status_code == 403:
            log_error(context, f"HTTP 403 (likely Cloudflare challenge): {resp.text[:200]}")
            return None

        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            wait_s = float(retry_after) if retry_after else min(2 ** attempt * 5, 60)
            log_error(context, f"rate limited (429), waiting {wait_s}s (attempt {attempt}/{max_retries})")
            time.sleep(wait_s)
            continue

        if resp.status_code in (500, 502, 503, 504):
            log_error(context, f"server error {resp.status_code} (attempt {attempt}/{max_retries})")
            time.sleep(min(2 ** attempt, 30))
            continue

        log_error(context, f"HTTP {resp.status_code}: {resp.text[:300]}")
        return None

    log_error(context, f"gave up after {max_retries} attempts")
    return None
