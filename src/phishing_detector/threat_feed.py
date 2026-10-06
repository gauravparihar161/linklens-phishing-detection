"""Local threat-feed cache and exact URL matching.

Submitted URLs are compared locally and are never sent to the feed provider.
The user initiates feed refresh after reviewing the provider's terms.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
FEED_PATH = ROOT / "data" / "cache" / "openphish_community.txt"
META_PATH = ROOT / "data" / "cache" / "openphish_community.json"
FEED_URL = "https://raw.githubusercontent.com/openphish/public_feed/refs/heads/main/feed.txt"
MAX_FEED_BYTES = 5_000_000


def _url_keys(value: str) -> set[str]:
    """Build strict full-URL keys with and without the scheme."""
    raw = str(value).strip()
    if not raw:
        return set()
    if "://" not in raw:
        raw = "https://" + raw
    try:
        parts = urlsplit(raw)
        if not parts.hostname:
            return set()
        host = parts.hostname.lower()
        if parts.port and not ((parts.scheme.lower() == "https" and parts.port == 443) or
                               (parts.scheme.lower() == "http" and parts.port == 80)):
            host = f"{host}:{parts.port}"
        path = parts.path or "/"
        # URL fragments are client-side and not sent to the server.
        without_fragment = urlunsplit((parts.scheme.lower(), host, path, parts.query, ""))
        schemeless = f"{host}{path}" + (f"?{parts.query}" if parts.query else "")
        return {without_fragment, schemeless}
    except ValueError:
        return set()


def load_feed() -> dict:
    """Load cached feed and metadata; absent cache is an unavailable state."""
    if not FEED_PATH.exists():
        return {"available": False, "urls": set(), "updated_at": None, "count": 0}
    try:
        meta = json.loads(META_PATH.read_text(encoding="utf-8")) if META_PATH.exists() else {}
        entries = FEED_PATH.read_text(encoding="utf-8").splitlines()
        keys: set[str] = set()
        for entry in entries:
            keys.update(_url_keys(entry))
        return {
            "available": bool(keys),
            "urls": keys,
            "updated_at": meta.get("updated_at"),
            "count": int(meta.get("entry_count", len(entries))),
            "source": meta.get("source", "OpenPhish Community Feed"),
        }
    except (OSError, ValueError, json.JSONDecodeError):
        return {"available": False, "urls": set(), "updated_at": None, "count": 0}


def lookup_url(value: str, feed: dict | None = None) -> bool | None:
    """Return True for a feed match, False for no match, or None if unavailable."""
    feed = feed or load_feed()
    if not feed.get("available"):
        return None
    return bool(_url_keys(value) & feed["urls"])


def refresh_feed(timeout_seconds: int = 20) -> dict:
    """Download the public feed, validate it, then atomically update local cache."""
    request = Request(FEED_URL, headers={"User-Agent": "LinkLens-Portfolio/0.1"})
    with urlopen(request, timeout=timeout_seconds) as response:
        payload = response.read(MAX_FEED_BYTES + 1)
    if len(payload) > MAX_FEED_BYTES:
        raise ValueError("Feed is larger than the configured safety limit.")
    content = payload.decode("utf-8", errors="strict")
    entries = [line.strip() for line in content.splitlines() if line.strip()]
    urls = [line for line in entries if line.startswith(("http://", "https://"))]
    if not urls:
        raise ValueError("The feed response contained no valid URL entries; cached data was left unchanged.")

    FEED_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Keep the temporary names distinct; both cache files share one stem.
    temp_feed = FEED_PATH.with_name(FEED_PATH.name + ".tmp")
    temp_meta = META_PATH.with_name(META_PATH.name + ".tmp")
    temp_feed.write_text("\n".join(urls) + "\n", encoding="utf-8")
    metadata = {
        "source": "OpenPhish Community Feed",
        "source_url": FEED_URL,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "entry_count": len(urls),
    }
    temp_meta.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    temp_feed.replace(FEED_PATH)
    temp_meta.replace(META_PATH)
    return metadata
