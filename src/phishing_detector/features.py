"""Shared URL normalization and human-readable signal extraction."""

from __future__ import annotations

import ipaddress
import re
from urllib.parse import urlsplit

SUSPICIOUS_TERMS = (
    "login", "verify", "account", "secure", "update", "password",
    "banking", "confirm", "signin", "wallet", "webscr", "support",
)


def normalize_url(value: str) -> str:
    """Match training normalization; scheme is shown separately, not learned yet."""
    value = str(value).strip()
    value = re.sub(r"^https?://", "", value, flags=re.IGNORECASE)
    return value.lower()


def scheme_status(value: str) -> str:
    """Return explicit HTTP state; a missing scheme remains unknown."""
    match = re.match(r"^([a-z][a-z0-9+.-]*)://", str(value).strip(), flags=re.IGNORECASE)
    if not match:
        return "unknown"
    scheme = match.group(1).lower()
    return scheme if scheme in {"http", "https"} else "other"


def _hostname(value: str) -> str:
    candidate = value.strip()
    if not re.match(r"^[a-z][a-z0-9+.-]*://", candidate, flags=re.I):
        candidate = "http://" + candidate
    try:
        return (urlsplit(candidate).hostname or "").lower()
    except ValueError:
        return ""


def explain_url(value: str) -> dict[str, object]:
    """Return simple lexical signals for the UI; these are not model explanations."""
    raw = str(value).strip()
    normalized = normalize_url(raw)
    host = _hostname(raw)
    try:
        ipaddress.ip_address(host)
        is_ip = True
    except ValueError:
        is_ip = False

    terms = [term for term in SUSPICIOUS_TERMS if term in normalized]
    return {
        "url_length": len(raw),
        "domain_length": len(host),
        "subdomain_count": max(0, len(host.split(".")) - 2) if host else 0,
        "dot_count": raw.count("."),
        "hyphen_count": raw.count("-"),
        "digit_count": sum(char.isdigit() for char in raw),
        "special_character_count": sum(char in "@?=&%_" for char in raw),
        "scheme_status": scheme_status(raw),
        "ip_address_host": is_ip,
        "suspicious_terms": terms,
        "hostname": host,
    }
