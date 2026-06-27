"""
Title cleaning, source-domain normalization, slug truncation, and the
keep/drop category classifier that decides which headlines are relevant
enough to score.
"""
import re
from urllib.parse import urlparse

import pandas as pd

from config import (
    KEEP_CATEGORIES,
    LOW_CREDIBILITY_SOURCES,
    SLUG_TOPIC_TOKENS,
    SLUG_TRUNCATION_MAX_WORDS,
    SLUG_TRUNCATION_MIN_WORDS,
    SLUG_TRUNCATION_SOURCES,
    SOURCE_CREDIBILITY_WEIGHTS,
)

_NUMERIC_PREFIX_RE = re.compile(r"^\d{5,}")
_REPEATED_DIGIT_RE = re.compile(r"(\b\d\b\s+){4,}")
_URL_ID_RE = re.compile(r"\b[A-Za-z0-9]{20,}\b")


def clean_source_domain(src):
    if not src or not isinstance(src, str):
        return ""
    s = src.lower().strip()
    if s.startswith("http"):
        try:
            s = urlparse(s).netloc
        except Exception:
            pass
    s = re.sub(r"^www\.", "", s)
    return s


def get_source_weight(source_domain):
    clean = clean_source_domain(source_domain)
    return SOURCE_CREDIBILITY_WEIGHTS.get(clean, 1.0)


def parse_gkg_date(val):
    if pd.isna(val):
        return pd.NaT
    s = str(val).strip()
    for fmt in ("%Y%m%d%H%M%S", "%Y%m%d%H%M%S.%f", "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S", "%m/%d/%Y %H:%M:%S"):
        try:
            return pd.Timestamp(pd.to_datetime(s, format=fmt), tz="UTC")
        except Exception:
            pass
    try:
        return pd.Timestamp(pd.to_datetime(s), tz="UTC")
    except Exception:
        return pd.NaT


def _is_aggregator_slug(title: str) -> bool:
    t = title.lower()
    hits = sum(1 for tok in SLUG_TOPIC_TOKENS if tok in t)
    return hits >= 1


def truncate_slug_title(title: str, source: str) -> str:
    """Cut overly long wire-aggregator/listicle titles down to a stable word count."""
    if not title or not isinstance(title, str):
        return title
    clean_src = clean_source_domain(source)
    words = title.split()
    if len(words) < SLUG_TRUNCATION_MIN_WORDS:
        return title
    if clean_src in SLUG_TRUNCATION_SOURCES:
        return " ".join(words[:SLUG_TRUNCATION_MAX_WORDS])
    if clean_src not in SOURCE_CREDIBILITY_WEIGHTS or SOURCE_CREDIBILITY_WEIGHTS.get(clean_src, 1.0) <= 0.7:
        if len(words) >= SLUG_TRUNCATION_MIN_WORDS:
            return " ".join(words[:SLUG_TRUNCATION_MAX_WORDS])
    return title


def is_clean_title(title):
    """Reject garbage titles: numeric junk, URL-id tokens, aggregator slugs, too short."""
    if not title or not isinstance(title, str):
        return False
    t = title.strip()
    if _NUMERIC_PREFIX_RE.match(t):
        return False
    if _REPEATED_DIGIT_RE.search(t):
        return False
    if _URL_ID_RE.search(t):
        return False
    if _is_aggregator_slug(t):
        return False
    words_count = len(t.split())
    if words_count < 3:
        high_impact_triggers = {
            "fed", "sec", "etf", "ftx", "ban", "hike", "cut",
            "sues", "crash", "cpi", "war", "rate", "crypto", "btc",
        }
        if words_count >= 1 and any(w in t.lower() for w in high_impact_triggers):
            return True
        return False
    return True


def classify_keep_with_reason(title, source=""):
    """
    Decide whether a headline is relevant enough to keep for scoring.

    Returns (keep: bool, reason: str) where `reason` is either the matched
    keep-category name, or a drop-reason code explaining why it was dropped.
    """
    if not title or not isinstance(title, str):
        return False, "empty_title"

    if _is_aggregator_slug(title):
        return False, "aggregator_slug"

    if not is_clean_title(title):
        if _NUMERIC_PREFIX_RE.match(title.strip()):
            return False, "numeric_prefix"
        if _REPEATED_DIGIT_RE.search(title.strip()):
            return False, "repeated_digits"
        if _URL_ID_RE.search(title.strip()):
            return False, "url_id_token"
        if len(title.strip().split()) < 3:
            return False, "title_too_short"
        return False, "dirty_title"

    clean_src = clean_source_domain(source)
    if clean_src in LOW_CREDIBILITY_SOURCES:
        return False, f"low_credibility_source:{clean_src}"

    t = title.lower()

    for cat_name, rules in KEEP_CATEGORIES.items():

        if cat_name == "geopolitical_major_powers":
            has_trigger = any(p in t for p in rules["require_any"])
            has_country = any(c in t for c in rules["require_country"])
            if has_trigger and has_country:
                return True, cat_name
            continue

        if cat_name == "major_us_company_news":
            has_company = any(c in t for c in rules["require_company"])
            has_event = any(p in t for p in rules["require_any"])
            if has_company and has_event:
                return True, cat_name
            continue

        if cat_name == "us_macro_data":
            has_macro = any(p in t for p in rules.get("require_any", []))
            has_signal = any(p in t for p in rules.get("require_data_signal", []))
            if has_macro and has_signal:
                return True, cat_name
            continue

        if cat_name == "fed_rates":
            has_trigger = any(p in t for p in rules.get("require_any", []))
            has_institution = any(p in t for p in rules.get("require_institution", []))
            has_exclusion = any(p in t for p in rules.get("exclude_any", []))
            if has_trigger and has_institution and not has_exclusion:
                return True, cat_name
            continue

        if cat_name == "black_swan_systemic":
            if not any(p in t for p in rules.get("require_any", [])):
                continue
            if any(p in t for p in rules.get("require_scale", [])):
                return True, cat_name
            continue

        if cat_name == "energy_crisis":
            if not any(p in t for p in rules.get("require_any", [])):
                continue
            if any(p in t for p in rules.get("require_market_signal", [])):
                return True, cat_name
            continue

        if cat_name == "major_currency":
            if not any(p in t for p in rules.get("require_any", [])):
                continue
            if any(p in t for p in rules.get("exclude_any", [])):
                return False, "currency_slug_noise"
            return True, cat_name

        if cat_name == "crypto_structural":
            if not any(p in t for p in rules.get("require_any", [])):
                continue
            if any(p in t for p in rules.get("exclude_any", [])):
                return False, "crypto_filler_content"
            return True, cat_name

        if not any(p in t for p in rules.get("require_any", [])):
            continue
        return True, cat_name

    return False, "no_category_match"


def classify_keep(title, source=""):
    kept, _ = classify_keep_with_reason(title, source)
    return kept


def convert_reason_code(rc):
    mapping = {
        "empty_title": "Dropped: Missing/Empty Title",
        "aggregator_slug": "Dropped: Wire Aggregator Slug Title",
        "numeric_prefix": "Dropped: Garbage Numeric Prefix Data",
        "repeated_digits": "Dropped: Structured Spam Ticker Line",
        "url_id_token": "Dropped: System Hash Parameter Slug",
        "title_too_short": "Dropped: Too Short (<3 words)",
        "dirty_title": "Dropped: Validation Check Corrupt Pattern",
        "no_category_match": "Dropped: Outside Market Focus Mandate",
        "crypto_filler_content": "Dropped: Crypto Filler / Promotional Content",
        "currency_slug_noise": "Dropped: Currency Roundup Slug",
        "entity_event_duplicate": "Dropped: Entity-Event Cluster Duplicate",
        "neutral_zero_score": "Dropped: Neutral Score \u2014 No Signal Contribution",
        "duplicate_headline": "Dropped: Duplicate Headline",
    }
    if rc.startswith("low_credibility_source:"):
        return f"Dropped: Low Credibility Source ({rc.split(':', 1)[1]})"
    return mapping.get(rc, f"Dropped: Filter Reason [{rc}]")


def human_drop_reason(rc):
    if not rc:
        return "Kept"
    return convert_reason_code(rc)
