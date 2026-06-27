"""
Tests for src.classify — title cleaning, source domain normalization,
slug truncation, and the keep/drop category classifier.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.classify import (
    classify_keep,
    classify_keep_with_reason,
    clean_source_domain,
    convert_reason_code,
    get_source_weight,
    human_drop_reason,
    is_clean_title,
    parse_gkg_date,
    truncate_slug_title,
)


# ---------------------------------------------------------------------------
# clean_source_domain
# ---------------------------------------------------------------------------
def test_clean_source_domain_strips_www_and_scheme():
    assert clean_source_domain("https://www.reuters.com/article/x") == "reuters.com"
    assert clean_source_domain("www.bloomberg.com") == "bloomberg.com"
    assert clean_source_domain("coindesk.com") == "coindesk.com"


def test_clean_source_domain_handles_empty():
    assert clean_source_domain("") == ""
    assert clean_source_domain(None) == ""


# ---------------------------------------------------------------------------
# get_source_weight
# ---------------------------------------------------------------------------
def test_get_source_weight_known_and_unknown():
    assert get_source_weight("reuters.com") == 1.3
    assert get_source_weight("https://www.zerohedge.com/foo") == 0.6
    assert get_source_weight("some-random-blog.net") == 1.0


# ---------------------------------------------------------------------------
# is_clean_title
# ---------------------------------------------------------------------------
def test_is_clean_title_rejects_numeric_prefix():
    assert is_clean_title("123456789 some headline text") is False


def test_is_clean_title_rejects_repeated_digits():
    assert is_clean_title("1 2 3 4 5 ticker spam line") is False


def test_is_clean_title_rejects_url_id_token():
    assert is_clean_title("article abcdefghijklmnopqrstuvwxyz123456") is False


def test_is_clean_title_rejects_too_short_without_trigger():
    assert is_clean_title("hello world") is False


def test_is_clean_title_allows_short_with_high_impact_trigger():
    assert is_clean_title("Fed cuts") is True


def test_is_clean_title_allows_normal_sentence():
    assert is_clean_title("Bitcoin surges past fifty thousand dollars today") is True


def test_is_clean_title_rejects_empty():
    assert is_clean_title("") is False
    assert is_clean_title(None) is False


# ---------------------------------------------------------------------------
# truncate_slug_title
# ---------------------------------------------------------------------------
def test_truncate_slug_title_truncates_known_slug_source():
    long_title = " ".join(["word"] * 16)
    result = truncate_slug_title(long_title, "zerohedge.com")
    assert len(result.split()) == 10


def test_truncate_slug_title_leaves_short_titles_alone():
    short_title = "Fed cuts rates by fifty basis points"
    assert truncate_slug_title(short_title, "zerohedge.com") == short_title


def test_truncate_slug_title_leaves_high_credibility_source_alone():
    long_title = " ".join(["word"] * 16)
    result = truncate_slug_title(long_title, "reuters.com")
    assert result == long_title


# ---------------------------------------------------------------------------
# classify_keep_with_reason
# ---------------------------------------------------------------------------
def test_classify_keep_fed_rate_cut():
    kept, reason = classify_keep_with_reason(
        "Federal Reserve cuts interest rates by fifty basis points", "reuters.com"
    )
    assert kept is True
    assert reason == "fed_rates"


def test_classify_keep_drops_low_credibility_source():
    kept, reason = classify_keep_with_reason(
        "Federal Reserve cuts interest rates by fifty basis points", "legacy.com"
    )
    assert kept is False
    assert reason.startswith("low_credibility_source")


def test_classify_keep_drops_empty_title():
    kept, reason = classify_keep_with_reason("", "reuters.com")
    assert kept is False
    assert reason == "empty_title"


def test_classify_keep_drops_no_category_match():
    kept, reason = classify_keep_with_reason(
        "Local bakery wins regional pastry award this weekend", "smalltownnews.com"
    )
    assert kept is False
    assert reason == "no_category_match"


def test_classify_keep_crypto_structural_excludes_filler():
    kept, reason = classify_keep_with_reason(
        "Beginners guide to bitcoin etf approved: what is bitcoin and how does it work", "somecryptosite.com"
    )
    assert kept is False
    assert reason == "crypto_filler_content"


def test_classify_keep_fed_rates_excludes_personal_finance_framing():
    kept, reason = classify_keep_with_reason(
        "How will the Fed rate hike affect your wallet and your savings", "somefinance.com"
    )
    assert kept is False


def test_classify_keep_thin_wrapper_matches_with_reason():
    assert classify_keep("Federal Reserve cuts interest rates", "reuters.com") is True


# ---------------------------------------------------------------------------
# reason code -> human string conversion
# ---------------------------------------------------------------------------
def test_convert_reason_code_known():
    assert convert_reason_code("empty_title") == "Dropped: Missing/Empty Title"


def test_convert_reason_code_low_credibility_includes_domain():
    result = convert_reason_code("low_credibility_source:legacy.com")
    assert "legacy.com" in result


def test_human_drop_reason_kept_when_empty():
    assert human_drop_reason("") == "Kept"
    assert human_drop_reason(None) == "Kept"


# ---------------------------------------------------------------------------
# parse_gkg_date
# ---------------------------------------------------------------------------
def test_parse_gkg_date_compact_format():
    result = parse_gkg_date("20240115120000")
    assert result.year == 2024
    assert result.month == 1
    assert result.day == 15


def test_parse_gkg_date_nat_passthrough():
    result = parse_gkg_date(pd.NA if hasattr(pd, "NA") else float("nan"))
    assert pd.isna(result)


def test_parse_gkg_date_unparsable_returns_nat():
    result = parse_gkg_date("not-a-real-date-string-at-all-zzz")
    assert pd.isna(result)
