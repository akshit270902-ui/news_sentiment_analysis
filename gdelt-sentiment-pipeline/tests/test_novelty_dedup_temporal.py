"""
Tests for src.novelty (article/novelty multipliers), src.dedup
(similarity + entity-event dedup, density clustering), and src.temporal
(sentiment spreading and cumulative decay).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.novelty import (
    article_multiplier_v2,
    build_first_seen_index,
    compute_novelty_multiplier,
)
from src.dedup import (
    apply_density_clustering,
    find_entity_event_duplicates,
    find_similarity_duplicates,
    normalize_ath_title,
)
from src.temporal import compute_cumulative_sentiment, spread_sentiment


def _ts(hour_offset):
    return pd.Timestamp("2024-01-01 00:00:00", tz="UTC") + pd.Timedelta(hours=hour_offset)


# ---------------------------------------------------------------------------
# normalize_ath_title
# ---------------------------------------------------------------------------
def test_normalize_ath_title_collapses_variants():
    a = normalize_ath_title("Bitcoin hits record high above $70,000")
    b = normalize_ath_title("BTC hits record high after rally")
    assert a == "bitcoin_ath_event"
    assert b == "bitcoin_ath_event"


def test_normalize_ath_title_leaves_unrelated_titles_distinct():
    a = normalize_ath_title("Federal Reserve cuts interest rates")
    b = normalize_ath_title("Exchange hacked overnight")
    assert a != b
    assert a != "bitcoin_ath_event"


# ---------------------------------------------------------------------------
# build_first_seen_index / article_multiplier_v2
# ---------------------------------------------------------------------------
def test_build_first_seen_index_picks_earliest_occurrence():
    df = pd.DataFrame({
        "title": ["FTX collapse shocks markets", "More on the FTX collapse fallout"],
        "dt": [_ts(0), _ts(5)],
    })
    first_seen = build_first_seen_index(df)
    assert first_seen["ftx collapse"] == _ts(0)


def test_article_multiplier_v2_fresh_trigger_gets_boost():
    first_seen = {}
    mult, is_aftermath = article_multiplier_v2(
        "Bitcoin ETF approved by regulators", _ts(0), first_seen
    )
    assert mult == 1.25
    assert is_aftermath is False


def test_article_multiplier_v2_aftermath_phrase_dampened():
    first_seen = {"ftx collapse": _ts(0)}
    mult, is_aftermath = article_multiplier_v2(
        "Markets still reeling in the aftermath of the ftx collapse", _ts(50), first_seen
    )
    assert mult < 1.0
    assert is_aftermath is True


def test_article_multiplier_v2_high_tier_anchor_floors_multiplier():
    first_seen = {"ftx collapse": _ts(0)}
    mult, _ = article_multiplier_v2(
        "Fallout of ftx collapse continues to drag on sentiment", _ts(200), first_seen
    )
    assert mult >= 0.25


# ---------------------------------------------------------------------------
# compute_novelty_multiplier
# ---------------------------------------------------------------------------
def test_compute_novelty_multiplier_decays_repeated_cluster_mentions():
    df = pd.DataFrame({
        "title": [
            "Silicon Valley Bank collapse shocks regulators",
            "Silicon Valley Bank collapse fallout continues",
            "Silicon Valley Bank collapse: what happens next",
        ],
        "dt": [_ts(0), _ts(2), _ts(4)],
    })
    mult = compute_novelty_multiplier(df)
    # later mentions of the same cluster should be weighted less than or equal
    # to the first mention
    assert mult.iloc[0] >= mult.iloc[1] >= mult.iloc[2]


# ---------------------------------------------------------------------------
# apply_density_clustering
# ---------------------------------------------------------------------------
def test_apply_density_clustering_flags_high_volume_window():
    base = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    # 20 headlines crammed into one 15-minute window, then a sparse tail
    dense_times = [base + pd.Timedelta(minutes=i) for i in range(20)]
    sparse_times = [base + pd.Timedelta(hours=h) for h in range(1, 10)]
    df = pd.DataFrame({
        "title": [f"headline {i}" for i in range(len(dense_times) + len(sparse_times))],
        "dt": dense_times + sparse_times,
    })
    result = apply_density_clustering(df)
    assert result["density_override"].any()


def test_apply_density_clustering_empty_df_passthrough():
    df = pd.DataFrame(columns=["title", "dt"])
    result = apply_density_clustering(df)
    assert result.empty


# ---------------------------------------------------------------------------
# find_similarity_duplicates
# ---------------------------------------------------------------------------
def test_find_similarity_duplicates_detects_near_identical_titles():
    df = pd.DataFrame({
        "title": [
            "Bitcoin surges past fifty thousand dollars today",
            "Bitcoin surges past fifty thousand dollars today!",
        ],
        "url": ["u1", "u2"],
        "dt": [_ts(0), _ts(1)],
    })
    dup_urls, kept_rows = find_similarity_duplicates(df, window_hours=24, similarity_threshold=0.70)
    assert "u2" in dup_urls
    assert len(kept_rows) == 1


def test_find_similarity_duplicates_keeps_distinct_titles():
    df = pd.DataFrame({
        "title": [
            "Federal Reserve cuts interest rates by fifty basis points",
            "Exchange hacked, millions of dollars stolen overnight",
        ],
        "url": ["u1", "u2"],
        "dt": [_ts(0), _ts(1)],
    })
    dup_urls, kept_rows = find_similarity_duplicates(df)
    assert dup_urls == set()
    assert len(kept_rows) == 2


def test_find_similarity_duplicates_respects_window(monkeypatch=None):
    df = pd.DataFrame({
        "title": [
            "Bitcoin surges past fifty thousand dollars today",
            "Bitcoin surges past fifty thousand dollars today",
        ],
        "url": ["u1", "u2"],
        "dt": [_ts(0), _ts(48)],  # outside default 24h window
    })
    dup_urls, kept_rows = find_similarity_duplicates(df, window_hours=24)
    assert dup_urls == set()
    assert len(kept_rows) == 2


# ---------------------------------------------------------------------------
# find_entity_event_duplicates
# ---------------------------------------------------------------------------
def test_find_entity_event_duplicates_collapses_same_window():
    df = pd.DataFrame({
        "title": [
            "FTX collapse rattles crypto markets",
            "FTX bankrupt amid fraud allegations",
            "Sam Bankman fried steps down as CEO",
        ],
        "url": ["u1", "u2", "u3"],
        "dt": [_ts(0), _ts(2), _ts(4)],
    })
    clusters = [(168, ["ftx collapse", "ftx bankrupt", "sam bankman"])]
    dup_urls = find_entity_event_duplicates(df, clusters)
    assert len(dup_urls) == 2
    assert "u1" not in dup_urls  # first occurrence kept


def test_find_entity_event_duplicates_no_match_returns_empty():
    df = pd.DataFrame({
        "title": ["Completely unrelated headline about weather"],
        "url": ["u1"],
        "dt": [_ts(0)],
    })
    clusters = [(168, ["ftx collapse", "ftx bankrupt", "sam bankman"])]
    dup_urls = find_entity_event_duplicates(df, clusters)
    assert dup_urls == set()


# ---------------------------------------------------------------------------
# temporal: spread_sentiment / compute_cumulative_sentiment
# ---------------------------------------------------------------------------
def _make_spread_input_row():
    return pd.DataFrame({
        "title": ["headline"],
        "opinion_score": [0.5],
        "fingpt_tier_weight": [3.0],
        "combined_significance": [1.0],
        "article_mult": [1.0],
        "novelty_mult": [1.0],
        "fingpt_impact_hours": [24],
        "is_aftermath": [False],
        "hour": [pd.Timestamp("2024-01-01 00:00:00", tz="UTC")],
    })


def test_spread_sentiment_produces_positive_sentiment_series():
    df = _make_spread_input_row()
    result = spread_sentiment(df)
    assert not result.empty
    assert (result["sentiment"] >= 0).all()


def test_compute_cumulative_sentiment_decays_over_time():
    idx = pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC")
    hourly = pd.DataFrame({"sentiment": [1.0, 0.0, 0.0, 0.0, 0.0]}, index=idx)
    result = compute_cumulative_sentiment(hourly, decay_per_hour=0.5)
    assert result["cumulative_sentiment"].iloc[0] == 1.0
    assert result["cumulative_sentiment"].iloc[1] == 0.5
    assert result["cumulative_sentiment"].iloc[2] == 0.25


def test_compute_cumulative_sentiment_empty_input():
    result = compute_cumulative_sentiment(pd.DataFrame(columns=["sentiment"]))
    assert result.empty
