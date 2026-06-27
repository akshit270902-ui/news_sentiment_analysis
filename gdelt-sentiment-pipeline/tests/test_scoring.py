"""
Tests for src.llm_scoring (prompt building, response parsing, keyword
fallback) and src.overrides (semantic overrides, score floor, soft-impact
dampening).
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm_scoring import (
    build_fingpt_prompt,
    keyword_fallback_score,
    matched_pipe_format,
    parse_fingpt_response,
)
from src.overrides import (
    apply_kept_score_floor,
    apply_semantic_overrides,
    apply_soft_impact_overrides,
)


# ---------------------------------------------------------------------------
# build_fingpt_prompt
# ---------------------------------------------------------------------------
def test_build_fingpt_prompt_contains_headline_and_chat_markers():
    prompt = build_fingpt_prompt("SEC approves spot Bitcoin ETF")
    assert "SEC approves spot Bitcoin ETF" in prompt
    assert "<|begin_of_text|>" in prompt
    assert "<|start_header_id|>assistant<|end_header_id|>" in prompt


# ---------------------------------------------------------------------------
# parse_fingpt_response — strict pipe format
# ---------------------------------------------------------------------------
def test_parse_fingpt_response_strict_pipe_format():
    raw = "Reasoning: clearly bullish.\nDATA: positive|0.85|0.90|72|4.5"
    result = parse_fingpt_response(raw, headline="irrelevant")
    assert result["sentiment"] == "positive"
    assert result["score"] == 0.85
    assert result["confidence"] == 0.90
    assert result["impact_hours"] == 72
    assert result["tier_weight"] == 4.5
    assert matched_pipe_format(raw) is True


def test_parse_fingpt_response_clamps_out_of_range_values():
    raw = "DATA: negative|-5.0|3.0|999|99"
    result = parse_fingpt_response(raw)
    assert result["score"] == -1.0
    assert result["confidence"] == 1.0
    assert result["impact_hours"] == 168
    assert result["tier_weight"] == 5.0


# ---------------------------------------------------------------------------
# parse_fingpt_response — loose numeric fallback
# ---------------------------------------------------------------------------
def test_parse_fingpt_response_loose_numeric_format():
    raw = "sentiment is positive with score 0.6 confidence 0.8 hours 24 weight 3"
    result = parse_fingpt_response(raw, headline="")
    assert result["sentiment"] == "positive"
    assert result["score"] == 0.6


# ---------------------------------------------------------------------------
# parse_fingpt_response — keyword fallback when nothing parses
# ---------------------------------------------------------------------------
def test_parse_fingpt_response_falls_back_to_keywords():
    raw = "I cannot determine a score for this."
    result = parse_fingpt_response(raw, headline="SEC approves bitcoin etf for institutional trading")
    assert result["sentiment"] == "positive"
    assert result["score"] > 0


def test_parse_fingpt_response_fallback_respects_detected_sentiment_word():
    raw = "negative outlook, no further data given"
    result = parse_fingpt_response(raw, headline="SEC approves bitcoin etf for institutional trading")
    # sentiment word found in raw output overrides keyword-fallback direction
    assert result["sentiment"] == "negative"
    assert result["score"] <= 0


# ---------------------------------------------------------------------------
# keyword_fallback_score
# ---------------------------------------------------------------------------
def test_keyword_fallback_score_positive_rule():
    result = keyword_fallback_score("Bitcoin halving begins as block subsidy halves")
    assert result["sentiment"] == "positive"
    assert result["tier_weight"] == 4.5


def test_keyword_fallback_score_negative_rule():
    result = keyword_fallback_score("Exchange hacked, funds stolen overnight")
    assert result["sentiment"] == "negative"
    assert result["score"] < 0


def test_keyword_fallback_score_no_match_returns_neutral():
    result = keyword_fallback_score("Local weather forecast for the weekend")
    assert result["sentiment"] == "neutral"
    assert result["score"] == 0.0


def test_keyword_fallback_score_minor_country_dampens_weight():
    base = keyword_fallback_score("Bitcoin halving begins as block subsidy halves")
    dampened = keyword_fallback_score("Brazil bitcoin halving begins as block subsidy halves")
    assert dampened["tier_weight"] <= base["tier_weight"]
    assert abs(dampened["score"]) <= abs(base["score"])


# ---------------------------------------------------------------------------
# apply_semantic_overrides
# ---------------------------------------------------------------------------
def test_apply_semantic_overrides_forces_negative_direction():
    df = pd.DataFrame({
        "title": ["Regulator bans crypto derivatives trading immediately"],
        "fingpt_sentiment": ["positive"],
        "opinion_score": [0.2],
    })
    result = apply_semantic_overrides(df)
    assert result.loc[0, "fingpt_sentiment"] == "negative"
    assert result.loc[0, "opinion_score"] < 0


def test_apply_semantic_overrides_leaves_correct_rows_unchanged():
    df = pd.DataFrame({
        "title": ["Some unrelated headline about cats"],
        "fingpt_sentiment": ["neutral"],
        "opinion_score": [0.0],
    })
    result = apply_semantic_overrides(df)
    assert result.loc[0, "fingpt_sentiment"] == "neutral"
    assert result.loc[0, "opinion_score"] == 0.0


# ---------------------------------------------------------------------------
# apply_kept_score_floor
# ---------------------------------------------------------------------------
def test_apply_kept_score_floor_floors_directional_near_zero():
    df = pd.DataFrame({
        "title": ["a", "b"],
        "url": ["u1", "u2"],
        "fingpt_sentiment": ["positive", "negative"],
        "opinion_score": [0.02, -0.01],
    })
    result, dropped = apply_kept_score_floor(df)
    assert result.loc[0, "opinion_score"] == 0.10
    assert result.loc[1, "opinion_score"] == -0.10
    assert dropped == []


def test_apply_kept_score_floor_drops_neutral_near_zero():
    df = pd.DataFrame({
        "title": ["a"],
        "url": ["u1"],
        "fingpt_sentiment": ["neutral"],
        "opinion_score": [0.0],
    })
    result, dropped = apply_kept_score_floor(df)
    assert len(result) == 0
    assert dropped == ["u1"]


# ---------------------------------------------------------------------------
# apply_soft_impact_overrides
# ---------------------------------------------------------------------------
def test_apply_soft_impact_overrides_dampens_matching_megacap_row():
    df = pd.DataFrame({
        "title": ["Microsoft bans cryptocurrency mining on its cloud platform"],
        "fingpt_tier_weight": [5.0],
        "opinion_score": [-0.8],
    })
    result = apply_soft_impact_overrides(df)
    assert result.loc[0, "fingpt_tier_weight"] <= 2.5
    assert result.loc[0, "opinion_score"] == -0.4


def test_apply_soft_impact_overrides_ignores_unrelated_row():
    df = pd.DataFrame({
        "title": ["Microsoft reports record quarterly earnings beat"],
        "fingpt_tier_weight": [3.0],
        "opinion_score": [0.5],
    })
    result = apply_soft_impact_overrides(df)
    assert result.loc[0, "fingpt_tier_weight"] == 3.0
    assert result.loc[0, "opinion_score"] == 0.5
