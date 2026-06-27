"""
LLM-based headline scoring: prompt construction for Llama-3-Instruct,
parsing of the structured pipe-delimited output, and a deterministic
keyword-based fallback for when generation fails to parse.
"""
import re

from config import (
    FINGPT_SCORE_SYSTEM,
    KEYWORD_FALLBACK_RULES,
    LLAMA3_FEW_SHOTS_USER,
    SINGLE_COUNTRY_MINOR_KEYWORDS,
)

_PIPE_RE = re.compile(
    r"(positive|negative|neutral)"
    r"\s*\|\s*(-?(?:1(?:\.0*)?|0(?:\.\d*)?))"
    r"\s*\|\s*(0?\.\d+|1\.0*|0\.0*)"
    r"\s*\|\s*(\d+)"
    r"\s*\|\s*([1-5](?:\.\d+)?)",
    re.IGNORECASE,
)


def build_fingpt_prompt(headline: str) -> str:
    """Build the full Llama-3-Instruct chat-formatted prompt for one headline."""
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{FINGPT_SCORE_SYSTEM}"
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{LLAMA3_FEW_SHOTS_USER.format(headline=headline)}"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def keyword_fallback_score(headline: str) -> dict:
    """Deterministic keyword-rule scoring, used when LLM output fails to parse."""
    t = headline.lower()
    best = None
    for (score, conf, impact_hrs, tier_w, kws) in KEYWORD_FALLBACK_RULES:
        if any(kw in t for kw in kws):
            if best is None or abs(score) > abs(best["score"]):
                best = {
                    "sentiment": "positive" if score > 0.05 else ("negative" if score < -0.05 else "neutral"),
                    "score": round(score, 4),
                    "confidence": round(conf, 4),
                    "impact_hours": impact_hrs,
                    "tier_weight": round(tier_w, 4),
                }
    if best is None:
        best = {
            "sentiment": "neutral",
            "score": 0.0,
            "confidence": 0.35,
            "impact_hours": 6,
            "tier_weight": 2.0,
        }
    if any(c in t for c in SINGLE_COUNTRY_MINOR_KEYWORDS):
        best["tier_weight"] = min(best["tier_weight"], 2.5)
        best["score"] = round(best["score"] * 0.6, 4)
        best["confidence"] = round(best["confidence"] * 0.85, 4)
    return best


def parse_fingpt_response(raw: str, headline: str = "") -> dict:
    """
    Parse the model's raw decoded output into a structured score dict.

    Tries the strict pipe-delimited format first, then a loose
    sentiment-word + first-4-numbers heuristic, then falls back to
    `keyword_fallback_score` (keeping any sentiment word that was found).
    """
    raw_clean = raw.strip()
    m = _PIPE_RE.search(raw_clean)
    if m:
        sentiment = m.group(1).lower()
        score = max(-1.0, min(1.0, float(m.group(2))))
        confidence = max(0.0, min(1.0, float(m.group(3))))
        impact_hrs = max(2, min(168, int(m.group(4))))
        tier_weight = max(1.0, min(5.0, float(m.group(5))))
        return {
            "sentiment": sentiment,
            "score": round(score, 4),
            "confidence": round(confidence, 4),
            "impact_hours": impact_hrs,
            "tier_weight": round(tier_weight, 4),
        }

    raw_lower = raw_clean.lower()
    if "positive" in raw_lower:
        sentiment = "positive"
    elif "negative" in raw_lower:
        sentiment = "negative"
    else:
        sentiment = None

    found_numbers = re.findall(r"[-+]?\d*\.\d+|\b\d+\b", raw_clean)
    floats = []
    for num in found_numbers:
        try:
            floats.append(float(num))
        except ValueError:
            continue

    if sentiment and len(floats) >= 4:
        try:
            score = max(-1.0, min(1.0, floats[0]))
            confidence = max(0.0, min(1.0, floats[1]))
            impact_hrs = max(2, min(168, int(floats[2])))
            tier_weight = max(1.0, min(5.0, floats[3]))
            return {
                "sentiment": sentiment,
                "score": round(score, 4),
                "confidence": round(confidence, 4),
                "impact_hours": impact_hrs,
                "tier_weight": round(tier_weight, 4),
            }
        except Exception:
            pass

    result = keyword_fallback_score(headline)
    if sentiment and sentiment != result["sentiment"]:
        result["sentiment"] = sentiment
        if sentiment == "positive" and result["score"] < 0:
            result["score"] = abs(result["score"])
        elif sentiment == "negative" and result["score"] > 0:
            result["score"] = -abs(result["score"])
    return result


def matched_pipe_format(raw: str) -> bool:
    """True if the raw decode matched the strict pipe-delimited format."""
    return bool(_PIPE_RE.search(raw))
