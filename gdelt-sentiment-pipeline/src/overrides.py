"""
Post-scoring corrections applied to the LLM output:

1. `apply_semantic_overrides`  — force sentiment direction when unambiguous
   keyword evidence contradicts what the LLM produced.
2. `apply_kept_score_floor`    — give a minimum non-zero score to directional
   articles that scored near zero, and drop true near-zero neutrals entirely.
3. `apply_soft_impact_overrides` — dampen headlines that mention a mega-cap
   name alongside a narrow/soft event that shouldn't carry full tier weight.
"""
from config import KEPT_SCORE_FLOOR, SEMANTIC_OVERRIDES, SOFT_IMPACT_OVERRIDES


def apply_semantic_overrides(df_keep):
    for direction, keywords in SEMANTIC_OVERRIDES:
        mask_trigger = df_keep["title"].str.lower().apply(
            lambda t: any(kw in t for kw in keywords)
        )
        mask_wrong = mask_trigger & (df_keep["fingpt_sentiment"] != direction)
        if mask_wrong.sum() == 0:
            continue
        for idx in df_keep.index[mask_wrong]:
            old_score = df_keep.at[idx, "opinion_score"]
            new_score = -max(abs(old_score), 0.40) if direction == "negative" else max(abs(old_score), 0.40)
            df_keep.at[idx, "fingpt_sentiment"] = direction
            df_keep.at[idx, "opinion_score"] = round(new_score, 4)
    return df_keep


def apply_kept_score_floor(df_keep):
    """Floor near-zero directional scores; drop near-zero neutral articles entirely."""
    zero_mask = df_keep["opinion_score"].abs() < KEPT_SCORE_FLOOR
    pos_floor = zero_mask & (df_keep["fingpt_sentiment"] == "positive")
    neg_floor = zero_mask & (df_keep["fingpt_sentiment"] == "negative")
    df_keep.loc[pos_floor, "opinion_score"] = KEPT_SCORE_FLOOR
    df_keep.loc[neg_floor, "opinion_score"] = -KEPT_SCORE_FLOOR

    neutral_zero = zero_mask & (df_keep["fingpt_sentiment"] == "neutral")
    dropped_urls = df_keep.loc[neutral_zero, "url"].tolist()
    df_keep = df_keep[~neutral_zero].copy()

    return df_keep, dropped_urls


def apply_soft_impact_overrides(df_keep):
    """Dampen tier weight and score for mega-cap headlines describing narrow/soft events."""
    for company, phrases in SOFT_IMPACT_OVERRIDES:
        mask = df_keep["title"].str.lower().apply(
            lambda t: company in t and any(p in t for p in phrases)
        )
        if mask.sum() == 0:
            continue
        df_keep.loc[mask, "fingpt_tier_weight"] = df_keep.loc[mask, "fingpt_tier_weight"].clip(upper=2.5)
        df_keep.loc[mask, "opinion_score"] = df_keep.loc[mask, "opinion_score"] * 0.5
    return df_keep
