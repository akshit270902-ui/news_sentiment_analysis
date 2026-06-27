"""
Novelty decay and aftermath/trigger multipliers.

These control how much weight a headline gets based on how "first" it is
within its event cluster, and whether it's a trigger event, a moderate
aftermath mention, or a stale "still reeling" follow-up story.
"""
import numpy as np
import pandas as pd

from config import (
    AFTERMATH_MODERATE,
    AFTERMATH_MODERATE_MULT,
    AFTERMATH_STRONG,
    AFTERMATH_STRONG_MULT,
    EVENT_CLUSTERS,
    HIGH_TIER_MULT_FLOOR,
    HIGH_TIER_MULT_FLOOR_ANCHORS,
    NOVELTY_DECAY,
    NOVELTY_WINDOW_H,
    TRIGGER_BOOST,
    TRIGGER_BOOST_PHRASES,
)


def compute_novelty_multiplier(df):
    """
    Within each known event cluster, decay weight for headlines that repeat
    the same story later — both by elapsed time and by repetition rank.
    """
    df_sorted = df.sort_values("dt")
    mult = pd.Series(1.0, index=df.index, dtype=float)
    for cluster_kws in EVENT_CLUSTERS:
        mask = df_sorted["title"].str.lower().apply(
            lambda t: any(kw in t for kw in cluster_kws)
        )
        cluster_idx = df_sorted.index[mask].tolist()
        if len(cluster_idx) < 2:
            continue
        first_dt = df_sorted.loc[cluster_idx[0], "dt"]
        for rank, idx in enumerate(cluster_idx):
            hours_since = (df_sorted.loc[idx, "dt"] - first_dt).total_seconds() / 3600
            if hours_since <= NOVELTY_WINDOW_H:
                time_decay = np.exp(-0.05 * hours_since)
                rank_decay = NOVELTY_DECAY ** rank
                mult[idx] *= min(time_decay, rank_decay)
    return mult


def build_first_seen_index(df):
    """Map each event-cluster phrase to the timestamp it was first mentioned."""
    first_seen = {}
    df_sorted = df.sort_values("dt")
    cluster_phrases = [kw.strip() for cluster in EVENT_CLUSTERS for kw in cluster]
    for row in df_sorted.itertuples(index=False):
        t = str(row.title).lower()
        for phrase in cluster_phrases:
            if phrase in t and phrase not in first_seen:
                first_seen[phrase] = row.dt
    return first_seen


def _article_multiplier_core(title, article_dt, first_seen):
    t = title.lower()
    matching_first = None
    for phrase, first_dt in first_seen.items():
        if phrase in t and first_dt < article_dt:
            if matching_first is None or first_dt < matching_first:
                matching_first = first_dt
    if matching_first is not None:
        hours_late = (article_dt - matching_first).total_seconds() / 3600
        if hours_late > 48:
            return AFTERMATH_STRONG_MULT * np.exp(-0.02 * hours_late), True
        elif hours_late > 6:
            staleness = np.exp(-0.03 * hours_late)
            is_aftermath = hours_late > 24
            if any(p in t for p in AFTERMATH_STRONG):
                return AFTERMATH_STRONG_MULT * staleness, True
            if any(p in t for p in AFTERMATH_MODERATE):
                return AFTERMATH_MODERATE_MULT * staleness, is_aftermath
            if any(p in t for p in TRIGGER_BOOST_PHRASES):
                return min(TRIGGER_BOOST, staleness), is_aftermath
            return staleness, is_aftermath
    if any(p in t for p in AFTERMATH_STRONG):
        return AFTERMATH_STRONG_MULT, True
    if any(p in t for p in AFTERMATH_MODERATE):
        return AFTERMATH_MODERATE_MULT, True
    if any(p in t for p in TRIGGER_BOOST_PHRASES):
        return TRIGGER_BOOST, False
    return 1.0, False


def article_multiplier_v2(title, article_dt, first_seen):
    """Aftermath/trigger multiplier, with a credibility floor for headline-grade events."""
    mult, is_aftermath = _article_multiplier_core(title, article_dt, first_seen)
    t = title.lower()
    if any(a in t for a in HIGH_TIER_MULT_FLOOR_ANCHORS):
        mult = max(mult, HIGH_TIER_MULT_FLOOR)
    return mult, is_aftermath
