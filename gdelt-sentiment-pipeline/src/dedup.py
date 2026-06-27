"""
Deduplication and density-clustering utilities.

Three independent dedup passes are applied upstream in the pipeline:
1. `apply_density_clustering`  — flags 15-minute windows with abnormally
   high headline volume (z-score > 2.5) for an importance boost.
2. Jaccard/exact-match dedup over a rolling time window (in the orchestrator
   script, using `normalize_ath_title` from this module).
3. `EVENT_DEDUP_CLUSTERS`-driven entity-event windowed dedup (also in the
   orchestrator), which collapses many headlines about the same person/event
   into a single representative one.
"""
import re

import numpy as np
import pandas as pd

_ATH_NORM_PHRASES = [
    "bitcoin hits record", "bitcoin all time high", "btc hits record",
    "bitcoin reaches record", "bitcoin surpasses record", "bitcoin new high",
    "bitcoin price record", "bitcoin record high", "btc record high",
    "bitcoin surges past", "bitcoin breaks record",
]


def normalize_ath_title(title: str) -> str:
    """Collapse all-time-high headline variants to a single canonical token."""
    t = title.lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    t_nospace = t.replace(" ", "")
    for phrase in _ATH_NORM_PHRASES:
        if phrase.replace(" ", "") in t_nospace:
            return "bitcoin_ath_event"
    return t


def apply_density_clustering(df):
    """Mark rows that fall in abnormally headline-dense 15-minute windows."""
    if df.empty:
        return df
    df_sorted = df.sort_values("dt").copy()
    df_sorted.set_index("dt", inplace=True)
    counts = df_sorted.resample("15Min").size()
    mean_val = counts.mean()
    std_val = counts.std() if counts.std() > 0 else 1.0
    z_scores = (counts - mean_val) / std_val
    high_density_intervals = z_scores[z_scores > 2.5].index

    df_sorted["density_override"] = False
    df_sorted["density_override"] = df_sorted["density_override"].astype(bool)
    for start_time in high_density_intervals:
        end_time = start_time + pd.Timedelta(minutes=15)
        df_sorted.loc[start_time:end_time, "density_override"] = True

    df_sorted.reset_index(inplace=True)
    return df_sorted


def find_similarity_duplicates(df_keep, window_hours=24, similarity_threshold=0.70):
    """
    Rolling-window near-duplicate detection over already-kept headlines.

    Two headlines within `window_hours` of each other are considered
    duplicates if their normalized titles match exactly, or if their
    word-set Jaccard-style overlap (intersection / smaller set size) meets
    `similarity_threshold`. ATH headlines are normalized to one canonical
    token so all phrasing variants of "Bitcoin hits new all-time high" collapse
    together.

    Returns (duplicate_urls: set, kept_rows: list) where `kept_rows` are the
    first-seen representative row objects (itertuples) for each cluster.
    """
    duplicate_urls = set()
    kept_rows = []
    history = []

    for row in df_keep.itertuples():
        t = str(row.title).lower()
        for sep in (" - ", " | ", " : "):
            t = t.split(sep)[0]
        ath_norm = normalize_ath_title(t)
        if ath_norm == "bitcoin_ath_event":
            norm = "bitcoin_ath_event"
            words = {"bitcoin", "ath", "event"}
        else:
            words = set(re.findall(r"\b[a-z0-9]+\b", t))
            norm = re.sub(r"[^a-z0-9]", "", t)

        cutoff = row.dt - pd.Timedelta(hours=window_hours)
        history = [h for h in history if h[0] >= cutoff]

        is_dup = False
        for h_dt, h_words, h_norm in history:
            if norm == h_norm:
                is_dup = True
                break
            if norm != "bitcoin_ath_event" and words and h_words:
                intersect = len(words.intersection(h_words))
                min_len = min(len(words), len(h_words))
                if min_len > 0 and (intersect / min_len) >= similarity_threshold:
                    is_dup = True
                    break

        if is_dup:
            duplicate_urls.add(row.url)
        else:
            kept_rows.append(row)
            history.append((row.dt, words, norm))

    return duplicate_urls, kept_rows


def find_entity_event_duplicates(df_keep, event_dedup_clusters):
    """
    For each (window_hours, anchor_keywords) cluster definition, collapse
    all matching headlines that fall within `window_hours` of an earlier
    matching headline down to a single representative.

    Returns the set of URLs to drop as entity-event duplicates.
    """
    entity_event_dup_urls = set()

    for window_h, anchors in event_dedup_clusters:
        mask = df_keep["title"].str.lower().apply(lambda t: any(a in t for a in anchors))
        cluster_df = df_keep[mask].copy().sort_values("dt")
        if len(cluster_df) < 2:
            continue

        to_drop_idx = set()
        processed_windows = []
        for idx, row_i in cluster_df.iterrows():
            if idx in to_drop_idx:
                continue
            already_covered = any(ws <= row_i["dt"] <= we for ws, we in processed_windows)
            if already_covered:
                to_drop_idx.add(idx)
                continue
            window_end = row_i["dt"] + pd.Timedelta(hours=window_h)
            processed_windows.append((row_i["dt"], window_end))
            same_window = cluster_df[
                (cluster_df["dt"] > row_i["dt"]) & (cluster_df["dt"] <= window_end)
            ]
            to_drop_idx.update(same_window.index.tolist())

        if to_drop_idx:
            entity_event_dup_urls.update(df_keep.loc[list(to_drop_idx), "url"].tolist())

    return entity_event_dup_urls
