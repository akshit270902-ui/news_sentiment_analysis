"""
Raw data ingestion: load from a previously enriched cache if available,
otherwise extract from the two raw GDELT parquet sources and resolve
headlines for any rows whose native title is missing or too short.
"""
import gc
import os

import pandas as pd

from config import CACHE_PATH_RO, CACHE_PATH_RW, INPUT_PATH_1, INPUT_PATH_2
from src.fetch_headlines import fetch_headlines_concurrent

STALE_SCORE_COLUMNS = (
    "fingpt_sentiment", "opinion_score", "fingpt_confidence",
    "fingpt_impact_hours", "fingpt_tier_weight", "opinion_label",
    "impact_score", "impact_score_raw", "combined_significance",
    "novelty_mult", "source_weight",
)


def _drop_stale_score_columns(df):
    stale_cols = [c for c in df.columns if c in STALE_SCORE_COLUMNS]
    if stale_cols:
        df = df.drop(columns=stale_cols)
    return df


def _extract_from_raw_sources():
    raw_src1 = pd.read_parquet(INPUT_PATH_1)
    raw_src2 = pd.read_parquet(INPUT_PATH_2)
    raw_src = pd.concat([raw_src1, raw_src2], axis=0, ignore_index=True)

    url_col = next((c for c in raw_src.columns if "url" in c.lower()), raw_src.columns[0])
    date_col = next((c for c in raw_src.columns if "date" in c.lower() or "time" in c.lower()), raw_src.columns[1])
    source_col = next((c for c in raw_src.columns if "source" in c.lower() or "domain" in c.lower()), None)
    title_col = next((c for c in raw_src.columns if "title" in c.lower() or "headline" in c.lower()), None)

    df = pd.DataFrame()
    df["url"] = raw_src[url_col].astype(str)
    df["datetime"] = raw_src[date_col]
    df["source"] = raw_src[source_col].astype(str) if source_col else ""

    if title_col and title_col in raw_src.columns:
        df["title"] = raw_src[title_col].astype(str).str.strip()
        slug_needed = df["title"].apply(lambda t: len(str(t).split()) < 4)
        if slug_needed.sum() > 0:
            sub_urls = df.loc[slug_needed, "url"].tolist()
            fetched = fetch_headlines_concurrent(sub_urls)
            df.loc[slug_needed, "title"] = fetched
    else:
        df["title"] = fetch_headlines_concurrent(df["url"].tolist())

    del raw_src, raw_src1, raw_src2
    gc.collect()
    return df


def load_or_extract_raw_data():
    """
    Load the cached enriched dataframe if present (preferring the writable
    cache over the read-only one), otherwise extract from raw GDELT
    parquet sources and persist a new cache. Always strips any stale
    scoring columns from a prior run.
    """
    if os.path.exists(CACHE_PATH_RW):
        df = pd.read_parquet(CACHE_PATH_RW)
        return _drop_stale_score_columns(df)

    if os.path.exists(CACHE_PATH_RO):
        df = pd.read_parquet(CACHE_PATH_RO)
        return _drop_stale_score_columns(df)

    df = _extract_from_raw_sources()
    df.to_parquet(CACHE_PATH_RW, index=False)
    return df
