"""
End-to-end GDELT news sentiment scoring pipeline.

Steps:
  1. Load cached/raw headline data, resolving missing titles.
  2. Truncate noisy wire-aggregator slug titles.
  3. Classify headlines into keep/drop categories.
  4. Flag headline-dense time windows.
  5. Deduplicate via rolling-window similarity match and entity-event windows.
  6. Run Llama-3 structured sentiment scoring (with keyword fallback parsing).
  7. Apply semantic overrides, score floor, and soft-impact dampening.
  8. Compute novelty/aftermath multipliers and the combined impact score.
  9. Write scored parquet, top-100 CSV, and the HTML headline explorer.
 10. Spread sentiment over time with decay and compute the cumulative index.
 11. If BTC OHLCV data is available, render the price-vs-sentiment chart.
"""
import os

import numpy as np
import pandas as pd

import config
from src.classify import (
    classify_keep_with_reason,
    get_source_weight,
    parse_gkg_date,
    truncate_slug_title,
)
from src.dedup import (
    apply_density_clustering,
    find_entity_event_duplicates,
    find_similarity_duplicates,
)
from src.data_ingestion import load_or_extract_raw_data
from src.html_report import build_html_explorer
from src.inference import run_batched_inference
from src.model_loader import load_model_and_tokenizer
from src.novelty import article_multiplier_v2, build_first_seen_index, compute_novelty_multiplier
from src.overrides import (
    apply_kept_score_floor,
    apply_semantic_overrides,
    apply_soft_impact_overrides,
)
from src.temporal import compute_cumulative_sentiment, spread_sentiment
from src.btc_chart import build_sentiment_price_chart


def run_pipeline():
    # 1. Ingest -------------------------------------------------------------
    df = load_or_extract_raw_data()
    df["dt"] = df["datetime"].apply(parse_gkg_date)
    df = df.dropna(subset=["dt"]).sort_values("dt").reset_index(drop=True)

    # 2. Slug truncation ------------------------------------------------------
    for idx, row in df.iterrows():
        truncated = truncate_slug_title(str(row["title"]), str(row.get("source", "")))
        if truncated != str(row["title"]):
            df.at[idx, "title"] = truncated

    # 3. Keep/drop classification --------------------------------------------
    df[["keep", "drop_reason"]] = df.apply(
        lambda row: pd.Series(classify_keep_with_reason(row["title"], row.get("source", ""))), axis=1
    )

    # 4. Density clustering ---------------------------------------------------
    df = apply_density_clustering(df)
    df_keep = df[df["keep"] == True].copy().reset_index(drop=True)

    # 5a. Rolling-window similarity dedup -------------------------------------
    duplicate_urls, kept_rows = find_similarity_duplicates(df_keep)
    if duplicate_urls:
        df.loc[df["url"].isin(duplicate_urls), "keep"] = False
        df.loc[df["url"].isin(duplicate_urls), "drop_reason"] = "duplicate_headline"
    df_keep = pd.DataFrame(kept_rows) if kept_rows else pd.DataFrame(columns=df_keep.columns)

    # 5b. Entity-event windowed dedup -----------------------------------------
    entity_event_dup_urls = find_entity_event_duplicates(df_keep, config.EVENT_DEDUP_CLUSTERS)
    if entity_event_dup_urls:
        df.loc[df["url"].isin(entity_event_dup_urls), "keep"] = False
        df.loc[df["url"].isin(entity_event_dup_urls), "drop_reason"] = "entity_event_duplicate"
        df_keep = df_keep[~df_keep["url"].isin(entity_event_dup_urls)].copy()

    # Article multiplier needs a first-seen index built before scoring -------
    first_seen_index = build_first_seen_index(df_keep)
    results = df_keep.apply(
        lambda r: article_multiplier_v2(r["title"], r["dt"], first_seen_index), axis=1
    )
    df_keep["article_mult"] = results.apply(lambda x: x[0])
    df_keep["is_aftermath"] = results.apply(lambda x: x[1])

    # 6. LLM scoring -----------------------------------------------------------
    model, tokenizer, eos_ids = load_model_and_tokenizer()
    headlines_list = df_keep["title"].tolist()
    parsed_structs = run_batched_inference(headlines_list, model, tokenizer, eos_ids)

    df_struct = pd.DataFrame(parsed_structs)
    df_keep = df_keep.reset_index(drop=True)
    df_keep["fingpt_sentiment"] = df_struct["sentiment"].values
    df_keep["opinion_score"] = df_struct["score"].values
    df_keep["fingpt_confidence"] = df_struct["confidence"].values
    df_keep["fingpt_impact_hours"] = df_struct["impact_hours"].values
    df_keep["fingpt_tier_weight"] = df_struct["tier_weight"].values
    df_keep["opinion_label"] = df_keep["fingpt_sentiment"]

    # 7. Post-scoring overrides --------------------------------------------------
    df_keep = apply_semantic_overrides(df_keep)
    df_keep, floor_dropped_urls = apply_kept_score_floor(df_keep)
    if floor_dropped_urls:
        df.loc[df["url"].isin(floor_dropped_urls), "keep"] = False
        df.loc[df["url"].isin(floor_dropped_urls), "drop_reason"] = "neutral_zero_score"
    df_keep = apply_soft_impact_overrides(df_keep)

    # 8. Combined impact score -----------------------------------------------------
    df_keep["novelty_mult"] = compute_novelty_multiplier(df_keep)
    df_keep["source_weight"] = df_keep["source"].apply(get_source_weight)
    df_keep["combined_significance"] = (
        df_keep["fingpt_confidence"] * df_keep["fingpt_tier_weight"] * df_keep["source_weight"]
    )
    df_keep["impact_score_raw"] = (
        df_keep["opinion_score"] * df_keep["combined_significance"]
        * df_keep["article_mult"] * df_keep["novelty_mult"]
    )
    if "density_override" in df_keep.columns:
        df_keep.loc[df_keep["density_override"] == True, "impact_score_raw"] *= 1.5
    df_keep["impact_score"] = np.tanh(df_keep["impact_score_raw"] / config.RAW_IMPACT_SCALE) * 5.0

    # 9. Persist outputs -------------------------------------------------------------
    score_lookup = {
        str(getattr(row, "url", "")): {
            "sentiment": str(getattr(row, "fingpt_sentiment", "")),
            "opinion": float(getattr(row, "opinion_score", 0.0) or 0.0),
            "impact": float(getattr(row, "impact_score", 0.0) or 0.0),
            "category": str(getattr(row, "drop_reason", "")),
        }
        for row in df_keep.itertuples(index=False)
    }

    df.to_parquet(config.OUTPUT_PATH, index=False)
    df_keep.sort_values(by="impact_score", key=abs, ascending=False).head(100).to_csv(
        config.TOP_PATH, index=False
    )
    build_html_explorer(df, score_lookup, config.HTML_REPORT_PATH)

    # 10. Temporal decay spreading ---------------------------------------------------
    df_keep["hour"] = df_keep["dt"].dt.floor("1h")
    time_series_agg = spread_sentiment(df_keep)
    cum_sentiment_df = compute_cumulative_sentiment(time_series_agg, config.DECAY_PER_HOUR)

    print(f"Scoring pipeline complete.")
    print(f"  Scored parquet -> {config.OUTPUT_PATH}")
    print(f"  Top 100 shocks -> {config.TOP_PATH}")
    print(f"  HTML explorer  -> {config.HTML_REPORT_PATH}")

    # 11. BTC price vs sentiment chart (optional) -------------------------------------
    if os.path.exists(config.BTC_PATH):
        merged = build_sentiment_price_chart(cum_sentiment_df, config.BTC_PATH, config.CHART_PATH)
        if not merged.empty:
            print(f"  Sentiment chart -> {config.CHART_PATH}")

    return df, df_keep, cum_sentiment_df


if __name__ == "__main__":
    run_pipeline()
