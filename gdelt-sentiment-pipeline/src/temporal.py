"""
Temporal aggregation of per-headline sentiment into an hourly time series,
and decay-weighted cumulative sentiment index over that series.
"""
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from config import DECAY_PER_HOUR


def spread_sentiment(df):
    """
    Spread each headline's (score, weight) forward over its impact window
    with exponential decay, then aggregate to an hourly weighted-average
    sentiment series.
    """
    one_hour_ns = np.int64(3_600_000_000_000)
    ts_chunks, sw_chunks, w_chunks = [], [], []

    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Temporal decay spreading"):
        effective_impact_hrs = max(2, int(row.fingpt_impact_hours) - (
            int(row.fingpt_impact_hours) // 4 if row.is_aftermath else 0
        ))
        base_w = float(
            row.fingpt_tier_weight * row.combined_significance
            * row.article_mult * row.novelty_mult
        )
        if hasattr(row, "density_override") and row.density_override:
            base_w *= 1.5

        score = float(row.opinion_score)
        hrs = effective_impact_hrs
        lam = np.log(20.0) / hrs
        h_arr = np.arange(hrs + 1, dtype=np.float64)
        w_arr = base_w * np.exp(-lam * h_arr)
        base_ns = np.int64(row.hour.value)
        ts_arr = base_ns + (h_arr * one_hour_ns).astype(np.int64)

        ts_chunks.append(ts_arr)
        sw_chunks.append(score * w_arr)
        w_chunks.append(w_arr)

    ts_flat = np.concatenate(ts_chunks)
    sw_flat = np.concatenate(sw_chunks)
    w_flat = np.concatenate(w_chunks)

    tmp = pd.DataFrame({"ts": ts_flat, "sw": sw_flat, "w": w_flat})
    grp = tmp.groupby("ts")[["sw", "w"]].sum()
    sentiment = np.clip(grp["sw"] / grp["w"], -1.0, 1.0)
    idx = pd.to_datetime(grp.index, utc=True, unit="ns")
    return pd.DataFrame({"sentiment": sentiment.values}, index=idx).sort_index()


def compute_cumulative_sentiment(time_series_agg, decay_per_hour=DECAY_PER_HOUR):
    """Apply an exponential-decay running accumulation over the hourly sentiment series."""
    if time_series_agg.empty:
        return pd.DataFrame(columns=["hourly_sentiment", "cumulative_sentiment"])

    all_hours = pd.date_range(
        start=time_series_agg.index.min().floor("1h"),
        end=time_series_agg.index.max().floor("1h"),
        freq="1h", tz="UTC",
    )
    hourly_signal = time_series_agg["sentiment"].reindex(all_hours, fill_value=0.0)

    cumulative = np.zeros(len(all_hours))
    for i in range(len(all_hours)):
        prev = cumulative[i - 1] if i > 0 else 0.0
        cumulative[i] = prev * decay_per_hour + float(hourly_signal.iloc[i])

    cum_df = pd.DataFrame(
        {"hourly_sentiment": hourly_signal.values, "cumulative_sentiment": cumulative},
        index=all_hours,
    )
    cum_df.index.name = "dt"
    return cum_df
