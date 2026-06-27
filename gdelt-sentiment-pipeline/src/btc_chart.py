"""
Builds the candlestick-vs-sentiment Plotly chart, joining cumulative
sentiment to hourly-resampled BTC/USDT OHLCV data.
"""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import CHART_PATH


def load_btc_ohlcv(btc_path):
    """Load a BTC OHLCV CSV (any common column-naming convention) and resample to 1h."""
    btc_raw = pd.read_csv(btc_path)
    lower_cols = {c.lower().strip(): c for c in btc_raw.columns}
    time_col = (
        lower_cols.get("timestamp") or lower_cols.get("time")
        or lower_cols.get("date") or btc_raw.columns[0]
    )

    if btc_raw[time_col].dtype == object:
        btc_raw["dt"] = pd.to_datetime(btc_raw[time_col], utc=True, errors="coerce")
    else:
        unit = "ms" if btc_raw[time_col].max() > 1e11 else "s"
        btc_raw["dt"] = pd.to_datetime(btc_raw[time_col], unit=unit, utc=True, errors="coerce")

    open_col = lower_cols.get("open") or btc_raw.columns[1]
    high_col = lower_cols.get("high") or btc_raw.columns[2]
    low_col = lower_cols.get("low") or btc_raw.columns[3]
    close_col = lower_cols.get("close") or btc_raw.columns[4]
    vol_col = lower_cols.get("volume") or btc_raw.columns[5]

    for c, name in [(open_col, "open"), (high_col, "high"), (low_col, "low"),
                     (close_col, "close"), (vol_col, "volume")]:
        btc_raw[name] = pd.to_numeric(btc_raw[c], errors="coerce")

    btc_raw = btc_raw.dropna(subset=["dt", "open", "close"]).set_index("dt").sort_index()
    return btc_raw[["open", "high", "low", "close", "volume"]].resample("1h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )


def build_sentiment_price_chart(cum_sentiment_df, btc_path, output_path=CHART_PATH):
    """Build and save the candlestick + cumulative-sentiment subplot chart. Returns the merged df."""
    btc_1h = load_btc_ohlcv(btc_path)
    merged = cum_sentiment_df.join(btc_1h, how="inner")

    if merged.empty:
        return merged

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.04, row_heights=[0.6, 0.4],
    )
    fig.add_trace(go.Candlestick(
        x=merged.index, open=merged["open"], high=merged["high"],
        low=merged["low"], close=merged["close"], name="BTC/USDT",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=merged.index, y=merged["cumulative_sentiment"], mode="lines",
        name="Cumulative Momentum Indicator",
        line=dict(color="#2563eb", width=2),
    ), row=2, col=1)
    fig.update_layout(
        title=dict(
            text="BTC/USDT Spot Price vs Multi-Factor Continuous Sentiment Vector Tracking",
            x=0.5, font=dict(family="monospace", size=14),
        ),
        template="plotly_white", height=850,
        margin=dict(l=60, r=40, t=80, b=40),
        xaxis=dict(rangeslider=dict(visible=False)),
        xaxis2=dict(rangeslider=dict(visible=False)),
    )
    fig.write_html(output_path)
    return merged
