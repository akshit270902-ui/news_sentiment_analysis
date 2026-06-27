# GDELT News Sentiment Scoring Pipeline

An end-to-end pipeline that takes raw GDELT news-event data, filters it down
to headlines that plausibly move Bitcoin price/risk sentiment, scores each
one with a locally-hosted quantized Llama-3-8B-Instruct model, and produces
a decay-weighted continuous sentiment index — plus an interactive HTML
explorer and an optional BTC price-vs-sentiment chart.

This repo is a refactor of a single linear Kaggle notebook script into a
testable package: configuration, filtering/classification, scoring, and
temporal aggregation are each separated into their own module so they can be
read, tested, and modified independently.

## Pipeline stages

1. **Ingest** (`src/data_ingestion.py`) — load a cached enriched dataframe if
   one exists, otherwise extract from the two raw GDELT parquet sources and
   resolve missing headline titles via async HTTP fetch
   (`src/fetch_headlines.py`).
2. **Slug truncation** (`src/classify.py`) — trim overly long wire-aggregator
   titles down to a stable word count for known noisy sources.
3. **Keep/drop classification** (`src/classify.py`) — match each headline
   against a set of keyword-based categories (Fed policy, macro data,
   crypto-structural events, geopolitics, bank collapses, etc.) defined in
   `config.py`. Anything that doesn't match a category, or matches an
   exclusion rule, is dropped with a recorded reason code.
4. **Density clustering** (`src/dedup.py`) — flag abnormally headline-dense
   15-minute windows for an importance boost later.
5. **Deduplication** (`src/dedup.py`) — two passes:
   - rolling-window near-duplicate detection (exact match or word-overlap
     similarity within a time window), and
   - entity-event windowed dedup, collapsing many headlines about the same
     named event into one representative headline.
6. **LLM scoring** (`src/model_loader.py`, `src/inference.py`,
   `src/llm_scoring.py`) — a 4-bit quantized Llama-3-8B-Instruct model scores
   each kept headline's Bitcoin-specific sentiment direction, confidence,
   impact duration, and tier weight, using a few-shot structured prompt. If
   the model's output doesn't parse, a deterministic keyword-rule fallback
   (`KEYWORD_FALLBACK_RULES` in `config.py`) takes over.
7. **Post-scoring overrides** (`src/overrides.py`) — semantic overrides force
   sentiment direction when keyword evidence is unambiguous; a score floor
   gives near-zero directional scores a minimum magnitude and drops
   near-zero neutral articles; soft-impact overrides dampen mega-cap
   headlines describing narrow/soft events.
8. **Impact scoring** (`src/novelty.py`, orchestrated in
   `scripts/run_pipeline.py`) — combines sentiment, confidence, source
   credibility, tier weight, novelty decay, and aftermath/trigger
   multipliers into a single bounded impact score.
9. **Outputs** — scored parquet, top-100-by-impact CSV, and an interactive
   HTML headline explorer (`src/html_report.py`).
10. **Temporal decay spreading** (`src/temporal.py`) — each headline's signal
    is spread forward over its impact window with exponential decay, then
    aggregated into an hourly cumulative sentiment index.
11. **BTC chart** (`src/btc_chart.py`) — if BTC OHLCV data is available,
    render a candlestick-vs-cumulative-sentiment Plotly chart.

## Structure

```
gdelt-sentiment-pipeline/
├── README.md
├── requirements.txt
├── .gitignore
├── config.py                  ← all paths, hyperparameters, keyword tables
├── src/
│   ├── fetch_headlines.py      ← async URL → headline resolution
│   ├── classify.py             ← cleaning, source weighting, keep/drop classifier
│   ├── dedup.py                ← density clustering, similarity & entity-event dedup
│   ├── novelty.py              ← novelty decay & aftermath/trigger multipliers
│   ├── llm_scoring.py          ← prompt building, response parsing, keyword fallback
│   ├── overrides.py            ← semantic overrides, score floor, soft-impact dampening
│   ├── temporal.py             ← sentiment spreading & cumulative decay index
│   ├── html_report.py          ← interactive headline explorer report
│   ├── btc_chart.py            ← BTC OHLCV loading & price/sentiment chart
│   ├── data_ingestion.py       ← cache loading / raw parquet extraction
│   ├── model_loader.py         ← quantized Llama-3 model + tokenizer loading
│   └── inference.py            ← batched generation loop
├── scripts/
│   └── run_pipeline.py         ← orchestrates the full pipeline end to end
└── tests/
    ├── test_classify.py
    ├── test_scoring.py
    └── test_novelty_dedup_temporal.py
```

## Running

```bash
pip install -r requirements.txt
python scripts/run_pipeline.py
```

Paths (input parquet sources, BTC CSV, output locations) and all scoring
thresholds live in `config.py` — adjust there rather than in code. Note the
default paths assume a Kaggle environment (`/kaggle/input/...`,
`/kaggle/working/...`); update `config.py` if running elsewhere.

The Llama-3 inference stage (`scripts/run_pipeline.py` step 6) requires a
GPU and `bitsandbytes` 4-bit quantization support, and expects
`huggingface_hub` to already be authenticated with access to
`meta-llama/Meta-Llama-3-8B-Instruct`.

