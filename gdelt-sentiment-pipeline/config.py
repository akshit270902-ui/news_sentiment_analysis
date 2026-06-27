"""
Central configuration for the GDELT news sentiment scoring pipeline.

Everything that is a "knob" — file paths, model name, thresholds, keyword
lists used for filtering/classification/scoring — lives here so the rest of
the codebase never hardcodes a magic string or number.
"""

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
INPUT_PATH_1     = "/kaggle/input/datasets/akshit27/gdelt-news/gdelt_news_raw_gkg.parquet"
INPUT_PATH_2     = "/kaggle/input/datasets/akshit27/gdelt-fed/gdelt_us_macro_releases_raw.parquet"
OUTPUT_PATH      = "/kaggle/working/gdelt_news_scored.parquet"
TOP_PATH         = "/kaggle/working/top100_headlines.csv"
CACHE_PATH_RO    = "/kaggle/input/datasets/akshit27/gdelt-f/gdelt_full_enriched.parquet"
CACHE_PATH_RW    = "/kaggle/working/gdelt_full_enriched.parquet"
HTML_REPORT_PATH = "/kaggle/working/headline_explorer.html"
BTC_PATH         = "/kaggle/input/datasets/akshit27/btcusd-g/BTCUSDT_1m.csv"
CHART_PATH       = "/kaggle/working/btc_sentiment_chart.html"

# ---------------------------------------------------------------------------
# Model / inference
# ---------------------------------------------------------------------------
BASE_MODEL             = "meta-llama/Meta-Llama-3-8B-Instruct"
BATCH_SIZE             = 32
MODEL_MAX_INPUT_TOKENS = 2048
MODEL_MAX_NEW_TOKENS   = 150

# ---------------------------------------------------------------------------
# Scoring / decay parameters
# ---------------------------------------------------------------------------
DECAY_PER_HOUR          = 0.90
NOVELTY_DECAY           = 0.50
NOVELTY_WINDOW_H        = 120
AFTERMATH_STRONG_MULT   = 0.30
AFTERMATH_MODERATE_MULT = 0.50
TRIGGER_BOOST           = 1.25
RAW_IMPACT_SCALE        = 2.5
KEPT_SCORE_FLOOR        = 0.10

# ---------------------------------------------------------------------------
# HTTP fetching (for slug -> headline resolution)
# ---------------------------------------------------------------------------
FETCH_TIMEOUT = 5
FETCH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}
MAX_CONCURRENT_REQUESTS = 500
MAX_PER_HOST            = 5

# ---------------------------------------------------------------------------
# Source credibility
# ---------------------------------------------------------------------------
LOW_CREDIBILITY_SOURCES = {
    "obituaries.com", "legacy.com", "courtlistener.com", "justia.com",
    "streetwisejournal.com", "ipsnews.net", "weeklyblitz.net",
    "industryleadersmagazine.com", "peakoil.com", "watcher.guru",
    "proactiveinvestors.com", "bolnews.com", "estonianworld.com",
    "latestly.com", "newsbtc.com", "banklesstimes.com", "lelezard.com",
    "wsau.com", "wkzo.com", "wtaq.com", "saltwire.com", "forextv.com",
    "marketscreener.com", "rttnews.com",
    "switzer.com.au", "industrialinfo.com",
}

SLUG_TRUNCATION_SOURCES = {
    "zerohedge.com", "marketwatch.com", "seekingalpha.com",
    "investing.com", "fxstreet.com", "forexlive.com", "newsbtc.com",
    "cryptoslate.com", "ambcrypto.com", "bitcoinist.com", "utoday.com",
    "u.today", "cryptopotato.com", "beincrypto.com", "dailyhodl.com",
    "cryptonews.com", "coinspeaker.com", "decrypt.co",
}

SLUG_TRUNCATION_MAX_WORDS = 10
SLUG_TRUNCATION_MIN_WORDS = 15

SOURCE_CREDIBILITY_WEIGHTS = {
    "reuters.com":              1.3,
    "bloomberg.com":            1.3,
    "wsj.com":                  1.2,
    "ft.com":                   1.2,
    "apnews.com":               1.2,
    "coindesk.com":             1.1,
    "financialtimes.com":       1.2,
    "forbes.com":               1.0,
    "cointelegraph.com":        1.0,
    "thehindubusinessline.com": 1.0,
    "businessinsider.in":       1.0,
    "foxbusiness.com":          0.9,
    "zerohedge.com":            0.6,
    "peakoil.com":              0.5,
    "bedfordgazette.com":       0.5,
    "chronicleonline.com":      0.5,
    "menafn.com":               0.7,
    "theepochtimes.com":        0.7,
    "business2community.com":   0.7,
}

# ---------------------------------------------------------------------------
# Soft impact overrides — dampen score when a mega-cap headline is really
# about a narrow/soft event (e.g. "Microsoft bans mining" is not a systemic
# crypto event, despite mentioning a mega-cap name)
# ---------------------------------------------------------------------------
SOFT_IMPACT_OVERRIDES = [
    ("microsoft", [
        "mining ban", "bans mining", "bans cryptocurrency mining",
        "bans cryptomining", "ban cryptocurrency mining",
    ]),
    ("amazon", ["mining ban", "bans mining", "bans cryptomining"]),
    ("google", ["mining ban", "bans mining", "bans cryptomining"]),
]

HIGH_TIER_MULT_FLOOR_ANCHORS = [
    "ftx", "bankman", "binance charged", "binance indicted",
    "sec sues", "doj charges", "exchange collapses", "exchange bankrupt",
    "exchange halts withdrawal", "stablecoin depeg", "usdt depeg",
    "usdc depeg", "tether insolvency", "bitcoin banned", "crypto banned",
    "luna collapse", "ust depeg", "celsius bankrupt",
]
HIGH_TIER_MULT_FLOOR = 0.25

SINGLE_COUNTRY_MINOR_KEYWORDS = [
    "brazil", "australia", "canada", "south korea", "singapore",
    "nigeria", "india crypto", "russia crypto", "indonesia crypto",
    "turkey crypto", "vietnam crypto", "thailand crypto",
    "honduras", "ecuador", "bolivia", "nepal", "pakistan crypto",
    "bangladesh crypto", "egypt crypto", "kenya crypto", "ghana crypto",
    "tanzania crypto", "ethiopia crypto", "cameroon crypto", "jordan crypto",
    "algeria crypto", "morocco crypto", "tunisia crypto",
]

# ---------------------------------------------------------------------------
# Aggregator / wire slug detection (e.g. "Newsquawk APAC market open")
# ---------------------------------------------------------------------------
SLUG_TOPIC_TOKENS = [
    "newsquawk", "apac", "dxy", "europe market open", "market open",
    "asia session", "daily markets", "morning note", "wrap:", "roundup:",
    "market wrap", "market update", "week ahead", "daily brief",
]

# ---------------------------------------------------------------------------
# Keep-category classification rules
# ---------------------------------------------------------------------------
KEEP_CATEGORIES = {
    "black_swan_systemic": {
        "require_any": [
            "halts operations", "state of emergency", "supply chain halted",
            "unprecedented stimulus", "global outage", "trading suspended",
            "liquidity crisis", "market crash", "forced liquidation",
            "circuit breaker triggered", "nationalization", "contagion risk",
            "emergency meeting", "hyperinflation", "default imminent",
            "global crisis", "global financial system", "systemic threat",
            "new covid strain", "new coronavirus strain", "covid variant",
            "covid lockdown", "pandemic fears", "pandemic shock",
            "global lockdown", "variant discovered", "variant detected",
            "omicron", "delta variant", "pandemic wave",
        ],
        "require_scale": [
            "markets", "stocks", "bitcoin", "crypto", "investors",
            "global", "worldwide", "who ", "variant spreads",
            "lockdown", "economy", "financial",
        ],
    },

    "us_macro_data": {
        "require_any": [
            "cpi ", "cpi:", "consumer price index",
            "pce ", "pce:", "personal consumption expenditure",
            "ppi ", "ppi:", "producer price index",
            "jolts report", "jolts data", "jolts survey",
            "job openings report", "job openings data", "labor turnover survey",
            "nonfarm payroll", "jobs report", "payrolls report",
            "unemployment rate", "jobless rate", "initial jobless claims",
            "gdp report", "gdp growth", "gdp contraction", "gdp data",
            "retail sales", "core inflation", "inflation data", "inflation report",
            "ism manufacturing", "ism services", "pmi data", "pmi report",
            "housing starts", "existing home sales", "durable goods orders",
            "consumer confidence", "consumer sentiment", "michigan sentiment",
            "trade balance report", "current account data",
        ],
        "require_data_signal": [
            "came in", "beat expectations", "missed expectations",
            "above forecast", "below forecast", "higher than expected",
            "lower than expected", "stronger than expected", "weaker than expected",
            "above estimates", "below estimates",
            "rose %", "fell %", "climbed %", "dropped %", "jumped %",
            "month-over-month", "year-over-year", " m/m", " y/y",
            "added jobs", "lost jobs", "shed jobs",
            "revised", "preliminary", "flash reading",
            "report shows", "data shows", "figures show", "reading of",
            "basis points", "percentage point",
            "bitcoin", " btc ", "crypto", "markets react", "risk assets",
            "dollar", "yields", "equities fall", "stocks fall",
            "fed may", "rate cut odds", "rate hike odds",
            "fed pricing", "bets on", "odds of",
        ],
    },

    "fed_rates": {
        "require_any": [
            "federal reserve", "fed reserve", " fomc ",
            "fed chair", "jerome powell", "janet yellen",
            "rate cut", "rate hike", "rate hold", "raises rates", "cuts rates",
            "interest rate decision", "rate decision", "holds rates",
            "basis points", "bps cut", "bps hike",
            "quantitative easing", "quantitative tightening",
            "fed balance sheet", "federal reserve balance sheet",
            "central bank balance sheet", "shrink the balance sheet",
            "expand the balance sheet", "balance sheet runoff",
            "tapering", "asset purchases",
            "monetary policy", "policy statement", "rate path",
            "dot plot", "forward guidance", "pivot",
            "dovish", "hawkish", "policy shift", "inflation target",
        ],
        "require_institution": [
            "federal reserve", "fed reserve", " fomc ", "fed chair",
            "jerome powell", "janet yellen", "central bank",
            "rate decision", "rate cut", "rate hike", "bps cut", "bps hike",
            "basis points", "interest rate",
        ],
        "exclude_any": [
            "opinion:", "op-ed", "analysis:", "commentary:",
            "blames", "says fed", "thinks fed", "argues that",
            "why the fed", "how the fed", "what the fed",
            "explains why", "here's why", "heres why",
            "could affect your", "affect your wallet", "affect your finances",
            "affect your bank", "affect your household", "affect your savings",
            "your household debt", "household debt burden",
            "kevin o'leary", "shark tank",
            "price prediction", "price target", "analyst predicts",
            "could reach", "might hit", "could hit",
            "did rate hikes kill", "killed the crypto",
            "mixed reactions", "draws mixed",
            "top finance", "top biz news", "key rate hikes",
            "2022 top", "[2022", "[2023",
            "how feds", "feds series", "feds rate",
            "heres how the fed", "here's how the fed",
            "your finances", "your wallet", "your bank account",
            "your savings", "your household",
            "how will the", "how could the",
            "what does the", "what will the",
            "consumers", "homeowners", "borrowers",
            "affect you", "affects you", "impact you",
            "impact on you", "impact on your",
            "generation hemp", "weighs in on",
            "global markets-festivity", "festivity on hold",
            "no rate cuts in 2024", "no rate cuts in 2023",
            "daily open", "daily markets", "stocks slip amid",
            "markets brace", "market brace",
            "meme-coin", "meme coin", "banking on rate cuts",
            "bet on rate cuts", "bets on rate cuts",
            "investors bet", "investors banking",
            "race record highs", "gold and bitcoin",
        ],
    },

    "us_china_sanctions_trade": {
        "require_any": [
            "us sanctions china", "united states sanctions china",
            "china sanctions us", "china sanctions america",
            "us imposes tariff", "us tariff on china", "trump tariff",
            "china tariff", "trade war", "trade dispute",
            "us blacklists", "entity list", "export controls china",
            "chip ban china", "semiconductor ban china",
            "us bans china", "china bans us",
            "us-china trade", "us china trade",
            "decoupling china", "tech war china",
        ],
    },

    "energy_crisis": {
        "require_any": [
            "oil price", "crude price", "brent crude", "wti crude",
            "opec cuts", "opec+ cuts", "opec production", "opec meeting",
            "oil supply shock", "oil embargo", "oil ban",
            "energy crisis", "energy shortage", "power shortage",
            "natural gas price", "gas shortage", "lng shortage",
            "oil sanctions", "russia oil", "iran oil",
            "oil jump", "oil jumps", "oil spikes", "oil surges",
        ],
        "require_market_signal": [
            "bitcoin", " btc ", "crypto", "markets", "stocks", "equities",
            "investors", "surge", "crash", "spike", "record", "rally",
            "risk", "inflation", "gdp", "opec", "brent", "wti",
        ],
    },

    "geopolitical_major_powers": {
        "require_any": [
            "declares war", "war declared", "invasion begins", "invades",
            "airstrike", "missile strike", "troops cross", "nuclear threat",
            "nuclear launch", "nuclear strike", "nuclear war",
            "military conflict", "military operation", "armed conflict",
            "ceasefire", "peace talks", "peace deal", "peace agreement",
            "war ends", "armistice", "sanctions escalate", "escalation",
            "tension", "tensions", "risk of war", "war fears",
        ],
        "require_country": [
            "united states", " usa ", " us ", "america",
            "china", "russia", "europe", "nato", "ukraine",
            "iran", "north korea", "israel", "taiwan",
            "middle east", "eu", "european union",
        ],
    },

    "major_currency": {
        "require_any": [
            "dollar collapse", "dollar crisis", "dxy falls", "dxy crashes",
            "dollar index", "usd weakens", "usd strengthens",
            "de-dollarization", "dollar dominance",
            "yuan devaluation", "renminbi devaluation", "pboc devalues",
            "yuan weakens", "yuan crashes", "yuan hits",
            "yen weakens", "yen crisis", "yen intervention", "boj intervenes",
            "yen hits record", "dollar yen",
            "euro weakens", "euro crisis", "euro parity", "euro crashes",
            "euro hits record", "ecb intervention",
            "currency war", "currency devaluation", "fx intervention",
            "reserve currency shift", "dollar replacement",
        ],
        "exclude_any": [
            "newsquawk", "apac", "europe market open", "market open",
            "asia session", "daily brief", "market wrap",
        ],
    },

    "bank_fund_collapse": {
        "require_any": [
            "bank collapse", "bank fails", "bank failure", "bank run",
            "bank seized", "fdic seizes", "bank bailout", "bank rescued",
            "bank insolvency", "bank bankrupt", "bank default",
            "bank collapses", "bank shutters", "bank closed by",
            "fund collapse", "fund fails", "fund liquidat", "fund bankrupt",
            "hedge fund collapses", "hedge fund fails", "hedge fund blows up",
            "softbank collapse", "softbank crisis", "softbank losses",
            "credit suisse", "lehman", "silicon valley bank", " svb ",
            "signature bank", "first republic bank",
            "systemic risk", "financial contagion", "banking crisis",
        ],
    },

    "major_us_company_news": {
        "require_company": [
            "microsoft", "nvidia", "nvda", "tesla", "meta ", "facebook",
            "amazon", "apple", "google", "alphabet",
        ],
        "require_any": [
            "layoffs", "mass layoffs", "job cuts", "workforce reduction",
            "stock crashes", "stock plunges", "stock collapses", "shares crash",
            "shares plunge", "shares tumble", "market cap wipes",
            "earnings", "revenue", "guidance", "outlook",
            "doj antitrust", "ftc antitrust", "us antitrust",
            "antitrust lawsuit", "monopoly ruling",
            "sec charges", "doj charges", "regulatory action",
            "data breach", "major breach", "cyberattack",
            "ceo resigns", "ceo fired", "ceo ousted",
            "accounting fraud", "fraud charges",
            "product launch", "new product", "quarterly beat", "beats estimates",
            "stock surges", "stock rallies", "record revenue",
        ],
    },

    "major_ai_news": {
        "require_any": [
            "gpt-5", "gpt5", "openai releases", "openai launches",
            "anthropic releases", "anthropic launches", "claude 3", "claude 4",
            "gemini ultra", "gemini pro", "google deepmind",
            "agi achieved", "artificial general intelligence",
            "ai regulation", "ai bill", "ai law", "ai ban",
            "ai executive order", "ai act", "eu ai act",
            "ai model beats", "ai surpasses human",
            "ai chips ban", "ai compute ban", "ai export controls",
            "openai valuation", "anthropic valuation",
            "ai model costs", "ai inference cost",
            "openai bankruptcy", "openai crisis", "openai collapse",
        ],
    },

    "crypto_structural": {
        "require_any": [
            "bitcoin etf approved", "bitcoin etf rejected", "bitcoin etf denied",
            "spot bitcoin etf", "sec approves bitcoin etf", "sec rejects bitcoin etf",
            "ethereum etf approved", "ethereum etf rejected",
            "bitcoin legal tender", "crypto legal tender",
            "country adopts bitcoin", "nation adopts bitcoin",
            "government adopts bitcoin", "state adopts bitcoin",
            "strategic bitcoin reserve", "national bitcoin reserve",
            "sovereign bitcoin reserve",
            "ban cryptocurrency", "bans cryptocurrency", "cryptocurrency banned",
            "ban cryptocurrencies", "cryptocurrencies banned", "ban crypto",
            "bans crypto", "crypto banned", "bans bitcoin", "ban bitcoin",
            "bitcoin banned", "bitcoin outlawed", "crypto outlawed", "bitcoin illegal",
            "watchdog bans", "regulator bans", "fca bans", "sec bans",
            "ftx collapse", "ftx bankrupt", "ftx fraud",
            "binance charged", "binance indicted", "binance bankrupt",
            "coinbase charged", "coinbase indicted",
            "exchange collapses", "exchange bankrupt", "exchange seized",
            "exchange halts withdrawal", "exchange halts withdrawals",
            "terra luna collapse", "luna collapse", "ust depeg",
            "tether insolvency", "usdt depeg", "usdc depeg", "stablecoin depeg",
            "bitcoin halving", "btc halving", "block subsidy halving",
            "microstrategy buys bitcoin", "tesla buys bitcoin", "blackrock bitcoin",
            "sovereign wealth buys bitcoin", "central bank buys bitcoin",
            "balance sheet to bitcoin", "converts balance sheet to bitcoin",
            "tesla balance sheet bitcoin", "moves balance sheet to bitcoin",
            "corporate treasury bitcoin", "treasury allocation bitcoin",
            "convert treasury to bitcoin",
            "sec sues", "doj charges crypto", "crypto regulation",
            "arrested for crypto", "indicted for crypto",
            "crypto hack", "exchange hacked", "bitcoin exploit",
            "crypto exploit", "wallet hacked", "defi hack",
            "hack costs", "stolen from exchange",
            "bitcoin price crashes", "bitcoin price surges", "bitcoin all-time high",
            "bitcoin hits record", "btc hits record", "bitcoin reaches",
            "hash rate", "haven asset", "safe haven crypto",
        ],
        "exclude_any": [
            "masterclass", "webinar", "attend the", "how to trade",
            "trading course", "trading app review", "robot review",
            "scam or not", "scam or legit", "is it a scam",
            "celebrate the bitcoin halving", "celebrate bitcoin halving",
            "come celebrate", "party with", "join us for",
            "is the bitcoin halving the right time to invest",
            "right time to invest in btc", "time to buy bitcoin",
            "should you invest", "should i invest",
            "what happens if bitcoin reaches", "is it possible",
            "what if bitcoin", "could bitcoin reach",
            "a new bitcoin all-time high before",
            "before the halving: is it possible",
            "network energy consumption", "energy consumption",
            "environmental impact", "carbon footprint",
            "bitcoin cash", "bch ",
            "brand video", "releases brand", "brand film",
            "1st brand", "new brand video", "releases video",
            "gold rush", "digital transformation spot",
            "alternate sales channels", "natural gas prices",
            "permian", "eps seek",
            "newsquawk", "apac trade", "europe market open",
            "q&a:", "what is bitcoin", "introduction to crypto",
            "guide to crypto", "guide to bitcoin", "beginners guide",
            "review 2020", "review 2021", "review 2022", "review 2023",
            "does this app", "trader experience", "read before trading",
            "read our review", "our verdict", "trading platform review",
            "trading signal", "trading bot review",
            "podcast episode", "interview with", "talks about",
            "explains why", "why you should",
            "price prediction", "price target", "analyst predicts",
            "could reach", "might hit", "could hit",
            "opinion:", "op-ed", "blames", "kevin o'leary", "shark tank",
            "young crypto investors", "still bullish", "survey",
            "study finds", "report finds", "poll finds",
            "extradited", "extradition",
            "plea agreement", "pleads guilty", "pleaded guilty",
            "guilty plea", "plea deal",
            "bankruptcy ceo", "bankruptcy judge", "bankruptcy proceedings",
            "bankruptcy hearing", "bankruptcy filing",
            "customer names", "creditor names", "creditor list",
            "sullivan & cromwell", "sullivan and cromwell",
            "us trustee", "media challenging", "secrecy in ftx",
            "ftx bankruptcy ceo makes", "paid by its customers",
            "tom brady sued", "celebrity sued", "investor sued",
            "sued by", "lawsuit accuses",
            "flip flops on", "flip-flops on",
            "intellectual father", "effective altruism",
            "has the ftx collapse killed",
            "puts binance into", "binance into trouble",
            "binance endures", "wild weeks",
            "a call for crypto regulations", "what should crypto regulation",
            "how will the", "impact on",
            "crypto goals are impacted", "singapore crypto goals",
            "crypto ice age",
            "race record highs investors bet",
            "meme-coin resurgence", "meme coin resurgence",
            "banking on rate cuts",
        ],
    },
}

# ---------------------------------------------------------------------------
# Semantic overrides — force sentiment direction when keyword evidence is
# unambiguous, regardless of what the LLM said
# ---------------------------------------------------------------------------
SEMANTIC_OVERRIDES = [
    ("negative", [
        "bans sale of crypto", "bans sale of bitcoin", "bans cryptocurrency derivatives",
        "bans crypto derivatives", "ban on crypto derivatives", "ban on bitcoin derivatives",
        "watchdog bans", "regulator bans", "fca bans", "sec bans",
        "city watchdog bans", "financial watchdog bans",
        "crack down on crypto", "crackdown on crypto", "crackdown on bitcoin",
        "ban on cryptocurrency", "ban on bitcoin",
        "re bans crypto", "again bans crypto", "bans crypto again",
        "re-bans crypto", "re bans bitcoin", "again bans bitcoin",
        "arrested", "arrested for", "indicted for", "indicted on",
        "charged with", "charged over", "pleads guilty", "found guilty",
        "sued by sec", "sec sues", "doj indicts", "doj charges",
        "fraud charges", "tax evasion", "money laundering charges",
        "ponzi scheme", "exit scam",
        "exchange seized", "exchange halts withdrawal", "exchange halts withdrawals",
        "hack costs", "hacked for", "stolen from", "funds stolen",
        "wallet drained", "exploit drains",
        "bitcoin price crashes", "btc crashes", "crypto crashes",
        "market crash", "forced liquidation", "circuit breaker triggered",
        "usdt depeg", "usdc depeg", "stablecoin depeg", "tether insolvency",
        "ust depeg", "luna collapse", "ftx collapse", "ftx bankrupt",
        "exchange collapses", "exchange bankrupt",
        "new covid strain", "new coronavirus strain", "covid variant",
        "covid lockdown", "pandemic fears", "pandemic shock",
        "global lockdown", "variant discovered", "variant detected",
        "omicron", "delta variant", "pandemic wave",
    ]),
    ("positive", [
        "bitcoin etf approved", "sec approves bitcoin etf", "spot bitcoin etf approved",
        "ethereum etf approved",
        "bitcoin legal tender", "crypto legal tender",
        "country adopts bitcoin", "nation adopts bitcoin",
        "strategic bitcoin reserve", "national bitcoin reserve",
        "rate cut announced", "fed cuts rates", "fed cuts interest",
        "ceasefire announced", "peace deal signed", "war ends",
        "bitcoin all-time high", "btc all-time high", "bitcoin hits record high",
        "bitcoin reaches new high",
        "microstrategy buys bitcoin", "tesla buys bitcoin",
        "blackrock bitcoin approved",
    ]),
]

# ---------------------------------------------------------------------------
# Event clustering — used for novelty decay & first-seen tracking
# ---------------------------------------------------------------------------
EVENT_CLUSTERS = [
    ["silicon valley bank", " svb "],
    ["signature bank"],
    ["credit suisse"],
    ["first republic bank", "first republic collapse"],
    ["ftx collapse", "ftx fraud", "ftx bankrupt", "sam bankman"],
    ["terra luna", "luna collapse", " ust ", "ust depeg", "do kwon"],
    ["celsius collapse", "celsius bankrupt"],
    ["ukraine", "russian invasion", "russia invades ukraine"],
    ["bitcoin halving", "btc halving", "block subsidy halving"],
    ["bitcoin etf approval", "bitcoin etf rejected", "spot bitcoin etf"],
    ["lehman brothers"],
    ["tether insolvency", "usdt depeg"],
    ["binance charged", "binance indicted", "cz binance"],
]

EVENT_DEDUP_CLUSTERS = [
    (48,  ["elon musk bitcoin", "musk bitcoin", "tesla bitcoin", "tesla balance sheet bitcoin",
           "elon musk considers", "elon musk inquired", "elon musk curious"]),
    (72,  ["sec sues ripple", "ripple lawsuit", "xrp lawsuit", "ripple xrp"]),
    (24,  ["bitcoin hits record", "bitcoin all-time high", "btc hits record",
           "bitcoin reaches", "bitcoin surpasses", "bitcoin surges past",
           "bitcoin breaks record", "bitcoin new all-time", "btc new high",
           "bitcoin price record", "bitcoin record high"]),
    (168, ["ftx collapse", "ftx bankrupt", "sam bankman"]),
    (168, ["terra luna", "luna collapse", "ust depeg"]),
    (168, ["silicon valley bank collapse", "svb collapse", "svb fails"]),
    (72,  ["bitcoin halving", "btc halving", "block reward halves"]),
    (48,  ["binance charged", "binance indicted", "cz binance arrested"]),
    (6,   ["spot bitcoin etf", "bitcoin etf", "s korea", "south korea authorities"]),
    (6,   ["no rate cuts", "daily open", "rate cuts in 2024", "rate cuts in 2023"]),
]

# ---------------------------------------------------------------------------
# Aftermath / trigger phrase sets — drive the article_multiplier
# ---------------------------------------------------------------------------
AFTERMATH_STRONG = [
    "fallout of", "fallout from", "in wake of", "in the wake of", "as a result of",
    "due to banking", "due to financial crisis", "due to crypto",
    "due to bitcoin crash", "due to market crash",
    "losses amid", "drops amid", "falls amid", "slides amid",
    "sinks amid", "tumbles amid", "slips amid", "declines amid",
    "plunges amid", "crashes amid", "selloff amid",
    "after the collapse", "after the crash", "after the crisis",
    "following the collapse", "following the crash", "following the crisis",
    "stemming from", "contagion from", "ripple from",
    "still reeling", "continues to fall", "continues to drop",
    "continues to decline", "remains under pressure from",
    "in fallout", "follows collapse", "follows crash",
]

AFTERMATH_MODERATE = [
    "amid concerns", "amid fears", "amid worries",
    "amid uncertainty", "amid volatility",
    "amid banking crisis", "amid financial crisis",
    "amid crypto crash", "amid bitcoin crash",
    "amid banking turmoil", "amid market turmoil",
    " continues ", "ongoing crisis",
    "in response to", "reaction to", "reacting to",
    "impact of the", "effects of the",
    "in the aftermath", "as crisis deepens",
    "as collapse continues", "as fallout", "as panic",
    "despite easing", "despite relief", "easing in banking",
    "banking crisis sentiment", "financial crisis fears",
    "amid sell-off", "amid selloff",
]

TRIGGER_BOOST_PHRASES = [
    " collapses", " collapse announced", " files for bankruptcy",
    " declares war", " invades", " invasion begins",
    " raises rates", " cuts rates", " hikes rates",
    " approved", " rejected", " bans ", " outlaws ",
    " halving begins", " halving complete", " reward halves",
    " depegs", " halts withdrawals", " seizes",
    " announces rate", " announces ban",
    " crashes to", " surges to", " soars to", " plunges to",
    " hits all-time high", " hits record",
]

# ---------------------------------------------------------------------------
# Keyword fallback scoring rules — used when the LLM output fails to parse.
# Each tuple: (score, confidence, impact_hours, tier_weight, keywords)
# ---------------------------------------------------------------------------
KEYWORD_FALLBACK_RULES = [
    (0.95, 0.92, 168, 5.0, ["bitcoin etf approved", "sec approves bitcoin etf", "spot bitcoin etf approved"]),
    (0.85, 0.88, 168, 4.5, ["bitcoin halving", "btc halving", "block subsidy halving"]),
    (0.80, 0.85,  72, 4.0, ["rate cut", "cuts rates", "bps cut", "dovish", "quantitative easing", "pivot"]),
    (0.75, 0.82,  72, 4.0, ["bitcoin legal tender", "country adopts bitcoin", "strategic bitcoin reserve"]),
    (0.70, 0.80,  24, 3.5, ["beats estimates", "record revenue", "stock surges", "stock rallies"]),
    (0.65, 0.75,  24, 3.0, ["ceasefire", "peace deal", "peace agreement", "war ends", "armistice"]),
    (0.60, 0.72,  24, 3.0, ["cpi falls", "inflation cools", "inflation drops", "inflation slows",
                              "gdp growth", "consumer confidence rises", "jobs added"]),
    (0.50, 0.68,  48, 3.0, ["bitcoin property", "bitcoin as property", "legal recognition of bitcoin",
                              "confirmed status of bitcoin", "proprietary injunction"]),
    (0.45, 0.65,  48, 2.5, ["reaches million users", "surpasses million users", "million crypto users",
                              "mainstream adoption", "growing adoption", "record wallet users"]),
    (0.45, 0.65,  24, 2.5, ["earnings beat", "product launch", "new product", "quarterly beat"]),
    (0.40, 0.62,  24, 2.5, ["microstrategy buys bitcoin", "tesla buys bitcoin", "blackrock bitcoin",
                              "balance sheet to bitcoin", "treasury allocation bitcoin"]),
    (0.05, 0.50,   6, 2.0, ["fomc meeting", "policy statement", "dot plot", "forward guidance"]),
    (-0.45, 0.65, 24, 2.5, ["rate hike", "raises rates", "bps hike", "hawkish", "quantitative tightening"]),
    (-0.50, 0.70, 24, 3.0, ["cpi hotter", "inflation surges", "inflation rises", "inflation beats",
                              "nonfarm payrolls beat", "jobs report beat", "unemployment falls"]),
    (-0.50, 0.72, 24, 3.0, ["crackdown on crypto", "crackdown on bitcoin", "regulatory crackdown",
                              "watchdog bans", "regulator bans", "fca bans",
                              "bans sale of crypto", "bans crypto derivatives",
                              "city watchdog bans", "financial watchdog bans"]),
    (-0.55, 0.75, 24, 3.0, ["arrested", "arrested for crypto", "arrested over crypto",
                              "indicted for", "indicted over", "charged with crypto",
                              "charged with fraud", "pleads guilty", "found guilty",
                              "sued by sec", "sec sues", "doj indicts", "doj charges",
                              "fraud charges", "tax evasion", "money laundering",
                              "ponzi scheme", "exit scam", "crypto scam",
                              "promoting cryptocurrency offerings"]),
    (-0.55, 0.75, 24, 3.0, ["hack costs", "hacked for", "exchange hacked",
                              "crypto hack", "defi hack", "wallet hacked",
                              "stolen from", "funds stolen", "bitcoin stolen",
                              "exploit drains", "reentrancy attack"]),
    (-0.60, 0.72, 24, 3.5, ["new covid strain", "new coronavirus strain", "covid variant",
                              "covid lockdown", "pandemic fears", "pandemic shock",
                              "global lockdown", "variant discovered", "variant detected",
                              "omicron", "delta variant", "pandemic wave"]),
    (-0.55, 0.72, 48, 3.5, ["trade war", "tariff", "sanctions", "export controls",
                              "chip ban", "tech war"]),
    (-0.60, 0.75, 48, 3.5, ["market crash", "liquidity crisis", "circuit breaker",
                              "forced liquidation"]),
    (-0.65, 0.78, 72, 4.0, ["declares war", "invasion begins", "invades",
                              "nuclear threat", "airstrike", "missile strike",
                              "military conflict"]),
    (-0.70, 0.80, 72, 4.0, ["bank collapse", "bank fails", "bank failure", "bank run",
                              "fdic seizes", "banking crisis", "financial contagion"]),
    (-0.75, 0.82, 72, 4.0, ["exchange collapses", "exchange bankrupt",
                              "exchange halts withdrawal",
                              "usdt depeg", "usdc depeg", "stablecoin depeg",
                              "tether insolvency"]),
    (-0.80, 0.85, 72, 4.5, ["cryptocurrency banned", "bitcoin banned", "bans crypto",
                              "crypto outlawed", "country bans bitcoin", "ban cryptocurrency",
                              "re bans crypto", "again bans crypto", "bans crypto again"]),
    (-0.85, 0.88, 168, 5.0, ["ftx collapse", "ftx bankrupt", "ftx fraud", "sam bankman",
                               "terra luna collapse", "luna collapse", "ust depeg",
                               "celsius bankrupt", "credit suisse"]),
    (-0.90, 0.92, 168, 5.0, ["bitcoin etf rejected", "sec rejects bitcoin etf"]),
    (-0.95, 0.95, 168, 5.0, ["global financial crisis", "systemic threat",
                               "hyperinflation", "default imminent", "global crisis"]),
]

# ---------------------------------------------------------------------------
# LLM prompt templates
# ---------------------------------------------------------------------------
FINGPT_SCORE_SYSTEM = (
    "You are an expert cryptocurrency quant researcher. Analyze the following news headline "
    "STRICTLY evaluating its price direction, liquidity expansion/contraction, and risk-appetite impact "
    "specifically on BITCOIN (BTC). Do not score for general stock or equity sentiment.\n\n"
    "You must generate text matching this structured output framework. First write an analytical reasoning "
    "sentence, followed by the exact structural pipe line data configuration prefix:\n"
    "Reasoning: <Provide 1-sentence analytical justification for directional pricing impact on Bitcoin>\n"
    "DATA: sentiment|score|confidence|impact_hours|tier_weight\n\n"
    "Rules for values:\n"
    "- sentiment: positive, negative, neutral\n"
    "- score: Continuous value from -1.0 (extreme directional selling pressure) to 1.0 (extreme buying pressure)\n"
    "- confidence: 0.0 to 1.0 tracking certainty\n"
    "- impact_hours: Select exactly from 2, 6, 24, 72, 168\n"
    "- tier_weight: 1.0 to 5.0 scale (5.0 represents historical systemic regime changes)\n\n"
    "IMPORTANT DIRECTIONAL RULES:\n"
    "- Regulatory bans, prohibitions, and watchdog enforcement actions are ALWAYS negative for Bitcoin\n"
    "- Arrests, indictments, and fraud charges involving crypto figures are ALWAYS negative\n"
    "- Exchange hacks, fund thefts, and security exploits are ALWAYS negative\n"
    "- ETF approvals, legal tender adoption, and sovereign reserves are ALWAYS positive\n"
    "- Rate cuts and dovish Fed policy are ALWAYS positive for Bitcoin\n"
    "- Rate hikes and hawkish policy are ALWAYS negative for Bitcoin\n"
    "- Pandemic shocks, new COVID variants, and global lockdowns are ALWAYS negative for Bitcoin"
)

LLAMA3_FEW_SHOTS_USER = (
    "Here are some examples of correct scoring:\n\n"
    "Headline: SEC approves first spot Bitcoin ETF opening institutional access\n"
    "Reasoning: Spot ETF approval permanently validates the asset class and builds immediate regulated capital pipeline access for global institutional wealth management.\n"
    "DATA: positive|0.98|0.99|168|5.0\n\n"
    "Headline: FTX exchange files for bankruptcy Sam Bankman-Fried resigns\n"
    "Reasoning: Catastrophic collapse of a major global trading entity triggers systemic contagion, margin liquidations, and long-term deficit of counterparty trust.\n"
    "DATA: negative|-0.95|0.98|168|5.0\n\n"
    "Headline: Federal Reserve raises interest rates by 75 basis points\n"
    "Reasoning: Aggressive rate hikes restrict dollar liquidity, increase bond yields, and suppress capital allocation to speculative risk assets like Bitcoin.\n"
    "DATA: negative|-0.75|0.92|24|4.0\n\n"
    "Headline: Federal Reserve cuts interest rates by 50 basis points as inflation falls\n"
    "Reasoning: Rate cuts trigger expansionary liquidity conditions and decrease capital yields, shifting institutional risk-on appetite into finite assets like Bitcoin.\n"
    "DATA: positive|0.80|0.90|72|4.5\n\n"
    "Now score this headline:\n"
    "Headline: {headline}"
)
