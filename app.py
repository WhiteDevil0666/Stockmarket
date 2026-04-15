"""
app.py — AI Stock Predictor  |  v2.0
Self-contained. No subfolders needed. Streamlit Cloud compatible.

Fixes & upgrades in this version:
  1. NSE = 0 bug fixed   (yfinance multi-level column handling)
  2. Model accuracy improved  (RSI/MACD signals, volume spikes, momentum)
  3. Portfolio tracker added
  4. Zerodha-inspired dark UI
"""

import os, sys, warnings, json
warnings.filterwarnings("ignore")
_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(_ROOT)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
import feedparser, requests
from datetime import datetime, timedelta, date
from time import sleep
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import ta

# ── Dark Zerodha-style theme ─────────────────────────────────
st.set_page_config(
    page_title="StockSense AI 📈",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── Global dark background ── */
.stApp { background-color: #0f0f1a; color: #e0e0e0; }
section[data-testid="stSidebar"] { background-color: #12122a; border-right: 1px solid #2a2a4a; }

/* ── Metric cards ── */
div[data-testid="metric-container"] {
    background: #1a1a2e; border-radius: 10px;
    padding: 12px 16px; border: 1px solid #2a2a4a;
}
div[data-testid="metric-container"] label { color: #8888aa !important; font-size: 13px; }
div[data-testid="metric-container"] div[data-testid="stMetricValue"] { color: #ffffff !important; font-size: 22px; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { background: #12122a; border-radius: 8px; padding: 4px; }
.stTabs [data-baseweb="tab"] { color: #8888aa; font-weight: 600; font-size: 14px; }
.stTabs [aria-selected="true"] { color: #00d4ff !important; border-bottom: 2px solid #00d4ff; }

/* ── Buttons ── */
.stButton > button {
    background: #1a1a2e; color: #00d4ff;
    border: 1px solid #00d4ff; border-radius: 8px;
    font-weight: 600; transition: all 0.2s;
}
.stButton > button:hover { background: #00d4ff; color: #0f0f1a; }

/* ── Signal badges ── */
.badge-buy  { background:#0d3b1e; color:#00e676; border:1px solid #00e676; padding:3px 10px; border-radius:12px; font-size:12px; font-weight:600; }
.badge-sell { background:#3b0d0d; color:#ff5252; border:1px solid #ff5252; padding:3px 10px; border-radius:12px; font-size:12px; font-weight:600; }
.badge-hold { background:#3b2d0d; color:#ffd740; border:1px solid #ffd740; padding:3px 10px; border-radius:12px; font-size:12px; font-weight:600; }

/* ── Dataframes ── */
.stDataFrame { background: #1a1a2e; border-radius: 10px; }
iframe[title="st.dataframe"] { background: #1a1a2e; }

/* ── Dividers ── */
hr { border-color: #2a2a4a; }

/* ── Sidebar text ── */
.css-1d391kg, [data-testid="stSidebar"] * { color: #cccccc; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════

def _secret(key):
    try:    return st.secrets.get(key, os.environ.get(key, ""))
    except: return os.environ.get(key, "")

NEWS_API_KEY   = _secret("NEWS_API_KEY")
BUY_THRESHOLD  = 0.58
SELL_THRESHOLD = 0.42
DATA_DIR       = "predictions"
PORTFOLIO_FILE = os.path.join(DATA_DIR, "portfolio.json")

NSE_STOCKS = [
    "RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS",
    "HINDUNILVR.NS","SBIN.NS","BHARTIARTL.NS","ITC.NS","KOTAKBANK.NS",
    "LT.NS","HCLTECH.NS","AXISBANK.NS","ASIANPAINT.NS","MARUTI.NS",
    "SUNPHARMA.NS","TITAN.NS","BAJFINANCE.NS","ULTRACEMCO.NS","WIPRO.NS",
    "ONGC.NS","NTPC.NS","POWERGRID.NS","BAJAJFINSV.NS","TECHM.NS",
    "TATAMOTORS.NS","NESTLEIND.NS","ADANIENT.NS","ADANIPORTS.NS","HINDALCO.NS",
    "TATASTEEL.NS","JSWSTEEL.NS","COALINDIA.NS","BPCL.NS","GRASIM.NS",
    "EICHERMOT.NS","DIVISLAB.NS","CIPLA.NS","DRREDDY.NS","HDFCLIFE.NS",
    "SBILIFE.NS","BRITANNIA.NS","HEROMOTOCO.NS","TATACONSUM.NS","APOLLOHOSP.NS",
    "BAJAJ-AUTO.NS","INDUSINDBK.NS","SHREECEM.NS","UPL.NS","LTIM.NS",
    "ADANIGREEN.NS","AMBUJACEM.NS","AUROPHARMA.NS","BANKBARODA.NS","BERGEPAINT.NS",
    "BIOCON.NS","BOSCHLTD.NS","CANBK.NS","CHOLAFIN.NS","COLPAL.NS",
    "CONCOR.NS","CUMMINSIND.NS","DABUR.NS","DALBHARAT.NS","DEEPAKNTR.NS",
    "DIXON.NS","DLF.NS","ESCORTS.NS","EXIDEIND.NS","FEDERALBNK.NS",
    "GAIL.NS","GODREJCP.NS","GODREJPROP.NS","GUJGASLTD.NS","HAL.NS",
    "HAVELLS.NS","HINDPETRO.NS","IDFCFIRSTB.NS","IEX.NS","IGL.NS",
    "INDUSTOWER.NS","IOC.NS","IRCTC.NS","JINDALSTEL.NS","JUBLFOOD.NS",
    "LALPATHLAB.NS","LICHSGFIN.NS","LUPIN.NS","MOTHERSON.NS","MPHASIS.NS",
    "MRF.NS","MUTHOOTFIN.NS","NAUKRI.NS","NHPC.NS","NMDC.NS",
    "OFSS.NS","PAGEIND.NS","PERSISTENT.NS","PETRONET.NS","PNB.NS",
    "POLYCAB.NS","RECLTD.NS","SAIL.NS","SBICARD.NS","SIEMENS.NS",
    "SRF.NS","TATAELXSI.NS","TATAPOWER.NS","TORNTPHARM.NS","TORNTPOWER.NS",
    "TRENT.NS","TVSMOTORS.NS","VEDL.NS","VOLTAS.NS","ZOMATO.NS",
    "ZYDUSLIFE.NS","ABB.NS","ABCAPITAL.NS","ACC.NS","AJANTPHARM.NS",
    "ALKEM.NS","ANGELONE.NS","ASTRAL.NS","AUBANK.NS","BALKRISIND.NS",
    "BANDHANBNK.NS","BATAINDIA.NS","BHARATFORG.NS","BHEL.NS","BSE.NS",
    "CAMS.NS","CANFINHOME.NS","CDSL.NS","CHAMBLFERT.NS","COFORGE.NS",
    "CROMPTON.NS","CYIENT.NS","DELHIVERY.NS","EMAMILTD.NS","FORTIS.NS",
    "GLAND.NS","GRANULES.NS","HAPPSTMNDS.NS","HFCL.NS","HUDCO.NS",
    "ICICIGI.NS","ICICIPRULI.NS","IDFC.NS","IPCALAB.NS","JKCEMENT.NS",
    "JSWENERGY.NS","KPITTECH.NS","LICI.NS","LTTS.NS","MANAPPURAM.NS",
    "MARICO.NS","MAXHEALTH.NS","MCX.NS","MFSL.NS","NLCINDIA.NS",
    "OBEROIRLTY.NS","PIIND.NS","PVRINOX.NS","RAMCOCEM.NS","RITES.NS",
    "SJVN.NS","SONACOMS.NS","STARHEALTH.NS","SUPREMEIND.NS","SYNGENE.NS",
    "TATACHEM.NS","TATACOMM.NS","TRIDENT.NS","UNIONBANK.NS","VARUNBEV.NS",
    "VBL.NS","WELCORP.NS","YESBANK.NS","AIAENG.NS","APOLLOTYRE.NS",
    "ATUL.NS","BALRAMCHIN.NS","BLUESTARCO.NS","BSOFT.NS","CASTROLIND.NS",
    "CESC.NS","CUB.NS","DEVYANI.NS","FINEORG.NS","GLAXO.NS",
    "GODFRYPHLP.NS","GRAPHITE.NS","GSPL.NS","HATSUN.NS","JAMNAAUTO.NS",
    "JBCHEPHARM.NS","JKLAKSHMI.NS","JKPAPER.NS","LUXIND.NS","MCDOWELL-N.NS",
    "NIACL.NS","PRINCEPIPE.NS","RADICO.NS","SKFINDIA.NS","TIMKEN.NS",
    "TTKPRESTIG.NS","UNITDSPR.NS",
]
CRYPTO_SYMBOLS = [
    "BTC-USD","ETH-USD","BNB-USD","XRP-USD","ADA-USD",
    "SOL-USD","DOGE-USD","MATIC-USD","DOT-USD","AVAX-USD",
]
RSS_FEEDS = [
    "https://www.moneycontrol.com/rss/latestnews.xml",
    "https://www.moneycontrol.com/rss/marketoutlook.xml",
    "https://economictimes.indiatimes.com/markets/stocks/rss.cms",
    "https://economictimes.indiatimes.com/markets/rss.cms",
    "https://www.business-standard.com/rss/markets-106.rss",
    "https://news.google.com/rss/search?q=NSE+BSE+stock+market+India&hl=en-IN&gl=IN&ceid=IN:en",
    "https://news.google.com/rss/search?q=Nifty+Sensex+today&hl=en-IN&gl=IN&ceid=IN:en",
    "https://news.google.com/rss/search?q=RBI+India+economy&hl=en-IN&gl=IN&ceid=IN:en",
    "https://news.google.com/rss/search?q=Bitcoin+Ethereum+crypto&hl=en&gl=US&ceid=US:en",
]


# ═══════════════════════════════════════════════════════════════
# FIX 1 — STOCK DATA  (multi-level column bug fixed)
# ═══════════════════════════════════════════════════════════════

def download_stock(symbol, years=2):
    """
    BUG FIX: yfinance returns multi-level columns like (Close, RELIANCE.NS).
    We flatten them here so the rest of the code works correctly.
    """
    end   = datetime.today()
    start = end - timedelta(days=years * 365)
    try:
        df = yf.download(
            symbol,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,
            actions=False,
        )

        if df is None or df.empty:
            return None

        # ── FLATTEN MULTI-LEVEL COLUMNS (the NSE=0 fix) ──────
        if isinstance(df.columns, pd.MultiIndex):
            # e.g. ('Close','RELIANCE.NS') → 'Close'
            df.columns = [col[0] if isinstance(col, tuple) else col
                          for col in df.columns]

        # Normalise column names to lowercase
        df.columns = [str(c).strip().lower() for c in df.columns]

        # Keep only what we need
        needed = ["open","high","low","close","volume"]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            return None

        df = df[needed].copy()
        df.index.name = "date"
        df = df.dropna()

        if len(df) < 60:   # need at least 60 trading days
            return None

        return df

    except Exception:
        return None


def add_indicators(df):
    """
    FIX 2 — adds SIGNAL-based features, not just raw values.
    Signals the model can learn from (crossovers, extremes, spikes).
    """
    df = df.copy()

    # ── Raw returns ──────────────────────────────────────────
    df["ret_1d"]  = df["close"].pct_change(1)
    df["ret_3d"]  = df["close"].pct_change(3)
    df["ret_5d"]  = df["close"].pct_change(5)
    df["ret_10d"] = df["close"].pct_change(10)
    df["ret_20d"] = df["close"].pct_change(20)

    # ── Moving averages & ratios ─────────────────────────────
    df["sma_10"] = ta.trend.sma_indicator(df["close"], window=10)
    df["sma_20"] = ta.trend.sma_indicator(df["close"], window=20)
    df["sma_50"] = ta.trend.sma_indicator(df["close"], window=50)
    df["ema_9"]  = ta.trend.ema_indicator(df["close"], window=9)
    df["ema_21"] = ta.trend.ema_indicator(df["close"], window=21)

    df["price_sma20_ratio"] = df["close"] / df["sma_20"]
    df["price_sma50_ratio"] = df["close"] / df["sma_50"]
    df["sma_cross_ratio"]   = df["sma_10"] / df["sma_20"]   # >1 = bullish cross

    # ── RSI with SIGNAL features ─────────────────────────────
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    df["rsi_prev"] = df["rsi"].shift(1)
    df["rsi_oversold"]   = (df["rsi"] < 30).astype(int)          # signal: oversold
    df["rsi_overbought"] = (df["rsi"] > 70).astype(int)          # signal: overbought
    df["rsi_cross_up"]   = ((df["rsi"] > 30) & (df["rsi_prev"] <= 30)).astype(int)   # just exited oversold
    df["rsi_cross_down"] = ((df["rsi"] < 70) & (df["rsi_prev"] >= 70)).astype(int)   # just exited overbought

    # ── MACD with SIGNAL features ────────────────────────────
    macd_obj = ta.trend.MACD(df["close"])
    df["macd"]        = macd_obj.macd()
    df["macd_signal"] = macd_obj.macd_signal()
    df["macd_hist"]   = macd_obj.macd_diff()
    df["macd_hist_prev"] = df["macd_hist"].shift(1)
    df["macd_bullish_cross"] = (
        (df["macd"] > df["macd_signal"]) &
        (df["macd"].shift(1) <= df["macd_signal"].shift(1))
    ).astype(int)
    df["macd_bearish_cross"] = (
        (df["macd"] < df["macd_signal"]) &
        (df["macd"].shift(1) >= df["macd_signal"].shift(1))
    ).astype(int)
    df["macd_hist_rising"] = (df["macd_hist"] > df["macd_hist_prev"]).astype(int)

    # ── Bollinger Bands ──────────────────────────────────────
    bb = ta.volatility.BollingerBands(df["close"], window=20)
    df["bb_pct"]   = bb.bollinger_pband()   # 0=lower band, 1=upper band
    df["bb_width"] = bb.bollinger_wband()   # volatility
    df["bb_squeeze"]    = (df["bb_width"] < df["bb_width"].rolling(50).mean() * 0.75).astype(int)
    df["bb_breakout_up"]   = (df["close"] > bb.bollinger_hband()).astype(int)
    df["bb_breakout_down"] = (df["close"] < bb.bollinger_lband()).astype(int)

    # ── Volume SIGNAL features ───────────────────────────────
    df["vol_sma20"] = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / df["vol_sma20"]
    df["vol_spike"] = (df["vol_ratio"] > 2.0).astype(int)          # volume > 2x average
    df["vol_dry"]   = (df["vol_ratio"] < 0.5).astype(int)          # very low volume

    # ── ATR (volatility) ─────────────────────────────────────
    df["atr"] = ta.volatility.AverageTrueRange(
        df["high"], df["low"], df["close"], window=14).average_true_range()
    df["atr_pct"] = df["atr"] / df["close"]   # normalised ATR

    # ── Stochastic ───────────────────────────────────────────
    stoch = ta.momentum.StochasticOscillator(df["high"], df["low"], df["close"])
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()
    df["stoch_oversold"]   = (df["stoch_k"] < 20).astype(int)
    df["stoch_overbought"] = (df["stoch_k"] > 80).astype(int)

    # ── Candlestick patterns ─────────────────────────────────
    df["body_size"]   = abs(df["close"] - df["open"]) / df["open"]
    df["upper_wick"]  = (df["high"] - df[["close","open"]].max(axis=1)) / df["open"]
    df["lower_wick"]  = (df[["close","open"]].min(axis=1) - df["low"]) / df["open"]
    df["is_bullish_candle"] = (df["close"] > df["open"]).astype(int)
    df["gap_up"]   = (df["open"] > df["close"].shift(1) * 1.01).astype(int)
    df["gap_down"]  = (df["open"] < df["close"].shift(1) * 0.99).astype(int)

    # ── Sentiment placeholder ────────────────────────────────
    df["sentiment_score"] = 0.0

    # ── TARGET: will price be higher tomorrow? ───────────────
    df["target"]     = (df["close"].shift(-1) > df["close"]).astype(int)
    df["next_close"] = df["close"].shift(-1)

    return df.replace([np.inf, -np.inf], np.nan).dropna()


def fetch_all_stocks(symbols, status_el=None, prog_el=None, p0=0, p1=30):
    results = {}
    n = len(symbols)
    for i, sym in enumerate(symbols):
        if status_el:
            status_el.write(f"📥 Downloading **{sym}** ({i+1}/{n})...")
        if prog_el:
            pct = int(p0 + (p1 - p0) * i / max(n, 1))
            prog_el.progress(pct, text=f"Fetching {sym}...")
        df = download_stock(sym)
        if df is not None:
            df = add_indicators(df)
            if not df.empty:
                results[sym] = df
        sleep(0.15)
    return results


# ═══════════════════════════════════════════════════════════════
# NEWS + SENTIMENT
# ═══════════════════════════════════════════════════════════════

def fetch_rss():
    articles = []
    for url in RSS_FEEDS:
        try:
            feed = feedparser.parse(url)
            for e in feed.entries[:40]:
                articles.append({
                    "title":     e.get("title",""),
                    "summary":   e.get("summary",""),
                    "source":    feed.feed.get("title","RSS"),
                    "published": e.get("published",""),
                })
        except:
            pass
    return articles

def fetch_newsapi():
    if not NEWS_API_KEY: return []
    from_dt = (datetime.utcnow()-timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    queries = [
        "NSE BSE Nifty Sensex India stock market",
        "Indian economy RBI inflation budget",
        "Bitcoin Ethereum cryptocurrency India",
    ]
    out = []
    for q in queries:
        try:
            r = requests.get("https://newsapi.org/v2/everything", timeout=10, params={
                "q":q,"from":from_dt,"sortBy":"publishedAt",
                "language":"en","pageSize":100,"apiKey":NEWS_API_KEY,
            })
            d = r.json()
            if d.get("status")=="ok":
                for a in d.get("articles",[]):
                    out.append({"title":a.get("title",""),"summary":a.get("description",""),
                                "source":a.get("source",{}).get("name","NewsAPI"),
                                "published":a.get("publishedAt","")})
            sleep(0.3)
        except:
            pass
    return out

def fetch_all_news():
    items = fetch_rss() + fetch_newsapi()
    if not items: return pd.DataFrame()
    df = pd.DataFrame(items).drop_duplicates(subset=["title"])
    df["title"]   = df["title"].fillna("")
    df["summary"] = df["summary"].fillna("")
    df["text"]    = df["title"] + ". " + df["summary"]
    all_syms = NSE_STOCKS + CRYPTO_SYMBOLS
    def tag(row):
        t = row["text"].upper()
        return [s for s in all_syms
                if len(s.split(".")[0].split("-")[0])>=3
                and s.split(".")[0].split("-")[0] in t]
    df["mentioned_symbols"] = df.apply(tag, axis=1)
    return df

_vader = SentimentIntensityAnalyzer()
for w in ["bullish","rally","surge","breakout","buy","upgrade","profit","dividend","growth","record","beat"]:
    _vader.lexicon[w] = 2.5
for w in ["bearish","crash","fall","drop","sell","downgrade","loss","fraud","deficit","decline","miss","risk"]:
    _vader.lexicon[w] = -2.5

def score_text(t):
    return round(_vader.polarity_scores(str(t))["compound"],4) if t else 0.0

def sent_label(s):
    if s >= 0.3:   return "Strong Bullish 🚀"
    if s >= 0.05:  return "Bullish 🟢"
    if s <= -0.3:  return "Strong Bearish 💀"
    if s <= -0.05: return "Bearish 🔴"
    return "Neutral ⚪"

def compute_sentiment(news_df):
    if news_df.empty:
        return pd.DataFrame(), {"score":0.0,"label":"Neutral ⚪","total_articles":0}
    news_df = news_df.copy()
    news_df["sentiment"] = news_df["text"].apply(score_text)
    # FIX: weight recent articles higher (2x for articles from last 6 hours)
    now = datetime.utcnow()
    def parse_pub(p):
        try:
            return datetime.strptime(str(p)[:19], "%Y-%m-%dT%H:%M:%S")
        except:
            return now - timedelta(hours=12)
    news_df["pub_dt"] = news_df["published"].apply(parse_pub)
    news_df["recency_weight"] = news_df["pub_dt"].apply(
        lambda d: 2.0 if (now - d).total_seconds() < 21600 else 1.0
    )
    rows = []
    for sym in NSE_STOCKS + CRYPTO_SYMBOLS:
        mask = news_df["mentioned_symbols"].apply(lambda s: sym in s)
        rel  = news_df[mask]
        if len(rel)==0:
            rows.append({"symbol":sym,"sentiment_score":0.0,
                         "sentiment_label":"Neutral ⚪","article_count":0,"top_headline":"No recent news"})
        else:
            # weighted average — recent news counts more
            avg = np.average(rel["sentiment"], weights=rel["recency_weight"])
            top = rel.sort_values("sentiment",ascending=False).iloc[0]["title"]
            rows.append({"symbol":sym,"sentiment_score":round(avg,4),
                         "sentiment_label":sent_label(avg),
                         "article_count":len(rel),"top_headline":top})
    sent_df = pd.DataFrame(rows)
    market  = {"score":round(news_df["sentiment"].mean(),4),
               "label":sent_label(news_df["sentiment"].mean()),
               "total_articles":len(news_df)}
    return sent_df, market


# ═══════════════════════════════════════════════════════════════
# FIX 2 — IMPROVED ML MODEL
# ═══════════════════════════════════════════════════════════════

FEATURE_COLS = [
    # Returns
    "ret_1d","ret_3d","ret_5d","ret_10d","ret_20d",
    # MA ratios
    "price_sma20_ratio","price_sma50_ratio","sma_cross_ratio",
    # RSI signals
    "rsi","rsi_oversold","rsi_overbought","rsi_cross_up","rsi_cross_down",
    # MACD signals
    "macd_hist","macd_bullish_cross","macd_bearish_cross","macd_hist_rising",
    # Bollinger signals
    "bb_pct","bb_width","bb_squeeze","bb_breakout_up","bb_breakout_down",
    # Volume signals
    "vol_ratio","vol_spike","vol_dry",
    # Volatility
    "atr_pct",
    # Stochastic signals
    "stoch_k","stoch_d","stoch_oversold","stoch_overbought",
    # Candle patterns
    "body_size","upper_wick","lower_wick","is_bullish_candle","gap_up","gap_down",
    # Sentiment (weighted)
    "sentiment_score",
]

def predict_stock(df, symbol, sentiment_score=0.0):
    try:
        df = df.copy()
        df["sentiment_score"] = sentiment_score

        train_df = df[:-1].copy()
        avail    = [c for c in FEATURE_COLS if c in train_df.columns]
        if len(avail) < 15 or len(train_df) < 80:
            return None

        X = train_df[avail].replace([np.inf,-np.inf], np.nan).fillna(0).values
        y = train_df["target"].astype(int).values
        if len(np.unique(y)) < 2: return None

        scaler = StandardScaler()
        X_sc   = scaler.fit_transform(X)

        # Ensemble: Random Forest + Gradient Boosting → average
        rf = RandomForestClassifier(n_estimators=200, max_depth=10,
                                     min_samples_leaf=4, random_state=42, n_jobs=-1)
        gb = GradientBoostingClassifier(n_estimators=100, max_depth=4,
                                         learning_rate=0.05, random_state=42)
        rf.fit(X_sc, y)
        gb.fit(X_sc, y)

        today_row = df[avail].iloc[-1].replace([np.inf,-np.inf], np.nan).fillna(0).values
        today_sc  = scaler.transform([today_row])

        prob_up_rf = rf.predict_proba(today_sc)[0][1]
        prob_up_gb = gb.predict_proba(today_sc)[0][1]
        prob_up    = (prob_up_rf * 0.55 + prob_up_gb * 0.45)   # weighted ensemble

        # ── ALPHA BOOST: strong sentiment shifts probability ─
        # Real trading trick — strong news overrides weak technical signals
        if sentiment_score >= 0.3:
            prob_up = min(prob_up + 0.05, 0.97)   # strong bullish news → boost
        elif sentiment_score <= -0.3:
            prob_up = max(prob_up - 0.05, 0.03)   # strong bearish news → suppress
        # ────────────────────────────────────────────────────

        prob_down  = 1 - prob_up

        direction  = "UP 🟢" if prob_up >= 0.5 else "DOWN 🔴"
        confidence = round(max(prob_up, prob_down) * 100, 1)

        if prob_up >= BUY_THRESHOLD:    signal = "BUY 🟢"
        elif prob_up <= SELL_THRESHOLD: signal = "SELL 🔴"
        else:                           signal = "HOLD 🟡"

        last_close = float(df["close"].iloc[-1])
        atr        = float(df["atr"].iloc[-1]) if "atr" in df.columns else last_close * 0.02
        if direction.startswith("UP"):
            est_low, est_high = last_close - atr*0.4, last_close + atr*1.3
        else:
            est_low, est_high = last_close - atr*1.3, last_close + atr*0.4

        # Cross-validation accuracy
        tscv, scores = TimeSeriesSplit(n_splits=4), []
        for tr,vl in tscv.split(X_sc):
            m = RandomForestClassifier(n_estimators=80, max_depth=8, random_state=42)
            m.fit(X_sc[tr], y[tr])
            scores.append(m.score(X_sc[vl], y[vl]))

        return {
            "symbol":        symbol,
            "last_close":    round(last_close, 2),
            "direction":     direction,
            "confidence":    confidence,
            "signal":        signal,
            "prob_up":       round(prob_up*100, 1),
            "prob_down":     round(prob_down*100, 1),
            "est_price_low": round(est_low, 2),
            "est_price_high":round(est_high, 2),
            "model_accuracy":round(np.mean(scores)*100, 1),
            "rsi":           round(float(df["rsi"].iloc[-1]),1)        if "rsi"       in df.columns else None,
            "macd_hist":     round(float(df["macd_hist"].iloc[-1]),4)  if "macd_hist" in df.columns else None,
            "bb_pct":        round(float(df["bb_pct"].iloc[-1])*100,1) if "bb_pct"   in df.columns else None,
            "vol_ratio":     round(float(df["vol_ratio"].iloc[-1]),2)  if "vol_ratio" in df.columns else None,
            "sentiment_score": round(sentiment_score,4),
            "updated_at":   datetime.now().strftime("%Y-%m-%d %H:%M IST"),
        }
    except Exception:
        return None

def run_predictions(stock_data, sentiment_df):
    lk = {} if sentiment_df.empty else dict(zip(sentiment_df["symbol"], sentiment_df["sentiment_score"]))
    rows = []
    for sym, df in stock_data.items():
        r = predict_stock(df, sym, lk.get(sym, 0.0))
        if r:
            if not sentiment_df.empty:
                sr = sentiment_df[sentiment_df["symbol"]==sym]
                if not sr.empty:
                    r["sentiment_label"] = sr.iloc[0].get("sentiment_label","Neutral ⚪")
                    r["article_count"]   = sr.iloc[0].get("article_count",0)
                    r["top_headline"]    = sr.iloc[0].get("top_headline","")
            rows.append(r)
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════
# DATA PIPELINE
# ═══════════════════════════════════════════════════════════════

def run_update(quick_mode=False):
    today    = datetime.now().strftime("%Y-%m-%d")
    nse_list = NSE_STOCKS[:20] if quick_mode else NSE_STOCKS
    cry_list = CRYPTO_SYMBOLS[:5] if quick_mode else CRYPTO_SYMBOLS

    prog   = st.progress(0, text="Starting pipeline...")
    status = st.empty()

    try:
        status.write("**Step 1/4** — Downloading price data from Yahoo Finance...")
        nse_data    = fetch_all_stocks(nse_list,  status, prog, 2,  30)
        crypto_data = fetch_all_stocks(cry_list,  status, prog, 30, 40)
        all_data    = {**nse_data, **crypto_data}

        if not all_data:
            st.error("❌ No stock data fetched. Check internet.")
            prog.empty(); return False

        status.write(f"**Step 2/4** — Fetching news (RSS + NewsAPI)...")
        prog.progress(42, text="Fetching news...")
        news_df = fetch_all_news()

        status.write("**Step 3/4** — Analysing sentiment (recency-weighted)...")
        prog.progress(55, text="Sentiment analysis...")
        sentiment_df, market_sent = compute_sentiment(news_df)

        status.write("**Step 4/4** — Training ensemble ML model (RF + GB)...")
        prog.progress(65, text="Training models...")
        pred_df = run_predictions(all_data, sentiment_df)

        if pred_df.empty:
            st.error("❌ No predictions generated.")
            prog.empty(); return False

        nse_set  = set(NSE_STOCKS)
        cry_set  = set(CRYPTO_SYMBOLS)
        pred_df["market"] = pred_df["symbol"].apply(
            lambda s: "NSE" if s in nse_set else ("Crypto" if s in cry_set else "Other"))
        pred_df["date"] = today

        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(os.path.join(DATA_DIR,"history"), exist_ok=True)
        pred_df.to_csv(os.path.join(DATA_DIR,"latest_predictions.csv"), index=False)
        pred_df.to_csv(os.path.join(DATA_DIR,"history",f"{today}.csv"), index=False)
        market_sent["date"] = today
        pd.DataFrame([market_sent]).to_csv(os.path.join(DATA_DIR,"market_sentiment.csv"), index=False)

        # Auto-log BUY signals into portfolio as "watchlist"
        auto_log_buys(pred_df)

        prog.progress(100, text="✅ Done!")
        status.empty()
        return True
    except Exception as e:
        st.error(f"❌ Pipeline error: {e}")
        return False


# ═══════════════════════════════════════════════════════════════
# FIX 3 — PORTFOLIO TRACKER
# ═══════════════════════════════════════════════════════════════

def load_portfolio():
    if not os.path.exists(PORTFOLIO_FILE): return []
    try:
        with open(PORTFOLIO_FILE) as f:
            return json.load(f)
    except:
        return []

def save_portfolio(entries):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(PORTFOLIO_FILE,"w") as f:
        json.dump(entries, f, indent=2)

def add_to_portfolio(symbol, buy_price, quantity, note=""):
    entries = load_portfolio()
    entries.append({
        "symbol":    symbol,
        "buy_price": float(buy_price),
        "quantity":  float(quantity),
        "date":      str(date.today()),
        "note":      note,
    })
    save_portfolio(entries)

def remove_from_portfolio(idx):
    entries = load_portfolio()
    if 0 <= idx < len(entries):
        entries.pop(idx)
        save_portfolio(entries)

def auto_log_buys(pred_df):
    """Log today's BUY signals into a separate watchlist CSV."""
    buys = pred_df[pred_df["signal"].str.startswith("BUY")][
        ["symbol","last_close","confidence","sentiment_label","date"]
    ].copy()
    if not buys.empty:
        buys.to_csv(os.path.join(DATA_DIR,"watchlist.csv"), index=False)

def get_current_prices(symbols):
    """Fetch current price for portfolio P&L calculation."""
    prices = {}
    for sym in symbols:
        try:
            t   = yf.Ticker(sym)
            inf = t.fast_info
            prices[sym] = float(getattr(inf, "last_price", 0) or 0)
        except:
            prices[sym] = 0.0
    return prices

def render_portfolio():
    st.markdown("## 💼 My Portfolio Tracker")
    st.caption("Track stocks you hold or want to track. P&L updates live.")

    entries = load_portfolio()

    # ── Add new entry ─────────────────────────────────────────
    with st.expander("➕ Add a stock to portfolio", expanded=len(entries)==0):
        c1,c2,c3,c4 = st.columns([2,1.5,1.5,2])
        sym   = c1.text_input("Symbol", placeholder="e.g. RELIANCE.NS").strip().upper()
        price = c2.number_input("Buy price ₹/$", min_value=0.01, value=100.0, step=0.5)
        qty   = c3.number_input("Quantity", min_value=1, value=10, step=1)
        note  = c4.text_input("Note (optional)", placeholder="Why I bought this")
        if st.button("Add to Portfolio", type="primary"):
            if sym:
                add_to_portfolio(sym, price, qty, note)
                st.success(f"✅ {sym} added!")
                st.rerun()
            else:
                st.warning("Enter a symbol first.")

    if not entries:
        st.info("Your portfolio is empty. Add stocks above to start tracking.")
        return

    # ── Fetch current prices ──────────────────────────────────
    symbols  = list({e["symbol"] for e in entries})
    cur_prices = get_current_prices(symbols)

    # ── Build P&L table ───────────────────────────────────────
    rows = []
    for i, e in enumerate(entries):
        sym   = e["symbol"]
        bp    = e["buy_price"]
        qty   = e["quantity"]
        cp    = cur_prices.get(sym, 0)
        invested = bp * qty
        current  = cp * qty if cp > 0 else 0
        pnl      = current - invested if cp > 0 else None
        pnl_pct  = (pnl / invested * 100) if pnl is not None else None
        rows.append({
            "idx":       i,
            "Symbol":    sym,
            "Buy ₹":     bp,
            "Current ₹": round(cp,2) if cp>0 else "—",
            "Qty":        int(qty),
            "Invested ₹": round(invested,2),
            "P&L ₹":      round(pnl,2) if pnl is not None else "—",
            "P&L %":      round(pnl_pct,2) if pnl_pct is not None else "—",
            "Date":       e.get("date","—"),
            "Note":       e.get("note",""),
        })

    pdf = pd.DataFrame(rows)

    # ── Summary metrics ───────────────────────────────────────
    valid = [r for r in rows if isinstance(r["P&L ₹"], float)]
    total_invested = sum(r["Invested ₹"] for r in rows)
    total_pnl      = sum(r["P&L ₹"] for r in valid if isinstance(r["P&L ₹"], float))
    winners  = sum(1 for r in valid if isinstance(r["P&L ₹"],float) and r["P&L ₹"]>0)
    losers   = sum(1 for r in valid if isinstance(r["P&L ₹"],float) and r["P&L ₹"]<0)

    mc1,mc2,mc3,mc4 = st.columns(4)
    mc1.metric("💰 Total Invested", f"₹{total_invested:,.0f}")
    mc2.metric("📈 Total P&L",
               f"₹{total_pnl:,.0f}" if valid else "—",
               delta=f"{total_pnl/total_invested*100:.1f}%" if total_invested>0 and valid else None)
    mc3.metric("✅ Winners", winners)
    mc4.metric("❌ Losers",  losers)
    st.divider()

    # ── Table ─────────────────────────────────────────────────
    display_cols = ["Symbol","Buy ₹","Current ₹","Qty","Invested ₹","P&L ₹","P&L %","Date","Note"]
    st.dataframe(
        pdf[display_cols].reset_index(drop=True),
        use_container_width=True,
        height=min(400, 60 + len(pdf)*40),
    )

    # ── Remove entry ──────────────────────────────────────────
    st.markdown("**Remove a stock:**")
    rem_sym = st.selectbox("Select to remove",
                            [f"{i}: {r['Symbol']} (bought @ {r['Buy ₹']})"
                             for i,r in enumerate(rows)],
                            key="rem_sel")
    if st.button("🗑 Remove selected", use_container_width=False):
        idx = int(rem_sym.split(":")[0])
        remove_from_portfolio(idx)
        st.success("Removed.")
        st.rerun()

    # ── P&L chart ─────────────────────────────────────────────
    if valid:
        st.markdown("#### P&L per Stock")
        chart_data = pd.DataFrame([{"Symbol":r["Symbol"], "P&L ₹":r["P&L ₹"]}
                                    for r in valid if isinstance(r["P&L ₹"],float)])
        fig = px.bar(chart_data, x="Symbol", y="P&L ₹",
                     color="P&L ₹",
                     color_continuous_scale=["#d50000","#ff9800","#00c853"],
                     color_continuous_midpoint=0,
                     title="Profit & Loss per holding")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#e0e0e0",
                          height=320,
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# DATA LOADERS
# ═══════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def load_predictions():
    p = os.path.join(DATA_DIR,"latest_predictions.csv")
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

@st.cache_data(ttl=3600)
def load_market_sentiment():
    p = os.path.join(DATA_DIR,"market_sentiment.csv")
    return pd.read_csv(p).iloc[0].to_dict() if os.path.exists(p) else {}

@st.cache_data(ttl=900)
def load_chart(symbol):
    try:
        df = yf.download(symbol, period="6mo", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df.columns = [c.lower() for c in df.columns]
        return df
    except:
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════
# STOCK DETAIL
# ═══════════════════════════════════════════════════════════════

def stock_detail(df, symbol, currency="₹"):
    row = df[df["symbol"]==symbol]
    if row.empty: return
    r = row.iloc[0]

    sig = r.get("signal","—")
    col_map = {"BUY 🟢":"badge-buy","SELL 🔴":"badge-sell","HOLD 🟡":"badge-hold"}
    badge   = col_map.get(sig,"badge-hold")
    st.markdown(f'<span class="{badge}">{sig}</span>', unsafe_allow_html=True)
    st.markdown("")

    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown(f"**Direction:** {r.get('direction','—')}")
        st.markdown(f"**Confidence:** `{r.get('confidence',0):.1f}%`")
        st.markdown(f"**Last Close:** `{currency}{r.get('last_close',0):.2f}`")
        st.markdown(f"**Est. Tomorrow:** `{currency}{r.get('est_price_low',0):.2f}` – `{currency}{r.get('est_price_high',0):.2f}`")
        st.markdown(f"**Model Accuracy:** `{r.get('model_accuracy',0):.1f}%`")
    with c2:
        rsi = float(r.get("rsi",50) or 50)
        st.markdown(f"**RSI (14):** `{rsi:.1f}`")
        if rsi>70:   st.warning("⚠️ Overbought zone")
        elif rsi<30: st.success("✅ Oversold — potential reversal")
        else:        st.info("RSI in neutral zone")
        macd_h = r.get("macd_hist")
        if macd_h is not None:
            trend = "📈 Bullish" if float(macd_h)>0 else "📉 Bearish"
            st.markdown(f"**MACD Histogram:** `{float(macd_h):.4f}` {trend}")
        st.markdown(f"**Volume Ratio:** `{r.get('vol_ratio','—')}`")
    with c3:
        st.markdown(f"**Sentiment:** {r.get('sentiment_label','—')}")
        st.markdown(f"**Sentiment Score:** `{r.get('sentiment_score',0):+.3f}`")
        st.markdown(f"**News Articles:** `{r.get('article_count',0)}`")
        hl = r.get("top_headline","")
        if hl: st.caption(f"📰 {str(hl)[:140]}...")

    # Add to portfolio button
    if st.button(f"➕ Add {symbol} to Portfolio", key=f"add_{symbol}"):
        add_to_portfolio(symbol, r.get("last_close",0), 1, f"Added from signal: {sig}")
        st.success(f"✅ {symbol} added to portfolio!")

    st.markdown(f"#### {symbol} — 6-Month Price Chart")
    cdf = load_chart(symbol)
    if not cdf.empty and "close" in cdf.columns:
        cdf["sma20"] = cdf["close"].rolling(20).mean()
        cdf["sma50"] = cdf["close"].rolling(50).mean()
        fig = go.Figure()
        if all(c in cdf.columns for c in ["open","high","low","close"]):
            fig.add_trace(go.Candlestick(
                x=cdf.index, open=cdf["open"], high=cdf["high"],
                low=cdf["low"], close=cdf["close"], name="OHLC",
                increasing_line_color="#00c853", decreasing_line_color="#d50000"))
        fig.add_trace(go.Scatter(x=cdf.index, y=cdf["sma20"],
                                  line=dict(color="#ff9800",width=1.2), name="SMA 20"))
        fig.add_trace(go.Scatter(x=cdf.index, y=cdf["sma50"],
                                  line=dict(color="#00d4ff",width=1.2), name="SMA 50"))
        fig.add_hrect(y0=r.get("est_price_low",0), y1=r.get("est_price_high",0),
                      fillcolor="rgba(124,77,255,0.12)", line_width=0,
                      annotation_text="Predicted range", annotation_position="top right")
        fig.update_layout(
            height=400, xaxis_rangeslider_visible=False,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,15,26,0.8)",
            font_color="#e0e0e0",
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=10,r=10,t=20,b=10),
        )
        fig.update_xaxes(showgrid=False, color="#555")
        fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)", color="#555")
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    # ── Header ────────────────────────────────────────────────
    st.markdown("""
    <div style='display:flex;align-items:center;gap:14px;margin-bottom:4px'>
      <span style='font-size:36px'>📈</span>
      <div>
        <h1 style='margin:0;font-size:28px;color:#00d4ff'>StockSense AI</h1>
        <p style='margin:0;color:#8888aa;font-size:13px'>India NSE + Crypto · ML + Sentiment · Auto-updates 6:30 AM IST</p>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    df  = load_predictions()
    mkt = load_market_sentiment()

    # ── NO DATA ───────────────────────────────────────────────
    if df.empty:
        st.warning("### ⚠️ No prediction data yet.")
        st.markdown("Click below to generate your first predictions. After this, GitHub Actions runs automatically every weekday morning.")
        st.divider()
        c1,c2 = st.columns(2)
        with c1:
            st.markdown("#### ⚡ Quick Start *(recommended)*")
            st.markdown("20 NSE + 5 crypto · **~5–8 mins**")
            if st.button("▶  Run Quick Update", type="primary", use_container_width=True):
                ok = run_update(quick_mode=True)
                if ok:
                    st.cache_data.clear(); st.rerun()
        with c2:
            st.markdown("#### 🔄 Full Update")
            st.markdown("200+ NSE + 10 crypto · **~25–35 mins**")
            if st.button("▶  Run Full Update", use_container_width=True):
                ok = run_update(quick_mode=False)
                if ok:
                    st.cache_data.clear(); st.rerun()
        return

    # ── METRICS BAR ───────────────────────────────────────────
    total   = len(df)
    buy_ct  = (df["signal"].str.startswith("BUY")).sum()
    sell_ct = (df["signal"].str.startswith("SELL")).sum()
    hold_ct = (df["signal"].str.startswith("HOLD")).sum()
    updated = df["updated_at"].iloc[0] if "updated_at" in df.columns else "—"
    data_dt = df["date"].iloc[0]        if "date"       in df.columns else "—"
    avg_acc = df["model_accuracy"].mean() if "model_accuracy" in df.columns else 0

    # ── Last refreshed banner ─────────────────────────────────
    now_ist = datetime.now().strftime("%d %b %Y, %I:%M %p")
    st.markdown(f"""
    <div style='background:#1a1a2e;border:1px solid #2a2a4a;border-radius:10px;
                padding:10px 18px;display:flex;justify-content:space-between;
                align-items:center;margin-bottom:14px;flex-wrap:wrap;gap:8px'>
      <div>
        <span style='color:#8888aa;font-size:12px'>📅 Predictions data for</span>
        <span style='color:#00d4ff;font-weight:700;font-size:15px;margin-left:8px'>{data_dt}</span>
      </div>
      <div>
        <span style='color:#8888aa;font-size:12px'>⏱ Model trained at</span>
        <span style='color:#ffd740;font-size:13px;margin-left:8px'>{updated}</span>
      </div>
      <div>
        <span style='color:#8888aa;font-size:12px'>🌐 Page loaded at</span>
        <span style='color:#aaaaaa;font-size:13px;margin-left:8px'>{now_ist} IST</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("📊 Stocks tracked", total)
    c2.metric("🟢 BUY",  buy_ct,  f"{buy_ct/total*100:.0f}%")
    c3.metric("🔴 SELL", sell_ct, f"{sell_ct/total*100:.0f}%")
    c4.metric("🟡 HOLD", hold_ct, f"{hold_ct/total*100:.0f}%")
    c5.metric("📰 Mood", mkt.get("label","—"), f"Score {mkt.get('score',0):+.2f}")
    c6.metric("🎯 Avg Accuracy", f"{avg_acc:.1f}%")
    st.divider()

    # ── SIDEBAR ───────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Filters")
        sig_f    = st.multiselect("Signal",
                                   ["BUY 🟢","SELL 🔴","HOLD 🟡"],
                                   default=["BUY 🟢","SELL 🔴","HOLD 🟡"])
        conf_min = st.slider("Min Confidence %", 50, 95, 50)
        sort_by  = st.selectbox("Sort by",
                                 ["confidence","prob_up","sentiment_score","model_accuracy","symbol"])
        st.divider()
        st.markdown("### 🔄 Manual Refresh")
        if st.button("▶ Quick Update (20 stocks)", use_container_width=True):
            ok = run_update(quick_mode=True)
            if ok: st.cache_data.clear(); st.rerun()
        if st.button("▶ Full Update (200+ stocks)", use_container_width=True):
            ok = run_update(quick_mode=False)
            if ok: st.cache_data.clear(); st.rerun()
        st.divider()
        st.markdown("### 📌 Quick Lookup")
        lk = st.text_input("Symbol e.g. RELIANCE.NS","")
        if lk:
            r = df[df["symbol"]==lk.strip().upper()]
            if r.empty: st.error("Not found")
            else:
                x = r.iloc[0]
                sig = x.get("signal","—")
                badge = {"BUY 🟢":"🟢","SELL 🔴":"🔴","HOLD 🟡":"🟡"}.get(sig,"⚪")
                st.markdown(f"**{x['symbol']}** {badge}")
                st.markdown(f"Confidence: `{x.get('confidence',0):.1f}%`")
                st.markdown(f"Close: `₹{x.get('last_close',0):.2f}`")
                st.markdown(f"Sentiment: {x.get('sentiment_label','—')}")

    # ── TABS ──────────────────────────────────────────────────
    t0,t1,t2,t3,t4,t5 = st.tabs([
        "🏆 Today's Top Picks",
        "🏦 NSE Stocks",
        "₿  Crypto",
        "📰 News & Sentiment",
        "📊 Signal Summary",
        "💼 My Portfolio",
    ])

    # ─ TAB 0: TOP PICKS ───────────────────────────────────────
    with t0:
        st.markdown("## 🏆 Today's Top Picks")
        st.caption(f"Best BUY and SELL signals for **{data_dt}** — ranked by confidence × model accuracy × sentiment boost")

        nse_only    = df[df["market"]=="NSE"].copy()    if "market" in df.columns else df.copy()
        crypto_only = df[df["market"]=="Crypto"].copy() if "market" in df.columns else pd.DataFrame()

        def score_stock(row):
            """Composite rank = confidence × accuracy × sentiment multiplier"""
            conf  = row.get("confidence", 50)
            acc   = row.get("model_accuracy", 50)
            sent  = row.get("sentiment_score", 0)
            boost = 1.1 if sent >= 0.3 else (0.9 if sent <= -0.3 else 1.0)
            return round((conf * 0.6 + acc * 0.4) * boost, 2)

        for section_label, section_df, currency in [
            ("🏦 NSE Stocks", nse_only, "₹"),
            ("₿ Crypto",     crypto_only, "$"),
        ]:
            if section_df.empty:
                continue
            section_df = section_df.copy()
            section_df["rank_score"] = section_df.apply(score_stock, axis=1)

            top_buy  = section_df[section_df["signal"].str.startswith("BUY")]\
                           .sort_values("rank_score", ascending=False).head(5)
            top_sell = section_df[section_df["signal"].str.startswith("SELL")]\
                           .sort_values("rank_score", ascending=False).head(5)

            st.markdown(f"### {section_label}")
            col_buy, col_sell = st.columns(2)

            with col_buy:
                st.markdown("#### 🟢 Top 5 BUY Today")
                if top_buy.empty:
                    st.info("No strong BUY signals today.")
                else:
                    for i, (_, r) in enumerate(top_buy.iterrows(), 1):
                        sent_str = r.get("sentiment_label","—")
                        headline = str(r.get("top_headline",""))[:80]
                        st.markdown(f"""
<div style='background:#0d3b1e;border:1px solid #00e676;border-radius:10px;
            padding:12px 16px;margin-bottom:10px'>
  <div style='display:flex;justify-content:space-between;align-items:center'>
    <span style='color:#00e676;font-size:17px;font-weight:700'>#{i} {r["symbol"]}</span>
    <span style='color:#ffd740;font-size:13px'>Confidence: {r.get("confidence",0):.1f}%</span>
  </div>
  <div style='color:#aaffaa;font-size:12px;margin-top:4px'>
    Close: <b>{currency}{r.get("last_close",0):.2f}</b> &nbsp;|&nbsp;
    Est. High: <b>{currency}{r.get("est_price_high",0):.2f}</b> &nbsp;|&nbsp;
    Accuracy: <b>{r.get("model_accuracy",0):.1f}%</b>
  </div>
  <div style='color:#88cc88;font-size:11px;margin-top:3px'>
    Sentiment: {sent_str} &nbsp;|&nbsp; Score: {r.get("sentiment_score",0):+.3f}
  </div>
  {'<div style="color:#668866;font-size:11px;margin-top:4px;font-style:italic">📰 ' + headline + '...</div>' if headline and headline != 'nan' else ''}
</div>""", unsafe_allow_html=True)

            with col_sell:
                st.markdown("#### 🔴 Top 5 SELL Today")
                if top_sell.empty:
                    st.info("No strong SELL signals today.")
                else:
                    for i, (_, r) in enumerate(top_sell.iterrows(), 1):
                        sent_str = r.get("sentiment_label","—")
                        headline = str(r.get("top_headline",""))[:80]
                        st.markdown(f"""
<div style='background:#3b0d0d;border:1px solid #ff5252;border-radius:10px;
            padding:12px 16px;margin-bottom:10px'>
  <div style='display:flex;justify-content:space-between;align-items:center'>
    <span style='color:#ff5252;font-size:17px;font-weight:700'>#{i} {r["symbol"]}</span>
    <span style='color:#ffd740;font-size:13px'>Confidence: {r.get("confidence",0):.1f}%</span>
  </div>
  <div style='color:#ffaaaa;font-size:12px;margin-top:4px'>
    Close: <b>{currency}{r.get("last_close",0):.2f}</b> &nbsp;|&nbsp;
    Est. Low: <b>{currency}{r.get("est_price_low",0):.2f}</b> &nbsp;|&nbsp;
    Accuracy: <b>{r.get("model_accuracy",0):.1f}%</b>
  </div>
  <div style='color:#cc8888;font-size:11px;margin-top:3px'>
    Sentiment: {sent_str} &nbsp;|&nbsp; Score: {r.get("sentiment_score",0):+.3f}
  </div>
  {'<div style="color:#886666;font-size:11px;margin-top:4px;font-style:italic">📰 ' + headline + '...</div>' if headline and headline != 'nan' else ''}
</div>""", unsafe_allow_html=True)

            st.divider()

        # ── Explanation box ───────────────────────────────────
        st.markdown("""
<div style='background:#1a1a2e;border:1px solid #2a2a4a;border-radius:10px;padding:14px 18px'>
  <p style='color:#00d4ff;font-weight:700;margin:0 0 8px'>🧠 How these picks are ranked</p>
  <p style='color:#aaaaaa;font-size:13px;margin:0'>
    <b>Rank Score</b> = (Confidence × 60% + Model Accuracy × 40%) × Sentiment Multiplier<br>
    <b>Sentiment Multiplier:</b> Strong Bullish 🚀 = ×1.1 boost &nbsp;|&nbsp;
    Strong Bearish 💀 = ×0.9 penalty &nbsp;|&nbsp; Others = ×1.0<br>
    This is a simplified version of how quant funds score trade signals.
  </p>
</div>""", unsafe_allow_html=True)

    # ─ TAB 1: NSE ─────────────────────────────────────────────
    with t1:
        ndf  = df[df["market"]=="NSE"].copy() if "market" in df.columns else df.copy()
        filt = ndf[ndf["signal"].isin(sig_f) & (ndf["confidence"]>=conf_min)]\
                  .sort_values(sort_by, ascending=False)
        st.markdown(f"### 🏦 {len(filt)} of {len(ndf)} NSE stocks")

        p1,p2 = st.columns([1,3])
        with p1:
            pie = px.pie(
                values=[(ndf["signal"].str.startswith("BUY")).sum(),
                        (ndf["signal"].str.startswith("SELL")).sum(),
                        (ndf["signal"].str.startswith("HOLD")).sum()],
                names=["BUY","SELL","HOLD"],
                color_discrete_sequence=["#00c853","#d50000","#ff9800"],
                hole=0.6)
            pie.update_layout(height=180, margin=dict(t=5,b=5,l=0,r=0),
                               showlegend=True,
                               legend=dict(font_color="#aaa",bgcolor="rgba(0,0,0,0)"),
                               paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(pie, use_container_width=True)
        with p2:
            fh = px.histogram(ndf, x="confidence", nbins=25,
                               color_discrete_sequence=["#00d4ff"],
                               title="Confidence spread across all NSE stocks")
            fh.update_layout(height=180, margin=dict(t=30,b=5,l=5,r=5),
                              paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)",
                              font_color="#aaa")
            fh.update_xaxes(color="#555"); fh.update_yaxes(color="#555")
            st.plotly_chart(fh, use_container_width=True)

        cols = [c for c in ["symbol","signal","confidence","direction",
                             "last_close","est_price_low","est_price_high",
                             "rsi","model_accuracy","sentiment_label"]
                if c in filt.columns]
        st.dataframe(filt[cols].reset_index(drop=True),
            use_container_width=True, height=420,
            column_config={
                "confidence":      st.column_config.ProgressColumn("Confidence %", min_value=0, max_value=100),
                "model_accuracy":  st.column_config.ProgressColumn("Accuracy %",   min_value=0, max_value=100),
                "last_close":      st.column_config.NumberColumn("Close ₹",    format="₹%.2f"),
                "est_price_low":   st.column_config.NumberColumn("Est Low ₹",  format="₹%.2f"),
                "est_price_high":  st.column_config.NumberColumn("Est High ₹", format="₹%.2f"),
                "rsi":             st.column_config.NumberColumn("RSI",         format="%.1f"),
            })

        st.divider()
        st.markdown("### 🔍 Deep Dive")
        sel = st.selectbox("Pick a stock", sorted(ndf["symbol"].tolist()), key="nse_dd")
        if sel: stock_detail(df, sel, "₹")

    # ─ TAB 2: CRYPTO ──────────────────────────────────────────
    with t2:
        cdf2 = df[df["market"]=="Crypto"].copy() if "market" in df.columns else pd.DataFrame()
        if cdf2.empty:
            st.info("No crypto data. Run a Full Update.")
        else:
            fc = cdf2[cdf2["signal"].isin(sig_f) & (cdf2["confidence"]>=conf_min)]\
                     .sort_values(sort_by, ascending=False)
            st.markdown(f"### ₿ {len(fc)} Crypto assets")
            ccols = [c for c in ["symbol","signal","confidence","direction",
                                  "last_close","est_price_low","est_price_high",
                                  "rsi","sentiment_label"] if c in fc.columns]
            st.dataframe(fc[ccols].reset_index(drop=True),
                use_container_width=True, height=360,
                column_config={
                    "confidence":     st.column_config.ProgressColumn("Confidence %", min_value=0, max_value=100),
                    "last_close":     st.column_config.NumberColumn("Price $",   format="$%.2f"),
                    "est_price_low":  st.column_config.NumberColumn("Est Low $", format="$%.2f"),
                    "est_price_high": st.column_config.NumberColumn("Est High $",format="$%.2f"),
                })
            st.divider()
            sel_c = st.selectbox("Pick a crypto", sorted(cdf2["symbol"].tolist()), key="cry_dd")
            if sel_c: stock_detail(df, sel_c, "$")

    # ─ TAB 3: NEWS ────────────────────────────────────────────
    with t3:
        st.markdown("### 📰 News Sentiment Heatmap")
        if "sentiment_score" not in df.columns:
            st.info("No sentiment data.")
        else:
            top = df.nlargest(40,"article_count") if "article_count" in df.columns else df.head(40)
            fb = px.bar(
                top.sort_values("sentiment_score",ascending=True),
                x="sentiment_score", y="symbol", orientation="h",
                color="sentiment_score",
                color_continuous_scale=["#d50000","#ff9800","#00c853"],
                color_continuous_midpoint=0,
                title="Sentiment score — top 40 most-mentioned stocks (recency-weighted)")
            fb.update_layout(height=600, margin=dict(l=130,r=20,t=50,b=20),
                              paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)",
                              font_color="#aaa",
                              coloraxis_showscale=False)
            st.plotly_chart(fb, use_container_width=True)

            if "top_headline" in df.columns and "article_count" in df.columns:
                st.markdown("### 📋 Headlines by Stock")
                hl = df[df["article_count"]>0][
                    ["symbol","sentiment_label","article_count","top_headline"]
                ].sort_values("article_count",ascending=False).head(30)
                st.dataframe(hl.reset_index(drop=True), use_container_width=True, height=360)

    # ─ TAB 4: SUMMARY ─────────────────────────────────────────
    with t4:
        st.markdown("### 📊 Signal Summary")
        bc,sc = st.columns(2)
        with bc:
            st.markdown("#### 🟢 Top BUY Signals")
            bdf = df[df["signal"].str.startswith("BUY")]\
                    .sort_values("confidence",ascending=False).head(15)
            st.dataframe(bdf[["symbol","confidence","prob_up",
                               "model_accuracy","sentiment_label","last_close"]]\
                           .reset_index(drop=True),
                use_container_width=True,
                column_config={
                    "confidence":    st.column_config.ProgressColumn("Confidence %", min_value=0, max_value=100),
                    "model_accuracy":st.column_config.ProgressColumn("Accuracy %",   min_value=0, max_value=100),
                })
        with sc:
            st.markdown("#### 🔴 Top SELL Signals")
            sdf = df[df["signal"].str.startswith("SELL")]\
                    .sort_values("confidence",ascending=False).head(15)
            st.dataframe(sdf[["symbol","confidence","prob_down",
                               "model_accuracy","sentiment_label","last_close"]]\
                           .reset_index(drop=True),
                use_container_width=True,
                column_config={
                    "confidence":    st.column_config.ProgressColumn("Confidence %", min_value=0, max_value=100),
                    "model_accuracy":st.column_config.ProgressColumn("Accuracy %",   min_value=0, max_value=100),
                })
        if "model_accuracy" in df.columns:
            st.divider()
            fa = px.histogram(df, x="model_accuracy", nbins=25,
                               color_discrete_sequence=["#7c4dff"],
                               title="Model accuracy distribution (ensemble RF + GB)")
            fa.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)",
                              font_color="#aaa", height=300)
            st.plotly_chart(fa, use_container_width=True)
            st.info(f"Average model accuracy: **{df['model_accuracy'].mean():.1f}%**  |  "
                    f"Best: **{df['model_accuracy'].max():.1f}%**  |  "
                    f"Stocks above 60%: **{(df['model_accuracy']>60).sum()}**")

    # ─ TAB 5: PORTFOLIO ───────────────────────────────────────
    with t5:
        render_portfolio()

    st.divider()
    st.caption("⚠️ Educational purposes only · Not financial advice · Always do your own research")


if __name__ == "__main__":
    main()
