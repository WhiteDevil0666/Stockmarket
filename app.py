"""
app.py — AI Stock Predictor (fully self-contained, no subfolders needed)
Everything is in this single file so Streamlit Cloud has no import issues.
"""

import os, sys, warnings
warnings.filterwarnings("ignore")

# Fix working directory
_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(_ROOT)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
import feedparser
import requests
from datetime import datetime, timedelta
from time import sleep
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import ta

st.set_page_config(
    page_title="AI Stock Predictor 🇮🇳",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
div[data-testid="metric-container"] {
    background: #f8f9fa; border-radius: 10px;
    padding: 10px; border: 1px solid #e0e0e0;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════

def _get_secret(key):
    try:
        return st.secrets.get(key, os.environ.get(key, ""))
    except:
        return os.environ.get(key, "")

NEWS_API_KEY   = _get_secret("NEWS_API_KEY")
BUY_THRESHOLD  = 0.60
SELL_THRESHOLD = 0.40
DATA_DIR       = "predictions"

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
    "https://news.google.com/rss/search?q=Bitcoin+Ethereum+crypto+price&hl=en&gl=US&ceid=US:en",
]


# ═══════════════════════════════════════════════════════════════
# STOCK DATA  (yfinance)
# ═══════════════════════════════════════════════════════════════

def download_stock(symbol, years=2):
    end   = datetime.today()
    start = end - timedelta(days=years * 365)
    try:
        df = yf.download(symbol,
                         start=start.strftime("%Y-%m-%d"),
                         end=end.strftime("%Y-%m-%d"),
                         progress=False, auto_adjust=True)
        if df.empty or len(df) < 60:
            return None
        df = df[["Open","High","Low","Close","Volume"]].copy()
        df.columns = ["open","high","low","close","volume"]
        df.index.name = "date"
        return df.dropna()
    except:
        return None


def add_indicators(df):
    df = df.copy()
    df["return_1d"]  = df["close"].pct_change(1)
    df["return_5d"]  = df["close"].pct_change(5)
    df["return_20d"] = df["close"].pct_change(20)
    df["sma_20"] = ta.trend.sma_indicator(df["close"], window=20)
    df["sma_50"] = ta.trend.sma_indicator(df["close"], window=50)
    df["ema_12"] = ta.trend.ema_indicator(df["close"], window=12)
    df["ema_26"] = ta.trend.ema_indicator(df["close"], window=26)
    df["price_sma20_ratio"] = df["close"] / df["sma_20"]
    df["price_sma50_ratio"] = df["close"] / df["sma_50"]
    df["sma20_sma50_ratio"] = df["sma_20"] / df["sma_50"]
    df["rsi_14"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    macd = ta.trend.MACD(df["close"])
    df["macd"]        = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"]   = macd.macd_diff()
    bb = ta.volatility.BollingerBands(df["close"], window=20)
    df["bb_upper"]   = bb.bollinger_hband()
    df["bb_lower"]   = bb.bollinger_lband()
    df["bb_percent"] = bb.bollinger_pband()
    df["bb_width"]   = bb.bollinger_wband()
    df["volume_sma20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma20"]
    df["atr_14"] = ta.volatility.AverageTrueRange(
        df["high"], df["low"], df["close"], window=14).average_true_range()
    stoch = ta.momentum.StochasticOscillator(df["high"], df["low"], df["close"])
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()
    df["target"]     = (df["close"].shift(-1) > df["close"]).astype(int)
    df["next_close"] = df["close"].shift(-1)
    return df.dropna()


def fetch_all_stocks(symbols, status_el=None, prog_el=None, prog_start=0, prog_end=30):
    results = {}
    total = len(symbols)
    for i, sym in enumerate(symbols):
        if status_el:
            status_el.write(f"Downloading {sym} ({i+1}/{total})...")
        if prog_el:
            prog_el.progress(int(prog_start + (prog_end - prog_start) * i / total),
                             text=f"Fetching {sym}...")
        df = download_stock(sym)
        if df is not None:
            df = add_indicators(df)
            results[sym] = df
        sleep(0.2)
    return results


# ═══════════════════════════════════════════════════════════════
# NEWS  (RSS + NewsAPI)
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
    if not NEWS_API_KEY:
        return []
    queries = [
        "NSE BSE Nifty Sensex India stock market",
        "Indian economy RBI inflation budget",
        "Bitcoin Ethereum cryptocurrency India",
    ]
    articles = []
    from_date = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    for q in queries:
        try:
            r = requests.get("https://newsapi.org/v2/everything", params={
                "q": q, "from": from_date, "sortBy": "publishedAt",
                "language": "en", "pageSize": 100, "apiKey": NEWS_API_KEY,
            }, timeout=10)
            data = r.json()
            if data.get("status") == "ok":
                for a in data.get("articles", []):
                    articles.append({
                        "title":     a.get("title",""),
                        "summary":   a.get("description",""),
                        "source":    a.get("source",{}).get("name","NewsAPI"),
                        "published": a.get("publishedAt",""),
                    })
            sleep(0.3)
        except:
            pass
    return articles


def fetch_all_news():
    all_articles = fetch_rss() + fetch_newsapi()
    if not all_articles:
        return pd.DataFrame()
    df = pd.DataFrame(all_articles).drop_duplicates(subset=["title"])
    df["title"]   = df["title"].fillna("")
    df["summary"] = df["summary"].fillna("")
    df["text"]    = df["title"] + ". " + df["summary"]
    all_syms = NSE_STOCKS + CRYPTO_SYMBOLS
    def tag(row):
        text = row["text"].upper()
        return [s for s in all_syms if len(s.split(".")[0].split("-")[0]) >= 3
                and s.split(".")[0].split("-")[0] in text]
    df["mentioned_symbols"] = df.apply(tag, axis=1)
    return df


# ═══════════════════════════════════════════════════════════════
# SENTIMENT  (VADER)
# ═══════════════════════════════════════════════════════════════

_vader = SentimentIntensityAnalyzer()
for w in ["bullish","rally","surge","breakout","buy","upgrade","profit","dividend","growth"]:
    _vader.lexicon[w] = 2.5
for w in ["bearish","crash","fall","drop","sell","downgrade","loss","fraud","deficit","decline"]:
    _vader.lexicon[w] = -2.5


def score_text(text):
    if not text: return 0.0
    return round(_vader.polarity_scores(str(text))["compound"], 4)


def sentiment_label(score):
    if score >= 0.05:  return "Bullish 🟢"
    if score <= -0.05: return "Bearish 🔴"
    return "Neutral ⚪"


def compute_sentiment(news_df):
    if news_df.empty:
        return pd.DataFrame(), {"score": 0.0, "label": "Neutral ⚪", "total_articles": 0}
    news_df = news_df.copy()
    news_df["sentiment"] = news_df["text"].apply(score_text)
    all_syms = NSE_STOCKS + CRYPTO_SYMBOLS
    rows = []
    for sym in all_syms:
        mask = news_df["mentioned_symbols"].apply(lambda s: sym in s)
        rel  = news_df[mask]
        if len(rel) == 0:
            rows.append({"symbol": sym, "sentiment_score": 0.0,
                         "sentiment_label": "Neutral ⚪", "article_count": 0,
                         "top_headline": "No recent news"})
        else:
            avg = rel["sentiment"].mean()
            top = rel.sort_values("sentiment", ascending=False).iloc[0]["title"]
            rows.append({"symbol": sym, "sentiment_score": round(avg,4),
                         "sentiment_label": sentiment_label(avg),
                         "article_count": len(rel), "top_headline": top})
    sent_df     = pd.DataFrame(rows)
    market_sent = {"score": round(news_df["sentiment"].mean(),4),
                   "label": sentiment_label(news_df["sentiment"].mean()),
                   "total_articles": len(news_df)}
    return sent_df, market_sent


# ═══════════════════════════════════════════════════════════════
# ML MODEL
# ═══════════════════════════════════════════════════════════════

FEATURE_COLS = [
    "return_1d","return_5d","return_20d",
    "price_sma20_ratio","price_sma50_ratio","sma20_sma50_ratio",
    "rsi_14","macd","macd_signal","macd_diff",
    "bb_percent","bb_width","volume_ratio","atr_14","stoch_k","stoch_d",
    "sentiment_score",
]


def predict_stock(df, symbol, sentiment_score=0.0):
    try:
        df = df.copy()
        df["sentiment_score"] = sentiment_score
        train_df = df[:-1].copy()
        avail = [c for c in FEATURE_COLS if c in train_df.columns]
        if len(avail) < 10 or len(train_df) < 60: return None
        X = train_df[avail].replace([np.inf,-np.inf], np.nan).fillna(0)
        y = train_df["target"].astype(int)
        if y.nunique() < 2: return None
        scaler  = StandardScaler()
        X_sc    = scaler.fit_transform(X)
        model   = RandomForestClassifier(n_estimators=150, max_depth=8,
                                          min_samples_leaf=5, random_state=42, n_jobs=-1)
        model.fit(X_sc, y)
        today_row = df[avail].iloc[-1].replace([np.inf,-np.inf], np.nan).fillna(0)
        prob_up   = model.predict_proba(scaler.transform([today_row.values]))[0][1]
        prob_down = 1 - prob_up
        direction = "UP 🟢" if prob_up >= 0.5 else "DOWN 🔴"
        confidence = max(prob_up, prob_down) * 100
        if prob_up >= BUY_THRESHOLD:   signal = "BUY 🟢"
        elif prob_up <= SELL_THRESHOLD: signal = "SELL 🔴"
        else:                           signal = "HOLD 🟡"
        last_close = float(df["close"].iloc[-1])
        atr        = float(df["atr_14"].iloc[-1]) if "atr_14" in df.columns else last_close * 0.02
        if direction.startswith("UP"):
            est_low, est_high = last_close - atr*0.5, last_close + atr*1.2
        else:
            est_low, est_high = last_close - atr*1.2, last_close + atr*0.5
        tscv   = TimeSeriesSplit(n_splits=3)
        scores = []
        for tr, vl in tscv.split(X_sc):
            m = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)
            m.fit(X_sc[tr], y.iloc[tr])
            scores.append(m.score(X_sc[vl], y.iloc[vl]))
        return {
            "symbol": symbol,
            "last_close": round(last_close,2),
            "direction": direction, "confidence": round(confidence,1),
            "signal": signal,
            "prob_up": round(prob_up*100,1), "prob_down": round(prob_down*100,1),
            "est_price_low": round(est_low,2), "est_price_high": round(est_high,2),
            "model_accuracy": round(np.mean(scores)*100,1),
            "rsi":         round(float(df["rsi_14"].iloc[-1]),1) if "rsi_14" in df.columns else None,
            "macd":        round(float(df["macd"].iloc[-1]),4)   if "macd" in df.columns else None,
            "bb_percent":  round(float(df["bb_percent"].iloc[-1])*100,1) if "bb_percent" in df.columns else None,
            "volume_ratio":round(float(df["volume_ratio"].iloc[-1]),2)   if "volume_ratio" in df.columns else None,
            "sentiment_score": round(sentiment_score,4),
            "updated_at":  datetime.now().strftime("%Y-%m-%d %H:%M IST"),
        }
    except:
        return None


def run_predictions(stock_data, sentiment_df):
    sent_lookup = {}
    if not sentiment_df.empty and "symbol" in sentiment_df.columns:
        sent_lookup = dict(zip(sentiment_df["symbol"], sentiment_df["sentiment_score"]))
    results = []
    for sym, df in stock_data.items():
        r = predict_stock(df, sym, sent_lookup.get(sym, 0.0))
        if r:
            if not sentiment_df.empty and "symbol" in sentiment_df.columns:
                row = sentiment_df[sentiment_df["symbol"] == sym]
                if not row.empty:
                    r["sentiment_label"] = row.iloc[0].get("sentiment_label","Neutral ⚪")
                    r["article_count"]   = row.iloc[0].get("article_count",0)
                    r["top_headline"]    = row.iloc[0].get("top_headline","")
            results.append(r)
    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════
# DATA PIPELINE  (runs inside Streamlit)
# ═══════════════════════════════════════════════════════════════

def run_update(quick_mode=False):
    today     = datetime.now().strftime("%Y-%m-%d")
    nse_list  = NSE_STOCKS[:20] if quick_mode else NSE_STOCKS
    cry_list  = CRYPTO_SYMBOLS[:5] if quick_mode else CRYPTO_SYMBOLS

    if quick_mode:
        st.info("⚡ Quick mode: 20 NSE + 5 crypto stocks (~5–8 min)")

    prog   = st.progress(0, text="Starting...")
    status = st.empty()

    try:
        status.write("**Step 1/4** — Downloading stock prices from Yahoo Finance...")
        nse_data    = fetch_all_stocks(nse_list,  status, prog, 0,  25)
        crypto_data = fetch_all_stocks(cry_list,  status, prog, 25, 35)
        all_data    = {**nse_data, **crypto_data}
        if not all_data:
            st.error("❌ No stock data downloaded. Check your internet.")
            return False

        status.write("**Step 2/4** — Fetching news (RSS + NewsAPI)...")
        prog.progress(40, text="Fetching news...")
        news_df = fetch_all_news()

        status.write("**Step 3/4** — Analysing news sentiment...")
        prog.progress(55, text="Running sentiment...")
        sentiment_df, market_sent = compute_sentiment(news_df)

        status.write("**Step 4/4** — Training ML models and generating predictions...")
        prog.progress(65, text="Training models (this takes a few minutes)...")
        predictions_df = run_predictions(all_data, sentiment_df)
        if predictions_df.empty:
            st.error("❌ No predictions generated.")
            return False

        nse_set  = set(NSE_STOCKS)
        cry_set  = set(CRYPTO_SYMBOLS)
        predictions_df["market"] = predictions_df["symbol"].apply(
            lambda s: "NSE" if s in nse_set else ("Crypto" if s in cry_set else "Other"))
        predictions_df["date"] = today

        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(os.path.join(DATA_DIR, "history"), exist_ok=True)
        predictions_df.to_csv(os.path.join(DATA_DIR, "latest_predictions.csv"), index=False)
        predictions_df.to_csv(os.path.join(DATA_DIR, "history", f"{today}.csv"), index=False)
        market_sent["date"] = today
        pd.DataFrame([market_sent]).to_csv(os.path.join(DATA_DIR, "market_sentiment.csv"), index=False)

        prog.progress(100, text="Done! ✅")
        status.empty()
        return True
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return False


# ═══════════════════════════════════════════════════════════════
# LOADERS
# ═══════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def load_predictions():
    p = os.path.join(DATA_DIR, "latest_predictions.csv")
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

@st.cache_data(ttl=3600)
def load_market_sentiment():
    p = os.path.join(DATA_DIR, "market_sentiment.csv")
    return pd.read_csv(p).iloc[0].to_dict() if os.path.exists(p) else {}

@st.cache_data(ttl=1800)
def load_chart(symbol):
    try:
        df = yf.download(symbol, period="6mo", progress=False, auto_adjust=True)
        df.columns = [c.lower() for c in df.columns]
        return df
    except:
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════
# STOCK DETAIL
# ═══════════════════════════════════════════════════════════════

def stock_detail(df, symbol, currency="₹"):
    row = df[df["symbol"] == symbol]
    if row.empty: return
    r = row.iloc[0]
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"**Signal:** {r.get('signal','—')}")
        st.markdown(f"**Direction:** {r.get('direction','—')}")
        st.markdown(f"**Confidence:** {r.get('confidence',0):.1f}%")
        st.markdown(f"**Last Close:** {currency}{r.get('last_close',0):.2f}")
        st.markdown(f"**Est. Tomorrow:** {currency}{r.get('est_price_low',0):.2f} – {currency}{r.get('est_price_high',0):.2f}")
    with c2:
        rsi = float(r.get("rsi",50) or 50)
        st.markdown(f"**RSI:** {rsi:.1f}")
        if rsi > 70:   st.warning("⚠️ Overbought (RSI > 70)")
        elif rsi < 30: st.success("✅ Oversold (RSI < 30)")
        else:          st.info(f"RSI neutral ({rsi:.0f})")
        st.markdown(f"**MACD:** {r.get('macd','—')}")
        st.markdown(f"**Volume Ratio:** {r.get('volume_ratio','—')}")
    with c3:
        st.markdown(f"**Sentiment:** {r.get('sentiment_label','—')}")
        st.markdown(f"**Score:** {r.get('sentiment_score',0):+.3f}")
        st.markdown(f"**News articles:** {r.get('article_count',0)}")
        hl = r.get("top_headline","")
        if hl: st.caption(f"📰 {str(hl)[:130]}...")

    st.markdown(f"#### {symbol} — 6-Month Chart")
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
                                  line=dict(color="#ff9800",width=1), name="SMA20"))
        fig.add_trace(go.Scatter(x=cdf.index, y=cdf["sma50"],
                                  line=dict(color="#2196f3",width=1), name="SMA50"))
        fig.add_hrect(y0=r.get("est_price_low",0), y1=r.get("est_price_high",0),
                      fillcolor="rgba(124,77,255,0.12)", line_width=0,
                      annotation_text="Predicted range", annotation_position="top right")
        fig.update_layout(height=380, xaxis_rangeslider_visible=False,
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          margin=dict(l=10,r=10,t=20,b=10))
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    st.markdown("# 📈 AI Stock Predictor — India & Crypto")
    st.markdown("##### Powered by Machine Learning + News Sentiment | Updates Every Morning 🌅")
    st.divider()

    df   = load_predictions()
    mkt  = load_market_sentiment()

    # ── NO DATA ───────────────────────────────────────────────
    if df.empty:
        st.warning("### ⚠️ No prediction data found yet.")
        st.markdown("Click a button below to generate your first predictions. This only needs to be done once — after that GitHub Actions updates the app every morning automatically.")
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### ⚡ Quick Start *(recommended)*")
            st.markdown("20 NSE + 5 crypto stocks · **~5–8 minutes**")
            if st.button("▶  Run Quick Update", type="primary", use_container_width=True):
                ok = run_update(quick_mode=True)
                if ok:
                    st.success("✅ Done!")
                    st.cache_data.clear()
                    st.rerun()
        with c2:
            st.markdown("#### 🔄 Full Update")
            st.markdown("All 200+ NSE + 10 crypto stocks · **~25–35 minutes**")
            if st.button("▶  Run Full Update", use_container_width=True):
                ok = run_update(quick_mode=False)
                if ok:
                    st.success("✅ Done!")
                    st.cache_data.clear()
                    st.rerun()
        return

    # ── METRICS ───────────────────────────────────────────────
    total   = len(df)
    buy_ct  = (df["signal"].str.startswith("BUY")).sum()
    sell_ct = (df["signal"].str.startswith("SELL")).sum()
    hold_ct = (df["signal"].str.startswith("HOLD")).sum()
    updated = df["updated_at"].iloc[0] if "updated_at" in df.columns else "—"

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("🕐 Updated",       updated.split(" ")[0] if updated != "—" else "—")
    c2.metric("🟢 BUY",  buy_ct,  f"{buy_ct/total*100:.0f}%")
    c3.metric("🔴 SELL", sell_ct, f"{sell_ct/total*100:.0f}%")
    c4.metric("🟡 HOLD", hold_ct, f"{hold_ct/total*100:.0f}%")
    c5.metric("📰 Mood", mkt.get("label","—"), f"{mkt.get('score',0):+.2f}")
    st.divider()

    # ── SIDEBAR ───────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Filters")
        sig_f    = st.multiselect("Signal", ["BUY 🟢","SELL 🔴","HOLD 🟡"],
                                   default=["BUY 🟢","SELL 🔴","HOLD 🟡"])
        conf_min = st.slider("Min Confidence %", 50, 95, 50)
        sort_by  = st.selectbox("Sort by", ["confidence","prob_up","sentiment_score","symbol"])
        st.divider()
        st.markdown("### 🔄 Refresh")
        if st.button("▶ Run Update Now", use_container_width=True):
            ok = run_update(quick_mode=False)
            if ok:
                st.cache_data.clear()
                st.rerun()
        st.divider()
        st.markdown("### 📌 Quick Lookup")
        lk = st.text_input("e.g. RELIANCE.NS", "")
        if lk:
            r = df[df["symbol"] == lk.strip().upper()]
            if r.empty:
                st.error("Not found")
            else:
                x = r.iloc[0]
                st.markdown(f"**{x['symbol']}**")
                for k in ["signal","confidence","last_close","sentiment_label"]:
                    st.markdown(f"{k}: `{x.get(k,'—')}`")

    # ── TABS ──────────────────────────────────────────────────
    t1, t2, t3, t4 = st.tabs(["🏦 NSE Stocks","₿ Crypto","📰 News Sentiment","📊 Summary"])

    with t1:
        ndf  = df[df["market"]=="NSE"].copy() if "market" in df.columns else df.copy()
        filt = ndf[ndf["signal"].isin(sig_f) & (ndf["confidence"]>=conf_min)].sort_values(sort_by, ascending=False)
        st.markdown(f"### 🏦 {len(filt)} NSE stocks")
        p1, p2 = st.columns([1,3])
        with p1:
            pie = px.pie(
                values=[(ndf["signal"].str.startswith("BUY")).sum(),
                        (ndf["signal"].str.startswith("SELL")).sum(),
                        (ndf["signal"].str.startswith("HOLD")).sum()],
                names=["BUY","SELL","HOLD"],
                color_discrete_sequence=["#00c853","#d50000","#ff9800"], hole=0.55)
            pie.update_layout(height=180, margin=dict(t=5,b=5,l=0,r=0),
                               showlegend=False, paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(pie, use_container_width=True)
        with p2:
            fh = px.histogram(ndf, x="confidence", nbins=20,
                               color_discrete_sequence=["#1565c0"], title="Confidence spread")
            fh.update_layout(height=180, margin=dict(t=30,b=5,l=5,r=5),
                              paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fh, use_container_width=True)

        cols = [c for c in ["symbol","signal","confidence","direction","last_close",
                             "est_price_low","est_price_high","rsi","sentiment_label"]
                if c in filt.columns]
        st.dataframe(filt[cols].reset_index(drop=True), use_container_width=True, height=420,
            column_config={
                "confidence":     st.column_config.ProgressColumn("Confidence %", min_value=0, max_value=100),
                "last_close":     st.column_config.NumberColumn("Close ₹",    format="₹%.2f"),
                "est_price_low":  st.column_config.NumberColumn("Est Low ₹",  format="₹%.2f"),
                "est_price_high": st.column_config.NumberColumn("Est High ₹", format="₹%.2f"),
                "rsi":            st.column_config.NumberColumn("RSI",         format="%.1f"),
            })
        st.divider()
        st.markdown("### 🔍 Deep Dive")
        sel = st.selectbox("Pick a stock", sorted(ndf["symbol"].tolist()), key="nse_dd")
        if sel: stock_detail(df, sel, "₹")

    with t2:
        cdf = df[df["market"]=="Crypto"].copy() if "market" in df.columns else pd.DataFrame()
        if cdf.empty:
            st.info("No crypto data yet — run a Full Update.")
        else:
            fc = cdf[cdf["signal"].isin(sig_f) & (cdf["confidence"]>=conf_min)].sort_values(sort_by, ascending=False)
            st.markdown(f"### ₿ {len(fc)} Crypto assets")
            ccols = [c for c in ["symbol","signal","confidence","direction","last_close",
                                  "est_price_low","est_price_high","rsi","sentiment_label"]
                     if c in fc.columns]
            st.dataframe(fc[ccols].reset_index(drop=True), use_container_width=True, height=360,
                column_config={
                    "confidence":     st.column_config.ProgressColumn("Confidence %", min_value=0, max_value=100),
                    "last_close":     st.column_config.NumberColumn("Price $",    format="$%.2f"),
                    "est_price_low":  st.column_config.NumberColumn("Est Low $",  format="$%.2f"),
                    "est_price_high": st.column_config.NumberColumn("Est High $", format="$%.2f"),
                })
            st.divider()
            sel_c = st.selectbox("Pick a crypto", sorted(cdf["symbol"].tolist()), key="cry_dd")
            if sel_c: stock_detail(df, sel_c, "$")

    with t3:
        st.markdown("### 📰 News Sentiment by Stock")
        if "sentiment_score" not in df.columns:
            st.info("No sentiment data.")
        else:
            top = df.nlargest(40,"article_count") if "article_count" in df.columns else df.head(40)
            fb = px.bar(top.sort_values("sentiment_score", ascending=True),
                        x="sentiment_score", y="symbol", orientation="h",
                        color="sentiment_score",
                        color_continuous_scale=["#d50000","#ff9800","#00c853"],
                        color_continuous_midpoint=0,
                        title="Sentiment score — top 40 most-mentioned stocks")
            fb.update_layout(height=570, margin=dict(l=120,r=20,t=50,b=20),
                              paper_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False)
            st.plotly_chart(fb, use_container_width=True)
            if "top_headline" in df.columns and "article_count" in df.columns:
                st.markdown("### 📋 Top Headlines")
                hl = df[df["article_count"]>0][["symbol","sentiment_label","article_count","top_headline"]]\
                       .sort_values("article_count",ascending=False).head(30)
                st.dataframe(hl.reset_index(drop=True), use_container_width=True, height=360)

    with t4:
        st.markdown("### 📊 Signal Summary")
        bc, sc = st.columns(2)
        with bc:
            st.markdown("#### 🟢 Top BUY Picks")
            bdf = df[df["signal"].str.startswith("BUY")].sort_values("confidence",ascending=False).head(15)
            st.dataframe(bdf[["symbol","confidence","prob_up","sentiment_label","last_close"]].reset_index(drop=True),
                use_container_width=True,
                column_config={"confidence": st.column_config.ProgressColumn("Confidence %", min_value=0, max_value=100)})
        with sc:
            st.markdown("#### 🔴 Top SELL Picks")
            sdf = df[df["signal"].str.startswith("SELL")].sort_values("confidence",ascending=False).head(15)
            st.dataframe(sdf[["symbol","confidence","prob_down","sentiment_label","last_close"]].reset_index(drop=True),
                use_container_width=True,
                column_config={"confidence": st.column_config.ProgressColumn("Confidence %", min_value=0, max_value=100)})
        if "model_accuracy" in df.columns:
            st.divider()
            fa = px.histogram(df, x="model_accuracy", nbins=20,
                               color_discrete_sequence=["#7c4dff"], title="Model accuracy distribution")
            fa.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=280)
            st.plotly_chart(fa, use_container_width=True)
            st.info(f"Average model accuracy: **{df['model_accuracy'].mean():.1f}%**")

    st.divider()
    st.caption("⚠️ Educational purposes only. Not financial advice. Always do your own research.")


if __name__ == "__main__":
    main()
