"""
config.py
---------
Central configuration. All symbols, API keys, and settings live here.
Supports both:
  - Streamlit Cloud (reads from st.secrets)
  - GitHub Actions  (reads from environment variables)
"""

import os

def _get_secret(key: str, default: str = "") -> str:
    """
    Read from Streamlit secrets when app is running,
    or from environment variables when GitHub Actions runs update_data.py.
    """
    try:
        import streamlit as st
        return st.secrets.get(key, os.environ.get(key, default))
    except Exception:
        return os.environ.get(key, default)


# ── API KEYS ──────────────────────────────────────────────────
NEWS_API_KEY = _get_secret("NEWS_API_KEY")

# ── SETTINGS ──────────────────────────────────────────────────
HISTORY_YEARS  = 2
PREDICT_DAYS   = 1
BUY_THRESHOLD  = 0.60
SELL_THRESHOLD = 0.40
DATA_DIR       = "predictions"

# ── NSE STOCKS (~200) ─────────────────────────────────────────
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
    "CROMPTON.NS","CYIENT.NS","DELHIVERY.NS","ELGIEQUIP.NS","EMAMILTD.NS",
    "ENDURANCE.NS","EQUITASBNK.NS","FORTIS.NS","GLAND.NS","GRANULES.NS",
    "HAPPSTMNDS.NS","HFCL.NS","HUDCO.NS","ICICIGI.NS","ICICIPRULI.NS",
    "IDFC.NS","IPCALAB.NS","IRB.NS","ISEC.NS","JKCEMENT.NS",
    "JSWENERGY.NS","KPITTECH.NS","LICI.NS","LTTS.NS","MANAPPURAM.NS",
    "MARICO.NS","MAXHEALTH.NS","MCX.NS","METROPOLIS.NS","MFSL.NS",
    "NLCINDIA.NS","OBEROIRLTY.NS","PIIND.NS","PVRINOX.NS","RAMCOCEM.NS",
    "RITES.NS","SJVN.NS","SONACOMS.NS","STARHEALTH.NS","SUPREMEIND.NS",
    "SYNGENE.NS","TATACHEM.NS","TATACOMM.NS","TRIDENT.NS","UNIONBANK.NS",
    "VARUNBEV.NS","VBL.NS","WELCORP.NS","YESBANK.NS",
    "ADANITRANS.NS","AIAENG.NS","APOLLOTYRE.NS","ATUL.NS","BAJAJHLDNG.NS",
    "BALRAMCHIN.NS","BLUESTARCO.NS","BSOFT.NS","CARBORUNIV.NS","CASTROLIND.NS",
    "CESC.NS","CUB.NS","DATAPATTNS.NS","DEVYANI.NS","FINEORG.NS",
    "GALAXYSURF.NS","GLAXO.NS","GODFRYPHLP.NS","GRAPHITE.NS","GSPL.NS",
    "HATSUN.NS","IBREALEST.NS","JAMNAAUTO.NS","JBCHEPHARM.NS","JKLAKSHMI.NS",
    "JKPAPER.NS","LUXIND.NS","MCDOWELL-N.NS","NIACL.NS","PRINCEPIPE.NS",
    "RADICO.NS","SKFINDIA.NS","TIMKEN.NS","TTKPRESTIG.NS","UNITDSPR.NS",
]

# ── CRYPTO ────────────────────────────────────────────────────
CRYPTO_SYMBOLS = [
    "BTC-USD","ETH-USD","BNB-USD","XRP-USD","ADA-USD",
    "SOL-USD","DOGE-USD","MATIC-USD","DOT-USD","AVAX-USD",
]

# ── RSS FEEDS (free, no key needed) ──────────────────────────
RSS_FEEDS = [
    "https://www.moneycontrol.com/rss/latestnews.xml",
    "https://www.moneycontrol.com/rss/marketoutlook.xml",
    "https://economictimes.indiatimes.com/markets/stocks/rss.cms",
    "https://economictimes.indiatimes.com/markets/rss.cms",
    "https://www.business-standard.com/rss/markets-106.rss",
    "https://news.google.com/rss/search?q=NSE+BSE+stock+market+India&hl=en-IN&gl=IN&ceid=IN:en",
    "https://news.google.com/rss/search?q=Nifty+Sensex+today&hl=en-IN&gl=IN&ceid=IN:en",
    "https://news.google.com/rss/search?q=RBI+India+economy+interest+rate&hl=en-IN&gl=IN&ceid=IN:en",
    "https://news.google.com/rss/search?q=Bitcoin+Ethereum+crypto+price&hl=en&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=cryptocurrency+India+regulation&hl=en-IN&gl=IN&ceid=IN:en",
]
