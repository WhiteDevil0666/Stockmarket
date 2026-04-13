"""
app.py
------
The Streamlit dashboard.
This file ONLY reads pre-computed CSV files — it does NOT train models.
All heavy lifting is done by update_data.py (runs daily via GitHub Actions).

Run locally:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
import os
from datetime import datetime, timedelta

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Predictor 🇮🇳",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #0d2137 100%);
        border-radius: 12px;
        padding: 16px 20px;
        margin: 6px 0;
        border-left: 4px solid #00d4ff;
    }
    .buy-card  { border-left-color: #00e676; }
    .sell-card { border-left-color: #ff5252; }
    .hold-card { border-left-color: #ffd740; }
    .stTabs [data-baseweb="tab"] { font-size: 16px; font-weight: 600; }
    div[data-testid="metric-container"] {
        background: #1a1a2e;
        border-radius: 10px;
        padding: 10px;
        border: 1px solid #2a2a4a;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)   # refresh cache every hour
def load_predictions():
    path = os.path.join("predictions", "latest_predictions.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df

@st.cache_data(ttl=3600)
def load_market_sentiment():
    path = os.path.join("predictions", "market_sentiment.csv")
    if not os.path.exists(path):
        return {}
    return pd.read_csv(path).iloc[0].to_dict()

@st.cache_data(ttl=1800)
def load_price_chart(symbol: str, period: str = "6mo"):
    """Fetch last 6 months of daily OHLCV for charting."""
    try:
        df = yf.download(symbol, period=period, progress=False, auto_adjust=True)
        df.columns = [c.lower() for c in df.columns]
        return df
    except:
        return pd.DataFrame()


def signal_color(signal: str) -> str:
    if signal.startswith("BUY"):
        return "🟢"
    elif signal.startswith("SELL"):
        return "🔴"
    return "🟡"


# ──────────────────────────────────────────────────────────────
# MAIN APP
# ──────────────────────────────────────────────────────────────

def main():
    # ── Header ────────────────────────────────────────────
    st.markdown("# 📈 AI Stock Predictor — India & Crypto")
    st.markdown("##### Powered by Machine Learning + News Sentiment | Updates Every Morning 🌅")
    st.divider()

    # ── Load data ─────────────────────────────────────────
    df = load_predictions()
    market_sent = load_market_sentiment()

    if df.empty:
        st.error("""
        ### ⚠️ No prediction data found yet.

        **First time setup?** Run the update script to generate predictions:
        ```bash
        python update_data.py
        ```
        This takes ~15-20 minutes the first time.
        """)
        st.stop()

    # ── Market Overview Bar ───────────────────────────────
    updated_at = df["updated_at"].iloc[0] if "updated_at" in df.columns else "Unknown"
    total      = len(df)
    buy_ct     = (df["signal"].str.startswith("BUY")).sum()
    sell_ct    = (df["signal"].str.startswith("SELL")).sum()
    hold_ct    = (df["signal"].str.startswith("HOLD")).sum()
    sent_score = market_sent.get("score", 0)
    sent_label = market_sent.get("label", "Neutral ⚪")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("🕐 Last Updated",  updated_at.split(" ")[0] if updated_at != "Unknown" else "—")
    col2.metric("🟢 BUY Signals",   buy_ct,  delta=f"{buy_ct/total*100:.0f}% of stocks")
    col3.metric("🔴 SELL Signals",  sell_ct, delta=f"{sell_ct/total*100:.0f}% of stocks")
    col4.metric("🟡 HOLD Signals",  hold_ct, delta=f"{hold_ct/total*100:.0f}% of stocks")
    col5.metric("📰 Market Mood",   sent_label, delta=f"Score: {sent_score:+.2f}")

    st.divider()

    # ── Tabs ──────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏦 NSE Stocks",
        "₿  Crypto",
        "📰 News Sentiment",
        "📊 Signal Summary",
    ])

    # ── SIDEBAR ───────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Filters")
        signal_filter = st.multiselect(
            "Filter by Signal",
            options=["BUY 🟢", "SELL 🔴", "HOLD 🟡"],
            default=["BUY 🟢", "SELL 🔴", "HOLD 🟡"],
        )
        conf_min = st.slider("Min Confidence (%)", 50, 95, 50)
        sort_by  = st.selectbox(
            "Sort by",
            ["confidence", "prob_up", "sentiment_score", "symbol"],
            index=0,
        )
        show_cols = st.multiselect(
            "Columns to show",
            ["symbol", "signal", "confidence", "direction", "last_close",
             "est_price_low", "est_price_high", "rsi", "sentiment_score",
             "sentiment_label", "article_count"],
            default=["symbol", "signal", "confidence", "direction",
                     "last_close", "est_price_low", "est_price_high",
                     "rsi", "sentiment_label"],
        )

        st.divider()
        st.markdown("### 📌 Quick Stock Lookup")
        lookup_sym = st.text_input("Enter NSE symbol (e.g. RELIANCE.NS)", "")

    # ── TAB 1: NSE STOCKS ─────────────────────────────────
    with tab1:
        nse_df = df[df["market"] == "NSE"].copy() if "market" in df.columns else df.copy()

        # Apply filters
        mask = (
            nse_df["signal"].isin(signal_filter) &
            (nse_df["confidence"] >= conf_min)
        )
        filtered = nse_df[mask].sort_values(sort_by, ascending=False)

        st.markdown(f"### 🏦 NSE Stocks — {len(filtered)} results")

        # Mini donut chart
        dcol1, dcol2 = st.columns([1, 3])
        with dcol1:
            pie_data = pd.Series({
                "BUY 🟢":  (nse_df["signal"].str.startswith("BUY")).sum(),
                "SELL 🔴": (nse_df["signal"].str.startswith("SELL")).sum(),
                "HOLD 🟡": (nse_df["signal"].str.startswith("HOLD")).sum(),
            })
            fig_pie = px.pie(
                values=pie_data.values,
                names=pie_data.index,
                color=pie_data.index,
                color_discrete_map={"BUY 🟢": "#00e676", "SELL 🔴": "#ff5252", "HOLD 🟡": "#ffd740"},
                hole=0.55,
            )
            fig_pie.update_layout(
                margin=dict(t=20, b=20, l=10, r=10),
                height=200,
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with dcol2:
            # Confidence histogram
            fig_hist = px.histogram(
                nse_df, x="confidence", color_discrete_sequence=["#00d4ff"],
                nbins=20, title="Confidence Distribution"
            )
            fig_hist.update_layout(
                height=200, margin=dict(t=30, b=10, l=10, r=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        # Main table
        display_cols = [c for c in show_cols if c in filtered.columns]
        st.dataframe(
            filtered[display_cols].reset_index(drop=True),
            use_container_width=True,
            height=450,
            column_config={
                "confidence":      st.column_config.ProgressColumn("Confidence %", min_value=0, max_value=100),
                "rsi":             st.column_config.NumberColumn("RSI", format="%.1f"),
                "last_close":      st.column_config.NumberColumn("Last Close ₹", format="₹%.2f"),
                "est_price_low":   st.column_config.NumberColumn("Est. Low ₹", format="₹%.2f"),
                "est_price_high":  st.column_config.NumberColumn("Est. High ₹", format="₹%.2f"),
                "sentiment_score": st.column_config.NumberColumn("Sentiment", format="%.3f"),
            },
        )

        # ── Individual Stock Deep Dive ─────────────────────
        st.divider()
        st.markdown("### 🔍 Individual Stock Deep Dive")
        nse_symbols = sorted(nse_df["symbol"].tolist())
        sel_sym = st.selectbox("Select a stock", nse_symbols, key="nse_select")

        if sel_sym:
            render_stock_detail(df, sel_sym, currency="₹")

    # ── TAB 2: CRYPTO ─────────────────────────────────────
    with tab2:
        crypto_df = df[df["market"] == "Crypto"].copy() if "market" in df.columns else pd.DataFrame()

        if crypto_df.empty:
            st.info("No crypto data available yet.")
        else:
            mask_c = (
                crypto_df["signal"].isin(signal_filter) &
                (crypto_df["confidence"] >= conf_min)
            )
            filtered_c = crypto_df[mask_c].sort_values(sort_by, ascending=False)

            st.markdown(f"### ₿ Crypto — {len(filtered_c)} results")

            display_cols_c = [c for c in show_cols if c in filtered_c.columns]
            st.dataframe(
                filtered_c[display_cols_c].reset_index(drop=True),
                use_container_width=True,
                height=380,
                column_config={
                    "confidence":      st.column_config.ProgressColumn("Confidence %", min_value=0, max_value=100),
                    "last_close":      st.column_config.NumberColumn("Last Price $", format="$%.2f"),
                    "est_price_low":   st.column_config.NumberColumn("Est. Low $", format="$%.2f"),
                    "est_price_high":  st.column_config.NumberColumn("Est. High $", format="$%.2f"),
                    "sentiment_score": st.column_config.NumberColumn("Sentiment", format="%.3f"),
                },
            )

            st.divider()
            st.markdown("### 🔍 Crypto Deep Dive")
            crypto_symbols = sorted(crypto_df["symbol"].tolist())
            sel_crypto = st.selectbox("Select crypto", crypto_symbols, key="crypto_select")
            if sel_crypto:
                render_stock_detail(df, sel_crypto, currency="$")

    # ── TAB 3: NEWS SENTIMENT ─────────────────────────────
    with tab3:
        st.markdown("### 📰 News Sentiment Heatmap")

        if "sentiment_score" in df.columns:
            # Heatmap of sentiment by stock
            top_n = df.nlargest(40, "article_count") if "article_count" in df.columns else df.head(40)

            fig_bar = px.bar(
                top_n.sort_values("sentiment_score", ascending=True),
                x="sentiment_score",
                y="symbol",
                orientation="h",
                color="sentiment_score",
                color_continuous_scale=["#ff5252", "#ffd740", "#00e676"],
                color_continuous_midpoint=0,
                title="Sentiment Score by Stock (top 40 most-mentioned)",
            )
            fig_bar.update_layout(
                height=600, margin=dict(l=120, r=20, t=50, b=20),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"), coloraxis_showscale=False,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # Top headlines
            st.markdown("### 📋 Top Headlines per Stock")
            if "top_headline" in df.columns and "article_count" in df.columns:
                headline_df = df[df["article_count"] > 0][
                    ["symbol", "sentiment_label", "article_count", "top_headline"]
                ].sort_values("article_count", ascending=False).head(30)
                st.dataframe(headline_df.reset_index(drop=True), use_container_width=True, height=400)

    # ── TAB 4: SIGNAL SUMMARY ─────────────────────────────
    with tab4:
        st.markdown("### 📊 Signal Summary & Statistics")

        col1, col2 = st.columns(2)

        with col1:
            # Top BUY recommendations
            st.markdown("#### 🟢 Top BUY Recommendations")
            buy_df = df[df["signal"].str.startswith("BUY")].sort_values(
                "confidence", ascending=False
            ).head(15)[["symbol", "confidence", "prob_up", "sentiment_label", "last_close"]]
            st.dataframe(buy_df.reset_index(drop=True), use_container_width=True,
                         column_config={"confidence": st.column_config.ProgressColumn("Confidence %", min_value=0, max_value=100)})

        with col2:
            # Top SELL recommendations
            st.markdown("#### 🔴 Top SELL Recommendations")
            sell_df = df[df["signal"].str.startswith("SELL")].sort_values(
                "prob_down", ascending=False
            ).head(15)[["symbol", "confidence", "prob_down", "sentiment_label", "last_close"]]
            st.dataframe(sell_df.reset_index(drop=True), use_container_width=True,
                         column_config={"confidence": st.column_config.ProgressColumn("Confidence %", min_value=0, max_value=100)})

        st.divider()

        # Accuracy distribution
        if "model_accuracy" in df.columns:
            st.markdown("#### 🎯 Model Accuracy Distribution")
            fig_acc = px.histogram(
                df, x="model_accuracy", color_discrete_sequence=["#7c4dff"],
                nbins=20, title="Cross-validation accuracy across all models"
            )
            fig_acc.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"), height=300,
            )
            st.plotly_chart(fig_acc, use_container_width=True)
            avg_acc = df["model_accuracy"].mean()
            st.info(f"📊 Average model accuracy across all stocks: **{avg_acc:.1f}%**")

    # ── SIDEBAR: Quick lookup ──────────────────────────────
    if lookup_sym:
        sym_clean = lookup_sym.strip().upper()
        row = df[df["symbol"] == sym_clean]
        if row.empty:
            st.sidebar.error(f"Symbol '{sym_clean}' not found.")
        else:
            r = row.iloc[0]
            st.sidebar.markdown(f"### {r['symbol']}")
            st.sidebar.markdown(f"**Signal:** {r.get('signal', '—')}")
            st.sidebar.markdown(f"**Confidence:** {r.get('confidence', '—'):.1f}%")
            st.sidebar.markdown(f"**Last Close:** {r.get('last_close', '—'):.2f}")
            st.sidebar.markdown(f"**Sentiment:** {r.get('sentiment_label', '—')}")

    # ── DISCLAIMER ────────────────────────────────────────
    st.divider()
    st.caption("""
    ⚠️ **Disclaimer:** This app is for **educational purposes only**.
    Predictions are based on historical patterns and news sentiment — they are NOT financial advice.
    Always do your own research before making any investment decisions.
    Past performance does not guarantee future results.
    """)


# ──────────────────────────────────────────────────────────────
# STOCK DETAIL COMPONENT
# ──────────────────────────────────────────────────────────────

def render_stock_detail(df: pd.DataFrame, symbol: str, currency: str = "₹"):
    """Renders a detailed card for a single stock."""
    row = df[df["symbol"] == symbol]
    if row.empty:
        st.warning(f"No prediction data for {symbol}")
        return
    r = row.iloc[0]

    dcol1, dcol2, dcol3 = st.columns([2, 2, 3])

    with dcol1:
        st.markdown(f"**Signal:** {r.get('signal', '—')}")
        st.markdown(f"**Direction:** {r.get('direction', '—')}")
        st.markdown(f"**Confidence:** {r.get('confidence', 0):.1f}%")
        st.markdown(f"**Last Close:** {currency}{r.get('last_close', 0):.2f}")
        st.markdown(f"**Est. Range:** {currency}{r.get('est_price_low', 0):.2f} – {currency}{r.get('est_price_high', 0):.2f}")

    with dcol2:
        st.markdown(f"**RSI (14):** {r.get('rsi', '—')}")
        rsi_val = r.get("rsi", 50)
        if rsi_val:
            if rsi_val > 70:
                st.warning("⚠️ RSI > 70 → Overbought")
            elif rsi_val < 30:
                st.success("✅ RSI < 30 → Oversold (potential buy)")
            else:
                st.info(f"RSI in neutral zone ({rsi_val:.0f})")
        st.markdown(f"**MACD:** {r.get('macd', '—')}")
        st.markdown(f"**BB %:** {r.get('bb_percent', '—')}")
        st.markdown(f"**Volume Ratio:** {r.get('volume_ratio', '—')}")

    with dcol3:
        st.markdown(f"**News Sentiment:** {r.get('sentiment_label', '—')}")
        st.markdown(f"**Sentiment Score:** {r.get('sentiment_score', 0):+.3f}")
        st.markdown(f"**Articles Found:** {r.get('article_count', 0)}")
        headline = r.get("top_headline", "")
        if headline:
            st.markdown(f"**Top Headline:** _{headline[:100]}..._")

    # Price chart
    st.markdown(f"#### {symbol} — Price Chart (6 months)")
    chart_df = load_price_chart(symbol)

    if not chart_df.empty:
        fig = go.Figure()

        # Candlestick
        if all(c in chart_df.columns for c in ["open", "high", "low", "close"]):
            fig.add_trace(go.Candlestick(
                x=chart_df.index,
                open=chart_df["open"], high=chart_df["high"],
                low=chart_df["low"],   close=chart_df["close"],
                name="OHLC",
                increasing_line_color="#00e676",
                decreasing_line_color="#ff5252",
            ))

        # SMA lines
        chart_df["sma20"] = chart_df["close"].rolling(20).mean()
        chart_df["sma50"] = chart_df["close"].rolling(50).mean()
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["sma20"],
                                  line=dict(color="#ffd740", width=1),  name="SMA 20"))
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["sma50"],
                                  line=dict(color="#00d4ff", width=1),  name="SMA 50"))

        # Prediction zone (tomorrow)
        tomorrow = chart_df.index[-1] + timedelta(days=1)
        fig.add_hrect(
            y0=r.get("est_price_low", 0), y1=r.get("est_price_high", 0),
            fillcolor="rgba(124,77,255,0.15)", line_width=0,
            annotation_text="Predicted Range", annotation_position="top right"
        )

        fig.update_layout(
            height=400,
            xaxis_rangeslider_visible=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=10, r=10, t=20, b=10),
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Chart data unavailable for this symbol.")


if __name__ == "__main__":
    main()
