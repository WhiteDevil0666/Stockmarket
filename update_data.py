"""
update_data.py
--------------
THE MAIN DAILY SCRIPT — runs every morning via GitHub Actions.

What it does (in order):
  1. Downloads fresh stock & crypto data (yfinance)
  2. Fetches overnight news (RSS + NewsAPI + Reddit)
  3. Scores news sentiment per stock
  4. Trains models and generates predictions
  5. Saves predictions/latest_predictions.csv
  6. Saves predictions/history/YYYY-MM-DD.csv  (for backtesting later)

GitHub Actions triggers this at 6:30 AM IST (1:00 AM UTC) every weekday.

You can also run it manually:
    python update_data.py
"""

import os
import sys
import pandas as pd
from datetime import datetime

# Make imports work regardless of where script is run from
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import NSE_STOCKS, CRYPTO_SYMBOLS, DATA_DIR
from data.fetch_stocks  import fetch_all_stocks
from data.fetch_news    import fetch_all_news
from sentiment.analyzer import compute_stock_sentiment, compute_market_sentiment
from model.predictor    import run_predictions


def run_daily_update():
    start_time = datetime.now()
    today      = start_time.strftime("%Y-%m-%d")
    print(f"\n{'='*60}")
    print(f"  DAILY STOCK PREDICTOR UPDATE — {today}")
    print(f"{'='*60}\n")

    # ── Step 1: Fetch stock data ───────────────────────────
    print("STEP 1/5 — Fetching stock & crypto data from Yahoo Finance")
    nse_data    = fetch_all_stocks(NSE_STOCKS,    label="NSE")
    crypto_data = fetch_all_stocks(CRYPTO_SYMBOLS, label="Crypto")
    all_stock_data = {**nse_data, **crypto_data}

    if not all_stock_data:
        print("❌ No stock data fetched. Aborting.")
        sys.exit(1)

    # ── Step 2: Fetch news ─────────────────────────────────
    print("\nSTEP 2/5 — Fetching news & social media")
    news_df = fetch_all_news()

    # ── Step 3: Sentiment analysis ─────────────────────────
    print("\nSTEP 3/5 — Running sentiment analysis")
    if not news_df.empty:
        sentiment_df = compute_stock_sentiment(news_df)
        market_sent  = compute_market_sentiment(news_df)
        print(f"  Overall market sentiment: {market_sent['label']} (score: {market_sent['score']:+.3f})")
        print(f"  Based on {market_sent['total_articles']} articles")
    else:
        sentiment_df = pd.DataFrame()
        market_sent  = {"score": 0.0, "label": "Neutral ⚪", "total_articles": 0}
        print("  ⚠ No news data — using neutral sentiment for all stocks")

    # ── Step 4: Run ML predictions ─────────────────────────
    print("\nSTEP 4/5 — Training models & generating predictions")
    predictions_df = run_predictions(all_stock_data, sentiment_df)

    if predictions_df.empty:
        print("❌ No predictions generated. Aborting.")
        sys.exit(1)

    # Add market type column
    nse_syms    = set(NSE_STOCKS)
    crypto_syms = set(CRYPTO_SYMBOLS)
    predictions_df["market"] = predictions_df["symbol"].apply(
        lambda s: "NSE" if s in nse_syms else ("Crypto" if s in crypto_syms else "Other")
    )
    predictions_df["date"] = today

    # ── Step 5: Save results ───────────────────────────────
    print("\nSTEP 5/5 — Saving results")
    os.makedirs(DATA_DIR, exist_ok=True)
    history_dir = os.path.join(DATA_DIR, "history")
    os.makedirs(history_dir, exist_ok=True)

    # Latest predictions (the app always reads this file)
    latest_path = os.path.join(DATA_DIR, "latest_predictions.csv")
    predictions_df.to_csv(latest_path, index=False)
    print(f"  ✓ Saved: {latest_path}  ({len(predictions_df)} rows)")

    # Daily snapshot for history
    history_path = os.path.join(history_dir, f"{today}.csv")
    predictions_df.to_csv(history_path, index=False)
    print(f"  ✓ Saved: {history_path}")

    # Save market sentiment summary
    market_sent["date"] = today
    sent_path = os.path.join(DATA_DIR, "market_sentiment.csv")
    pd.DataFrame([market_sent]).to_csv(sent_path, index=False)
    print(f"  ✓ Saved: {sent_path}")

    # Summary
    elapsed = (datetime.now() - start_time).seconds
    buy_count  = (predictions_df["signal"].str.startswith("BUY")).sum()
    sell_count = (predictions_df["signal"].str.startswith("SELL")).sum()
    hold_count = (predictions_df["signal"].str.startswith("HOLD")).sum()

    print(f"""
{'='*60}
  UPDATE COMPLETE in {elapsed}s
  {len(predictions_df)} stocks processed
  BUY signals:  {buy_count}
  SELL signals: {sell_count}
  HOLD signals: {hold_count}
{'='*60}
""")


if __name__ == "__main__":
    run_daily_update()
