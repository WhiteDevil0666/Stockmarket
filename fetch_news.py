"""
data/fetch_news.py
------------------
Fetches financial news from 2 FREE sources:
  1. RSS Feeds   — Moneycontrol, Economic Times, Business Standard, Google News
  2. NewsAPI     — requires your free API key (100 requests/day on free tier)

Reddit removed. No other paid APIs used.
"""

import os, sys, feedparser, requests, pandas as pd
from datetime import datetime, timedelta
from time import sleep

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import NEWS_API_KEY, RSS_FEEDS, NSE_STOCKS, CRYPTO_SYMBOLS


# ── 1. RSS FEEDS (totally free, no login) ────────────────────

def fetch_rss_news(max_per_feed: int = 40) -> list:
    articles = []
    for url in RSS_FEEDS:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:max_per_feed]:
                articles.append({
                    "title":     entry.get("title", ""),
                    "summary":   entry.get("summary", ""),
                    "url":       entry.get("link", ""),
                    "source":    feed.feed.get("title", "RSS"),
                    "published": entry.get("published", ""),
                })
        except Exception as e:
            print(f"  ⚠ RSS error: {e}")
    print(f"  📰 RSS feeds: {len(articles)} articles")
    return articles


# ── 2. NEWSAPI (free key — 100 req/day) ──────────────────────

def fetch_newsapi(query: str, days_back: int = 1) -> list:
    if not NEWS_API_KEY:
        print("  ⚠ NEWS_API_KEY not set — skipping NewsAPI")
        return []

    from_date = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        r = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": query, "from": from_date,
                "sortBy": "publishedAt", "language": "en",
                "pageSize": 100, "apiKey": NEWS_API_KEY,
            },
            timeout=10,
        )
        data = r.json()
        if data.get("status") != "ok":
            print(f"  ⚠ NewsAPI: {data.get('message','unknown error')}")
            return []
        articles = []
        for a in data.get("articles", []):
            articles.append({
                "title":     a.get("title", ""),
                "summary":   a.get("description", ""),
                "url":       a.get("url", ""),
                "source":    a.get("source", {}).get("name", "NewsAPI"),
                "published": a.get("publishedAt", ""),
            })
        print(f"  📰 NewsAPI '{query[:40]}': {len(articles)} articles")
        return articles
    except Exception as e:
        print(f"  ✗ NewsAPI failed: {e}")
        return []


def fetch_all_newsapi() -> list:
    """3 queries = 3 of your 100 free daily NewsAPI requests."""
    queries = [
        "NSE BSE Nifty Sensex India stock market",
        "Indian economy RBI inflation GDP budget",
        "Bitcoin Ethereum cryptocurrency India",
    ]
    all_articles = []
    for q in queries:
        all_articles.extend(fetch_newsapi(q, days_back=1))
        sleep(0.3)
    return all_articles


# ── TAG articles with stock symbols ──────────────────────────

def tag_article_with_stocks(article: dict, all_symbols: list) -> list:
    text = (article["title"] + " " + article["summary"]).upper()
    matched = []
    for sym in all_symbols:
        ticker = sym.split(".")[0].split("-")[0]
        if len(ticker) >= 3 and ticker in text:
            matched.append(sym)
    return matched


# ── MASTER fetch function ─────────────────────────────────────

def fetch_all_news() -> pd.DataFrame:
    print("\n📡 Fetching news (RSS + NewsAPI)...")
    all_articles = []
    all_articles.extend(fetch_rss_news())
    all_articles.extend(fetch_all_newsapi())

    if not all_articles:
        print("  ⚠ No articles fetched!")
        return pd.DataFrame()

    df = pd.DataFrame(all_articles)
    df = df.drop_duplicates(subset=["title"])
    df["title"]   = df["title"].fillna("")
    df["summary"] = df["summary"].fillna("")
    df["text"]    = df["title"] + ". " + df["summary"]

    all_symbols = NSE_STOCKS + CRYPTO_SYMBOLS
    df["mentioned_symbols"] = df.apply(
        lambda row: tag_article_with_stocks(row, all_symbols), axis=1
    )
    print(f"\n✅ Total unique articles: {len(df)}")
    return df


if __name__ == "__main__":
    df = fetch_all_news()
    print(df[["title", "source", "mentioned_symbols"]].head(10).to_string())
