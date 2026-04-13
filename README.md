# 📈 AI Stock Predictor — India & Crypto

> Predicts NSE stocks & crypto daily using Machine Learning + News Sentiment  
> 100% free. Auto-updates every morning. Deployed on Streamlit Cloud.

---

## 🔧 What You Need (All Free)

| Thing | Where to get it |
|---|---|
| GitHub account | Already have it ✅ |
| NewsAPI key | Already have it ✅ |
| Streamlit account | streamlit.io/cloud (sign in with GitHub) |

---

## 🚀 Setup — Step by Step

### STEP 1 — Create your GitHub repo

1. Go to **github.com** → click **"New repository"**
2. Name it: `stock-predictor`
3. Set it to **Public**
4. Click **"Create repository"**

---

### STEP 2 — Upload all project files

Upload every file from this project keeping the same folder structure:
```
stock-predictor/
├── app.py
├── update_data.py
├── config.py
├── requirements.txt
├── data/
│   ├── __init__.py
│   ├── fetch_stocks.py
│   └── fetch_news.py
├── sentiment/
│   ├── __init__.py
│   └── analyzer.py
├── model/
│   ├── __init__.py
│   └── predictor.py
├── predictions/          ← empty folder, just upload the .gitkeep file
└── .github/
    └── workflows/
        └── daily_update.yml
```

---

### STEP 3 — Add your NewsAPI key as a GitHub Secret

1. In your GitHub repo → click **Settings**
2. Left sidebar → **Secrets and variables** → **Actions**
3. Click **"New repository secret"**
4. Name: `NEWS_API_KEY`
5. Value: paste your NewsAPI key
6. Click **"Add secret"** ✅

---

### STEP 4 — Run the first update manually

1. In your repo → click the **Actions** tab
2. Click **"Daily Stock Prediction Update"** in the left panel
3. Click **"Run workflow"** → **"Run workflow"** (green button)
4. ⏳ Wait 20–30 minutes (first run downloads 2 years of data for 200+ stocks)
5. When it finishes, you'll see a new commit in your repo with the predictions CSV ✅

---

### STEP 5 — Deploy on Streamlit Cloud

1. Go to **streamlit.io/cloud** → sign in with GitHub
2. Click **"New app"**
3. Fill in:
   - **Repository:** your-github-username/stock-predictor
   - **Branch:** main
   - **Main file path:** `app.py`
4. Click **"Advanced settings"** → **"Secrets"** tab
5. Paste this (replacing with your real key):
   ```
   NEWS_API_KEY = "your_actual_newsapi_key_here"
   ```
6. Click **"Deploy"** 🎉

Your app will be live at: `https://your-app-name.streamlit.app`

---

### STEP 6 — Done! Daily auto-updates

Every weekday at **6:30 AM IST**, GitHub Actions will:
1. Fetch fresh price data from Yahoo Finance
2. Pull overnight news from 10+ RSS feeds + NewsAPI
3. Run sentiment analysis on every article
4. Train ML models and generate predictions
5. Commit updated CSV → Streamlit auto-shows fresh data

---

## 📊 News Sources Used

| Source | Type | Cost |
|---|---|---|
| Moneycontrol | RSS Feed | Free |
| Economic Times Markets | RSS Feed | Free |
| Business Standard | RSS Feed | Free |
| Google News (NSE/BSE) | RSS Feed | Free |
| Google News (Crypto) | RSS Feed | Free |
| NewsAPI | API (your key) | Free (100 req/day) |

---

## 🆓 Free Tier Usage Summary

| Service | Free Limit | Our Daily Usage |
|---|---|---|
| Yahoo Finance (yfinance) | Unlimited | ~210 calls |
| RSS Feeds | Unlimited | ~400 articles |
| NewsAPI | 100 req/day | 3 req/day ✅ |
| GitHub Actions | 2000 min/month | ~25 min/day ✅ |
| Streamlit Cloud | 1 free app | 1 app ✅ |

---

## ❓ FAQ

**Q: What if GitHub Actions fails?**  
A: Go to Actions tab → click the failed run → read the error log. Most common issue is a timeout — just re-run it manually.

**Q: Can I add more stocks?**  
A: Yes — edit `config.py` and add Yahoo Finance symbols to `NSE_STOCKS`. Indian stocks use `.NS` suffix (e.g. `TATASTEEL.NS`).

**Q: How accurate are predictions?**  
A: Typically 52–62% accuracy per stock (shown in the dashboard). Use as one signal among many — not as financial advice.

**Q: My NewsAPI shows 0 articles?**  
A: Free tier has a 1-month lookback limit and 100 req/day cap. RSS feeds will still work even if NewsAPI hits its limit.

---

⚠️ **Disclaimer:** Educational purposes only. Not financial advice. Always do your own research.

*Built with: Python · Streamlit · scikit-learn · yfinance · VADER · NewsAPI · GitHub Actions*
