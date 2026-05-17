"""
fetcher.py — 資料擷取模組
支援 RSS Feeds、NewsAPI、Reddit (PRAW) 三種資料來源
Alternative Data 亮點：整合 Reddit 社群輿情
"""

import feedparser
import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fetcher")

# ─── 財經 RSS 來源清單 ────────────────────────────────────────────────────────
RSS_SOURCES = {
    "yahoo_finance":  "https://finance.yahoo.com/news/rssindex",
    "reuters_biz":    "https://feeds.reuters.com/reuters/businessNews",
    "cnbc":           "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "seeking_alpha":  "https://seekingalpha.com/feed.xml",
    "moneyudn":       "https://money.udn.com/rssfeed/news/1001/5590",   # 聯合新聞網財經
    "cnyes":          "https://feeds.feedburner.com/cnyes/HOT",          # 鉅亨網熱門
}

# ─── 去重用的 seen-hash 記憶體快取 (重啟後清空；生產環境改用 Redis/DB) ──────
_seen_hashes: set[str] = set()


def _make_hash(title: str, url: str) -> str:
    """以標題+URL 產生 MD5，用於去重"""
    raw = f"{title.strip().lower()}|{url.strip()}"
    return hashlib.md5(raw.encode()).hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ─── 1. RSS Fetcher ───────────────────────────────────────────────────────────

def fetch_rss(source_name: str, url: str, max_items: int = 20) -> list[dict]:
    """
    解析單一 RSS feed，回傳標準化的新聞 list。
    每筆包含：id, source, title, summary, url, published_at, raw_text
    """
    logger.info(f"[RSS] 抓取 {source_name} ...")
    try:
        feed = feedparser.parse(url)
    except Exception as e:
        logger.error(f"[RSS] {source_name} 解析失敗: {e}")
        return []

    articles = []
    for entry in feed.entries[:max_items]:
        title   = getattr(entry, "title",   "") or ""
        summary = getattr(entry, "summary", "") or ""
        link    = getattr(entry, "link",    "") or ""
        pub_raw = getattr(entry, "published", None)

        # 統一時間格式
        if pub_raw:
            try:
                pub_dt = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                published_at = pub_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            except Exception:
                published_at = _now_iso()
        else:
            published_at = _now_iso()

        # 去重
        h = _make_hash(title, link)
        if h in _seen_hashes:
            continue
        _seen_hashes.add(h)

        raw_text = f"{title}. {summary}".strip()

        articles.append({
            "id":           h,
            "source":       source_name,
            "source_type":  "rss",
            "title":        title,
            "summary":      summary,
            "url":          link,
            "published_at": published_at,
            "fetched_at":   _now_iso(),
            "raw_text":     raw_text,
        })

    logger.info(f"[RSS] {source_name} → {len(articles)} 筆新文章")
    return articles


def fetch_all_rss(sources: dict = RSS_SOURCES, max_items: int = 20) -> list[dict]:
    """抓取所有 RSS 來源，合併去重後回傳"""
    all_articles = []
    for name, url in sources.items():
        articles = fetch_rss(name, url, max_items)
        all_articles.extend(articles)
        time.sleep(0.5)   # 友善延遲，避免被 rate-limit
    logger.info(f"[RSS] 共取得 {len(all_articles)} 筆")
    return all_articles


# ─── 2. NewsAPI Fetcher ───────────────────────────────────────────────────────

def fetch_newsapi(
    api_key: str,
    query: str = "stock market OR interest rate OR Fed OR earnings",
    language: str = "en",
    page_size: int = 30,
) -> list[dict]:
    """
    使用 NewsAPI /v2/everything 抓取最新財經新聞。
    免費方案：100 requests/day，延遲最多 24h。
    申請：https://newsapi.org/register
    """
    if not api_key:
        logger.warning("[NewsAPI] 未設定 API Key，略過")
        return []

    url = "https://newsapi.org/v2/everything"
    params = {
        "q":        query,
        "language": language,
        "sortBy":   "publishedAt",
        "pageSize": page_size,
        "apiKey":   api_key,
    }

    logger.info(f"[NewsAPI] 查詢: {query[:50]}...")
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.error(f"[NewsAPI] 請求失敗: {e}")
        return []

    articles = []
    for item in data.get("articles", []):
        title       = item.get("title", "") or ""
        description = item.get("description", "") or ""
        link        = item.get("url", "") or ""
        pub_raw     = item.get("publishedAt", "") or ""
        source_name = item.get("source", {}).get("name", "NewsAPI")

        if title in ("[Removed]", ""):
            continue

        h = _make_hash(title, link)
        if h in _seen_hashes:
            continue
        _seen_hashes.add(h)

        articles.append({
            "id":           h,
            "source":       source_name,
            "source_type":  "newsapi",
            "title":        title,
            "summary":      description,
            "url":          link,
            "published_at": pub_raw or _now_iso(),
            "fetched_at":   _now_iso(),
            "raw_text":     f"{title}. {description}".strip(),
        })

    logger.info(f"[NewsAPI] → {len(articles)} 筆")
    return articles


# ─── 3. Reddit Alternative Data Fetcher ──────────────────────────────────────

def fetch_reddit_posts(
    client_id: str,
    client_secret: str,
    user_agent: str = "financial-monitor/1.0",
    subreddits: list[str] = None,
    limit: int = 25,
) -> list[dict]:
    """
    爬取 Reddit 財經版熱門貼文（Alternative Data 亮點）。
    申請 Reddit API：https://www.reddit.com/prefs/apps
    需要 praw：pip install praw
    """
    if not client_id or not client_secret:
        logger.warning("[Reddit] 未設定 credentials，略過")
        return []

    try:
        import praw
    except ImportError:
        logger.warning("[Reddit] 請先 pip install praw")
        return []

    if subreddits is None:
        subreddits = ["stocks", "investing", "wallstreetbets", "SecurityAnalysis"]

    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )

    articles = []
    for sub_name in subreddits:
        try:
            subreddit = reddit.subreddit(sub_name)
            for post in subreddit.hot(limit=limit):
                title = post.title or ""
                body  = post.selftext or ""
                link  = f"https://reddit.com{post.permalink}"
                score = post.score
                pub_dt = datetime.fromtimestamp(post.created_utc, tz=timezone.utc)

                h = _make_hash(title, link)
                if h in _seen_hashes:
                    continue
                _seen_hashes.add(h)

                articles.append({
                    "id":           h,
                    "source":       f"r/{sub_name}",
                    "source_type":  "reddit",
                    "title":        title,
                    "summary":      body[:500],
                    "url":          link,
                    "published_at": pub_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "fetched_at":   _now_iso(),
                    "raw_text":     f"{title}. {body[:300]}".strip(),
                    "reddit_score": score,    # 額外欄位：社群熱度分數
                })
            logger.info(f"[Reddit] r/{sub_name} → {limit} 筆")
        except Exception as e:
            logger.error(f"[Reddit] r/{sub_name} 失敗: {e}")

    return articles


# ─── 4. 統一入口 ──────────────────────────────────────────────────────────────

def fetch_all(config: dict) -> list[dict]:
    """
    統一擷取所有來源，回傳合併後的文章 list。

    config 範例：
    {
        "newsapi_key":       "YOUR_KEY",
        "reddit_client_id":  "YOUR_ID",
        "reddit_secret":     "YOUR_SECRET",
        "rss_sources":       RSS_SOURCES,       # 可覆蓋預設清單
        "newsapi_query":     "stock market",
        "reddit_subreddits": ["stocks", "investing"],
    }
    """
    all_articles: list[dict] = []

    # RSS
    sources = config.get("rss_sources", RSS_SOURCES)
    all_articles.extend(fetch_all_rss(sources))

    # NewsAPI
    newsapi_key = config.get("newsapi_key", "")
    if newsapi_key:
        query = config.get("newsapi_query", "stock market OR earnings OR Fed")
        all_articles.extend(fetch_newsapi(newsapi_key, query))

    # Reddit
    r_id  = config.get("reddit_client_id", "")
    r_sec = config.get("reddit_secret", "")
    if r_id and r_sec:
        subs = config.get("reddit_subreddits", ["stocks", "investing", "wallstreetbets"])
        all_articles.extend(fetch_reddit_posts(r_id, r_sec, subreddits=subs))

    logger.info(f"[fetch_all] 總計 {len(all_articles)} 筆（已去重）")
    return all_articles


# ─── 快速測試 ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 只跑 RSS（不需要 API Key）
    articles = fetch_all_rss(max_items=5)
    for a in articles[:3]:
        print(f"\n[{a['source']}] {a['title'][:80]}")
        print(f"  URL: {a['url'][:60]}...")
        print(f"  Published: {a['published_at']}")
    print(f"\n共 {len(articles)} 筆文章")
