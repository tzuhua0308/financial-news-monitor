"""
signal_engine.py — 訊號生成模組
將情緒分數 + 來源可信度 + 提及頻率 → 交易訊號
面試亮點：展示你能把 NLP 輸出轉成可執行的量化訊號
"""

import logging
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("signal_engine")

# ─── 來源可信度權重（可自行調整）────────────────────────────────────────────
SOURCE_WEIGHTS = {
    "reuters_biz":    1.0,
    "cnbc":           0.9,
    "yahoo_finance":  0.8,
    "seeking_alpha":  0.75,
    "moneyudn":       0.7,
    "cnyes":          0.7,
    # NewsAPI 來源
    "Bloomberg":      1.0,
    "Financial Times":1.0,
    "The Wall Street Journal": 1.0,
    # Reddit（社群輿情，權重較低但有參考價值）
    "r/stocks":       0.4,
    "r/investing":    0.45,
    "r/wallstreetbets": 0.3,
    "r/SecurityAnalysis": 0.55,
}
DEFAULT_WEIGHT = 0.6   # 未知來源的預設權重

# ─── 關鍵字清單（可從 config/keywords.yaml 載入）──────────────────────────
WATCHLIST_KEYWORDS = {
    "macro": ["Fed", "FOMC", "interest rate", "inflation", "CPI", "GDP",
              "recession", "quantitative", "ECB", "BOJ", "central bank"],
    "equity": ["earnings", "EPS", "revenue", "guidance", "upgrade", "downgrade",
               "buyback", "dividend", "IPO", "merger", "acquisition"],
    "risk": ["bankruptcy", "default", "recall", "fraud", "investigation",
             "layoff", "warning", "crash", "selloff"],
}

# ─── 股票代號簡易 NER（可替換為 spaCy）──────────────────────────────────────
TICKER_PATTERN = re.compile(r"\b([A-Z]{2,5})\b")
COMMON_TICKERS = {
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM",
    "GS", "BAC", "AMD", "INTC", "TSM", "ASML", "BABA", "2330",
}


def extract_tickers(text: str) -> list[str]:
    """從文字中找出疑似股票代號"""
    found = TICKER_PATTERN.findall(text)
    return list({t for t in found if t in COMMON_TICKERS})


def extract_themes(text: str) -> list[str]:
    """比對關鍵字，回傳命中的主題"""
    text_lower = text.lower()
    themes = []
    for theme, keywords in WATCHLIST_KEYWORDS.items():
        if any(kw.lower() in text_lower for kw in keywords):
            themes.append(theme)
    return themes


# ─── 訊號資料結構 ─────────────────────────────────────────────────────────────

@dataclass
class TradingSignal:
    signal_id:       str
    article_id:      str
    generated_at:    str
    source:          str
    title:           str
    url:             str
    tickers:         list[str]
    themes:          list[str]

    # 情緒欄位
    raw_compound:    float   # FinBERT/VADER compound (-1 → +1)
    weighted_score:  float   # 來源可信度加權後
    sentiment_label: str

    # 訊號決策
    signal:          str     # BUY / SELL / WATCH / IGNORE
    confidence:      float   # 0–1，訊號可信度
    reason:          str     # 人類可讀的決策理由


def generate_signal(article: dict) -> Optional[TradingSignal]:
    """
    核心訊號生成邏輯。
    輸入：來自 sentiment.analyze_articles() 的單篇文章 dict
    輸出：TradingSignal 或 None（若文章不值得關注）
    """
    sentiment = article.get("sentiment", {})
    compound  = sentiment.get("compound", 0.0)
    label     = sentiment.get("label", "neutral")
    source    = article.get("source", "unknown")

    # 1. 來源可信度加權
    weight = SOURCE_WEIGHTS.get(source, DEFAULT_WEIGHT)
    weighted = round(compound * weight, 4)

    # 2. 提取股票代號與主題
    raw_text = article.get("raw_text", "")
    tickers  = extract_tickers(article.get("title", "") + " " + raw_text)
    themes   = extract_themes(raw_text)

    # 3. 訊號規則（可依需求調整閾值）
    #    |weighted| >= 0.3  → 強訊號
    #    |weighted| >= 0.15 → 弱訊號（WATCH）
    #    < 0.15             → 忽略
    abs_w = abs(weighted)

    if abs_w >= 0.3:
        if weighted > 0:
            signal, confidence = "BUY",  round(min(abs_w * 1.5, 1.0), 2)
        else:
            signal, confidence = "SELL", round(min(abs_w * 1.5, 1.0), 2)
        reason = (
            f"強{'正面' if weighted>0 else '負面'}情緒 (加權分={weighted:+.3f})，"
            f"來源權重={weight}，主題={themes or ['general']}"
        )
    elif abs_w >= 0.15:
        signal     = "WATCH"
        confidence = round(abs_w * 2, 2)
        reason     = (
            f"中性偏{'正' if weighted>0 else '負'} (加權分={weighted:+.3f})，"
            f"建議持續觀察，主題={themes or ['general']}"
        )
    else:
        return None   # 情緒不顯著，不產生訊號

    import uuid
    signal_id = uuid.uuid4().hex[:12]

    return TradingSignal(
        signal_id      = signal_id,
        article_id     = article.get("id", ""),
        generated_at   = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        source         = source,
        title          = article.get("title", ""),
        url            = article.get("url", ""),
        tickers        = tickers,
        themes         = themes,
        raw_compound   = compound,
        weighted_score = weighted,
        sentiment_label= label,
        signal         = signal,
        confidence     = confidence,
        reason         = reason,
    )


def process_articles(enriched_articles: list[dict]) -> list[TradingSignal]:
    """批次處理，過濾出有效訊號"""
    signals = []
    for article in enriched_articles:
        sig = generate_signal(article)
        if sig:
            signals.append(sig)
            logger.info(
                f"[Signal] {sig.signal:4s} | {sig.weighted_score:+.3f} | "
                f"tickers={sig.tickers} | {sig.title[:50]}"
            )
    logger.info(f"[Signal] {len(enriched_articles)} 篇 → {len(signals)} 個有效訊號")
    return signals


# ─── SQLite 持久化 ────────────────────────────────────────────────────────────

DB_PATH = Path(__file__).parent.parent / "data" / "signals.db"


def init_db(db_path: Path = DB_PATH):
    """初始化資料庫 schema"""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            signal_id       TEXT PRIMARY KEY,
            article_id      TEXT,
            generated_at    TEXT,
            source          TEXT,
            title           TEXT,
            url             TEXT,
            tickers         TEXT,
            themes          TEXT,
            raw_compound    REAL,
            weighted_score  REAL,
            sentiment_label TEXT,
            signal          TEXT,
            confidence      REAL,
            reason          TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            id              TEXT PRIMARY KEY,
            source          TEXT,
            source_type     TEXT,
            title           TEXT,
            url             TEXT,
            published_at    TEXT,
            fetched_at      TEXT,
            sentiment_label TEXT,
            compound        REAL
        )
    """)
    conn.commit()
    conn.close()
    logger.info(f"[DB] 初始化完成：{db_path}")


def save_signals(signals: list[TradingSignal], db_path: Path = DB_PATH):
    """將訊號批次寫入 SQLite"""
    if not signals:
        return
    conn = sqlite3.connect(db_path)
    conn.executemany("""
        INSERT OR IGNORE INTO signals
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, [
        (s.signal_id, s.article_id, s.generated_at,
         s.source, s.title, s.url,
         ",".join(s.tickers), ",".join(s.themes),
         s.raw_compound, s.weighted_score, s.sentiment_label,
         s.signal, s.confidence, s.reason)
        for s in signals
    ])
    conn.commit()
    conn.close()
    logger.info(f"[DB] 已儲存 {len(signals)} 筆訊號")


def save_articles(enriched_articles: list[dict], db_path: Path = DB_PATH):
    """將文章（含情緒）寫入 SQLite"""
    conn = sqlite3.connect(db_path)
    conn.executemany("""
        INSERT OR IGNORE INTO articles
        VALUES (?,?,?,?,?,?,?,?,?)
    """, [
        (a["id"], a["source"], a["source_type"],
         a["title"], a["url"],
         a["published_at"], a["fetched_at"],
         a["sentiment"]["label"], a["sentiment"]["compound"])
        for a in enriched_articles
    ])
    conn.commit()
    conn.close()


if __name__ == "__main__":
    # 用假資料跑完整 signal_engine 流程測試
    mock_enriched = [
        {
            "id": "abc1", "source": "reuters_biz", "source_type": "rss",
            "title": "Fed signals aggressive rate hike, markets plunge",
            "raw_text": "Federal Reserve signals aggressive rate hike as inflation soars. Markets crash.",
            "url": "https://example.com/1", "published_at": "2025-01-01T00:00:00Z", "fetched_at": "2025-01-01T00:00:00Z",
            "summary": "",
            "sentiment": {"label": "negative", "score": 0.93, "compound": -0.82,
                          "positive": 0.02, "neutral": 0.05, "negative": 0.93, "model": "vader"},
        },
        {
            "id": "abc2", "source": "cnbc", "source_type": "rss",
            "title": "NVDA beats earnings, raises guidance; stock up 8%",
            "raw_text": "NVIDIA beats Q4 earnings estimates and raises forward guidance. Stock surges 8% after hours.",
            "url": "https://example.com/2", "published_at": "2025-01-01T00:00:00Z", "fetched_at": "2025-01-01T00:00:00Z",
            "summary": "",
            "sentiment": {"label": "positive", "score": 0.95, "compound": 0.87,
                          "positive": 0.95, "neutral": 0.04, "negative": 0.01, "model": "vader"},
        },
        {
            "id": "abc3", "source": "r/stocks", "source_type": "reddit",
            "title": "What do you think about AAPL's upcoming earnings?",
            "raw_text": "AAPL earnings coming up next week. Expectations are mixed.",
            "url": "https://reddit.com/r/stocks/3", "published_at": "2025-01-01T00:00:00Z", "fetched_at": "2025-01-01T00:00:00Z",
            "summary": "",
            "sentiment": {"label": "neutral", "score": 0.6, "compound": 0.05,
                          "positive": 0.2, "neutral": 0.6, "negative": 0.2, "model": "vader"},
        },
    ]

    init_db()
    signals = process_articles(mock_enriched)
    save_signals(signals)
    save_articles(mock_enriched)

    print(f"\n=== 生成 {len(signals)} 個訊號 ===")
    for s in signals:
        print(f"\n  訊號：{s.signal:4s}  信心：{s.confidence:.0%}  加權分：{s.weighted_score:+.3f}")
        print(f"  股票：{s.tickers or '未辨識'}  主題：{s.themes}")
        print(f"  標題：{s.title[:60]}")
        print(f"  原因：{s.reason}")
