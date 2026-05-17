"""
scheduler.py — 主控排程
整合 fetcher → sentiment → signal_engine → notifier 的完整 Pipeline
APScheduler 定時執行，支援每 N 分鐘抓取 + 每日彙整推播
"""

import logging
import os
from pathlib import Path

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from fetcher import fetch_all
from notifier import NotificationManager
from sentiment import SentimentAnalyzer
from signal_engine import init_db, process_articles, save_articles, save_signals

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("scheduler")

# ─── 從環境變數讀取 API Keys（不要 hardcode！）────────────────────────────
CONFIG = {
    # RSS 使用預設清單，無需 Key
    "newsapi_key":        os.getenv("NEWSAPI_KEY", ""),
    "reddit_client_id":   os.getenv("REDDIT_CLIENT_ID", ""),
    "reddit_secret":      os.getenv("REDDIT_SECRET", ""),
    "reddit_subreddits":  ["stocks", "investing", "wallstreetbets"],
    "newsapi_query":      "stock market OR earnings OR Fed OR inflation OR NVDA OR TSLA",
}

NOTIFY_CONFIG = {
    "line_token":         os.getenv("LINE_NOTIFY_TOKEN", ""),
    "telegram_token":     os.getenv("TELEGRAM_BOT_TOKEN", ""),
    "telegram_chat_id":   os.getenv("TELEGRAM_CHAT_ID", ""),
    "min_confidence":     0.4,
    "only_signals":       ["BUY", "SELL", "WATCH"],
}


# ─── Pipeline 核心函數 ────────────────────────────────────────────────────────

_analyzer = SentimentAnalyzer(prefer="finbert")   # 全域重用，避免重複載入模型
_notifier = NotificationManager(**NOTIFY_CONFIG)
_daily_signals: list = []                          # 當天訊號暫存，供每日彙整


def run_pipeline():
    """完整 Pipeline：抓取 → NLP → 訊號 → 推播"""
    global _daily_signals
    logger.info("=" * 50)
    logger.info("▶ Pipeline 啟動")

    # Step 1: 抓取新聞
    articles = fetch_all(CONFIG)
    if not articles:
        logger.info("▷ 無新文章，略過")
        return

    # Step 2: 情緒分析
    enriched = _analyzer.analyze_articles(articles)

    # Step 3: 生成訊號
    signals = process_articles(enriched)

    # Step 4: 儲存到 SQLite
    save_articles(enriched)
    save_signals(signals)

    # Step 5: 推播
    pushed = _notifier.push_batch(signals)

    # 累積每日訊號（供晚間彙整）
    _daily_signals.extend(signals)

    logger.info(
        f"▶ Pipeline 完成：{len(articles)} 篇 → "
        f"{len(signals)} 訊號 → {pushed} 則推播"
    )
    logger.info("=" * 50)


def run_daily_summary():
    """每日 22:00 推播當天訊號彙整並清空暫存"""
    global _daily_signals
    logger.info("[Daily] 傳送每日彙整...")
    _notifier.push_daily_summary(_daily_signals)
    _daily_signals = []
    logger.info("[Daily] 彙整完成，訊號暫存已清空")


# ─── 啟動排程 ─────────────────────────────────────────────────────────────────

def main():
    init_db()
    run_pipeline()   # 啟動時先跑一次

    scheduler = BlockingScheduler(timezone="Asia/Taipei")

    # 每 30 分鐘抓取一次（交易時段可調短為 15 分鐘）
    scheduler.add_job(
        run_pipeline,
        trigger=IntervalTrigger(minutes=30),
        id="pipeline",
        name="News Pipeline",
        replace_existing=True,
    )

    # 每日 22:00 推播彙整
    scheduler.add_job(
        run_daily_summary,
        trigger=CronTrigger(hour=22, minute=0),
        id="daily_summary",
        name="Daily Summary",
        replace_existing=True,
    )

    logger.info("🚀 Scheduler 啟動，每 30 分鐘執行一次 Pipeline")
    logger.info("   每日 22:00 推播彙整報告")
    logger.info("   Ctrl+C 停止")

    try:
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("⏹ Scheduler 已停止")


if __name__ == "__main__":
    main()
