"""
notifier.py — 自動推播模組
支援 LINE Notify 和 Telegram Bot
面試亮點：端對端的自動化，展示系統整合能力
"""

import logging
from datetime import datetime, timezone

import requests

logger = logging.getLogger("notifier")

SIGNAL_EMOJI = {"BUY": "🟢", "SELL": "🔴", "WATCH": "🟡", "IGNORE": "⚪"}


def _format_message(signal) -> str:
    """將 TradingSignal 格式化成推播訊息"""
    emoji  = SIGNAL_EMOJI.get(signal.signal, "⚪")
    tickers = ", ".join(signal.tickers) if signal.tickers else "N/A"
    themes  = ", ".join(signal.themes)  if signal.themes  else "general"

    return (
        f"{emoji} [{signal.signal}] 財經情緒訊號\n"
        f"──────────────────\n"
        f"📰 {signal.title[:80]}\n"
        f"📊 情緒分數：{signal.weighted_score:+.3f}  信心：{signal.confidence:.0%}\n"
        f"🏷️  股票：{tickers}  |  主題：{themes}\n"
        f"🔗 {signal.url[:60]}{'...' if len(signal.url)>60 else ''}\n"
        f"🕐 {signal.generated_at}\n"
        f"💡 {signal.reason[:100]}"
    )


# ─── LINE Notify ──────────────────────────────────────────────────────────────

class LINENotifier:
    """
    LINE Notify 推播（台灣面試官最熟悉）。
    申請 Token：https://notify-bot.line.me/zh_TW/
    免費，每日上限 1000 則。
    """

    API_URL = "https://notify-api.line.me/api/notify"

    def __init__(self, token: str):
        self.token   = token
        self.headers = {"Authorization": f"Bearer {token}"}

    def send(self, message: str) -> bool:
        if not self.token:
            logger.warning("[LINE] 未設定 Token")
            return False
        try:
            resp = requests.post(
                self.API_URL,
                headers=self.headers,
                data={"message": f"\n{message}"},
                timeout=10,
            )
            if resp.status_code == 200:
                logger.info("[LINE] 推播成功")
                return True
            logger.error(f"[LINE] 推播失敗 HTTP {resp.status_code}: {resp.text}")
            return False
        except Exception as e:
            logger.error(f"[LINE] 推播例外：{e}")
            return False

    def send_signal(self, signal) -> bool:
        return self.send(_format_message(signal))

    def send_daily_summary(self, signals: list) -> bool:
        """每日彙整推播"""
        if not signals:
            return self.send("📋 今日無顯著財經情緒訊號")

        buy  = sum(1 for s in signals if s.signal == "BUY")
        sell = sum(1 for s in signals if s.signal == "SELL")
        watch= sum(1 for s in signals if s.signal == "WATCH")
        avg  = sum(s.weighted_score for s in signals) / len(signals)

        tickers_all = []
        for s in signals:
            tickers_all.extend(s.tickers)
        top_tickers = ", ".join(
            sorted(set(tickers_all), key=lambda t: tickers_all.count(t), reverse=True)[:5]
        ) or "N/A"

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        msg = (
            f"📊 每日情緒彙整 {today}\n"
            f"──────────────────\n"
            f"🟢 BUY 訊號：{buy}   🔴 SELL：{sell}   🟡 WATCH：{watch}\n"
            f"📈 平均加權情緒分：{avg:+.3f}\n"
            f"🏷️  熱門股票：{top_tickers}\n"
            f"──────────────────\n"
            f"強烈訊號（TOP 3）：\n"
        )
        top3 = sorted(signals, key=lambda s: abs(s.weighted_score), reverse=True)[:3]
        for i, s in enumerate(top3, 1):
            emoji = SIGNAL_EMOJI.get(s.signal, "⚪")
            msg += f"{i}. {emoji}{s.signal} {s.weighted_score:+.3f} {s.title[:40]}\n"

        return self.send(msg)


# ─── Telegram Bot ─────────────────────────────────────────────────────────────

class TelegramNotifier:
    """
    Telegram Bot 推播。
    1. BotFather 建立 Bot 取得 token
    2. 對 Bot 傳訊息後訪問 https://api.telegram.org/bot<TOKEN>/getUpdates 取得 chat_id
    pip install python-telegram-bot  （或直接用 requests）
    """

    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id   = chat_id
        self.api_url   = f"https://api.telegram.org/bot{bot_token}"

    def send(self, message: str, parse_mode: str = "HTML") -> bool:
        if not self.bot_token or not self.chat_id:
            logger.warning("[Telegram] 未設定 token 或 chat_id")
            return False
        try:
            resp = requests.post(
                f"{self.api_url}/sendMessage",
                json={
                    "chat_id":    self.chat_id,
                    "text":       message,
                    "parse_mode": parse_mode,
                },
                timeout=10,
            )
            if resp.ok:
                logger.info("[Telegram] 推播成功")
                return True
            logger.error(f"[Telegram] 推播失敗：{resp.text}")
            return False
        except Exception as e:
            logger.error(f"[Telegram] 推播例外：{e}")
            return False

    def send_signal(self, signal) -> bool:
        msg = _format_message(signal)
        # Telegram 不支援部分特殊字元，轉為 HTML 標籤
        msg = msg.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return self.send(f"<pre>{msg}</pre>")

    def send_daily_summary(self, signals: list) -> bool:
        if not signals:
            return self.send("📋 今日無顯著財經情緒訊號")
        avg = sum(s.weighted_score for s in signals) / len(signals)
        buy  = sum(1 for s in signals if s.signal == "BUY")
        sell = sum(1 for s in signals if s.signal == "SELL")
        watch= sum(1 for s in signals if s.signal == "WATCH")
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        msg = (
            f"<b>📊 每日情緒彙整 {today}</b>\n"
            f"🟢 BUY: {buy}  🔴 SELL: {sell}  🟡 WATCH: {watch}\n"
            f"平均情緒分：<code>{avg:+.3f}</code>\n\n"
        )
        for s in sorted(signals, key=lambda x: abs(x.weighted_score), reverse=True)[:5]:
            emoji = SIGNAL_EMOJI.get(s.signal, "⚪")
            msg += f"{emoji} <b>{s.signal}</b> {s.weighted_score:+.3f} — {s.title[:50]}\n"
        return self.send(msg, parse_mode="HTML")


# ─── 統一推播管理器 ───────────────────────────────────────────────────────────

class NotificationManager:
    """
    同時管理多個推播渠道。
    只推播高信心訊號（confidence >= min_confidence）。
    """

    def __init__(
        self,
        line_token:      str = "",
        telegram_token:  str = "",
        telegram_chat_id:str = "",
        min_confidence:  float = 0.4,
        only_signals:    list[str] = None,  # 只推 BUY/SELL，不推 WATCH
    ):
        self.notifiers       = []
        self.min_confidence  = min_confidence
        self.only_signals    = only_signals or ["BUY", "SELL", "WATCH"]

        if line_token:
            self.notifiers.append(LINENotifier(line_token))
        if telegram_token and telegram_chat_id:
            self.notifiers.append(TelegramNotifier(telegram_token, telegram_chat_id))

        if not self.notifiers:
            logger.warning("[Notifier] 未設定任何推播渠道（LINE/Telegram token 為空）")

    def push_signal(self, signal) -> bool:
        """推播單一訊號"""
        if signal.signal not in self.only_signals:
            return False
        if signal.confidence < self.min_confidence:
            logger.debug(f"[Notifier] 信心不足 {signal.confidence:.0%}，略過")
            return False

        success = True
        for notifier in self.notifiers:
            ok = notifier.send_signal(signal)
            success = success and ok
        return success

    def push_batch(self, signals: list) -> int:
        """批次推播，回傳成功推播數"""
        pushed = 0
        for signal in signals:
            if self.push_signal(signal):
                pushed += 1
        logger.info(f"[Notifier] 批次推播 {pushed}/{len(signals)} 筆")
        return pushed

    def push_daily_summary(self, signals: list):
        """推播每日彙整"""
        for notifier in self.notifiers:
            notifier.send_daily_summary(signals)


if __name__ == "__main__":
    # 乾跑測試（不需要真實 token，只印出訊息格式）
    from signal_engine import TradingSignal

    mock_signal = TradingSignal(
        signal_id="test001",
        article_id="abc1",
        generated_at="2025-05-17T08:00:00Z",
        source="reuters_biz",
        title="Fed signals aggressive rate hike, global markets plunge sharply",
        url="https://reuters.com/example",
        tickers=["SPY", "QQQ"],
        themes=["macro", "risk"],
        raw_compound=-0.82,
        weighted_score=-0.82,
        sentiment_label="negative",
        signal="SELL",
        confidence=0.85,
        reason="強負面情緒 (加權分=-0.820)，來源權重=1.0，主題=['macro', 'risk']",
    )

    print("=== 推播訊息預覽 ===\n")
    print(_format_message(mock_signal))
    print("\n✅ notifier.py 格式測試完成（未實際發送）")
