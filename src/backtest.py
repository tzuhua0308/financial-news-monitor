"""
backtest.py — 訊號回測模組
將情緒訊號與模擬價格走勢做相關性分析
面試亮點：展示你能驗證 NLP 訊號的實際預測力（alpha 分析）
"""

import logging
import random
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger("backtest")

DB_PATH = Path(__file__).parent.parent / "data" / "signals.db"


# ─── 模擬股價（無 yfinance 時的 fallback）────────────────────────────────────

def simulate_price_returns(
    signal_df: pd.DataFrame,
    noise_std: float = 0.015,
    signal_strength: float = 0.008,
    seed: int = 42,
) -> pd.DataFrame:
    """
    模擬訊號發生後 1 日的股價報酬率。
    BUY 訊號 → 偏正報酬；SELL → 偏負；WATCH → 接近零
    加入隨機噪音模擬真實市場不確定性。

    真實場景請替換為：
        yfinance.download(ticker, start=date, end=date+1d)["Close"].pct_change()
    """
    rng = random.Random(seed)
    results = []
    for _, row in signal_df.iterrows():
        if row["signal"] == "BUY":
            mu = signal_strength * row["confidence"]
        elif row["signal"] == "SELL":
            mu = -signal_strength * row["confidence"]
        else:
            mu = 0.0

        ret = mu + rng.gauss(0, noise_std)
        results.append(round(ret, 5))

    signal_df = signal_df.copy()
    signal_df["next_day_return"] = results
    return signal_df


# ─── 載入訊號 ────────────────────────────────────────────────────────────────

def load_signals_df(db_path: Path = DB_PATH) -> pd.DataFrame:
    if not db_path.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(
        "SELECT * FROM signals ORDER BY generated_at ASC", conn
    )
    conn.close()
    df["generated_at"] = pd.to_datetime(df["generated_at"])
    df["weighted_score"] = df["weighted_score"].astype(float)
    df["confidence"] = df["confidence"].astype(float)
    return df


# ─── 核心回測指標 ──────────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    total_signals:      int
    buy_signals:        int
    sell_signals:       int
    watch_signals:      int

    # 準確率（方向正確）
    buy_accuracy:       float   # BUY 訊號後次日確實上漲的比率
    sell_accuracy:      float   # SELL 訊號後次日確實下跌的比率
    overall_accuracy:   float

    # 報酬統計
    avg_return_buy:     float
    avg_return_sell:    float   # 做空收益（正 = 賺錢）
    avg_return_watch:   float
    cumulative_pnl:     float   # 假設每筆 1 單位等值，累積 PnL

    # 風險指標
    sharpe_approx:      float   # 簡化 Sharpe（年化）
    max_drawdown:       float

    # 情緒-報酬相關性
    sentiment_return_corr: float   # weighted_score vs next_day_return

    # 時間序列（供繪圖）
    equity_curve:       pd.Series = field(default_factory=pd.Series)
    signal_df:          pd.DataFrame = field(default_factory=pd.DataFrame)


def run_backtest(
    db_path: Path = DB_PATH,
    use_real_prices: bool = False,
) -> Optional[BacktestResult]:
    """
    執行回測。

    use_real_prices=True 時嘗試用 yfinance 抓真實價格；
    False 時使用 simulate_price_returns 模擬（預設，不需網路）。
    """
    df = load_signals_df(db_path)
    if df.empty:
        logger.warning("[Backtest] 無訊號資料")
        return None

    # ── 取得次日報酬 ─────────────────────────────────────────────────────────
    if use_real_prices:
        df = _fetch_real_returns(df)
    else:
        df = simulate_price_returns(df)

    df = df.dropna(subset=["next_day_return"])

    # ── 分組統計 ─────────────────────────────────────────────────────────────
    buy_df   = df[df["signal"] == "BUY"]
    sell_df  = df[df["signal"] == "SELL"]
    watch_df = df[df["signal"] == "WATCH"]

    def accuracy(subset: pd.DataFrame, direction: str) -> float:
        if subset.empty:
            return 0.0
        if direction == "up":
            return (subset["next_day_return"] > 0).mean()
        return (subset["next_day_return"] < 0).mean()

    buy_acc   = accuracy(buy_df, "up")
    sell_acc  = accuracy(sell_df, "down")
    overall   = (
        (buy_df["next_day_return"] > 0).sum()
        + (sell_df["next_day_return"] < 0).sum()
    ) / max(len(buy_df) + len(sell_df), 1)

    # ── 模擬 PnL（多/空各 1 單位等值）───────────────────────────────────────
    pnl_series = pd.Series(dtype=float)
    for _, row in df.sort_values("generated_at").iterrows():
        ret = row["next_day_return"]
        if row["signal"] == "BUY":
            pnl_series = pd.concat([pnl_series, pd.Series([ret])])
        elif row["signal"] == "SELL":
            pnl_series = pd.concat([pnl_series, pd.Series([-ret])])  # 做空

    equity = (1 + pnl_series).cumprod() if not pnl_series.empty else pd.Series([1.0])
    equity.index = range(len(equity))

    # ── Sharpe（簡化版，假設無風險利率 = 0）──────────────────────────────────
    if not pnl_series.empty and pnl_series.std() > 0:
        sharpe = (pnl_series.mean() / pnl_series.std()) * (252 ** 0.5)
    else:
        sharpe = 0.0

    # ── 最大回撤 ─────────────────────────────────────────────────────────────
    peak = equity.cummax()
    dd   = ((equity - peak) / peak)
    max_dd = dd.min() if not dd.empty else 0.0

    # ── 情緒-報酬相關性 ───────────────────────────────────────────────────────
    corr = df["weighted_score"].corr(df["next_day_return"])

    return BacktestResult(
        total_signals   = len(df),
        buy_signals     = len(buy_df),
        sell_signals    = len(sell_df),
        watch_signals   = len(watch_df),
        buy_accuracy    = round(buy_acc, 4),
        sell_accuracy   = round(sell_acc, 4),
        overall_accuracy= round(overall, 4),
        avg_return_buy  = round(buy_df["next_day_return"].mean()  if not buy_df.empty  else 0, 5),
        avg_return_sell = round(-sell_df["next_day_return"].mean() if not sell_df.empty else 0, 5),
        avg_return_watch= round(watch_df["next_day_return"].mean() if not watch_df.empty else 0, 5),
        cumulative_pnl  = round(equity.iloc[-1] - 1, 4),
        sharpe_approx   = round(sharpe, 3),
        max_drawdown    = round(max_dd, 4),
        sentiment_return_corr = round(corr, 4),
        equity_curve    = equity,
        signal_df       = df,
    )


def _fetch_real_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    用 yfinance 抓真實次日報酬（需 pip install yfinance）。
    若個股代號為空則抓 SPY（大盤代理）。
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("[Backtest] yfinance 未安裝，改用模擬資料")
        return simulate_price_returns(df)

    results = []
    for _, row in df.iterrows():
        ticker = row.get("tickers", "SPY") or "SPY"
        ticker = ticker.split(",")[0].strip() or "SPY"
        date   = pd.Timestamp(row["generated_at"]).date()
        end    = date + timedelta(days=3)
        try:
            hist = yf.download(ticker, start=str(date), end=str(end), progress=False)
            if len(hist) >= 2:
                ret = float(hist["Close"].pct_change().iloc[1])
            else:
                ret = 0.0
        except Exception:
            ret = 0.0
        results.append(ret)

    df = df.copy()
    df["next_day_return"] = results
    return df
