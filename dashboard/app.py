"""
dashboard/app.py — 完整監控 Dashboard
四個 Tab：即時訊號 / 情緒趨勢 / 來源分析 / 回測報告
執行：streamlit run dashboard/app.py
"""

import sqlite3
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from backtest import load_signals_df, simulate_price_returns

DB_PATH = Path(__file__).parent.parent / "data" / "signals.db"

# ─── 頁面設定 ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Financial Sentiment Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    div[data-testid="stTabs"] button { font-size: 14px; }
    .block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)


# ─── 資料載入 ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def load_data():
    if not DB_PATH.exists():
        return pd.DataFrame(), pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    signals  = pd.read_sql("SELECT * FROM signals  ORDER BY generated_at DESC", conn)
    articles = pd.read_sql("SELECT * FROM articles ORDER BY fetched_at  DESC LIMIT 1000", conn)
    conn.close()
    if not signals.empty:
        signals["generated_at"] = pd.to_datetime(signals["generated_at"])
        signals["weighted_score"] = signals["weighted_score"].astype(float)
        signals["confidence"]     = signals["confidence"].astype(float)
    if not articles.empty:
        articles["fetched_at"] = pd.to_datetime(articles["fetched_at"])
        articles["compound"]   = articles["compound"].astype(float)
    return signals, articles


signals_df, articles_df = load_data()
has_data = not signals_df.empty

# ─── Header ──────────────────────────────────────────────────────────────────
hc1, hc2 = st.columns([3, 1])
with hc1:
    st.markdown("## 📊 Financial News Sentiment Monitor")
    st.caption("Real-time NLP signal tracking · FinBERT + Alternative Data Pipeline")
with hc2:
    if has_data:
        latest = signals_df["generated_at"].max()
        st.markdown(
            f"<div style='text-align:right;color:#6b7280;font-size:13px;padding-top:14px'>"
            f"🔄 最後更新<br><b>{latest.strftime('%m/%d %H:%M')}</b></div>",
            unsafe_allow_html=True,
        )

# ─── KPI Bar ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
if has_data:
    buy_n   = (signals_df.signal == "BUY").sum()
    sell_n  = (signals_df.signal == "SELL").sum()
    watch_n = (signals_df.signal == "WATCH").sum()
    avg_sc  = signals_df.weighted_score.mean()
    today_n = (signals_df.generated_at >= pd.Timestamp.now(tz="UTC").normalize()).sum()
    c1.metric("📋 總訊號",  len(signals_df), f"今日 {today_n}")
    c2.metric("🟢 BUY",     buy_n)
    c3.metric("🔴 SELL",    sell_n)
    c4.metric("🟡 WATCH",   watch_n)
    c5.metric("平均情緒分", f"{avg_sc:+.3f}")
else:
    for c in [c1, c2, c3, c4, c5]:
        c.metric("—", "—")

st.divider()

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["🚦 即時訊號", "📈 情緒趨勢", "🗂️ 來源分析", "🔬 回測報告"]
)

# ══════════════════════════════════════════════════════
# TAB 1 — 即時訊號
# ══════════════════════════════════════════════════════
with tab1:
    if not has_data:
        st.info("尚無資料。請先執行 `python src/scheduler.py` 啟動 Pipeline。")
    else:
        fc1, fc2, fc3 = st.columns([1, 1, 2])
        with fc1:
            sig_filter = st.multiselect(
                "訊號類型", ["BUY", "SELL", "WATCH"],
                default=["BUY", "SELL", "WATCH"],
            )
        with fc2:
            min_conf = st.slider("最低信心度", 0.0, 1.0, 0.0, 0.05)
        with fc3:
            days_back = st.slider("顯示最近 N 天", 1, 30, 7)

        cutoff   = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days_back)
        filtered = signals_df[
            signals_df.signal.isin(sig_filter) &
            (signals_df.confidence >= min_conf) &
            (signals_df.generated_at >= cutoff)
        ].copy()

        st.caption(f"顯示 {len(filtered)} / {len(signals_df)} 筆訊號")

        display = filtered[[
            "generated_at", "signal", "weighted_score", "confidence",
            "tickers", "themes", "source", "title",
        ]].copy()
        display.columns = ["時間", "訊號", "情緒分", "信心", "股票", "主題", "來源", "標題"]
        display["時間"]   = display["時間"].dt.strftime("%m/%d %H:%M")
        display["情緒分"] = display["情緒分"].map(lambda x: f"{x:+.3f}")
        display["信心"]   = display["信心"].map(lambda x: f"{x:.0%}")
        display["標題"]   = display["標題"].str[:55] + "…"

        def color_signal(val):
            m = {"BUY": "color:#059669;font-weight:600",
                 "SELL": "color:#dc2626;font-weight:600",
                 "WATCH": "color:#d97706;font-weight:600"}
            return m.get(val, "")

        st.dataframe(
            display.style.map(color_signal, subset=["訊號"]),
            use_container_width=True, height=420, hide_index=True,
        )

        st.markdown("**篩選範圍內訊號分布**")
        bc1, bc2, bc3 = st.columns(3)
        total_f = max(len(filtered), 1)
        bc1.progress(int((filtered.signal == "BUY").sum()   / total_f * 100), text=f"BUY   {(filtered.signal=='BUY').sum()}")
        bc2.progress(int((filtered.signal == "SELL").sum()  / total_f * 100), text=f"SELL  {(filtered.signal=='SELL').sum()}")
        bc3.progress(int((filtered.signal == "WATCH").sum() / total_f * 100), text=f"WATCH {(filtered.signal=='WATCH').sum()}")


# ══════════════════════════════════════════════════════
# TAB 2 — 情緒趨勢
# ══════════════════════════════════════════════════════
with tab2:
    if articles_df.empty:
        st.info("暫無文章資料。")
    else:
        st.subheader("情緒分數時間序列")
        granularity = st.radio("時間粒度", ["1h", "4h", "1D"], horizontal=True, index=1)

        ts = (
            articles_df.set_index("fetched_at")["compound"]
            .resample(granularity).agg(["mean", "count"])
            .reset_index()
        )
        ts.columns = ["time", "avg_compound", "count"]

        fig = go.Figure()
        fig.add_hrect(y0=0.15,  y1=1,  fillcolor="rgba(5,150,105,0.06)",  line_width=0)
        fig.add_hrect(y0=-1, y1=-0.15, fillcolor="rgba(220,38,38,0.06)",  line_width=0)
        fig.add_bar(
            x=ts["time"], y=ts["count"], yaxis="y2", name="文章數",
            marker_color="rgba(156,163,175,0.3)",
        )
        fig.add_scatter(
            x=ts["time"], y=ts["avg_compound"],
            mode="lines+markers", name="平均情緒分",
            line=dict(color="#4F46E5", width=2.5),
            fill="tozeroy", fillcolor="rgba(79,70,229,0.07)",
        )
        fig.add_hline(y=0.15,  line_dash="dot", line_color="#059669", opacity=0.5,
                      annotation_text="BUY +0.15",  annotation_position="top left")
        fig.add_hline(y=-0.15, line_dash="dot", line_color="#dc2626", opacity=0.5,
                      annotation_text="SELL -0.15", annotation_position="bottom left")
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.25)
        fig.update_layout(
            height=340, margin=dict(l=0,r=0,t=20,b=0),
            yaxis=dict(title="情緒分數", range=[-1, 1]),
            yaxis2=dict(title="文章數", overlaying="y", side="right", showgrid=False),
            legend=dict(x=0, y=1.08, orientation="h"),
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

        # 分布直方圖
        st.subheader("情緒分數分布")
        fig2 = px.histogram(
            articles_df, x="compound", nbins=50,
            color="sentiment_label",
            color_discrete_map={"positive": "#059669", "neutral": "#9ca3af", "negative": "#dc2626"},
            opacity=0.85, labels={"compound": "Compound Score", "sentiment_label": "情緒"},
            height=260,
        )
        fig2.update_layout(margin=dict(l=0,r=0,t=10,b=0), bargap=0.04)
        st.plotly_chart(fig2, use_container_width=True)

        # 個股情緒排行
        if has_data:
            st.subheader("個股情緒排行")
            rows = []
            for _, row in signals_df.iterrows():
                for t in str(row.get("tickers", "")).split(","):
                    t = t.strip()
                    if t:
                        rows.append({"ticker": t, "score": row["weighted_score"]})
            if rows:
                tdf = (
                    pd.DataFrame(rows)
                    .groupby("ticker")
                    .agg(avg_score=("score", "mean"), mentions=("score", "count"))
                    .sort_values("avg_score", ascending=False)
                    .reset_index()
                )
                fig3 = px.bar(
                    tdf, x="ticker", y="avg_score", color="avg_score",
                    color_continuous_scale=["#dc2626", "#f9fafb", "#059669"],
                    color_continuous_midpoint=0, text="mentions",
                    labels={"avg_score": "平均情緒分", "ticker": "股票", "mentions": "提及次數"},
                    height=280,
                )
                fig3.update_traces(texttemplate="%{text}次", textposition="outside")
                fig3.update_layout(margin=dict(l=0,r=0,t=10,b=0), coloraxis_showscale=False)
                st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════
# TAB 3 — 來源分析
# ══════════════════════════════════════════════════════
with tab3:
    if articles_df.empty:
        st.info("暫無文章資料。")
    else:
        sc1, sc2 = st.columns(2)

        with sc1:
            st.subheader("來源文章數")
            src_count = articles_df.source.value_counts().reset_index()
            src_count.columns = ["source", "count"]
            fig4 = px.pie(
                src_count.head(8), names="source", values="count",
                color_discrete_sequence=px.colors.qualitative.Pastel,
                hole=0.4, height=320,
            )
            fig4.update_layout(margin=dict(l=0,r=0,t=10,b=0))
            fig4.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig4, use_container_width=True)

        with sc2:
            st.subheader("來源平均情緒分")
            src_sent = (
                articles_df.groupby("source")["compound"]
                .agg(["mean", "count"])
                .reset_index()
                .sort_values("mean")
            )
            src_sent.columns = ["source", "avg_sentiment", "articles"]
            fig5 = px.bar(
                src_sent, x="avg_sentiment", y="source", orientation="h",
                color="avg_sentiment",
                color_continuous_scale=["#dc2626", "#f3f4f6", "#059669"],
                color_continuous_midpoint=0, text="articles",
                labels={"avg_sentiment": "平均情緒分", "source": "來源", "articles": "篇數"},
                height=320,
            )
            fig5.update_traces(texttemplate="%{text}篇", textposition="outside")
            fig5.update_layout(margin=dict(l=0,r=0,t=10,b=0), coloraxis_showscale=False)
            st.plotly_chart(fig5, use_container_width=True)

        st.subheader("各來源情緒分類分布")
        cross     = pd.crosstab(articles_df["source"], articles_df["sentiment_label"])
        cross_pct = cross.div(cross.sum(axis=1), axis=0) * 100
        fig6 = px.bar(
            cross_pct.reset_index().melt(id_vars="source"),
            x="source", y="value", color="sentiment_label",
            color_discrete_map={"positive": "#059669", "neutral": "#9ca3af", "negative": "#dc2626"},
            labels={"value": "比率 (%)", "source": "來源", "variable": "情緒"},
            barmode="stack", height=300,
        )
        fig6.update_layout(margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig6, use_container_width=True)


# ══════════════════════════════════════════════════════
# TAB 4 — 回測報告
# ══════════════════════════════════════════════════════
with tab4:
    st.subheader("🔬 情緒訊號回測分析")
    st.caption("假設每筆 BUY/SELL 訊號以當日收盤買入，次日收盤賣出，等額資金")

    bt_c1, bt_c2 = st.columns([3, 1])
    with bt_c2:
        noise   = st.slider("市場噪音 σ",  0.005, 0.030, 0.015, 0.005,
                            help="模擬報酬的隨機標準差，反映市場不確定性")
        sig_str = st.slider("訊號強度",    0.002, 0.020, 0.008, 0.001,
                            help="情緒分數對報酬的影響幅度")

    if not has_data:
        st.info("尚無訊號資料，請先執行 Pipeline。")
    else:
        df_bt = load_signals_df()
        df_bt = simulate_price_returns(df_bt, noise_std=noise, signal_strength=sig_str)

        buy_df   = df_bt[df_bt.signal == "BUY"]
        sell_df  = df_bt[df_bt.signal == "SELL"]
        watch_df = df_bt[df_bt.signal == "WATCH"]

        def accuracy(sub, direction):
            if sub.empty: return 0.0
            return (sub.next_day_return > 0).mean() if direction == "up" else (sub.next_day_return < 0).mean()

        pnl_list = []
        for _, row in df_bt.sort_values("generated_at").iterrows():
            if row.signal == "BUY":
                pnl_list.append(float(row.next_day_return))
            elif row.signal == "SELL":
                pnl_list.append(-float(row.next_day_return))

        pnl_s  = pd.Series(pnl_list)
        equity = (1 + pnl_s).cumprod() if len(pnl_s) > 0 else pd.Series([1.0])
        equity.index = range(len(equity))
        sharpe = (pnl_s.mean() / pnl_s.std() * (252 ** 0.5)) if len(pnl_s) > 1 and pnl_s.std() > 0 else 0.0
        max_dd = ((equity - equity.cummax()) / equity.cummax()).min() if len(equity) > 0 else 0.0
        corr   = df_bt.weighted_score.astype(float).corr(df_bt.next_day_return)
        pnl_total = equity.iloc[-1] - 1

        # KPI
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("累積 PnL",      f"{pnl_total:+.2%}")
        k2.metric("Sharpe (年化)", f"{sharpe:.2f}")
        k3.metric("最大回撤",      f"{max_dd:.2%}")
        k4.metric("BUY 準確率",    f"{accuracy(buy_df,'up'):.1%}")
        k5.metric("SELL 準確率",   f"{accuracy(sell_df,'down'):.1%}")
        k6.metric("情緒-報酬相關", f"{corr:.4f}")

        st.divider()

        row1_c1, row1_c2 = st.columns([3, 2])

        with row1_c1:
            st.markdown("**策略權益曲線**")
            fig_eq = go.Figure()
            fig_eq.add_scatter(
                x=list(range(len(equity))), y=equity.values,
                mode="lines", name="策略",
                line=dict(color="#4F46E5", width=2.5),
                fill="tozeroy", fillcolor="rgba(79,70,229,0.08)",
            )
            fig_eq.add_hline(y=1.0, line_dash="dash", line_color="gray", opacity=0.4)
            fig_eq.update_layout(
                height=280, margin=dict(l=0,r=0,t=10,b=0),
                xaxis_title="交易筆數", yaxis_title="資產倍數（起始=1.0）",
                showlegend=False,
            )
            st.plotly_chart(fig_eq, use_container_width=True)

        with row1_c2:
            st.markdown("**BUY vs SELL 報酬分布**")
            dist_rows = (
                [{"類型": "BUY", "報酬": r.next_day_return} for _, r in buy_df.iterrows()] +
                [{"類型": "SELL (做空)", "報酬": -r.next_day_return} for _, r in sell_df.iterrows()]
            )
            if dist_rows:
                fig_dist = px.histogram(
                    pd.DataFrame(dist_rows), x="報酬", color="類型", nbins=25,
                    color_discrete_map={"BUY": "#059669", "SELL (做空)": "#dc2626"},
                    barmode="overlay", opacity=0.75, height=280,
                )
                fig_dist.update_layout(margin=dict(l=0,r=0,t=10,b=0), legend_title="")
                st.plotly_chart(fig_dist, use_container_width=True)

        # 散點圖
        st.markdown("**情緒分數 vs 次日報酬**")
        fig_sc = px.scatter(
            df_bt, x="weighted_score", y="next_day_return",
            color="signal",
            color_discrete_map={"BUY": "#059669", "SELL": "#dc2626", "WATCH": "#d97706"},
            trendline="ols",
            trendline_scope="overall",
            trendline_color_override="#4F46E5",
            labels={
                "weighted_score": "加權情緒分",
                "next_day_return": "次日報酬率",
                "signal": "訊號",
            },
            opacity=0.65, height=320,
        )
        fig_sc.add_vline(x=0.15,  line_dash="dot", line_color="#059669", opacity=0.4)
        fig_sc.add_vline(x=-0.15, line_dash="dot", line_color="#dc2626", opacity=0.4)
        fig_sc.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.3)
        fig_sc.update_layout(margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig_sc, use_container_width=True)

        st.caption(
            "⚠️ 目前使用模擬股價資料。"
            "替換為真實股價：安裝 `yfinance` 後，在 `backtest.py` 改為 `use_real_prices=True`。"
            "本系統為研究用途，非投資建議。"
        )

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#9ca3af;font-size:12px'>"
    "Financial News Sentiment Monitor &nbsp;·&nbsp; FinBERT + VADER &nbsp;·&nbsp; "
    "SQLite &nbsp;·&nbsp; Streamlit &nbsp;·&nbsp; Plotly"
    "</div>",
    unsafe_allow_html=True,
)
