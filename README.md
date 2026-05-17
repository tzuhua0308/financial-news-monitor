# 📊 Financial News Sentiment Monitor
### Real-time Alternative Data Pipeline | NLP Trading Signal System

> **作品集定位**：展示 Alternative Data 處理、金融 NLP、端對端自動化 Pipeline 能力  
> 面向職缺：量化交易員、金融研究員、資料分析師

---

## 🏗️ 系統架構

```
RSS / NewsAPI / Reddit（Alt Data）
        ↓  APScheduler 每 30 分鐘觸發
    fetcher.py   →  去重 + 標準化
        ↓
   sentiment.py  →  FinBERT 情緒分析 (備援 VADER)
        ↓
 signal_engine.py →  加權訊號生成 + SQLite 儲存
        ↓
   notifier.py   →  LINE Notify / Telegram Bot 推播
        ↓
  dashboard/app.py  →  Streamlit 即時監控介面
```

---

## ✨ 技術亮點

| 亮點 | 說明 |
|------|------|
| **Alternative Data** | 整合 Reddit 社群輿情（r/stocks、WSB），非傳統財經資料 |
| **Domain NLP** | 使用 **FinBERT**（金融領域微調 BERT），比通用 BERT 更準確 |
| **自訂詞典** | VADER 加入財經關鍵字（bullish/bearish/rate hike 等），提升準確率 |
| **信號加權** | 依來源可信度（Reuters > Reddit）對情緒分數加權，避免噪音 |
| **端對端 Pipeline** | 從原始資料到推播全自動，展示系統整合能力 |
| **去重機制** | MD5 hash 去重，避免重複分析與推播 |

---

## 🚀 快速開始

### 1. 安裝依賴

```bash
# 建議使用虛擬環境
python -m venv venv && source venv/bin/activate

# 安裝套件
pip install -r requirements.txt

# FinBERT 需要 PyTorch（CPU 版，較小）
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 2. 設定 API Keys

```bash
cp .env.example .env
# 編輯 .env 填入你的 Keys
```

| 服務 | 申請連結 | 費用 |
|------|----------|------|
| NewsAPI | https://newsapi.org/register | 免費（100次/天） |
| Reddit API | https://www.reddit.com/prefs/apps | 免費 |
| LINE Notify | https://notify-bot.line.me/zh_TW/ | 免費（1000則/天） |
| Telegram Bot | BotFather on Telegram | 免費 |

### 3. 執行

```bash
# 只跑一次 Pipeline（測試用）
cd src && python scheduler.py --once

# 啟動持續排程（每 30 分鐘）
cd src && python scheduler.py

# 啟動 Dashboard
streamlit run dashboard/app.py
```

---

## 📁 專案結構

```
financial-news-monitor/
├── src/
│   ├── fetcher.py        # RSS + NewsAPI + Reddit 擷取，去重
│   ├── sentiment.py      # FinBERT / VADER 情緒分析
│   ├── signal_engine.py  # 加權訊號生成 + SQLite 持久化
│   ├── notifier.py       # LINE Notify + Telegram 推播
│   └── scheduler.py      # APScheduler 主控排程
├── dashboard/
│   └── app.py            # Streamlit 即時監控介面
├── data/
│   └── signals.db        # SQLite 訊號資料庫（自動建立）
├── requirements.txt
└── .env.example
```

---

## 📈 訊號邏輯說明

### 情緒分數計算

```
weighted_score = FinBERT_compound × source_weight
```

| 來源 | 可信度權重 |
|------|-----------|
| Reuters, Bloomberg | 1.0 |
| CNBC | 0.9 |
| Yahoo Finance | 0.8 |
| Reddit r/investing | 0.45 |
| Reddit r/wallstreetbets | 0.3 |

### 訊號閾值

| weighted_score | 訊號 |
|---------------|------|
| ≥ +0.30 | 🟢 **BUY** |
| +0.15 ~ +0.30 | 🟡 WATCH（偏正） |
| -0.15 ~ +0.15 | — 忽略（中性） |
| -0.15 ~ -0.30 | 🟡 WATCH（偏負） |
| ≤ -0.30 | 🔴 **SELL** |

> ⚠️ 本系統為研究用途，非投資建議

---

## 🔍 Q & A

**Q：為什麼用 FinBERT 而不是 ChatGPT API？**  
A：FinBERT 在金融情緒分類任務上有 benchmarked 的準確率（FPB dataset F1 ≈ 0.88），且可完全本地運行，無 API 費用與隱私疑慮，適合高頻批次處理。

**Q：Reddit 資料的價值是什麼？**  
A：散戶情緒有時是市場短期走勢的領先指標（如 GME 事件），屬於典型 Alternative Data，可作為傳統資料的補充信號。

**Q：如何驗證訊號的有效性？**  
A：將歷史訊號與次日股價漲跌做相關性分析，計算 precision、recall，可進一步建立回測框架驗證 alpha。

---

## 🛠️ 可擴充方向

- [ ] 加入 SEC EDGAR 財報 NLP 分析
- [ ] 整合 Yahoo Finance 股價 API，做訊號與價格的相關性回測
- [ ] 升級為 Kafka + Flink 的即時串流架構
- [ ] 多語言支援（中文財經新聞 → 使用 `ckiplab/bert-base-chinese`）
- [ ] 部署到 GCP / AWS，加 Docker Compose

---

*Built with FinBERT · feedparser · APScheduler · Streamlit · SQLite*
