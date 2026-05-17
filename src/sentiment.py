"""
sentiment.py — NLP 情緒分析模組
主力：FinBERT（金融領域微調 BERT）
備援：VADER（無需 GPU，輕量快速）
面試亮點：展示你懂 domain-specific NLP，而非只用通用模型
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger("sentiment")

SentimentLabel = Literal["positive", "neutral", "negative"]


@dataclass
class SentimentResult:
    label: SentimentLabel          # 情緒分類
    score: float                   # 該分類的置信度 0–1
    compound: float                # 複合分數 -1（最負）→ +1（最正）
    positive: float = 0.0
    neutral:  float = 0.0
    negative: float = 0.0
    model:    str   = "unknown"    # 使用哪個模型（finbert / vader）


# ─── 文字前處理 ───────────────────────────────────────────────────────────────

def _clean_text(text: str, max_tokens: int = 512) -> str:
    """
    金融文本前處理：去除 HTML tag、URL、多餘空白。
    FinBERT 最大輸入 512 tokens，粗估 1 token ≈ 4 字元。
    """
    text = re.sub(r"<[^>]+>", " ", text)           # 去 HTML
    text = re.sub(r"http\S+", " ", text)            # 去 URL
    text = re.sub(r"[^\w\s\.\,\!\?\-\%\$]", " ", text)  # 保留財經常用符號
    text = re.sub(r"\s+", " ", text).strip()
    # 截斷（避免超過 FinBERT token 上限）
    char_limit = max_tokens * 4
    return text[:char_limit]


# ─── 1. FinBERT 分析器 ────────────────────────────────────────────────────────

class FinBERTAnalyzer:
    """
    使用 ProsusAI/finbert（Hugging Face），針對金融文本微調的 BERT。
    比通用 BERT 在財經情緒分析上準確率高約 10–15%。

    安裝：pip install transformers torch
    首次執行會自動下載模型（約 440MB）。
    """

    MODEL_NAME = "ProsusAI/finbert"

    def __init__(self):
        self._pipeline = None

    def _load(self):
        """Lazy load：只有第一次呼叫 analyze 時才載入模型"""
        if self._pipeline is not None:
            return

        try:
            from transformers import pipeline
            logger.info("[FinBERT] 載入模型中（首次約需 30 秒）...")
            self._pipeline = pipeline(
                "text-classification",
                model=self.MODEL_NAME,
                top_k=None,          # 回傳所有分類的分數
                truncation=True,
                max_length=512,
            )
            logger.info("[FinBERT] 模型載入完成")
        except ImportError:
            logger.error("[FinBERT] 請安裝：pip install transformers torch")
            raise

    def analyze(self, text: str) -> SentimentResult:
        self._load()
        cleaned = _clean_text(text)
        if not cleaned:
            return SentimentResult("neutral", 1.0, 0.0, model="finbert")

        raw = self._pipeline(cleaned)[0]
        # raw 格式：[{"label": "positive", "score": 0.92}, ...]
        scores = {r["label"].lower(): r["score"] for r in raw}

        pos = scores.get("positive", 0.0)
        neu = scores.get("neutral",  0.0)
        neg = scores.get("negative", 0.0)

        # 複合分數：positive → +1，negative → -1
        compound = round(pos - neg, 4)

        # 取最高置信度的標籤
        label = max(scores, key=scores.get)
        score = scores[label]

        return SentimentResult(
            label=label,
            score=round(score, 4),
            compound=compound,
            positive=round(pos, 4),
            neutral=round(neu, 4),
            negative=round(neg, 4),
            model="finbert",
        )

    def analyze_batch(self, texts: list[str], batch_size: int = 16) -> list[SentimentResult]:
        """批次處理，GPU 環境下效率更高"""
        self._load()
        cleaned = [_clean_text(t) for t in texts]
        results = []

        for i in range(0, len(cleaned), batch_size):
            batch = cleaned[i : i + batch_size]
            raw_batch = self._pipeline(batch)
            for raw in raw_batch:
                scores   = {r["label"].lower(): r["score"] for r in raw}
                pos      = scores.get("positive", 0.0)
                neg      = scores.get("negative", 0.0)
                compound = round(pos - neg, 4)
                label    = max(scores, key=scores.get)
                results.append(SentimentResult(
                    label=label,
                    score=round(scores[label], 4),
                    compound=compound,
                    positive=round(pos, 4),
                    neutral=round(scores.get("neutral", 0.0), 4),
                    negative=round(neg, 4),
                    model="finbert",
                ))

        return results


# ─── 2. VADER 備援分析器 ──────────────────────────────────────────────────────

class VADERAnalyzer:
    """
    VADER（Valence Aware Dictionary and sEntiment Reasoner）
    - 不需 GPU，安裝輕量，適合資源受限環境
    - 為通用英文情緒詞典，在財經文本上表現略遜於 FinBERT
    - 可作為 FinBERT 的 fallback 或快速原型

    安裝：pip install nltk
    """

    def __init__(self):
        self._sia = None

    def _load(self):
        if self._sia is not None:
            return
        try:
            import nltk
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            nltk.download("vader_lexicon", quiet=True)
            self._sia = SentimentIntensityAnalyzer()
            # 擴充金融領域詞彙（自訂詞典）
            finance_lexicon = {
                "bullish": 2.5, "bearish": -2.5,
                "rally": 2.0, "selloff": -2.0, "crash": -3.0,
                "outperform": 2.0, "underperform": -2.0,
                "beat": 1.5, "miss": -1.5, "downgrade": -2.0, "upgrade": 2.0,
                "record high": 3.0, "52-week high": 2.5,
                "layoff": -2.0, "bankruptcy": -3.5, "default": -3.0,
                "dividend": 1.5, "buyback": 1.5, "acquisition": 1.0,
                "rate hike": -1.5, "rate cut": 1.5, "quantitative easing": 1.0,
            }
            self._sia.lexicon.update(finance_lexicon)
            logger.info("[VADER] 分析器載入完成（含金融詞典擴充）")
        except ImportError:
            logger.error("[VADER] 請安裝：pip install nltk")
            raise

    def analyze(self, text: str) -> SentimentResult:
        self._load()
        cleaned = _clean_text(text)
        vs = self._sia.polarity_scores(cleaned)

        compound = round(vs["compound"], 4)
        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"

        # 用 compound 對應的區間估算 score
        score = round(abs(compound) if label != "neutral" else vs["neu"], 4)

        return SentimentResult(
            label=label,
            score=score,
            compound=compound,
            positive=round(vs["pos"], 4),
            neutral=round(vs["neu"], 4),
            negative=round(vs["neg"], 4),
            model="vader",
        )


# ─── 3. 統一分析器（自動 fallback）──────────────────────────────────────────

class SentimentAnalyzer:
    """
    對外統一接口。
    優先嘗試 FinBERT；若未安裝 transformers/torch 則自動切換 VADER。

    使用方式：
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze("Apple beats Q3 earnings estimates")
        print(result.label, result.compound)
    """

    def __init__(self, prefer: Literal["finbert", "vader"] = "finbert"):
        self.prefer = prefer
        self._analyzer = None

    def _init_analyzer(self):
        if self._analyzer is not None:
            return

        if self.prefer == "finbert":
            try:
                self._analyzer = FinBERTAnalyzer()
                self._analyzer._load()   # 提前觸發載入，確認環境
                logger.info("[SentimentAnalyzer] 使用 FinBERT")
            except Exception as e:
                logger.warning(f"[SentimentAnalyzer] FinBERT 不可用 ({e})，切換 VADER")
                self._analyzer = VADERAnalyzer()
        else:
            self._analyzer = VADERAnalyzer()
            logger.info("[SentimentAnalyzer] 使用 VADER")

    def analyze(self, text: str) -> SentimentResult:
        self._init_analyzer()
        return self._analyzer.analyze(text)

    def analyze_articles(self, articles: list[dict]) -> list[dict]:
        """
        批次分析文章 list（來自 fetcher.py 的格式）。
        在每篇文章 dict 中加入 sentiment 欄位。
        """
        self._init_analyzer()
        enriched = []

        # FinBERT 支援批次，VADER 只能逐筆
        if isinstance(self._analyzer, FinBERTAnalyzer):
            texts   = [a["raw_text"] for a in articles]
            results = self._analyzer.analyze_batch(texts)
            for article, result in zip(articles, results):
                enriched.append({**article, "sentiment": _result_to_dict(result)})
        else:
            for article in articles:
                result = self._analyzer.analyze(article["raw_text"])
                enriched.append({**article, "sentiment": _result_to_dict(result)})

        logger.info(
            f"[SentimentAnalyzer] 分析完成 {len(enriched)} 筆，"
            f"正面:{sum(1 for a in enriched if a['sentiment']['label']=='positive')} "
            f"中性:{sum(1 for a in enriched if a['sentiment']['label']=='neutral')} "
            f"負面:{sum(1 for a in enriched if a['sentiment']['label']=='negative')}"
        )
        return enriched


def _result_to_dict(r: SentimentResult) -> dict:
    return {
        "label":    r.label,
        "score":    r.score,
        "compound": r.compound,
        "positive": r.positive,
        "neutral":  r.neutral,
        "negative": r.negative,
        "model":    r.model,
    }


# ─── 快速測試 ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_headlines = [
        "Apple beats Q3 earnings estimates, stock surges 5%",
        "Fed signals another rate hike as inflation remains elevated",
        "Tesla faces recall of 1.8 million vehicles over safety concerns",
        "Markets close flat ahead of CPI data release",
        "Goldman Sachs upgrades Taiwan Semiconductor to Buy with $180 target",
    ]

    print("=== VADER 快速測試（無需下載模型）===\n")
    analyzer = SentimentAnalyzer(prefer="vader")

    for headline in test_headlines:
        result = analyzer.analyze(headline)
        bar = "+" * int(max(result.compound * 10, 0)) + "-" * int(max(-result.compound * 10, 0))
        print(f"  [{result.label:8s}] {result.compound:+.3f} |{bar:<10}| {headline[:60]}")

    print("\n\n=== 批次分析示範 ===\n")
    mock_articles = [
        {"raw_text": h, "title": h, "id": str(i), "source": "test",
         "source_type": "test", "summary": "", "url": "", "published_at": "", "fetched_at": ""}
        for i, h in enumerate(test_headlines)
    ]
    enriched = analyzer.analyze_articles(mock_articles)
    for a in enriched:
        s = a["sentiment"]
        print(f"  compound={s['compound']:+.3f}  label={s['label']:8s}  {a['title'][:55]}")
