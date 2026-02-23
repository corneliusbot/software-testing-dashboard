#!/usr/bin/env python3
"""
Earnings NLP Enhancement Module
================================
Provides contextual sentiment analysis, management guidance detection,
Q-over-Q keyword acceleration, and company-level thesis scoring for
earnings call transcripts.

Usage:
    # As module (imported by app.py):
    from earnings_nlp import EarningsNLP
    nlp = EarningsNLP()
    results = nlp.get_cached_results()

    # Standalone processing:
    python earnings_nlp.py                    # Process sample companies
    python earnings_nlp.py --process-all      # Process all 324 transcripts
    python earnings_nlp.py --tickers DDOG ZS  # Process specific tickers
"""

import json
import os
import re
import sys
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
TRANSCRIPT_DIR = DATA_DIR / "transcripts"
CACHE_FILE = DATA_DIR / "earnings_nlp_cache.json"

# Sample companies for initial processing (thesis-relevant + high-mention companies)
SAMPLE_TICKERS = ["DDOG", "DT", "CRWD", "PANW", "ZS", "QLYS", "TENB", "ESTC", "GTLB", "TEAM", "PATH", "FROG", "OKTA", "RPD", "NOW", "ABNB"]

# Thesis-relevant keywords (same as app.py but flattened for NLP use)
THESIS_KEYWORDS = [
    "testing", "quality assurance", "QA ", "test automation", "code quality",
    "unit test", "regression test", "continuous testing",
    "validation", "validate", "verification", "verify", "code review",
    "governance", "guardrails", "compliance", "audit", "oversight", "responsible AI",
    "security testing", "vulnerability", "penetration test",
    "AI safety", "model safety", "hallucination", "alignment", "red team",
    "observability", "monitoring", "telemetry", "tracing",
]

# Keyword groups for aggregation
KEYWORD_GROUPS = {
    "Testing / QA": ["testing", "quality assurance", "QA ", "test automation", "code quality",
                     "unit test", "regression test", "continuous testing"],
    "Validation": ["validation", "validate", "verification", "verify", "code review"],
    "Governance": ["governance", "guardrails", "compliance", "audit", "oversight", "responsible AI"],
    "Security": ["security testing", "vulnerability", "penetration test"],
    "AI Safety": ["AI safety", "model safety", "hallucination", "alignment", "red team"],
    "Observability": ["observability", "monitoring", "telemetry", "tracing"],
}

# Management guidance patterns
BULLISH_PATTERNS = [
    r"invest(?:ing|ment|ed)?\s+(?:in|more\s+in)\s+(?:testing|validation|quality|governance|security|observability)",
    r"(?:quality|testing|validation|governance|security)\s+(?:is|remains?|continues?\s+to\s+be)\s+(?:a\s+)?(?:top\s+)?priority",
    r"(?:testing|QA|validation|governance)\s+(?:spend|budget|investment)\s+(?:increas|grow|expand|doubl)",
    r"governance\s+mandate",
    r"customer(?:s)?\s+(?:asking|demanding|requiring)\s+(?:for\s+)?(?:testing|validation|quality|governance)",
    r"(?:expand|grow|increas|ramp|doubl|scal)(?:ing|ed)?\s+(?:our\s+)?(?:testing|QA|validation|quality|governance|security|observability)",
    r"(?:unprecedented|strong|growing|increasing)\s+demand\s+(?:for\s+)?(?:testing|observability|validation|security|governance|quality)",
    r"shift(?:ing)?\s+(?:resources?\s+)?(?:to|toward)?\s+(?:testing|validation|quality|governance)",
    r"(?:more|greater|additional)\s+(?:investment|focus|emphasis|attention)\s+(?:on|in|toward)\s+(?:testing|validation|quality|governance|security)",
]

BEARISH_PATTERNS = [
    r"(?:reduc|cut|decreas|eliminat|automat(?:ing)?\s+away)(?:ing|ed|e)?\s+(?:QA|testing|validation|quality)\s+(?:headcount|budget|team|spend|staff)",
    r"(?:cut|reduc|lower|decreas)(?:ting|ed|e)?\s+(?:our\s+)?(?:testing|QA|validation)\s+(?:budget|spend|investment|cost)",
    r"(?:AI|automation)\s+(?:replac|eliminat|reduc)(?:ing|ed|es)?\s+(?:the\s+need\s+for\s+)?(?:manual\s+)?(?:testing|QA|validation)",
    r"(?:less|fewer|reduced)\s+(?:need\s+for\s+)?(?:manual\s+)?(?:testing|QA|validation)",
]

CUSTOMER_SIGNAL_PATTERNS = [
    r"customer(?:s)?\s+(?:are\s+)?(?:asking|demanding|requiring|requesting|looking)\s+(?:for|us\s+to)",
    r"demand\s+(?:for|from)\s+(?:customer|enterprise|client)(?:s)?\s+(?:for\s+)?(?:testing|validation|security|governance|observability|quality)",
    r"enterprise(?:s)?\s+(?:need|require|want|demand|adopt)(?:ing|s|ed)?\s+(?:testing|validation|security|governance|observability|quality)",
    r"(?:customer|client|enterprise)\s+(?:demand|adoption|interest|uptake)\s+(?:for\s+)?(?:testing|observability|security|validation|governance)",
    r"(?:win|won|winning|deal)\s+(?:because|due\s+to)\s+(?:our\s+)?(?:testing|validation|security|governance|observability)",
]


def _get_openai_client():
    """Create OpenAI client with API key from environment."""
    try:
        import openai
    except ImportError:
        raise ImportError("openai package required. pip install openai")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        # Try loading from .zshrc
        zshrc = Path.home() / ".zshrc"
        if zshrc.exists():
            for line in zshrc.read_text().splitlines():
                if "OPENAI_API_KEY" in line and "export" in line:
                    match = re.search(r'OPENAI_API_KEY="([^"]+)"', line)
                    if match:
                        api_key = match.group(1)
                        os.environ["OPENAI_API_KEY"] = api_key
                        break
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment or ~/.zshrc")

    return openai.OpenAI(api_key=api_key)


def extract_keyword_snippets(text: str, keywords: list[str], window: int = 500) -> list[dict]:
    """
    Extract snippets around keyword occurrences with metadata.
    Returns list of {keyword, snippet, position, section}.
    """
    text_lower = text.lower()
    snippets = []
    seen_positions = set()

    for kw in keywords:
        kw_lower = kw.lower().strip()
        start = 0
        while True:
            idx = text_lower.find(kw_lower, start)
            if idx == -1:
                break
            # Skip if too close to an already-extracted snippet
            too_close = any(abs(idx - p) < window // 2 for p in seen_positions)
            if not too_close:
                snippet_start = max(0, idx - window // 2)
                snippet_end = min(len(text), idx + len(kw) + window // 2)
                snippet = text[snippet_start:snippet_end].strip()
                snippet = snippet.replace("\r\n", " ").replace("\n", " ")
                # Clean up multiple spaces
                snippet = re.sub(r"\s+", " ", snippet)

                snippets.append({
                    "keyword": kw.strip(),
                    "snippet": snippet,
                    "position": idx,
                    "char_pct": round(idx / max(len(text), 1) * 100, 1),
                })
                seen_positions.add(idx)

            start = idx + len(kw)
            if len(snippets) >= 50:  # Cap per transcript
                break
        if len(snippets) >= 50:
            break

    return snippets


def classify_snippets_batch(client, snippets: list[dict], ticker: str, company: str) -> list[dict]:
    """
    Send a batch of snippets to GPT-4o for contextual sentiment classification.
    Returns enriched snippets with sentiment labels.
    """
    if not snippets:
        return []

    # Process in batches of 15 to stay within token limits
    batch_size = 15
    results = []

    for i in range(0, len(snippets), batch_size):
        batch = snippets[i:i + batch_size]

        # Build the prompt
        snippet_texts = []
        for j, s in enumerate(batch):
            snippet_texts.append(f"[{j+1}] Keyword: \"{s['keyword']}\"\nContext: \"{s['snippet']}\"")

        prompt = f"""You are analyzing earnings call transcripts from {company} ({ticker}) for signals about software testing, QA, validation, quality assurance, and AI/agentic impact on testing/development.

FOCUS SPECIFICALLY ON: software testing, QA, validation, quality assurance, AI/agentic impact on testing, automated testing, release velocity requiring more testing, governance/compliance requirements for software, observability of AI systems.

For each snippet below, classify the sentiment into EXACTLY ONE category:
- INCREASING: Company is spending more, prioritizing, expanding testing/QA/validation, or acknowledging need for more testing due to AI/agentic development
- DECREASING: Company is cutting testing budgets, reducing QA headcount, or automating away testing roles
- CUSTOMER_DEMAND: Company mentions customer/enterprise demand for testing/validation/quality assurance (most valuable signal)
- NEUTRAL: Testing/QA mentioned but no clear directional signal about spending/priority

Also rate confidence 1-5 (5 = very clear signal, 1 = ambiguous).

CRITICAL: If the snippet contains management quotes about testing/QA/validation/quality/observability/governance, extract COMPLETE SENTENCES or 2-3 sentence passages that provide full context. Extract ONLY quotes that are SPECIFICALLY about:
- Software testing, QA, validation, quality assurance
- AI/agentic impact on testing workflows
- Automated testing, release velocity requiring more testing
- Governance/compliance requirements for software
- Observability of AI systems
- Customer demands for testing/validation capabilities

REJECT generic business quotes like "Customer interest has been strong" or "budgets being allocated that way" that could apply to any business area.

Each quote should be a complete thought that clearly relates to the testing/validation thesis. Include enough context that a reader understands WHY this quote matters for software testing/validation spend.

Snippets:
{chr(10).join(snippet_texts)}

Respond in JSON format ONLY (no markdown, no code fences):
[
  {{"id": 1, "sentiment": "INCREASING", "confidence": 4, "key_quote": "complete relevant sentence about testing/QA or null"}},
  ...
]"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o",  # Use GPT-4o (not mini) for quote quality
                messages=[
                    {"role": "system", "content": "You classify earnings call snippets for testing/QA relevance and extract complete, contextual management quotes. Respond only with valid JSON arrays. No markdown formatting."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=3000,  # More tokens for longer quotes
            )

            raw = response.choices[0].message.content.strip()
            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)

            classifications = json.loads(raw)

            for cls in classifications:
                idx = cls.get("id", 0) - 1
                if 0 <= idx < len(batch):
                    batch[idx]["sentiment"] = cls.get("sentiment", "NEUTRAL")
                    batch[idx]["confidence"] = cls.get("confidence", 1)
                    quote = cls.get("key_quote")
                    if quote and quote != "null" and len(quote) > 10:
                        batch[idx]["key_quote"] = quote

        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error for {ticker} batch {i}: {e}")
            # Fallback: mark as NEUTRAL
            for s in batch:
                s["sentiment"] = "NEUTRAL"
                s["confidence"] = 1
        except Exception as e:
            logger.warning(f"API error for {ticker} batch {i}: {e}")
            for s in batch:
                s["sentiment"] = "NEUTRAL"
                s["confidence"] = 1
            time.sleep(2)  # Back off on errors

        results.extend(batch)
        time.sleep(0.5)  # Rate limit courtesy

    return results


def detect_management_guidance(text: str) -> dict:
    """
    Detect management guidance patterns in transcript text.
    Returns {bullish: [...], bearish: [...], customer_signal: [...]}.
    """
    results = {"bullish": [], "bearish": [], "customer_signal": []}

    text_lower = text.lower()

    for pattern in BULLISH_PATTERNS:
        for match in re.finditer(pattern, text_lower):
            start = max(0, match.start() - 200)
            end = min(len(text), match.end() + 200)
            context = text[start:end].replace("\n", " ").strip()
            context = re.sub(r"\s+", " ", context)
            results["bullish"].append({
                "pattern": pattern[:60],
                "match": match.group(),
                "context": context,
            })

    for pattern in BEARISH_PATTERNS:
        for match in re.finditer(pattern, text_lower):
            start = max(0, match.start() - 200)
            end = min(len(text), match.end() + 200)
            context = text[start:end].replace("\n", " ").strip()
            context = re.sub(r"\s+", " ", context)
            results["bearish"].append({
                "pattern": pattern[:60],
                "match": match.group(),
                "context": context,
            })

    for pattern in CUSTOMER_SIGNAL_PATTERNS:
        for match in re.finditer(pattern, text_lower):
            start = max(0, match.start() - 200)
            end = min(len(text), match.end() + 200)
            context = text[start:end].replace("\n", " ").strip()
            context = re.sub(r"\s+", " ", context)
            results["customer_signal"].append({
                "pattern": pattern[:60],
                "match": match.group(),
                "context": context,
            })

    return results


def compute_qoq_acceleration(transcripts_by_ticker: dict) -> dict:
    """
    Compute quarter-over-quarter keyword acceleration.

    Returns:
    {
        "quarterly_totals": [{quarter, total_mentions, by_group: {...}}],
        "by_ticker": {
            "DDOG": [{quarter, mentions, growth_rate}],
            ...
        },
        "acceleration": {
            "overall_trend": "accelerating" | "decelerating" | "stable",
            "latest_growth_rate": float,
            "growth_rate_change": float,
        }
    }
    """
    # Collect per-quarter, per-ticker keyword counts
    quarter_data = defaultdict(lambda: defaultdict(int))  # quarter -> group -> count
    ticker_quarter_data = defaultdict(lambda: defaultdict(int))  # ticker -> quarter -> count

    for ticker, transcripts in transcripts_by_ticker.items():
        for t in transcripts:
            event_date = t.get("event_date", "")
            if not event_date:
                continue

            try:
                dt = datetime.strptime(event_date[:10], "%Y-%m-%d")
            except (ValueError, TypeError):
                continue

            month = dt.month
            year = dt.year
            if month <= 3:
                qtr = f"CY Q1 {year}"
            elif month <= 6:
                qtr = f"CY Q2 {year}"
            elif month <= 9:
                qtr = f"CY Q3 {year}"
            else:
                qtr = f"CY Q4 {year}"

            text_lower = t.get("text", "").lower()

            for group_name, keywords in KEYWORD_GROUPS.items():
                count = sum(text_lower.count(kw.lower()) for kw in keywords)
                quarter_data[qtr][group_name] += count
                ticker_quarter_data[ticker][qtr] += count

    # Sort quarters chronologically
    def quarter_sort_key(q):
        parts = q.split()  # "CY Q1 2025"
        return (int(parts[2]), int(parts[1][1]))

    sorted_quarters = sorted(quarter_data.keys(), key=quarter_sort_key)

    # Build quarterly totals
    quarterly_totals = []
    for qtr in sorted_quarters:
        total = sum(quarter_data[qtr].values())
        quarterly_totals.append({
            "quarter": qtr,
            "total_mentions": total,
            "by_group": dict(quarter_data[qtr]),
        })

    # Build per-ticker timeseries
    by_ticker = {}
    for ticker in ticker_quarter_data:
        ticker_quarters = sorted(
            [q for q in ticker_quarter_data[ticker].keys()],
            key=quarter_sort_key
        )
        entries = []
        prev_count = None
        for qtr in ticker_quarters:
            count = ticker_quarter_data[ticker][qtr]
            growth_rate = None
            if prev_count is not None and prev_count > 0:
                growth_rate = round((count - prev_count) / prev_count * 100, 1)
            entries.append({
                "quarter": qtr,
                "mentions": count,
                "growth_rate": growth_rate,
            })
            prev_count = count
        by_ticker[ticker] = entries

    # Compute overall acceleration
    if len(quarterly_totals) >= 3:
        totals = [q["total_mentions"] for q in quarterly_totals]
        growth_rates = []
        for i in range(1, len(totals)):
            if totals[i - 1] > 0:
                growth_rates.append((totals[i] - totals[i - 1]) / totals[i - 1] * 100)

        latest_growth = growth_rates[-1] if growth_rates else 0
        if len(growth_rates) >= 2:
            growth_change = growth_rates[-1] - growth_rates[-2]
        else:
            growth_change = 0

        if growth_change > 5:
            trend = "accelerating"
        elif growth_change < -5:
            trend = "decelerating"
        else:
            trend = "stable"

        acceleration = {
            "overall_trend": trend,
            "latest_growth_rate": round(latest_growth, 1),
            "growth_rate_change": round(growth_change, 1),
        }
    else:
        acceleration = {
            "overall_trend": "insufficient_data",
            "latest_growth_rate": 0,
            "growth_rate_change": 0,
        }

    return {
        "quarterly_totals": quarterly_totals,
        "by_ticker": by_ticker,
        "acceleration": acceleration,
    }


def compute_thesis_score(
    ticker: str,
    company: str,
    sentiment_results: list[dict],
    guidance_results: list[dict],
    qoq_data: dict,
) -> dict:
    """
    Compute a composite thesis score for a company.

    Score components:
    - Sentiment balance (INCREASING + CUSTOMER_DEMAND vs DECREASING): 0-4 pts
    - Management guidance signals (bullish vs bearish): 0-3 pts
    - Keyword trend (acceleration): 0-2 pts
    - Volume/confidence bonus: 0-1 pt

    Returns company thesis score dict.
    """
    # Aggregate sentiments across all transcripts
    all_sentiments = []
    all_quotes = []
    for result in sentiment_results:
        for s in result.get("snippets", []):
            all_sentiments.append(s)
            if s.get("key_quote"):
                all_quotes.append(s["key_quote"])

    # Sentiment balance
    increasing = sum(1 for s in all_sentiments if s.get("sentiment") == "INCREASING")
    decreasing = sum(1 for s in all_sentiments if s.get("sentiment") == "DECREASING")
    customer_demand = sum(1 for s in all_sentiments if s.get("sentiment") == "CUSTOMER_DEMAND")
    neutral = sum(1 for s in all_sentiments if s.get("sentiment") == "NEUTRAL")
    total_classified = len(all_sentiments)

    if total_classified > 0:
        positive_ratio = (increasing + customer_demand * 1.5) / total_classified
        negative_ratio = decreasing / total_classified
        sentiment_score = min(4, max(0, (positive_ratio - negative_ratio) * 6 + 2))
    else:
        sentiment_score = 2  # Neutral default

    # Management guidance score
    total_bullish = sum(len(g.get("bullish", [])) for g in guidance_results)
    total_bearish = sum(len(g.get("bearish", [])) for g in guidance_results)
    total_customer = sum(len(g.get("customer_signal", [])) for g in guidance_results)

    if total_bullish + total_bearish > 0:
        guidance_ratio = (total_bullish + total_customer * 1.5) / (total_bullish + total_bearish + total_customer + 1)
        guidance_score = min(3, guidance_ratio * 3)
    else:
        guidance_score = 1.5  # Neutral default

    # Keyword trend score
    ticker_qoq = qoq_data.get("by_ticker", {}).get(ticker, [])
    if len(ticker_qoq) >= 2:
        growth_rates = [e["growth_rate"] for e in ticker_qoq if e.get("growth_rate") is not None]
        if growth_rates:
            avg_growth = sum(growth_rates) / len(growth_rates)
            if avg_growth > 20:
                trend_score = 2.0
                keyword_trend = "accelerating"
            elif avg_growth > 0:
                trend_score = 1.5
                keyword_trend = "growing"
            elif avg_growth > -10:
                trend_score = 1.0
                keyword_trend = "stable"
            else:
                trend_score = 0.5
                keyword_trend = "declining"
        else:
            trend_score = 1.0
            keyword_trend = "stable"
    else:
        trend_score = 1.0
        keyword_trend = "insufficient_data"

    # Volume/confidence bonus
    high_confidence = sum(1 for s in all_sentiments if s.get("confidence", 0) >= 4)
    volume_bonus = min(1.0, high_confidence / max(total_classified, 1) * 2)

    # Composite score
    thesis_score = round(sentiment_score + guidance_score + trend_score + volume_bonus, 1)

    # Determine direction
    if thesis_score >= 7:
        direction = "strongly_confirming"
    elif thesis_score >= 5.5:
        direction = "confirming"
    elif thesis_score >= 4:
        direction = "neutral"
    elif thesis_score >= 2.5:
        direction = "challenging"
    else:
        direction = "strongly_challenging"

    # Get best quotes (high confidence, positive sentiment) with deduplication
    key_quotes = []
    seen_quotes = set()
    
    # Helper function to normalize quotes for deduplication
    def normalize_quote(quote):
        # Remove quotes, trim, lowercase for comparison
        return re.sub(r'["\']', '', quote.lower().strip())
    
    # Sort by confidence descending, then by sentiment value
    sentiment_priority = {"CUSTOMER_DEMAND": 0, "INCREASING": 1, "DECREASING": 2, "NEUTRAL": 3}
    sorted_sentiments = sorted(all_sentiments, 
                              key=lambda x: (-x.get("confidence", 0), sentiment_priority.get(x.get("sentiment", "NEUTRAL"), 3)))
    
    for s in sorted_sentiments:
        q = s.get("key_quote")
        if not q or len(q) <= 10:
            continue
            
        normalized = normalize_quote(q)
        
        # Check for duplicates or very similar quotes
        is_duplicate = False
        for seen in seen_quotes:
            # Check if quotes are substantially similar (>80% overlap)
            if len(set(normalized.split()) & set(seen.split())) / max(len(set(normalized.split())), len(set(seen.split()))) > 0.8:
                is_duplicate = True
                break
        
        if not is_duplicate:
            key_quotes.append(q)
            seen_quotes.add(normalized)
            
        if len(key_quotes) >= 5:
            break

    return {
        "ticker": ticker,
        "company": company,
        "thesis_score": thesis_score,
        "direction": direction,
        "key_quotes": key_quotes,
        "keyword_trend": keyword_trend,
        "quarters_analyzed": len(ticker_qoq),
        "sentiment_breakdown": {
            "INCREASING": increasing,
            "DECREASING": decreasing,
            "CUSTOMER_DEMAND": customer_demand,
            "NEUTRAL": neutral,
        },
        "guidance_counts": {
            "bullish": total_bullish,
            "bearish": total_bearish,
            "customer_signal": total_customer,
        },
        "score_components": {
            "sentiment": round(sentiment_score, 2),
            "guidance": round(guidance_score, 2),
            "trend": round(trend_score, 2),
            "confidence_bonus": round(volume_bonus, 2),
        },
    }


class EarningsNLP:
    """
    Main interface for earnings NLP analysis.
    Handles loading transcripts, processing through GPT, caching results.
    """

    def __init__(self, cache_path: Optional[Path] = None):
        self.cache_path = cache_path or CACHE_FILE
        self._cache = None
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = _get_openai_client()
        return self._client

    def load_cache(self) -> dict:
        """Load cached NLP results from disk."""
        if self._cache is not None:
            return self._cache

        if self.cache_path.exists():
            try:
                self._cache = json.loads(self.cache_path.read_text())
                return self._cache
            except Exception:
                pass

        self._cache = {
            "processed_at": None,
            "tickers_processed": [],
            "sentiment_results": {},   # ticker -> [per-transcript results]
            "guidance_results": {},    # ticker -> [per-transcript guidance]
            "qoq_acceleration": {},    # overall Q-over-Q data
            "thesis_scores": {},       # ticker -> score dict
        }
        return self._cache

    def save_cache(self):
        """Save NLP results to disk."""
        if self._cache:
            self._cache["processed_at"] = datetime.now().isoformat()
            self.cache_path.write_text(json.dumps(self._cache, indent=2, default=str))
            logger.info(f"Cache saved to {self.cache_path}")

    def load_transcripts(self, tickers: Optional[list[str]] = None) -> dict:
        """
        Load transcript files, optionally filtering to specific tickers.
        Returns {ticker: [transcript_dicts]}.
        """
        if not TRANSCRIPT_DIR.exists():
            logger.warning(f"Transcript directory not found: {TRANSCRIPT_DIR}")
            return {}

        result = {}
        for ticker_dir in sorted(TRANSCRIPT_DIR.iterdir()):
            if not ticker_dir.is_dir():
                continue
            ticker = ticker_dir.name
            if tickers and ticker not in tickers:
                continue

            transcripts = []
            for fpath in sorted(ticker_dir.glob("*.json")):
                try:
                    data = json.loads(fpath.read_text())
                    transcripts.append(data)
                except Exception as e:
                    logger.warning(f"Error loading {fpath}: {e}")

            if transcripts:
                result[ticker] = transcripts

        return result

    def process_ticker(self, ticker: str, transcripts: list[dict], use_llm: bool = True) -> dict:
        """
        Process all transcripts for a single ticker.
        Returns {sentiment_results: [...], guidance_results: [...]}.
        """
        logger.info(f"Processing {ticker} ({len(transcripts)} transcripts)...")

        sentiment_results = []
        guidance_results = []

        for t in transcripts:
            text = t.get("text", "")
            company = t.get("company", ticker)
            event_date = t.get("event_date", "")
            title = t.get("title", "")

            if not text:
                continue

            # Extract keyword snippets
            snippets = extract_keyword_snippets(text, THESIS_KEYWORDS, window=500)

            if snippets and use_llm:
                # Classify with GPT-4o-mini
                classified = classify_snippets_batch(self.client, snippets, ticker, company)
            else:
                # Fallback: mark all as NEUTRAL (no LLM)
                classified = snippets
                for s in classified:
                    s["sentiment"] = "NEUTRAL"
                    s["confidence"] = 1

            # Detect management guidance (regex-based, no LLM needed)
            guidance = detect_management_guidance(text)

            sentiment_results.append({
                "event_date": event_date,
                "title": title,
                "snippets": classified,
                "snippet_count": len(classified),
            })

            guidance_results.append({
                "event_date": event_date,
                "title": title,
                **guidance,
            })

        return {
            "sentiment_results": sentiment_results,
            "guidance_results": guidance_results,
        }

    def process_sample(self, tickers: Optional[list[str]] = None, use_llm: bool = True):
        """
        Process sample tickers (or specified list) and cache results.
        """
        tickers = tickers or SAMPLE_TICKERS
        cache = self.load_cache()

        # Load transcripts
        all_transcripts = self.load_transcripts(tickers=None)  # Load all for Q-over-Q
        sample_transcripts = {t: ts for t, ts in all_transcripts.items() if t in tickers}

        logger.info(f"Processing {len(sample_transcripts)} tickers ({sum(len(v) for v in sample_transcripts.values())} transcripts)")

        # Process each ticker
        for ticker, transcripts in sample_transcripts.items():
            # Skip if already cached (unless forced)
            if ticker in cache.get("sentiment_results", {}) and not use_llm:
                logger.info(f"Skipping {ticker} (already cached)")
                continue

            result = self.process_ticker(ticker, transcripts, use_llm=use_llm)
            cache["sentiment_results"][ticker] = result["sentiment_results"]
            cache["guidance_results"][ticker] = result["guidance_results"]

        # Compute Q-over-Q acceleration (using ALL transcripts for broad view)
        cache["qoq_acceleration"] = compute_qoq_acceleration(all_transcripts)

        # Compute thesis scores for processed tickers
        for ticker in sample_transcripts:
            company = sample_transcripts[ticker][0].get("company", ticker) if sample_transcripts[ticker] else ticker
            score = compute_thesis_score(
                ticker=ticker,
                company=company,
                sentiment_results=cache["sentiment_results"].get(ticker, []),
                guidance_results=cache["guidance_results"].get(ticker, []),
                qoq_data=cache["qoq_acceleration"],
            )
            cache["thesis_scores"][ticker] = score

        cache["tickers_processed"] = list(set(cache.get("tickers_processed", []) + list(sample_transcripts.keys())))

        self._cache = cache
        self.save_cache()

        logger.info(f"Done! Processed {len(sample_transcripts)} tickers")

    def get_cached_results(self) -> dict:
        """
        Load cached results for dashboard consumption.
        Returns the full cache dict with all analysis results.
        """
        return self.load_cache()

    def get_sentiment_summary(self) -> dict:
        """
        Get aggregate sentiment breakdown across all processed tickers.
        Returns {INCREASING: n, DECREASING: n, CUSTOMER_DEMAND: n, NEUTRAL: n}.
        """
        cache = self.load_cache()
        totals = {"INCREASING": 0, "DECREASING": 0, "CUSTOMER_DEMAND": 0, "NEUTRAL": 0}

        for ticker, results in cache.get("sentiment_results", {}).items():
            for result in results:
                for snippet in result.get("snippets", []):
                    sent = snippet.get("sentiment", "NEUTRAL")
                    if sent in totals:
                        totals[sent] += 1

        return totals

    def get_thesis_rankings(self) -> list[dict]:
        """
        Get company thesis scores sorted by score (descending).
        """
        cache = self.load_cache()
        scores = list(cache.get("thesis_scores", {}).values())
        return sorted(scores, key=lambda x: x.get("thesis_score", 0), reverse=True)

    def get_key_quotes(self, limit: int = 20) -> list[dict]:
        """
        Get the most impactful quotes across all processed tickers.
        Returns [{ticker, company, event_date, quote, sentiment, confidence}].
        """
        cache = self.load_cache()
        quotes = []

        for ticker, results in cache.get("sentiment_results", {}).items():
            for result in results:
                for snippet in result.get("snippets", []):
                    if snippet.get("key_quote") and snippet.get("confidence", 0) >= 3:
                        quotes.append({
                            "ticker": ticker,
                            "event_date": result.get("event_date", ""),
                            "title": result.get("title", ""),
                            "quote": snippet["key_quote"],
                            "keyword": snippet.get("keyword", ""),
                            "sentiment": snippet.get("sentiment", "NEUTRAL"),
                            "confidence": snippet.get("confidence", 1),
                        })

        # Sort by confidence descending, then by sentiment value
        sentiment_order = {"CUSTOMER_DEMAND": 0, "INCREASING": 1, "DECREASING": 2, "NEUTRAL": 3}
        quotes.sort(key=lambda x: (-x["confidence"], sentiment_order.get(x["sentiment"], 3)))

        return quotes[:limit]

    def get_qoq_data(self) -> dict:
        """Get Q-over-Q acceleration data for charting."""
        cache = self.load_cache()
        return cache.get("qoq_acceleration", {})

    def get_guidance_summary(self) -> dict:
        """
        Get aggregate management guidance counts.
        Returns {bullish: n, bearish: n, customer_signal: n, by_ticker: {...}}.
        """
        cache = self.load_cache()
        totals = {"bullish": 0, "bearish": 0, "customer_signal": 0}
        by_ticker = {}

        for ticker, results in cache.get("guidance_results", {}).items():
            ticker_totals = {"bullish": 0, "bearish": 0, "customer_signal": 0}
            for result in results:
                for key in ["bullish", "bearish", "customer_signal"]:
                    count = len(result.get(key, []))
                    totals[key] += count
                    ticker_totals[key] += count
            by_ticker[ticker] = ticker_totals

        return {**totals, "by_ticker": by_ticker}


def main():
    """CLI entrypoint for processing earnings transcripts."""
    parser = argparse.ArgumentParser(description="Earnings NLP Enhancement")
    parser.add_argument("--process-all", action="store_true", help="Process all transcripts (expensive)")
    parser.add_argument("--tickers", nargs="+", help="Process specific tickers")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM classification (regex only)")
    parser.add_argument("--force", action="store_true", help="Force re-processing even if cached")
    args = parser.parse_args()

    nlp = EarningsNLP()

    if args.process_all:
        # Load all available tickers
        all_transcripts = nlp.load_transcripts()
        tickers = list(all_transcripts.keys())
        logger.info(f"Processing ALL {len(tickers)} tickers...")
    elif args.tickers:
        tickers = args.tickers
    else:
        tickers = SAMPLE_TICKERS

    nlp.process_sample(tickers=tickers, use_llm=not args.no_llm)

    # Print summary
    summary = nlp.get_sentiment_summary()
    rankings = nlp.get_thesis_rankings()
    quotes = nlp.get_key_quotes(limit=10)

    print("\n" + "=" * 60)
    print("EARNINGS NLP ANALYSIS SUMMARY")
    print("=" * 60)

    print(f"\nSentiment Breakdown:")
    for sent, count in summary.items():
        bar = "█" * min(count, 40)
        print(f"  {sent:20s} {count:4d} {bar}")

    print(f"\nCompany Thesis Rankings:")
    print(f"  {'Ticker':<8} {'Score':>6} {'Direction':<22} {'Trend':<15} {'Qtrs':>4}")
    print(f"  {'-'*8} {'-'*6} {'-'*22} {'-'*15} {'-'*4}")
    for r in rankings:
        print(f"  {r['ticker']:<8} {r['thesis_score']:>6.1f} {r['direction']:<22} {r['keyword_trend']:<15} {r['quarters_analyzed']:>4}")

    if quotes:
        print(f"\nTop Quotes:")
        for q in quotes[:5]:
            icon = {"INCREASING": "📈", "DECREASING": "📉", "CUSTOMER_DEMAND": "🎯", "NEUTRAL": "⚪"}.get(q["sentiment"], "⚪")
            print(f"  {icon} [{q['ticker']}] \"{q['quote']}\"")

    print(f"\nResults cached to: {CACHE_FILE}")


if __name__ == "__main__":
    main()