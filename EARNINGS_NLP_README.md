# Earnings NLP Enhancement

## Overview

The `earnings_nlp.py` module enhances the Earnings Call Intelligence tab with AI-powered contextual sentiment analysis. Instead of just counting keyword mentions, it classifies **what companies are actually saying** about testing, validation, governance, and security.

## The Problem

Before: "We're cutting our testing budget" and "We're doubling our testing budget" both count as +1 for "testing."

After: The first gets classified as **DECREASING** and the second as **INCREASING**, with management quotes extracted and a composite thesis score computed per company.

## Architecture

```
earnings_nlp.py
├── extract_keyword_snippets()      # Find keywords + surrounding 500 chars
├── classify_snippets_batch()       # GPT-4o-mini sentiment classification
├── detect_management_guidance()    # Regex pattern matching (no LLM)
├── compute_qoq_acceleration()     # Quarter-over-quarter trend analysis
├── compute_thesis_score()          # Composite company scoring
└── class EarningsNLP               # Main interface + caching
```

## What It Does

### 1. Contextual Sentiment Extraction (GPT-4o-mini)
For each keyword hit in a transcript, extracts surrounding 500 characters and classifies:
- **INCREASING** — company is spending more / prioritizing this
- **DECREASING** — company is cutting / deprioritizing this
- **CUSTOMER_DEMAND** — company mentions customer demand (most valuable signal)
- **NEUTRAL** — mentioned but no directional signal

Processes in batches of 15 snippets per API call for efficiency.

### 2. Management Guidance Detection (Regex)
Flags specific language patterns — no LLM needed, instant:
- **BULLISH:** "investing in validation", "quality is a priority", "testing spend increasing"
- **BEARISH:** "reducing QA headcount", "automating away testing", "cutting testing budget"
- **CUSTOMER_SIGNAL:** "customers asking for", "demand for testing/validation"

### 3. Quarter-over-Quarter Acceleration
Computes keyword mention counts per quarter across ALL companies (not just sample):
- Absolute counts per quarter per keyword group
- Growth rate (Q-over-Q % change)
- Acceleration (is growth rate itself increasing?)

### 4. Company Thesis Scoring (0-10)
Each company gets a composite score:

| Component | Weight | Description |
|-----------|--------|-------------|
| Sentiment balance | 0-4 pts | INCREASING + CUSTOMER_DEMAND vs DECREASING |
| Management guidance | 0-3 pts | Bullish vs bearish pattern matches |
| Keyword trend | 0-2 pts | Q-over-Q acceleration |
| Confidence bonus | 0-1 pt | High-confidence classifications |

Scores map to directions:
- 7-10: Strongly Confirming
- 5.5-7: Confirming
- 4-5.5: Neutral
- 2.5-4: Challenging
- 0-2.5: Strongly Challenging

## Usage

### Generate Cache (CLI)

```bash
# Process sample companies (10 tickers, ~38 transcripts)
cd ~/clawd/dashboards/software-shift
source .venv/bin/activate
python earnings_nlp.py

# Process specific tickers
python earnings_nlp.py --tickers DDOG CRWD ZS

# Process ALL transcripts (expensive — ~$2-5 in API costs)
python earnings_nlp.py --process-all

# Skip LLM, regex patterns only (free, instant)
python earnings_nlp.py --no-llm

# Force re-processing even if cached
python earnings_nlp.py --force
```

### Dashboard Integration

The Earnings Intel tab in `app.py` automatically loads `data/earnings_nlp_cache.json` and shows:
1. **Sentiment Breakdown** — Bar chart of INCREASING/DECREASING/CUSTOMER_DEMAND/NEUTRAL
2. **Management Guidance** — Bar chart of bullish/bearish/customer signal pattern matches
3. **Q-over-Q Acceleration** — Stacked area chart of keyword mentions per quarter
4. **Company Thesis Ranking** — Table sorted by thesis score with progress bars
5. **Key Quotes** — Most impactful management quotes with sentiment badges

If the cache doesn't exist, a friendly info message prompts the user to run the processor.

## File Structure

```
data/
├── earnings_nlp_cache.json    # Cached NLP results (~200KB)
├── transcripts/
│   ├── DDOG/
│   │   ├── DDOG_Q2_2025_Earnings_Call_2025-08-07.json
│   │   └── ...
│   └── ... (96 ticker folders)
```

## Cache Format

```json
{
  "processed_at": "2026-02-19T...",
  "tickers_processed": ["DDOG", "DT", ...],
  "sentiment_results": {
    "DDOG": [{
      "event_date": "2026-02-10",
      "title": "Q4 2025 Earnings Call",
      "snippets": [{
        "keyword": "observability",
        "snippet": "...context...",
        "sentiment": "INCREASING",
        "confidence": 4,
        "key_quote": "We're seeing unprecedented demand..."
      }]
    }]
  },
  "guidance_results": { ... },
  "qoq_acceleration": {
    "quarterly_totals": [...],
    "by_ticker": { ... },
    "acceleration": {
      "overall_trend": "accelerating",
      "latest_growth_rate": 12.5,
      "growth_rate_change": 3.2
    }
  },
  "thesis_scores": {
    "DDOG": {
      "ticker": "DDOG",
      "thesis_score": 8.5,
      "direction": "strongly_confirming",
      "key_quotes": [...],
      "keyword_trend": "accelerating",
      "quarters_analyzed": 3,
      "sentiment_breakdown": { ... },
      "score_components": { ... }
    }
  }
}
```

## Costs

- **Sample run (10 tickers):** ~$0.10-0.30 in GPT-4o-mini API costs
- **Full run (96 tickers):** ~$2-5 in API costs
- Results are cached so you don't re-pay on rerun

## Sample Companies

Default sample: DDOG, DT, CRWD, PANW, ZS, QLYS, TENB, ESTC, GTLB, TEAM

These were chosen as the most thesis-relevant companies across testing, security, observability, governance, and dev tools.
