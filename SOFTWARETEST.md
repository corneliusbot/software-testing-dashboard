# SOFTWARETEST.md — Dashboard Development Guide

**Purpose:** Authoritative reference for all future work on the Software Spend Shift dashboard. Every sub-agent working on `app.py` or related files MUST read this first. It captures design decisions, hard rules, and lessons learned from Steve's feedback.

**Last updated:** 2026-02-20

---

## 1. Project Overview

**Thesis:** Enterprise software spend is shifting from 70/20/10 (build/maintain/test) → 20/40/40 (build/validate/operate+govern) due to AI agent explosion. This dashboard tracks evidence for/against that shift.

**End user:** Steve Goulden (CFA, equity research analyst). He uses this to build an investment thesis around Leapwork (private company) and the broader software testing/validation sector.

**Stack:** Streamlit (Python), served on port 8502 via Cloudflare tunnel. Single file: `app.py` (~3,100 lines).

---

## 2. Tab Structure (8 tabs)

| Tab | Name | Content |
|-----|------|---------|
| tab0 | 🎯 Thesis Overview | Composite evidence score (74/100), dimension breakdown (BC/VE/GR/MN), signal strength over time |
| tab1 | 📰 Intelligence Feed | Scored articles with filters by category (A-E) and dimension. Includes scoring key legend. |
| tab2 | 📈 Company Signals | Stock basket comparisons (Testing/Governance vs Dev Tools vs Broad Software). Revenue divergence. |
| tab3 | 🎙️ Earnings Intel | Transcript NLP analysis. Management guidance patterns. Q-over-Q keyword trends. Company thesis rankings. |
| tab4 | 💰 Funding & M&A | Deal cards for testing/validation sector funding rounds and acquisitions. |
| tab5 | 🔍 Search Trends | Google Trends data for testing vs building vs AI coding tool search terms. |
| tab6 | 🚨 Incident Tracker | Security/failure incidents from Category C feeds. AI-attribution analysis. Pain index. |
| tab7 | 📈 Leading Indicators | Google Trends (extended), job market data, additional leading signals. |

---

## 3. Hard Rules (Non-Negotiable)

These are lessons from Steve's direct feedback. Violating any of these will require rework.

### 3.1 Earnings Intel — Quarter Restriction
**RULE: Only show CY Q2 2025, CY Q3 2025, CY Q4 2025 in the Earnings Intel tab.**

```python
VALID_CY_QUARTERS = ["CY Q2 2025", "CY Q3 2025", "CY Q4 2025"]
```

Do NOT dynamically expand this to all available quarters. Steve specifically wants this restricted window. The broader quarter set (`ALL_CY_QUARTERS`) exists for internal computation but must not be used for display filtering in the Earnings tab.

**Why:** Earlier quarters have sparse data and add noise. The thesis tracking starts mid-2025.

### 3.2 Chronological Ordering — Always
**RULE: Any chart, table, or list that involves time periods MUST be in chronological order.**

This sounds obvious but has been broken twice. Specifically:
- Q-over-Q keyword acceleration chart must sort quarters chronologically
- Article lists should default to newest-first
- Incident timelines must be date-ordered

**Implementation:** When using `pd.Categorical` for quarters, always provide an explicitly sorted `categories` list. Parse quarters with a sort key:
```python
def quarter_sort_key(q):
    # "CY Q2 2025" → (2025, 2)
    parts = q.replace("CY ", "").split()
    return (int(parts[1]), int(parts[0][1]))
```

### 3.3 No "Customer Demand" as a Standalone Category
**RULE: Sentiment/guidance categories are ONLY: INCREASING, DECREASING, NEUTRAL.**

The earnings NLP may output `CUSTOMER_DEMAND` as a raw category. This MUST be folded into `INCREASING` before display — customer demand signals are positive indicators.

Never show "Customer Demand" or "CUSTOMER_DEMAND" as a label in any chart, table, or metric. Always remap:
```python
if sentiment == "CUSTOMER_DEMAND":
    sentiment = "INCREASING"
```

**Why:** It doesn't make sense as a peer category alongside increasing/decreasing. It IS an increasing signal.

### 3.4 Foreign Language Title Translation
**RULE: Any article with a non-English title must show an English translation underneath.**

- Pre-computed translations stored in `data/title_translations.json`
- Display format: original title, then `🌐 _Translated title_` in italics below
- Detection: Unicode heuristic (non-Latin character ratio > 20-30%)
- Translation: GPT-4o-mini via `scripts/translate_titles.py`
- Cache loaded with `@st.cache_data` — no API calls at page load time
- Applies to: Intelligence Feed (tab1), Earnings Intel (tab3), Incident Tracker (tab6)

### 3.5 Funding & M&A — Styled Cards
**RULE: Deal entries must use styled HTML card divs, not raw markdown.**

Use the `deal-card` CSS pattern (dark background `#161b22`, border `#30363d`, rounded corners). Raw markdown rendering in Streamlit breaks formatting with mixed HTML. Always use `st.markdown(html, unsafe_allow_html=True)`.

### 3.6 Scoring Key Must Be Visible
**RULE: The Intelligence Feed tab must show a scoring key/legend near the top.**

Using `st.info()`, explain:
- **R** = Relevance (1-5): How relevant to the Leapwork thesis
- **S** = Strength (1-5): Evidence quality (1=anecdote, 5=hard data/exec quote)
- **C** = Composite (-25 to +25): R × S × Direction (+1 supports, 0 neutral, -1 challenges)
- **Dimensions:** BC=Build Compression, VE=Validation Expansion, GR=Governance & Risk, MN=Market Narrative

---

## 4. Article Scoring System (V2)

### 4.1 Four Dimensions
| Code | Name | What It Captures |
|------|------|-----------------|
| BC | Build Compression | AI coding adoption, time-to-production shrinking, citizen dev |
| VE | Validation Expansion | Testing budgets growing, QA automation, platform consolidation |
| GR | Governance & Risk | Security breaches, regulatory mandates, audit failures |
| MN | Market Narrative Shift | Language like "software sprawl", "AI risk", "operational resilience" |

### 4.2 Score Fields
```json
{
  "relevance": 3,           // 1-5
  "dimensions": ["BC", "VE"], // any dimension scoring >= 2
  "dimension_scores": {"BC": 3, "VE": 4, "GR": 0, "MN": 1},
  "strength": 2,            // 1-5 (1=anecdote, 5=hard data)
  "direction": 1,           // -1, 0, +1
  "source_category": "B",   // A-E
  "composite_score": 6,     // relevance × strength × direction
  "schema_version": 2
}
```

### 4.3 Source Categories
| Cat | Name | Sources |
|-----|------|---------|
| A | Enterprise Budget | Gartner, Forrester, IDC, McKinsey, CIO.com, HBR |
| B | Dev Velocity | GitHub, Vercel, OpenAI, Anthropic, Product Hunt |
| C | Security & Failures | KrebsOnSecurity, Dark Reading, The Register, BleepingComputer |
| D | Compliance & Regulation | EU AI Act, NIST, SEC, FCA, FINRA, ISO |
| E | Testing Market | Tricentis, TechCrunch testing, Datadog, testing startups |

### 4.4 Scoring Scripts
- `score_articles.py` — Sequential scorer, reliable, ~20 articles/min
- `score_fast.py` / `score_fast3.py` — Concurrent scorer (ThreadPoolExecutor + httpx), ~500 articles/min
- Use fast scorer for bulk work, sequential for incremental daily scoring

---

## 5. Data Files

| File | Contents | Updated By |
|------|----------|-----------|
| `data/news_articles.json` | 7,121 articles from scanner_v2 | `scanner_v2.py` |
| `data/article_scores.json` | V2 scores keyed by article hash | `score_articles.py` |
| `data/earnings_nlp_cache.json` | Transcript NLP results (sentiment, guidance, Q-over-Q) | `earnings_nlp.py` |
| `data/earnings_quotes.json` | Curated management quotes (manually added via dashboard form) | `app.py` form |
| `data/title_translations.json` | English translations of non-English article titles | `scripts/translate_titles.py` |
| `data/funding_deals.json` | Funding/M&A deals in testing sector | Manual / scanner |
| `data/transcripts/` | Koyfin earnings call transcripts (JSON per company per quarter) | `koyfin_transcripts.py` |
| `data/manual_entries.json` | Manual evidence entries added via dashboard | `app.py` form |
| `data/job_market.json` | Job posting data for testing roles | Scanner |
| `data/market_sizing.json` | TAM/market size data points | Manual |

---

## 6. Composite Evidence Score (Thesis Overview)

The headline number on tab0. Weighted formula:

| Source | Weight | Description |
|--------|--------|-------------|
| Earnings management commentary | 30% | From earnings NLP — guidance sentiment, keyword trends |
| Enterprise budget surveys/reports | 25% | Category A articles scored positively |
| Funding/M&A activity | 20% | Deal volume and valuations in testing space |
| News/article sentiment | 15% | Overall article composite scores |
| Search/job trends | 10% | Google Trends + job posting signals |

**Current score: 74/100 CONFIRMING** (as of Feb 20, 2026)

---

## 7. Earnings NLP Pipeline

### 7.1 Process
1. `koyfin_transcripts.py` pulls transcripts via reverse-engineered Koyfin API
2. `earnings_nlp.py` processes transcripts:
   - Keyword detection (testing, validation, QA, governance, etc.)
   - Contextual sentiment (INCREASING / DECREASING / NEUTRAL)
   - Guidance language detection
   - Q-over-Q acceleration metrics
3. Results cached in `data/earnings_nlp_cache.json`
4. `app.py` reads cache and displays in Earnings Intel tab

### 7.2 Companies Covered (16)
DDOG, DT, CRWD, ZS, PANW, GTLB, TEAM, FROG, CRM, NOW, OKTA, RPD, PATH, QLYS, TENB, ESTC

### 7.3 Display Rules
- Company thesis ranking table shows: Rank, Ticker, Score, Sentiment breakdown, Guidance breakdown
- Top 5 companies get expandable detail cards with quotes
- Management quotes show with source context (not fragments)
- Auto-extracted quotes (from NLP) labeled as such, distinguished from curated quotes

---

## 8. Search Trends

### 8.1 Term Groups
| Group | Terms | Color |
|-------|-------|-------|
| Testing & QA | "AI test automation", "software testing AI", "test automation platform" | Blue |
| AI Development | "AI code generation", "GitHub Copilot", "vibe coding" | Red |
| AI Coding Tools | "Claude Code", "ChatGPT Codex" | Purple (#AB47BC) |

### 8.2 Implementation Notes
- Google Trends (pytrends) max 5 terms per request — split into batches
- Use bridge term for cross-batch normalization if needed
- Show both individual term charts and group average comparison

---

## 9. Incident Tracker Reliability

**Steve asked about this directly.** Be transparent:

- Incidents themselves are **real** — sourced from established security feeds (Category C)
- AI-attribution scoring is **LLM-inferred** — GPT classifying whether AI/automation was a factor
- No ground truth for AI attribution — treat as directional signal only
- Timeline and severity: **reliable**
- "AI caused this" claims: **60-70% reliable** — useful for patterns, don't cite specifics without verifying
- Pain Index is a composite of incidents × severity × AI-attribution — useful for trend, not absolute measure

---

## 10. Common Mistakes to Avoid

| Mistake | What Happened | Rule |
|---------|--------------|------|
| Expanded quarters beyond Q2-Q4 2025 | Sub-agent "helpfully" showed all 7 quarters, adding noise | Stick to VALID_CY_QUARTERS for earnings display |
| Q-over-Q chart not chronological | Quarter categories were in insertion order from cache, not sorted | Always sort with explicit quarter_sort_key() |
| "Customer Demand" shown as category | Raw NLP output leaked into display | Remap to INCREASING before any display |
| Foreign titles untranslated | Scanner collects Korean/Japanese articles, shown as-is | Pre-translate and cache, show 🌐 translation |
| Funding cards in raw markdown | Mixed HTML/markdown breaks Streamlit rendering | Use styled HTML divs with unsafe_allow_html |
| Scoring key missing | Users don't know what R:2 S:1 C:+0 means | Always show st.info() legend in Intelligence Feed |
| BS fallback vol too low | Portfolio dashboard option pricing used 18% vol, showed $6.22 vs $14.31 actual | Use 28% vol for BS fallback, prefer sidecar JSON |
| Sidecar JSON stale | option_prices.json wasn't being refreshed | Run scripts/fetch_option_prices.py regularly |

---

## 11. File Structure

```
dashboards/software-shift/
├── app.py                    # Main dashboard (3,100 lines)
├── earnings_nlp.py           # Transcript NLP processor
├── koyfin_transcripts.py     # Transcript fetcher (Koyfin API)
├── scanner.py                # V1 scanner (deprecated)
├── scanner_v2.py             # V2 multi-source scanner (A-E categories)
├── score_articles.py         # Sequential article scorer
├── score_fast.py             # Concurrent scorer (preferred for bulk)
├── scripts/
│   └── translate_titles.py   # Pre-compute title translations
├── data/
│   ├── news_articles.json    # All collected articles
│   ├── article_scores.json   # V2 scores
│   ├── earnings_nlp_cache.json
│   ├── earnings_quotes.json
│   ├── title_translations.json
│   ├── funding_deals.json
│   ├── job_market.json
│   ├── market_sizing.json
│   ├── manual_entries.json
│   └── transcripts/          # Per-company transcript JSONs
├── SOFTWARETEST.md           # THIS FILE — read before any changes
├── UPGRADE_PLAN.md           # Original upgrade plan (historical)
└── *.md                      # Various change logs (historical)
```

---

## 12. Development Workflow

1. **Read this file first** — every time, no exceptions
2. **Be surgical** — app.py is 3,100 lines; don't rewrite sections that work
3. **Test compilation:** `python3 -c "import py_compile; py_compile.compile('app.py', doraise=True)"`
4. **Auto-reload:** Streamlit on port 8502 auto-reloads on file save
5. **Cloudflare tunnel:** Free tier, URL changes on restart. Check with `ps aux | grep cloudflared`
6. **Cache decorators:** All disk-reading functions use `@st.cache_data` — maintain this pattern
7. **Sub-agents:** When spawning sub-agents for dashboard work, include "Read SOFTWARETEST.md first" in the task prompt

---

*This is a living document. Update it when Steve gives new feedback or when new mistakes are discovered.*
