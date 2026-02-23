#!/usr/bin/env python3
"""
Fast concurrent article scorer — runs multiple API calls in parallel.
Uses asyncio + OpenAI async client for ~10x speedup over sequential.
"""

import asyncio
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(__file__).parent / "data"
NEWS_STORE = DATA_DIR / "news_articles.json"
SCORES_FILE = DATA_DIR / "article_scores.json"

BATCH_SIZE = 25          # articles per API call
CONCURRENCY = 8          # parallel API calls
MODEL = "gpt-4o-mini"
DIMENSIONS = ["BC", "VE", "GR", "MN"]

# Same system prompt as score_articles.py
SYSTEM_PROMPT = """You are an expert analyst scoring news articles for relevance to the Leapwork investment thesis.

## THE THESIS

Enterprise software spend is shifting from roughly **70/20/10** (build / maintain / test) toward **20/40/40** (build / validate / operate+govern).

**Why?** AI coding agents (Copilot, Cursor, Devin, Replit Agent, etc.) are compressing the "build" phase — code is written in hours instead of weeks. This creates an EXPLOSION of software surface area that must be tested, validated, secured, governed, and monitored. The bottleneck is no longer "can we build it?" but "can we trust it?"

**Implication:** Companies selling testing, QA automation, validation platforms, AI governance, and operational resilience tools (like Leapwork) are entering a massive TAM expansion.

## YOUR TASK

For each article, provide scores along FOUR independent dimensions plus metadata.

### DIMENSIONS (score each 0-5 independently)

**BC — Build Compression** (the CAUSE)
How much does this article relate to AI making software creation faster/cheaper?
- AI coding tools, copilots, code generation, agent-built software
- "Ship in hours", "citizen developer", "vibe coding", "10x developer"
- Low-code/no-code growth, AI-generated apps
- 0 = unrelated, 3 = moderately related, 5 = directly about build compression

**VE — Validation Expansion** (the THESIS itself)
How much does this article relate to testing/QA/validation growing in importance or spend?
- Testing budget growth, QA automation platforms, testing market consolidation
- Release quality complaints, production bugs from rushed releases
- Test automation tools, CI/CD testing, shift-left testing
- 0 = unrelated, 3 = moderately related, 5 = directly about validation expansion

**GR — Governance & Risk** (the REGULATORY DRIVER)
How much does this article relate to governance, compliance, or risk from AI/automation?
- Security breaches caused by AI or automation speed
- Regulatory mandates (EU AI Act, NIST AI framework, SEC cyber rules)
- Audit failures, model validation requirements, AI safety
- 0 = unrelated, 3 = moderately related, 5 = directly about governance/risk

**MN — Market Narrative Shift** (the SENTIMENT)
How much does this article use language or frame narratives that support the spend-shift?
- "Software sprawl", "AI risk", "automation governance", "operational resilience"
- Industry analysts projecting testing/governance market growth
- Executive quotes about need for validation/trust
- 0 = no narrative signal, 3 = some narrative alignment, 5 = strong narrative signal

### OTHER FIELDS

**relevance** (1-5): Overall relevance to the Leapwork thesis.
- 1 = barely related or off-topic noise
- 2 = tangentially related (general AI/tech)
- 3 = moderately related (software industry, adjacent to thesis)
- 4 = quite relevant (directly about testing, QA, governance, or AI coding)
- 5 = bull's-eye (executive quotes about spend shift, market data on testing growth)

**strength** (1-5): How strong is the EVIDENCE in this article?
- 1 = weak anecdote, clickbait, or vague mention
- 2 = industry commentary without data
- 3 = credible analysis or concrete example
- 4 = data-backed report or named executive quote
- 5 = hard data (market sizing, revenue numbers, survey results, earnings data)

**direction** (-1, 0, or +1):
- +1 = SUPPORTS the thesis (testing/governance spend growing, build costs shrinking, more incidents from AI speed)
- -1 = CHALLENGES the thesis (testing budgets being cut, AI making testing obsolete, governance seen as unnecessary)
-  0 = NEUTRAL or ambiguous

### OUTPUT FORMAT

Respond with a JSON object containing a "results" array. Each element:
```json
{"results": [{"idx": <number>, "r": <1-5>, "bc": <0-5>, "ve": <0-5>, "gr": <0-5>, "mn": <0-5>, "s": <1-5>, "d": <-1|0|1>}, ...]}
```

Where: idx=article index, r=relevance, bc/ve/gr/mn=dimension scores, s=strength, d=direction.
You MUST return ALL articles in the batch. Return one entry per article.

ONLY output the JSON object. No explanation, no markdown fences."""


def get_api_key():
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    zshrc = Path.home() / ".zshrc"
    if zshrc.exists():
        for line in zshrc.read_text().splitlines():
            if "OPENAI_API_KEY" in line and "export" in line:
                match = re.search(r'OPENAI_API_KEY="([^"]+)"', line)
                if match:
                    os.environ["OPENAI_API_KEY"] = match.group(1)
                    return match.group(1)
    raise ValueError("OPENAI_API_KEY not found")


def load_articles():
    return json.loads(NEWS_STORE.read_text()) if NEWS_STORE.exists() else []


def load_scores():
    if SCORES_FILE.exists():
        data = json.loads(SCORES_FILE.read_text())
        return data if isinstance(data, dict) else {}
    return {}


def save_scores(scores):
    SCORES_FILE.write_text(json.dumps(scores, indent=2))


def is_v2_score(entry):
    return "dimension_scores" in entry


def build_v2_entry(result, article):
    relevance = max(1, min(5, result.get("r", 1)))
    bc = max(0, min(5, result.get("bc", 0)))
    ve = max(0, min(5, result.get("ve", 0)))
    gr = max(0, min(5, result.get("gr", 0)))
    mn = max(0, min(5, result.get("mn", 0)))
    strength = max(1, min(5, result.get("s", 1)))
    direction = result.get("d", 0)
    if direction not in (-1, 0, 1):
        direction = 0

    dimension_scores = {"BC": bc, "VE": ve, "GR": gr, "MN": mn}
    dimensions = [dim for dim, score in dimension_scores.items() if score >= 2]
    source_category = article.get("source_category", "?")
    composite = relevance * strength * direction

    return {
        "relevance": relevance,
        "dimensions": dimensions,
        "dimension_scores": dimension_scores,
        "strength": strength,
        "direction": direction,
        "source_category": source_category,
        "composite_score": composite,
        "scored_at": datetime.now().isoformat(),
        "schema_version": 2,
    }


async def score_batch_async(client, articles_batch, batch_idx, semaphore, batch_num, total_batches):
    """Score a batch using async API call with semaphore for concurrency control."""
    async with semaphore:
        lines = []
        for i, article in enumerate(articles_batch):
            title = article.get("title", "").strip()
            source = article.get("source", "").strip()
            snippet = article.get("snippet", "").strip()
            cat = article.get("source_category", "?")
            entry = f"{batch_idx + i}. [{source}|{cat}] {title}"
            if snippet:
                entry += f" — {snippet[:150]}"
            lines.append(entry)

        prompt = "Score these articles:\n\n" + "\n".join(lines)

        for attempt in range(3):
            try:
                response = await client.chat.completions.create(
                    model=MODEL,
                    max_tokens=4096,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                    timeout=120,
                )

                text = response.choices[0].message.content.strip()
                parsed = json.loads(text)

                if isinstance(parsed, dict):
                    for v in parsed.values():
                        if isinstance(v, list):
                            print(f"  ✅ Batch {batch_num}/{total_batches}: {len(v)} scored", flush=True)
                            return v
                    return []
                elif isinstance(parsed, list):
                    print(f"  ✅ Batch {batch_num}/{total_batches}: {len(parsed)} scored", flush=True)
                    return parsed
                return []

            except Exception as e:
                err_str = str(e)
                if "rate_limit" in err_str.lower() or "429" in err_str:
                    wait = 10 * (attempt + 1)
                    print(f"  ⏳ Batch {batch_num}: rate limited, waiting {wait}s (attempt {attempt+1}/3)", flush=True)
                    await asyncio.sleep(wait)
                elif attempt < 2:
                    print(f"  ⚠️ Batch {batch_num}: {e}, retrying ({attempt+1}/3)", flush=True)
                    await asyncio.sleep(3)
                else:
                    print(f"  ❌ Batch {batch_num}: failed after 3 attempts: {e}", flush=True)
                    return []


async def main():
    from openai import AsyncOpenAI

    api_key = get_api_key()
    client = AsyncOpenAI(api_key=api_key)

    articles = load_articles()
    scores = load_scores()

    # Find articles needing V2 scoring
    to_score = []
    for i, article in enumerate(articles):
        uid = article.get("uid", str(i))
        if uid not in scores or not is_v2_score(scores[uid]):
            to_score.append((i, uid, article))

    v2_count = sum(1 for s in scores.values() if is_v2_score(s))
    print(f"📊 Total articles:  {len(articles)}")
    print(f"📊 Already scored:  {len(scores)}")
    print(f"📊 V2 scored:       {v2_count}")
    print(f"📊 To score (v2):   {len(to_score)}")
    print(f"📊 Concurrency:     {CONCURRENCY}")
    print(f"📊 Batch size:      {BATCH_SIZE}")

    if not to_score:
        print("\n✅ All articles already have V2 scores!")
        return

    # Build batches
    batches = []
    for start in range(0, len(to_score), BATCH_SIZE):
        batches.append(to_score[start:start + BATCH_SIZE])

    total_batches = len(batches)
    print(f"\n🚀 Scoring {len(to_score)} articles in {total_batches} batches ({CONCURRENCY} concurrent)...\n")

    semaphore = asyncio.Semaphore(CONCURRENCY)
    scored_total = 0
    failed_total = 0
    start_time = time.time()

    # Process in waves of concurrent batches, saving after each wave
    wave_size = CONCURRENCY * 2  # process 2x concurrency then save

    for wave_start in range(0, total_batches, wave_size):
        wave_batches = batches[wave_start:wave_start + wave_size]
        
        tasks = []
        for bi, batch in enumerate(wave_batches):
            batch_articles = [item[2] for item in batch]
            global_batch_num = wave_start + bi + 1
            # Use 0-based indexing within each batch so LLM returns idx 0..N-1
            task = score_batch_async(client, batch_articles, 0, semaphore, global_batch_num, total_batches)
            tasks.append((batch, task))

        # Run wave concurrently with timeout
        try:
            results_list = await asyncio.wait_for(
                asyncio.gather(*[t[1] for t in tasks], return_exceptions=True),
                timeout=180  # 3 min max per wave
            )
        except asyncio.TimeoutError:
            print(f"  ⏰ Wave timed out after 3min, moving on...", flush=True)
            continue

        # Process results
        wave_scored = 0
        for (batch, _), results in zip(tasks, results_list):
            if isinstance(results, Exception):
                print(f"  ❌ Wave exception: {results}", flush=True)
                failed_total += 1
                continue
            if not results:
                failed_total += 1
                continue

            batch_uids = [item[1] for item in batch]
            batch_articles = [item[2] for item in batch]

            for result in results:
                idx = result.get("idx", -1)
                if 0 <= idx < len(batch):
                    uid = batch_uids[idx]
                    article = batch_articles[idx]
                    scores[uid] = build_v2_entry(result, article)
                    wave_scored += 1
                    scored_total += 1

        # Save after each wave
        save_scores(scores)
        elapsed = time.time() - start_time
        rate = scored_total / elapsed * 60 if elapsed > 0 else 0
        v2_now = sum(1 for s in scores.values() if is_v2_score(s))
        remaining = len(to_score) - scored_total
        eta_min = remaining / rate if rate > 0 else 999
        print(f"  💾 Saved. Progress: {v2_now} V2 scored | {scored_total}/{len(to_score)} this run | {rate:.0f}/min | ETA: {eta_min:.0f}min", flush=True)

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  SCORING COMPLETE")
    print(f"{'='*60}")
    print(f"  Scored:          {scored_total}")
    print(f"  Failed batches:  {failed_total}")
    print(f"  Total V2 scores: {sum(1 for s in scores.values() if is_v2_score(s))}")
    print(f"  Time:            {elapsed/60:.1f} minutes")
    print(f"  Rate:            {scored_total/elapsed*60:.0f} articles/min")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
