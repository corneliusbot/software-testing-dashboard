#!/usr/bin/env python3
"""
Fast article scorer v2 — ThreadPoolExecutor for true parallel HTTP with timeouts.
Each batch gets its own thread with httpx timeout guarantees.
"""

import json
import os
import re
import sys
import time
import signal
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(__file__).parent / "data"
NEWS_STORE = DATA_DIR / "news_articles.json"
SCORES_FILE = DATA_DIR / "article_scores.json"

BATCH_SIZE = 25
WORKERS = 8
MODEL = "gpt-4o-mini"
DIMENSIONS = ["BC", "VE", "GR", "MN"]

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
- 0 = unrelated, 3 = moderately related, 5 = directly about build compression

**VE — Validation Expansion** (the THESIS itself)
How much does this article relate to testing/QA/validation growing in importance or spend?
- Testing budget growth, QA automation platforms, testing market consolidation
- 0 = unrelated, 3 = moderately related, 5 = directly about validation expansion

**GR — Governance & Risk** (the REGULATORY DRIVER)
How much does this article relate to governance, compliance, or risk from AI/automation?
- Security breaches, regulatory mandates, audit failures, AI safety
- 0 = unrelated, 3 = moderately related, 5 = directly about governance/risk

**MN — Market Narrative Shift** (the SENTIMENT)
How much does this article use language that supports the spend-shift narrative?
- "Software sprawl", "AI risk", "operational resilience", market growth projections
- 0 = no narrative signal, 3 = some alignment, 5 = strong narrative signal

### OTHER FIELDS

**relevance** (1-5): Overall relevance to the Leapwork thesis (1=noise, 5=bull's-eye)
**strength** (1-5): Evidence quality (1=anecdote, 5=hard data)
**direction** (-1, 0, or +1): +1=supports thesis, -1=challenges, 0=neutral

### OUTPUT FORMAT

Respond with a JSON object: {"results": [{"idx": <number>, "r": <1-5>, "bc": <0-5>, "ve": <0-5>, "gr": <0-5>, "mn": <0-5>, "s": <1-5>, "d": <-1|0|1>}, ...]}
Return ALL articles. ONLY output JSON, no explanation."""


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
    return {
        "relevance": relevance,
        "dimensions": dimensions,
        "dimension_scores": dimension_scores,
        "strength": strength,
        "direction": direction,
        "source_category": article.get("source_category", "?"),
        "composite_score": relevance * strength * direction,
        "scored_at": datetime.now().isoformat(),
        "schema_version": 2,
    }


def score_one_batch(client, articles_batch, batch_num, total_batches):
    """Score a single batch synchronously (called from thread pool)."""
    lines = []
    for i, article in enumerate(articles_batch):
        title = article.get("title", "").strip()
        source = article.get("source", "").strip()
        snippet = article.get("snippet", "").strip()
        cat = article.get("source_category", "?")
        entry = f"{i}. [{source}|{cat}] {title}"
        if snippet:
            entry += f" — {snippet[:150]}"
        lines.append(entry)

    prompt = "Score these articles:\n\n" + "\n".join(lines)

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                max_tokens=4096,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
                timeout=90,
            )

            text = response.choices[0].message.content.strip()
            parsed = json.loads(text)

            results = []
            if isinstance(parsed, dict):
                for v in parsed.values():
                    if isinstance(v, list):
                        results = v
                        break
            elif isinstance(parsed, list):
                results = parsed

            return batch_num, results

        except Exception as e:
            err_str = str(e).lower()
            if "rate_limit" in err_str or "429" in err_str:
                wait = 15 * (attempt + 1)
                print(f"  ⏳ B{batch_num}: rate limited, wait {wait}s", flush=True)
                time.sleep(wait)
            elif attempt < 2:
                time.sleep(3)
            else:
                print(f"  ❌ B{batch_num}: {str(e)[:80]}", flush=True)
                return batch_num, []


def main():
    from openai import OpenAI

    api_key = get_api_key()
    client = OpenAI(api_key=api_key, timeout=90, max_retries=2)

    articles = load_articles()
    scores = load_scores()

    to_score = []
    for i, article in enumerate(articles):
        uid = article.get("uid", str(i))
        if uid not in scores or not is_v2_score(scores[uid]):
            to_score.append((i, uid, article))

    v2_count = sum(1 for s in scores.values() if is_v2_score(s))
    print(f"📊 Total articles:  {len(articles)}")
    print(f"📊 V2 scored:       {v2_count}")
    print(f"📊 To score:        {len(to_score)}")

    if not to_score:
        print("✅ All done!")
        return

    # Build batches
    batches = []
    for start in range(0, len(to_score), BATCH_SIZE):
        batches.append(to_score[start:start + BATCH_SIZE])

    total = len(batches)
    print(f"🚀 {len(to_score)} articles → {total} batches × {BATCH_SIZE}, {WORKERS} workers\n")

    scored_total = 0
    failed_total = 0
    start_time = time.time()
    save_interval = 20  # save every N batches

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {}
        for bi, batch in enumerate(batches):
            batch_articles = [item[2] for item in batch]
            f = pool.submit(score_one_batch, client, batch_articles, bi + 1, total)
            futures[f] = batch

        completed = 0
        for future in as_completed(futures):
            batch = futures[future]
            completed += 1

            try:
                batch_num, results = future.result(timeout=120)
            except Exception as e:
                print(f"  ❌ batch exception: {e}", flush=True)
                failed_total += 1
                continue

            if not results:
                failed_total += 1
                continue

            batch_uids = [item[1] for item in batch]
            batch_articles = [item[2] for item in batch]

            batch_scored = 0
            for result in results:
                idx = result.get("idx", -1)
                if 0 <= idx < len(batch):
                    uid = batch_uids[idx]
                    scores[uid] = build_v2_entry(result, batch_articles[idx])
                    batch_scored += 1
                    scored_total += 1

            # Periodic save
            if completed % save_interval == 0 or completed == len(batches):
                save_scores(scores)
                elapsed = time.time() - start_time
                rate = scored_total / elapsed * 60 if elapsed > 0 else 0
                v2_now = sum(1 for s in scores.values() if is_v2_score(s))
                remaining = len(to_score) - scored_total
                eta = remaining / rate if rate > 0 else 999
                print(f"  💾 [{completed}/{total}] V2={v2_now} | +{scored_total} this run | {rate:.0f}/min | ETA {eta:.0f}min", flush=True)

    # Final save
    save_scores(scores)
    elapsed = time.time() - start_time
    v2_final = sum(1 for s in scores.values() if is_v2_score(s))
    print(f"\n{'='*60}")
    print(f"  DONE — {scored_total} scored, {failed_total} failed, {v2_final} V2 total")
    print(f"  Time: {elapsed/60:.1f}min, Rate: {scored_total/elapsed*60:.0f}/min")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
