#!/usr/bin/env python3
"""
Reliable concurrent article scorer using ThreadPoolExecutor + httpx.
Real thread parallelism with hard timeouts on every HTTP call.
"""

import json
import os
import re
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime
import httpx

DATA_DIR = Path(__file__).parent / "data"
NEWS_STORE = DATA_DIR / "news_articles.json"
SCORES_FILE = DATA_DIR / "article_scores.json"

BATCH_SIZE = 25
WORKERS = 6
MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """You are an expert analyst scoring news articles for relevance to the Leapwork investment thesis.

## THE THESIS
Enterprise software spend is shifting from 70/20/10 (build/maintain/test) toward 20/40/40 (build/validate/operate+govern) because AI coding agents compress the build phase, creating an explosion of software that must be tested, validated, secured, and governed.

## DIMENSIONS (score each 0-5)
- BC (Build Compression): AI making software creation faster/cheaper
- VE (Validation Expansion): Testing/QA/validation growing in importance
- GR (Governance & Risk): Governance, compliance, risk from AI/automation
- MN (Market Narrative Shift): Language/framing supporting spend-shift

## OTHER FIELDS
- r (relevance 1-5): Overall relevance (1=noise, 5=bull's-eye)
- s (strength 1-5): Evidence quality (1=anecdote, 5=hard data)
- d (direction -1/0/+1): +1=supports, -1=challenges, 0=neutral

## OUTPUT
{"results": [{"idx": <number>, "r": <1-5>, "bc": <0-5>, "ve": <0-5>, "gr": <0-5>, "mn": <0-5>, "s": <1-5>, "d": <-1|0|1>}, ...]}
Return ALL articles. ONLY JSON."""


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

def is_v2(entry):
    return "dimension_scores" in entry

def build_entry(result, article):
    r = max(1, min(5, result.get("r", 1)))
    bc = max(0, min(5, result.get("bc", 0)))
    ve = max(0, min(5, result.get("ve", 0)))
    gr = max(0, min(5, result.get("gr", 0)))
    mn = max(0, min(5, result.get("mn", 0)))
    s = max(1, min(5, result.get("s", 1)))
    d = result.get("d", 0)
    if d not in (-1, 0, 1):
        d = 0
    ds = {"BC": bc, "VE": ve, "GR": gr, "MN": mn}
    return {
        "relevance": r, "dimensions": [k for k, v in ds.items() if v >= 2],
        "dimension_scores": ds, "strength": s, "direction": d,
        "source_category": article.get("source_category", "?"),
        "composite_score": r * s * d,
        "scored_at": datetime.now().isoformat(), "schema_version": 2,
    }


def score_batch(api_key, articles_batch, batch_num, total):
    """Score one batch using httpx with hard timeout."""
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
            with httpx.Client(timeout=httpx.Timeout(connect=10, read=90, write=10, pool=10)) as client:
                resp = client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": MODEL,
                        "max_tokens": 4096,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": 0.0,
                        "response_format": {"type": "json_object"},
                    },
                )

            if resp.status_code == 429:
                wait = 20 * (attempt + 1)
                print(f"  ⏳ B{batch_num}: 429, wait {wait}s", flush=True)
                time.sleep(wait)
                continue

            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"].strip()
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

        except httpx.TimeoutException:
            if attempt < 2:
                print(f"  ⏳ B{batch_num}: timeout, retry {attempt+1}", flush=True)
                time.sleep(5)
            else:
                print(f"  ❌ B{batch_num}: timeout x3", flush=True)
                return batch_num, []

        except Exception as e:
            if attempt < 2:
                time.sleep(3)
            else:
                print(f"  ❌ B{batch_num}: {str(e)[:60]}", flush=True)
                return batch_num, []

    return batch_num, []


def main():
    api_key = get_api_key()
    articles = load_articles()
    scores = load_scores()

    to_score = []
    for i, article in enumerate(articles):
        uid = article.get("uid", str(i))
        if uid not in scores or not is_v2(scores[uid]):
            to_score.append((i, uid, article))

    v2_count = sum(1 for s in scores.values() if is_v2(s))
    print(f"📊 V2={v2_count} | To score={len(to_score)} | Total={len(articles)}", flush=True)

    if not to_score:
        print("✅ All done!", flush=True)
        return

    batches = [to_score[i:i+BATCH_SIZE] for i in range(0, len(to_score), BATCH_SIZE)]
    total = len(batches)
    print(f"🚀 {total} batches × {BATCH_SIZE}, {WORKERS} workers", flush=True)

    scored_total = 0
    failed_total = 0
    start_time = time.time()
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {}
        for bi, batch in enumerate(batches):
            batch_articles = [item[2] for item in batch]
            f = pool.submit(score_batch, api_key, batch_articles, bi + 1, total)
            futures[f] = batch

        completed = 0
        for future in as_completed(futures):
            batch = futures[future]
            completed += 1

            try:
                batch_num, results = future.result(timeout=300)
            except Exception as e:
                print(f"  ❌ future error: {e}", flush=True)
                failed_total += 1
                continue

            if not results:
                failed_total += 1
                if completed % 10 == 0:
                    print(f"  [{completed}/{total}] (batch had no results)", flush=True)
                continue

            batch_uids = [item[1] for item in batch]
            batch_articles = [item[2] for item in batch]

            batch_scored = 0
            with lock:
                for result in results:
                    idx = result.get("idx", -1)
                    if 0 <= idx < len(batch):
                        uid = batch_uids[idx]
                        scores[uid] = build_entry(result, batch_articles[idx])
                        batch_scored += 1
                        scored_total += 1

                # Save every 5 completed batches
                if completed % 5 == 0 or completed == total:
                    save_scores(scores)

            elapsed = time.time() - start_time
            rate = scored_total / elapsed * 60 if elapsed > 0 else 0
            remaining = len(to_score) - scored_total
            eta = remaining / rate if rate > 0 else 999

            if completed % 5 == 0 or completed == total:
                v2_now = v2_count + scored_total
                print(f"  [{completed}/{total}] V2={v2_now} +{scored_total} | {rate:.0f}/min | ETA {eta:.0f}m | fail={failed_total}", flush=True)

    save_scores(scores)
    elapsed = time.time() - start_time
    v2_final = sum(1 for s in scores.values() if is_v2(s))
    print(f"\n{'='*60}", flush=True)
    print(f"  DONE — +{scored_total} scored, {failed_total} failed", flush=True)
    print(f"  V2 total: {v2_final} | Time: {elapsed/60:.1f}min | {scored_total/max(elapsed,1)*60:.0f}/min", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
