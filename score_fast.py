#!/usr/bin/env python3
"""
Fast concurrent article scorer — runs multiple API calls in parallel.
Uses the same scoring schema as score_articles.py but with async concurrency.
"""

import asyncio
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
NEWS_STORE = DATA_DIR / "news_articles.json"
SCORES_FILE = DATA_DIR / "article_scores.json"

MODEL = "gpt-4o-mini"
BATCH_SIZE = 20
CONCURRENCY = 5  # Number of parallel API calls
DIMENSIONS = ["BC", "VE", "GR", "MN"]

# Import system prompt from main scorer
sys.path.insert(0, str(Path(__file__).parent))
from score_articles import SYSTEM_PROMPT, build_v2_entry, is_v2_score


def get_api_key():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        zshrc = Path.home() / ".zshrc"
        if zshrc.exists():
            for line in zshrc.read_text().splitlines():
                if "OPENAI_API_KEY" in line and "export" in line:
                    match = re.search(r'OPENAI_API_KEY="([^"]+)"', line)
                    if match:
                        api_key = match.group(1)
                        os.environ["OPENAI_API_KEY"] = api_key
                        break
    return api_key


async def score_batch_async(client, articles_batch, start_idx, batch_num, total_batches):
    """Score a single batch asynchronously."""
    import httpx
    
    lines = []
    for i, article in enumerate(articles_batch):
        title = article.get("title", "").strip()
        source = article.get("source", "").strip()
        snippet = article.get("snippet", "").strip()
        cat = article.get("source_category", "?")
        entry = f"{start_idx + i}. [{source}|{cat}] {title}"
        if snippet:
            entry += f" — {snippet[:150]}"
        lines.append(entry)

    prompt = "Score these articles:\n\n" + "\n".join(lines)

    try:
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=MODEL,
            max_tokens=4096,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )

        text = response.choices[0].message.content.strip()

        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                for v in parsed.values():
                    if isinstance(v, list):
                        return batch_num, parsed, v
                return batch_num, None, []
            elif isinstance(parsed, list):
                return batch_num, None, parsed
            return batch_num, None, []
        except json.JSONDecodeError:
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if match:
                return batch_num, None, json.loads(match.group())
            return batch_num, None, []
    except Exception as e:
        print(f"  Batch {batch_num}/{total_batches}: ❌ Error: {e}", flush=True)
        return batch_num, None, []


async def main():
    from openai import OpenAI
    
    api_key = get_api_key()
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found")
        sys.exit(1)

    client = OpenAI(api_key=api_key, timeout=120.0)

    # Load articles and scores
    articles = json.loads(NEWS_STORE.read_text())
    existing_scores = json.loads(SCORES_FILE.read_text()) if SCORES_FILE.exists() else {}

    # Find unscored articles
    to_score = []
    for i, article in enumerate(articles):
        uid = article.get("uid", str(i))
        if uid not in existing_scores:
            to_score.append((i, uid, article))
        elif not is_v2_score(existing_scores[uid]):
            to_score.append((i, uid, article))

    v2_count = sum(1 for s in existing_scores.values() if is_v2_score(s))
    print(f"📊 Total articles:  {len(articles)}")
    print(f"📊 Already scored:  {len(existing_scores)}")
    print(f"📊 V2 scored:       {v2_count}")
    print(f"📊 To score (v2):   {len(to_score)}")

    if not to_score:
        print("\n✅ Nothing to score! All articles have v2 scores.")
        return

    # Build batches
    batches = []
    for batch_start in range(0, len(to_score), BATCH_SIZE):
        batch = to_score[batch_start:batch_start + BATCH_SIZE]
        batches.append(batch)

    total_batches = len(batches)
    print(f"\n🚀 Scoring {len(to_score)} articles in {total_batches} batches of {BATCH_SIZE}")
    print(f"   Concurrency: {CONCURRENCY} parallel API calls\n")

    scored = 0
    failed = 0
    semaphore = asyncio.Semaphore(CONCURRENCY)

    async def process_batch(batch_idx, batch):
        nonlocal scored, failed
        async with semaphore:
            batch_articles = [item[2] for item in batch]
            batch_uids = [item[1] for item in batch]
            batch_start = batch_idx * BATCH_SIZE

            batch_num, _, results = await score_batch_async(
                client, batch_articles, batch_start, batch_idx + 1, total_batches
            )

            if results:
                batch_scored = 0
                for result in results:
                    idx = result.get("idx", -1)
                    local_idx = idx - batch_start
                    if 0 <= local_idx < len(batch):
                        uid = batch_uids[local_idx]
                        article = batch_articles[local_idx]
                        existing_scores[uid] = build_v2_entry(result, article)
                        batch_scored += 1
                        scored += 1
                print(f"  Batch {batch_num}/{total_batches}: ✅ {batch_scored}/{len(batch)} scored", flush=True)
            else:
                failed += 1
                print(f"  Batch {batch_num}/{total_batches}: ❌ No results", flush=True)

            # Save after each batch
            SCORES_FILE.write_text(json.dumps(existing_scores, indent=2))

    # Process all batches with concurrency
    tasks = [process_batch(i, batch) for i, batch in enumerate(batches)]
    await asyncio.gather(*tasks)

    # Final save
    SCORES_FILE.write_text(json.dumps(existing_scores, indent=2))

    print(f"\n{'='*60}")
    print(f"  SCORING COMPLETE")
    print(f"{'='*60}")
    print(f"  Scored:          {scored}")
    print(f"  Failed batches:  {failed}")
    print(f"  Total scores:    {len(existing_scores)}")
    
    v2_final = sum(1 for s in existing_scores.values() if is_v2_score(s))
    print(f"  V2 scored:       {v2_final}")
    print(f"  Remaining:       {len(articles) - v2_final}")


if __name__ == "__main__":
    asyncio.run(main())
