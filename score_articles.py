#!/usr/bin/env python3
"""
Score articles using 4-dimension classification system (v2).

Thesis: Enterprise spend shifting from 70/20/10 (build/maintain/test) → 20/40/40
(build/validate/operate+govern) due to AI agent explosion.

Each article gets classified along FOUR dimensions:
  BC — Build Compression:     AI coding adoption, time-to-production shrinking
  VE — Validation Expansion:  Testing budgets growing, QA automation, platform consolidation
  GR — Governance & Risk:     Security breaches, regulatory mandates, audit failures
  MN — Market Narrative Shift: Language like "software sprawl", "AI risk", "operational resilience"

Scoring schema v2:
  relevance       (1-5)        — overall relevance to the Leapwork thesis
  dimension_scores {BC,VE,GR,MN} — independent 0-5 score per dimension
  dimensions      [list]       — any dimension scoring >= 2
  strength        (1-5)        — evidence quality (1=anecdote, 5=hard data/exec quote)
  direction       (-1,0,+1)    — supports, neutral, or challenges thesis
  source_category (A-E or ?)   — inherited from article metadata
  composite_score              — relevance × strength × direction  (range -25 to +25)

Usage:
  python3 score_articles.py                    # Score unscored articles only
  python3 score_articles.py --rescore-all      # Rescore everything with new schema
  python3 score_articles.py --stats            # Show scoring stats
  python3 score_articles.py --batch-size 30    # Custom batch size
  python3 score_articles.py --sample 50        # Score only N articles (for testing)
  python3 score_articles.py --dry-run          # Show what would be scored without calling API
"""

import argparse
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
SCORES_V1_FILE = DATA_DIR / "article_scores_v1.json"

DEFAULT_BATCH_SIZE = 50
MODEL = "gpt-4o-mini"

DIMENSIONS = ["BC", "VE", "GR", "MN"]

# ---------------------------------------------------------------------------
# V2 System Prompt — 4-Dimension Classification
# ---------------------------------------------------------------------------
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

### SCORING RULES

- An article about AI coding tools that make building faster → high BC, direction +1 (supports thesis — build compresses)
- An article about testing/QA market growth → high VE, direction +1
- An article about AI REPLACING testers entirely → high VE, direction -1 (challenges thesis)
- An article about a security breach from rushed AI code → high GR and possibly BC, direction +1
- An article about EU AI Act mandating testing → high GR, direction +1
- An article about "enterprises need more governance" → high MN, direction +1
- Completely off-topic articles (food, sports, unrelated medical) → all dimensions 0, relevance 1, strength 1, direction 0

### OUTPUT FORMAT

Respond with a JSON object containing a "results" array. Each element:
```json
{"results": [{"idx": <number>, "r": <1-5>, "bc": <0-5>, "ve": <0-5>, "gr": <0-5>, "mn": <0-5>, "s": <1-5>, "d": <-1|0|1>}, ...]}
```

Where: idx=article index, r=relevance, bc/ve/gr/mn=dimension scores, s=strength, d=direction.
You MUST return ALL articles in the batch. Return one entry per article.

ONLY output the JSON object. No explanation, no markdown fences."""


def load_articles():
    """Load articles from news store."""
    if NEWS_STORE.exists():
        return json.loads(NEWS_STORE.read_text())
    return []


def load_scores():
    """Load existing scores (v2 format)."""
    if SCORES_FILE.exists():
        data = json.loads(SCORES_FILE.read_text())
        # Handle both dict format (uid→score) and list format
        if isinstance(data, dict):
            return data
        return {}
    return {}


def save_scores(scores):
    """Save scores to disk."""
    SCORES_FILE.write_text(json.dumps(scores, indent=2))


def is_v2_score(score_entry):
    """Check if a score entry is in v2 format (has dimension_scores)."""
    return "dimension_scores" in score_entry


def score_batch(articles_batch, start_idx):
    """Score a batch of articles using OpenAI API with v2 prompt."""
    from openai import OpenAI
    
    # Get API key from environment or .zshrc
    import os
    import re
    from pathlib import Path
    
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
    
    client = OpenAI(api_key=api_key, timeout=120.0)

    # Build the prompt with article titles + snippets
    lines = []
    for i, article in enumerate(articles_batch):
        title = article.get("title", "").strip()
        source = article.get("source", "").strip()
        snippet = article.get("snippet", "").strip()
        cat = article.get("source_category", "?")
        # Include snippet if available (truncated) for better scoring
        entry = f"{start_idx + i}. [{source}|{cat}] {title}"
        if snippet:
            entry += f" — {snippet[:150]}"
        lines.append(entry)

    prompt = "Score these articles:\n\n" + "\n".join(lines)

    response = client.chat.completions.create(
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
        # OpenAI may wrap array in object
        if isinstance(parsed, dict):
            for v in parsed.values():
                if isinstance(v, list):
                    return v
            return []
        elif isinstance(parsed, list):
            return parsed
        return []
    except json.JSONDecodeError:
        print(f"  ⚠️ Failed to parse batch response, attempting extraction...")
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return []


def build_v2_entry(result, article):
    """Convert raw LLM result into v2 score entry."""
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


def print_stats(scores, articles=None):
    """Print comprehensive scoring statistics."""
    if not scores:
        print("No scores to analyze.")
        return

    v1_count = 0
    v2_count = 0
    v2_entries = []

    for uid, entry in scores.items():
        if is_v2_score(entry):
            v2_count += 1
            v2_entries.append(entry)
        else:
            v1_count += 1

    print(f"\n{'='*60}")
    print(f"  ARTICLE SCORING STATISTICS")
    print(f"{'='*60}")
    print(f"  Total scored:   {len(scores)}")
    print(f"  V1 (old):       {v1_count}")
    print(f"  V2 (new):       {v2_count}")

    if articles:
        print(f"  Total articles:  {len(articles)}")
        print(f"  Unscored:       {len(articles) - len(scores)}")

    if not v2_entries:
        print("\n  No v2 scores yet. Run scorer to generate them.")
        return

    # Direction distribution
    directions = Counter(e["direction"] for e in v2_entries)
    print(f"\n  Direction Distribution (v2):")
    print(f"    +1 (supports):   {directions.get(1, 0):>5}  ({directions.get(1,0)/len(v2_entries)*100:.1f}%)")
    print(f"     0 (neutral):    {directions.get(0, 0):>5}  ({directions.get(0,0)/len(v2_entries)*100:.1f}%)")
    print(f"    -1 (challenges): {directions.get(-1, 0):>5}  ({directions.get(-1,0)/len(v2_entries)*100:.1f}%)")

    # Relevance distribution
    relevances = [e["relevance"] for e in v2_entries]
    print(f"\n  Relevance (1-5):")
    for r in range(1, 6):
        cnt = relevances.count(r)
        bar = "█" * (cnt * 40 // len(v2_entries))
        print(f"    {r}: {cnt:>5}  {bar}")

    # Strength distribution
    strengths = [e["strength"] for e in v2_entries]
    print(f"\n  Strength (1-5):")
    for s in range(1, 6):
        cnt = strengths.count(s)
        bar = "█" * (cnt * 40 // len(v2_entries))
        print(f"    {s}: {cnt:>5}  {bar}")

    # Dimension coverage
    dim_counts = Counter()
    for e in v2_entries:
        for dim in e.get("dimensions", []):
            dim_counts[dim] += 1

    print(f"\n  Dimension Coverage (articles with score >= 2):")
    for dim in DIMENSIONS:
        cnt = dim_counts.get(dim, 0)
        bar = "█" * (cnt * 40 // max(len(v2_entries), 1))
        print(f"    {dim}: {cnt:>5}  {bar}")

    # Average dimension scores
    print(f"\n  Average Dimension Scores:")
    for dim in DIMENSIONS:
        vals = [e["dimension_scores"][dim] for e in v2_entries if dim in e.get("dimension_scores", {})]
        if vals:
            avg = sum(vals) / len(vals)
            print(f"    {dim}: {avg:.2f}")

    # Composite score stats
    composites = [e["composite_score"] for e in v2_entries]
    print(f"\n  Composite Score (relevance × strength × direction):")
    print(f"    Mean:   {sum(composites)/len(composites):>+.2f}")
    print(f"    Min:    {min(composites):>+d}")
    print(f"    Max:    {max(composites):>+d}")
    pos = sum(1 for c in composites if c > 0)
    neg = sum(1 for c in composites if c < 0)
    zero = sum(1 for c in composites if c == 0)
    print(f"    Positive: {pos}, Negative: {neg}, Zero: {zero}")

    # Source category breakdown
    cat_counts = Counter(e.get("source_category", "?") for e in v2_entries)
    print(f"\n  Source Categories:")
    for cat in sorted(cat_counts.keys()):
        print(f"    {cat}: {cat_counts[cat]:>5}")

    print(f"\n{'='*60}")


def print_sample_results(scores, articles, n=20):
    """Print sample v2 scored articles for quality verification."""
    # Find v2-scored articles with highest composite scores
    v2_scored = []
    uid_to_article = {a.get("uid"): a for a in articles}

    for uid, entry in scores.items():
        if is_v2_score(entry):
            article = uid_to_article.get(uid, {})
            v2_scored.append((uid, entry, article))

    if not v2_scored:
        print("No v2 scores to display.")
        return

    # Sort by absolute composite score (most interesting first)
    v2_scored.sort(key=lambda x: abs(x[1]["composite_score"]), reverse=True)

    print(f"\n{'='*80}")
    print(f"  TOP {min(n, len(v2_scored))} SCORED ARTICLES (by |composite_score|)")
    print(f"{'='*80}")

    for uid, entry, article in v2_scored[:n]:
        title = article.get("title", "???")[:70]
        dims = ",".join(entry["dimensions"]) or "none"
        ds = entry["dimension_scores"]
        d_str = f"BC={ds['BC']} VE={ds['VE']} GR={ds['GR']} MN={ds['MN']}"
        dir_sym = {1: "↑", -1: "↓", 0: "→"}[entry["direction"]]
        print(f"\n  {dir_sym} [{entry['composite_score']:>+3d}] {title}")
        print(f"    rel={entry['relevance']} str={entry['strength']} dims=[{dims}] {d_str} cat={entry['source_category']}")

    print(f"\n{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="Score articles with 4-dimension classification (v2)")
    parser.add_argument("--rescore-all", action="store_true",
                        help="Force re-score ALL articles with v2 schema")
    parser.add_argument("--stats", action="store_true",
                        help="Show scoring statistics and exit")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Articles per API call (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--sample", type=int, default=None,
                        help="Score only N articles (for testing)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be scored without calling API")
    args = parser.parse_args()

    articles = load_articles()
    existing_scores = load_scores()

    # Stats mode
    if args.stats:
        print_stats(existing_scores, articles)
        return

    # Build uid→article index
    uid_to_idx = {}
    for i, article in enumerate(articles):
        uid = article.get("uid", str(i))
        uid_to_idx[uid] = i

    # Determine what needs scoring
    if args.rescore_all:
        # Rescore everything with v2 — clear all existing scores
        print("🔄 RESCORE ALL mode — will re-score all articles with v2 schema")
        to_score = [(i, article.get("uid", str(i)), article) for i, article in enumerate(articles)]
    else:
        # Score only unscored articles, OR articles still on v1 schema
        to_score = []
        for i, article in enumerate(articles):
            uid = article.get("uid", str(i))
            if uid not in existing_scores:
                to_score.append((i, uid, article))
            elif not is_v2_score(existing_scores[uid]):
                # Has v1 score but not v2 — needs upgrading
                to_score.append((i, uid, article))

    print(f"📊 Total articles:  {len(articles)}")
    print(f"📊 Already scored:  {len(existing_scores)}")
    v2_count = sum(1 for s in existing_scores.values() if is_v2_score(s))
    print(f"📊 V2 scored:       {v2_count}")
    print(f"📊 To score (v2):   {len(to_score)}")

    if not to_score:
        print("\n✅ Nothing to score! All articles have v2 scores.")
        print_stats(existing_scores, articles)
        return

    # Apply sample limit
    if args.sample:
        to_score = to_score[:args.sample]
        print(f"📊 Sample limit:    {args.sample} articles")

    if args.dry_run:
        print(f"\n🔍 DRY RUN — would score {len(to_score)} articles in {(len(to_score) + args.batch_size - 1) // args.batch_size} batches")
        for i, (idx, uid, article) in enumerate(to_score[:10]):
            print(f"  {i+1}. [{article.get('source_category','?')}] {article.get('title','???')[:80]}")
        if len(to_score) > 10:
            print(f"  ... and {len(to_score) - 10} more")
        return

    # Process in batches
    scored = 0
    failed_batches = 0
    batch_size = args.batch_size

    total_batches = (len(to_score) + batch_size - 1) // batch_size
    print(f"\n🚀 Scoring {len(to_score)} articles in {total_batches} batches of {batch_size}...\n")

    for batch_start in range(0, len(to_score), batch_size):
        batch = to_score[batch_start:batch_start + batch_size]
        batch_articles = [item[2] for item in batch]
        batch_uids = [item[1] for item in batch]
        batch_num = batch_start // batch_size + 1

        print(f"  Batch {batch_num}/{total_batches}: scoring {len(batch)} articles...", end=" ", flush=True)

        try:
            results = score_batch(batch_articles, batch_start)

            # Map results back to UIDs
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

            # Save after each batch
            save_scores(existing_scores)
            print(f"✅ {batch_scored}/{len(batch)} scored")

            # Rate limiting
            time.sleep(0.5)

        except Exception as e:
            print(f"❌ Failed: {e}")
            failed_batches += 1
            time.sleep(2)
            continue

    # Summary
    print(f"\n{'='*60}")
    print(f"  SCORING COMPLETE")
    print(f"{'='*60}")
    print(f"  Scored:          {scored}")
    print(f"  Failed batches:  {failed_batches}")
    print(f"  Total scores:    {len(existing_scores)}")

    # Print stats and samples
    print_stats(existing_scores, articles)
    print_sample_results(existing_scores, articles, n=20)


if __name__ == "__main__":
    main()
