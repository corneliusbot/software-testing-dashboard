#!/usr/bin/env python3
"""
News scanner for Software Spend Shift dashboard.
Fetches articles from Google News RSS, deduplicates against existing store, appends new ones.
Run via cron or manually: python3 scanner.py
"""

import json
import hashlib
import re
import urllib.parse
import time
from datetime import datetime
from pathlib import Path
from time import mktime

import feedparser

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
NEWS_STORE = DATA_DIR / "news_articles.json"

SEARCH_QUERIES = [
    "software testing AI",
    "AI code quality",
    "software QA spending",
    "AI governance software",
    "testing automation AI agents",
    "code review AI",
    "software validation AI",
    "shift left testing",
    "AI generated code quality",
    "software testing market growth",
    "DevOps testing automation",
    "AI guardrails software",
    "continuous testing AI",
    "software quality assurance trends",
]

CONFIRM_KEYWORDS = [
    "growth", "increase", "shift", "demand", "surge", "rising", "expand",
    "boom", "record", "accelerat", "invest", "adoption", "critical",
    "essential", "necessary", "skyrocket", "outpac", "priorit", "doubl",
    "tripl", "billion", "million funding", "market size",
]

CHALLENGE_KEYWORDS = [
    "overhyped", "decline", "unnecessary", "slow", "cut", "layoff",
    "downturn", "shrink", "overblown", "skeptic", "bubble", "waste",
    "diminish", "obsolete", "replaced",
]


def classify_article(title: str, snippet: str) -> str:
    text = (title + " " + snippet).lower()
    confirm_score = sum(1 for kw in CONFIRM_KEYWORDS if kw in text)
    challenge_score = sum(1 for kw in CHALLENGE_KEYWORDS if kw in text)
    if confirm_score > challenge_score and confirm_score >= 1:
        return "confirms"
    elif challenge_score > confirm_score and challenge_score >= 1:
        return "challenges"
    return "neutral"


def load_store() -> list[dict]:
    if NEWS_STORE.exists():
        try:
            return json.loads(NEWS_STORE.read_text())
        except Exception:
            return []
    return []


def save_store(articles: list[dict]):
    NEWS_STORE.write_text(json.dumps(articles, indent=2, default=str))


def article_uid(link: str, title: str) -> str:
    return hashlib.md5((link or title).encode()).hexdigest()


def scan():
    existing = load_store()
    seen_ids = {a.get("uid") for a in existing}
    new_articles = []

    for q in SEARCH_QUERIES:
        try:
            encoded = urllib.parse.quote_plus(q)
            url = f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en"
            feed = feedparser.parse(url)

            for entry in feed.entries:
                uid = article_uid(entry.get("link", ""), entry.get("title", ""))
                if uid in seen_ids:
                    continue
                seen_ids.add(uid)

                pub_date = None
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    try:
                        pub_date = datetime.fromtimestamp(mktime(entry.published_parsed)).isoformat()
                    except Exception:
                        pass

                title = entry.get("title", "")
                snippet = entry.get("summary", entry.get("description", ""))
                snippet = re.sub(r"<[^>]+>", "", snippet).strip()[:300]

                source = ""
                if " - " in title:
                    parts = title.rsplit(" - ", 1)
                    source = parts[-1].strip()
                    title = parts[0].strip()

                new_articles.append({
                    "uid": uid,
                    "date": pub_date,
                    "source": source,
                    "title": title,
                    "link": entry.get("link", ""),
                    "snippet": snippet,
                    "signal": classify_article(title, snippet),
                    "query": q,
                    "scanned_at": datetime.now().isoformat(),
                })

            time.sleep(0.3)
        except Exception as e:
            print(f"Error scanning '{q}': {e}")
            continue

    if new_articles:
        all_articles = existing + new_articles
        # Sort by date descending
        all_articles.sort(
            key=lambda x: x.get("date") or "1970-01-01",
            reverse=True,
        )
        save_store(all_articles)
        print(f"✅ Added {len(new_articles)} new articles (total: {len(all_articles)})")
    else:
        print(f"No new articles found (existing: {len(existing)})")

    return len(new_articles)


if __name__ == "__main__":
    scan()
