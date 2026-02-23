#!/usr/bin/env python3
"""
Scanner V2 — Multi-Source Intelligence Collector for Software Spend Shift Dashboard.

Replaces the original scanner.py's 14 generic Google News queries with a structured
multi-source collector covering 5 intelligence categories for the Leapwork thesis:
enterprise spend shifting from building → testing/validation as AI agents explode.

Categories:
    A: Enterprise Budget Signals  (Gartner, Forrester, IDC, McKinsey, CIO.com)
    B: Dev Velocity / Build Compression  (GitHub, Vercel, OpenAI, Product Hunt)
    C: Security & Failure Signals  (KrebsOnSecurity, Dark Reading, The Register)
    D: Compliance & Regulation  (NIST, EU AI Act, SEC)
    E: Testing Market Movements  (TechCrunch, Datadog, Dynatrace)

Usage:
    python3 scanner_v2.py                    # Run all categories
    python3 scanner_v2.py --category A       # Run one category
    python3 scanner_v2.py --category A B     # Run specific categories
    python3 scanner_v2.py --stats            # Show collection stats
    python3 scanner_v2.py --dry-run          # Scan but don't save
    python3 scanner_v2.py --verbose          # Extra logging

Dependencies: feedparser, requests (stdlib for the rest)
"""

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.parse
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from time import mktime

import feedparser
import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
NEWS_STORE = DATA_DIR / "news_articles.json"

# ---------------------------------------------------------------------------
# HTTP settings
# ---------------------------------------------------------------------------
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
REQUEST_TIMEOUT = 12  # seconds
RSS_DELAY = 1.5       # seconds between RSS fetches
GNEWS_DELAY = 2.0     # seconds between Google News fetches (more cautious)

# ---------------------------------------------------------------------------
# Source definitions — RSS feeds per category
# ---------------------------------------------------------------------------
# Each entry: (source_key, url, category)
# Feeds verified 2026-02-19. Dead feeds commented with reason.

RSS_FEEDS = [
    # ── Category A: Enterprise Budget ──────────────────────────────────────
    # ("gartner", "https://www.gartner.com/en/newsroom/press-releases/rss", "A"),  # 403 blocked
    ("forrester_blog",   "https://www.forrester.com/blogs/feed",                              "A"),
    ("cio_com",          "https://www.cio.com/feed/",                                         "A"),
    # ("hbr_tech", "https://hbr.org/topic/technology/feed", "A"),  # 404
    ("mckinsey_insights","https://www.mckinsey.com/insights/rss",                             "A"),
    ("zdnet_ai",         "https://www.zdnet.com/topic/artificial-intelligence/rss.xml",       "A"),
    ("venturebeat_ai",   "https://venturebeat.com/category/ai/feed/",                         "A"),

    # ── Category B: Dev Velocity / Build Compression ──────────────────────
    ("github_blog",      "https://github.blog/feed/",                                         "B"),
    ("vercel_blog",      "https://vercel.com/atom",                                           "B"),
    ("openai_blog",      "https://openai.com/blog/rss.xml",                                   "B"),
    # ("anthropic_news", "https://www.anthropic.com/rss.xml", "B"),  # 404 — no public RSS found
    ("product_hunt",     "https://www.producthunt.com/feed",                                  "B"),
    ("thenewstack",      "https://thenewstack.io/feed/",                                      "B"),
    ("siliconangle",     "https://siliconangle.com/feed/",                                    "B"),

    # ── Category C: Security & Failure Signals ────────────────────────────
    ("krebs_on_security","https://krebsonsecurity.com/feed/",                                 "C"),
    ("dark_reading",     "https://www.darkreading.com/rss.xml",                               "C"),
    ("the_register",     "https://www.theregister.com/headlines.atom",                        "C"),
    ("bleeping_computer","https://www.bleepingcomputer.com/feed/",                            "C"),
    ("cloudflare_blog",  "https://blog.cloudflare.com/rss/",                                  "C"),
    ("hacker_news",      "https://news.ycombinator.com/rss",                                  "C"),
    # ("hnrss_incidents", "https://hnrss.org/newest?q=production+incident+AI", "C"),  # SSL handshake failure

    # ── Category D: Compliance & Regulation ───────────────────────────────
    ("nist_news",        "https://www.nist.gov/news-events/news/rss.xml",                     "D"),
    ("devops_com",       "https://devops.com/feed/",                                          "D"),
    # Note: EU AI Act and SEC don't have clean RSS; covered by Google News queries below.

    # ── Category E: Testing Market ────────────────────────────────────────
    ("techcrunch_testing","https://techcrunch.com/tag/testing/feed/",                          "E"),
    # ("tricentis_blog", "https://www.tricentis.com/blog/feed", "E"),  # 404
    ("datadog_blog",     "https://www.datadoghq.com/feed/",                                   "E"),
    ("dynatrace_blog",   "https://www.dynatrace.com/news/blog/feed/",                         "E"),
    ("sdtimes",          "https://sdtimes.com/feed/",                                         "E"),
    ("infoworld",        "https://www.infoworld.com/feed/",                                   "E"),
]

# ---------------------------------------------------------------------------
# Google News queries per category (supplementary)
# ---------------------------------------------------------------------------
GOOGLE_NEWS_QUERIES = {
    "A": [
        "enterprise AI governance budget",
        "CIO software testing investment",
        "IT spend validation governance 2026",
        "software quality assurance market growth",
    ],
    "B": [
        "AI code generation adoption enterprise",
        "GitHub Copilot usage statistics",
        "citizen developer AI low-code growth",
        "vibe coding AI ship faster",
    ],
    "C": [
        "AI system production failure incident",
        "automation outage software release rollback",
        "AI generated code vulnerability security",
        "production incident AI deployment",
    ],
    "D": [
        "EU AI Act compliance software",
        "AI regulation mandatory testing audit",
        "SEC cybersecurity disclosure software",
        "software liability AI governance regulation",
    ],
    "E": [
        "test automation AI startup funding",
        "software testing company acquisition",
        "AI test generation autonomous QA",
        "testing platform consolidation market",
    ],
}

# ---------------------------------------------------------------------------
# Category metadata
# ---------------------------------------------------------------------------
CATEGORY_NAMES = {
    "A": "Enterprise Budget",
    "B": "Dev Velocity",
    "C": "Security/Failures",
    "D": "Compliance",
    "E": "Testing Market",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_verbose = False

def log(msg: str):
    """Print if verbose mode on."""
    if _verbose:
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def make_uid(url: str) -> str:
    """MD5 hash of normalised URL."""
    normalised = url.strip().rstrip("/").lower()
    return hashlib.md5(normalised.encode()).hexdigest()


def title_similar(a: str, b: str, threshold: float = 0.85) -> bool:
    """Return True if two titles are near-duplicates."""
    a_low, b_low = a.lower(), b.lower()
    # Quick length check — if lengths differ by >40%, can't be similar enough
    if abs(len(a_low) - len(b_low)) > max(len(a_low), len(b_low)) * 0.4:
        return False
    return SequenceMatcher(None, a_low, b_low).ratio() >= threshold


def clean_html(text: str) -> str:
    """Strip HTML tags and collapse whitespace."""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&#\d+;", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_published(entry) -> str | None:
    """Extract a YYYY-MM-DD date from a feedparser entry."""
    for attr in ("published_parsed", "updated_parsed"):
        parsed = getattr(entry, attr, None)
        if parsed:
            try:
                return datetime.fromtimestamp(mktime(parsed)).strftime("%Y-%m-%d")
            except Exception:
                pass
    # Fallback: try parsing date strings directly
    for attr in ("published", "updated"):
        raw = getattr(entry, attr, None)
        if raw:
            try:
                # feedparser stores as string sometimes
                from email.utils import parsedate_to_datetime
                dt = parsedate_to_datetime(raw)
                return dt.strftime("%Y-%m-%d")
            except Exception:
                pass
    return None


def extract_source_from_title(title: str) -> tuple[str, str]:
    """
    Google News titles have format 'Headline - Source Name'.
    Returns (cleaned_title, source).
    """
    if " - " in title:
        parts = title.rsplit(" - ", 1)
        if len(parts[1].strip()) < 60:  # reasonable source name length
            return parts[0].strip(), parts[1].strip()
    return title, ""


# ---------------------------------------------------------------------------
# Data store
# ---------------------------------------------------------------------------
def load_store() -> list[dict]:
    """Load existing articles from disk."""
    if NEWS_STORE.exists():
        try:
            data = json.loads(NEWS_STORE.read_text())
            if isinstance(data, list):
                return data
        except Exception as e:
            log(f"Warning: Could not load {NEWS_STORE}: {e}")
    return []


def save_store(articles: list[dict]):
    """Write articles to disk, sorted by date descending."""
    articles.sort(
        key=lambda x: x.get("published") or x.get("date") or "1970-01-01",
        reverse=True,
    )
    NEWS_STORE.write_text(json.dumps(articles, indent=2, default=str))


def build_dedup_index(articles: list[dict]) -> tuple[set, list[str]]:
    """
    Build dedup structures from existing articles.
    Returns (set of UIDs, list of titles for similarity check).
    """
    uids = set()
    titles = []
    for a in articles:
        uid = a.get("uid")
        if uid:
            uids.add(uid)
        # Also add URL-based UID for backwards compat with v1 data
        url = a.get("url") or a.get("link") or ""
        if url:
            uids.add(make_uid(url))
        title = a.get("title", "")
        if title:
            titles.append(title)
    return uids, titles


def is_duplicate(uid: str, title: str, seen_uids: set, seen_titles: list[str]) -> bool:
    """Check if an article is a duplicate by UID or title similarity."""
    if uid in seen_uids:
        return True
    # Title similarity check — only check recent titles for performance.
    # The quick length pre-filter in title_similar keeps this fast.
    check_window = seen_titles[-200:] if len(seen_titles) > 200 else seen_titles
    for existing_title in check_window:
        if title_similar(title, existing_title):
            return True
    return False


# ---------------------------------------------------------------------------
# Collectors
# ---------------------------------------------------------------------------
def fetch_rss_feed(source_key: str, url: str, category: str) -> list[dict]:
    """
    Fetch and parse a single RSS/Atom feed.
    Returns list of article dicts.
    """
    articles = []
    try:
        headers = {"User-Agent": USER_AGENT}
        resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT, allow_redirects=True)

        if resp.status_code != 200:
            log(f"⚠ {source_key}: HTTP {resp.status_code}")
            return []

        feed = feedparser.parse(resp.content)

        if not feed.entries:
            log(f"⚠ {source_key}: No entries in feed")
            return []

        log(f"✓ {source_key}: {len(feed.entries)} entries")

        for entry in feed.entries:
            link = entry.get("link", "")
            if not link:
                continue

            title = entry.get("title", "")
            if not title:
                continue

            # Clean up title (some feeds include HTML)
            title = clean_html(title)

            snippet = entry.get("summary", entry.get("description", ""))
            snippet = clean_html(snippet)[:500]

            published = parse_published(entry)

            articles.append({
                "uid": make_uid(link),
                "title": title,
                "url": link,
                "source": source_key,
                "source_category": category,
                "published": published,
                "snippet": snippet,
                "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            })

    except requests.exceptions.Timeout:
        log(f"✗ {source_key}: Timeout")
    except requests.exceptions.ConnectionError as e:
        log(f"✗ {source_key}: Connection error")
    except Exception as e:
        log(f"✗ {source_key}: {type(e).__name__}: {e}")

    return articles


def fetch_google_news(query: str, category: str) -> list[dict]:
    """
    Fetch articles from Google News RSS for a search query.
    Returns list of article dicts.
    """
    articles = []
    try:
        encoded = urllib.parse.quote_plus(query)
        url = f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en"

        headers = {"User-Agent": USER_AGENT}
        resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT, allow_redirects=True)

        if resp.status_code != 200:
            log(f"⚠ GNews '{query}': HTTP {resp.status_code}")
            return []

        feed = feedparser.parse(resp.content)
        log(f"✓ GNews '{query}': {len(feed.entries)} entries")

        for entry in feed.entries:
            link = entry.get("link", "")
            if not link:
                continue

            raw_title = entry.get("title", "")
            if not raw_title:
                continue

            title, source_name = extract_source_from_title(raw_title)
            title = clean_html(title)

            snippet = entry.get("summary", entry.get("description", ""))
            snippet = clean_html(snippet)[:500]

            published = parse_published(entry)

            # Use a composite source key for Google News results
            source_key = f"gnews_{source_name.lower().replace(' ', '_')[:30]}" if source_name else "gnews"

            articles.append({
                "uid": make_uid(link),
                "title": title,
                "url": link,
                "source": source_key,
                "source_category": category,
                "published": published,
                "snippet": snippet,
                "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            })

    except requests.exceptions.Timeout:
        log(f"✗ GNews '{query}': Timeout")
    except requests.exceptions.ConnectionError:
        log(f"✗ GNews '{query}': Connection error")
    except Exception as e:
        log(f"✗ GNews '{query}': {type(e).__name__}: {e}")

    return articles


# ---------------------------------------------------------------------------
# Main collection logic
# ---------------------------------------------------------------------------
def collect_category(category: str) -> list[dict]:
    """
    Collect all articles for a given category (A-E).
    Runs RSS feeds first, then Google News queries.
    Returns raw (un-deduped) list of articles.
    """
    articles = []

    # RSS feeds for this category
    cat_feeds = [(s, u, c) for s, u, c in RSS_FEEDS if c == category]
    for source_key, url, cat in cat_feeds:
        results = fetch_rss_feed(source_key, url, cat)
        articles.extend(results)
        time.sleep(RSS_DELAY)

    # Google News queries for this category
    queries = GOOGLE_NEWS_QUERIES.get(category, [])
    for query in queries:
        results = fetch_google_news(query, category)
        articles.extend(results)
        time.sleep(GNEWS_DELAY)

    return articles


def collect(categories: list[str] | None = None, dry_run: bool = False) -> dict:
    """
    Main collection entry point.

    Args:
        categories: List of category letters to collect (None = all).
        dry_run: If True, don't save to disk.

    Returns:
        Dict with stats per category and overall.
    """
    if categories is None:
        categories = list(CATEGORY_NAMES.keys())

    # Load existing data
    existing = load_store()
    seen_uids, seen_titles = build_dedup_index(existing)
    initial_count = len(existing)

    stats = {}
    all_new = []

    for cat in categories:
        if cat not in CATEGORY_NAMES:
            print(f"⚠ Unknown category '{cat}', skipping")
            continue

        print(f"\n{'─'*60}")
        print(f"Collecting Category {cat}: {CATEGORY_NAMES[cat]}")
        print(f"{'─'*60}")

        raw = collect_category(cat)
        new_for_cat = []

        for article in raw:
            uid = article["uid"]
            title = article["title"]

            if is_duplicate(uid, title, seen_uids, seen_titles):
                continue

            # Mark as seen
            seen_uids.add(uid)
            seen_titles.append(title)
            new_for_cat.append(article)

        all_new.extend(new_for_cat)

        # Count existing for this category
        existing_cat = sum(1 for a in existing if a.get("source_category") == cat)
        total_cat = existing_cat + len(new_for_cat)

        stats[cat] = {
            "new": len(new_for_cat),
            "total": total_cat,
        }

        print(f"  → {len(new_for_cat)} new articles ({total_cat} total in category {cat})")

    # Merge and save
    if all_new and not dry_run:
        merged = existing + all_new

        # Backfill source_category for v1 articles that don't have it
        for article in merged:
            if "source_category" not in article:
                article["source_category"] = "?"

        save_store(merged)

    # Overall stats
    total_new = len(all_new)
    total_all = len(existing) + total_new
    stats["_total"] = {"new": total_new, "total": total_all}

    return stats


def print_stats_summary(stats: dict):
    """Pretty-print collection stats."""
    print(f"\n{'═'*60}")
    print("COLLECTION SUMMARY")
    print(f"{'═'*60}")

    for cat in sorted(CATEGORY_NAMES.keys()):
        if cat in stats:
            s = stats[cat]
            name = CATEGORY_NAMES[cat]
            print(f"  Category {cat} ({name:20s}): {s['new']:4d} new  ({s['total']:,} total)")

    total = stats.get("_total", {})
    print(f"{'─'*60}")
    print(f"  {'Total':32s}: {total.get('new', 0):4d} new  ({total.get('total', 0):,} total)")
    print(f"{'═'*60}")


def show_collection_stats():
    """Show stats about the existing article collection without fetching."""
    articles = load_store()

    if not articles:
        print("No articles in store.")
        return

    print(f"\n{'═'*60}")
    print("ARTICLE COLLECTION STATS")
    print(f"{'═'*60}")
    print(f"  Total articles: {len(articles):,}")

    # By category
    cat_counts = {}
    for a in articles:
        cat = a.get("source_category", "?")
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    print(f"\n  By Category:")
    for cat in sorted(cat_counts.keys()):
        name = CATEGORY_NAMES.get(cat, "Unknown/V1")
        print(f"    {cat}: {name:25s} — {cat_counts[cat]:,} articles")

    # By source (top 20)
    source_counts = {}
    for a in articles:
        src = a.get("source", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1

    print(f"\n  Top 20 Sources:")
    for src, count in sorted(source_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"    {src:35s} — {count:,}")

    # Date range
    dates = [a.get("published") or a.get("date") for a in articles]
    dates = [d for d in dates if d]
    if dates:
        print(f"\n  Date range: {min(dates)} → {max(dates)}")

    # Freshness
    fetched = [a.get("fetched_at") or a.get("scanned_at") for a in articles]
    fetched = [f for f in fetched if f]
    if fetched:
        print(f"  Last fetch:  {max(fetched)}")

    print(f"{'═'*60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    global _verbose

    parser = argparse.ArgumentParser(
        description="Scanner V2 — Multi-Source Intelligence Collector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 scanner_v2.py                    # Run all categories
  python3 scanner_v2.py --category A       # Run one category
  python3 scanner_v2.py --category A B     # Run specific categories
  python3 scanner_v2.py --stats            # Show collection stats
  python3 scanner_v2.py --dry-run          # Scan but don't save
  python3 scanner_v2.py --verbose          # Extra logging
        """,
    )
    parser.add_argument(
        "--category", "-c",
        nargs="+",
        choices=list(CATEGORY_NAMES.keys()),
        help="Categories to collect (default: all)",
    )
    parser.add_argument(
        "--stats", "-s",
        action="store_true",
        help="Show collection stats and exit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch but don't save to disk",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()
    _verbose = args.verbose

    if args.stats:
        show_collection_stats()
        return

    categories = args.category  # None means all
    print(f"Scanner V2 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Categories: {', '.join(categories) if categories else 'ALL'}")
    if args.dry_run:
        print("⚠ DRY RUN — results will not be saved")

    start = time.time()
    stats = collect(categories=categories, dry_run=args.dry_run)
    elapsed = time.time() - start

    print_stats_summary(stats)
    print(f"\nCompleted in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
