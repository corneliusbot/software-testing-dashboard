#!/usr/bin/env python3
"""Pre-compute English translations for non-English article titles.

Reads data/news_articles.json, detects non-English titles via Unicode heuristics,
translates them using GPT-4o-mini, and saves a cache to data/title_translations.json.
"""
import json
import os
import re
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
NEWS_FILE = DATA_DIR / "news_articles.json"
CACHE_FILE = DATA_DIR / "title_translations.json"

# Regex: matches if title has substantial non-ASCII content
# CJK Unified, Hangul, Katakana, Hiragana, Cyrillic, Arabic, Thai, Devanagari, etc.
NON_LATIN_RE = re.compile(
    r'[\u0400-\u04FF'   # Cyrillic
    r'\u0600-\u06FF'    # Arabic
    r'\u0900-\u097F'    # Devanagari
    r'\u0E00-\u0E7F'    # Thai
    r'\u1100-\u11FF'    # Hangul Jamo
    r'\u3000-\u303F'    # CJK Symbols
    r'\u3040-\u309F'    # Hiragana
    r'\u30A0-\u30FF'    # Katakana
    r'\u4E00-\u9FFF'    # CJK Unified
    r'\uAC00-\uD7AF'    # Hangul Syllables
    r'\uF900-\uFAFF'    # CJK Compatibility
    r']'
)


def is_non_english(title: str) -> bool:
    """Heuristic: title is non-English if >30% of non-whitespace chars are non-Latin."""
    if not title or len(title) < 4:
        return False
    non_ws = title.replace(" ", "")
    if not non_ws:
        return False
    hits = NON_LATIN_RE.findall(non_ws)
    return len(hits) / len(non_ws) > 0.3


def translate_batch(titles: dict[str, str], api_key: str) -> dict[str, str]:
    """Translate a dict of {uid: title} using OpenAI GPT-4o-mini. Returns {uid: english_title}."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    results = {}

    # Process in batches of 15 to stay within token limits
    items = list(titles.items())
    batch_size = 15

    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        prompt_lines = []
        for uid, title in batch:
            prompt_lines.append(f"[{uid}] {title}")

        prompt = (
            "Translate each of the following article titles to English. "
            "Return ONLY a JSON object mapping the ID in brackets to the English translation. "
            "Keep translations concise and natural (newspaper headline style).\n\n"
            + "\n".join(prompt_lines)
        )

        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content
            parsed = json.loads(content)
            results.update(parsed)
            print(f"  Translated batch {i // batch_size + 1}: {len(parsed)} titles")
        except Exception as e:
            print(f"  ERROR translating batch {i // batch_size + 1}: {e}", file=sys.stderr)

    return results


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    if not NEWS_FILE.exists():
        print(f"ERROR: {NEWS_FILE} not found", file=sys.stderr)
        sys.exit(1)

    articles = json.loads(NEWS_FILE.read_text())
    print(f"Loaded {len(articles)} articles")

    # Load existing cache to avoid re-translating
    existing: dict[str, str] = {}
    if CACHE_FILE.exists():
        try:
            existing = json.loads(CACHE_FILE.read_text())
            print(f"Loaded {len(existing)} existing translations")
        except Exception:
            pass

    # Find non-English titles not yet translated
    to_translate: dict[str, str] = {}
    for a in articles:
        uid = a.get("uid", "")
        title = a.get("title", "")
        if uid and is_non_english(title) and uid not in existing:
            to_translate[uid] = title

    print(f"Found {len(to_translate)} non-English titles needing translation")

    if not to_translate:
        print("Nothing to translate — cache is up to date")
        # Still save (may have loaded existing)
        CACHE_FILE.write_text(json.dumps(existing, ensure_ascii=False, indent=2))
        return

    translations = translate_batch(to_translate, api_key)
    existing.update(translations)

    CACHE_FILE.write_text(json.dumps(existing, ensure_ascii=False, indent=2))
    print(f"Saved {len(existing)} total translations to {CACHE_FILE}")


if __name__ == "__main__":
    main()
