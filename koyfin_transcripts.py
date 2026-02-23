#!/usr/bin/env python3
"""
Koyfin Earnings Transcript Bulk Downloader

Uses Koyfin's internal API to:
1. Resolve tickers → Koyfin IDs (KIDs)
2. List available transcripts per company
3. Download full transcript text for Q3 + Q4 earnings calls
4. Save to data/transcripts/ as JSON

Auth: Requires Koyfin auth_token (JWT from browser cookie).
Pass via --token or set KOYFIN_TOKEN env var.

Usage:
    python3 koyfin_transcripts.py --token <JWT> [--tickers MSFT,AAPL,CRM]
    python3 koyfin_transcripts.py --token <JWT> --file tickers.txt
    python3 koyfin_transcripts.py --token <JWT> --nasdaq-software
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

BASE_URL = "https://app.koyfin.com/api"
DATA_DIR = Path(__file__).parent / "data" / "transcripts"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Track progress for resumption
PROGRESS_FILE = DATA_DIR / "_progress.json"

# Software / tech / Nasdaq-heavy tickers (expandable)
NASDAQ_SOFTWARE = [
    # Mega-cap tech
    "MSFT", "AAPL", "GOOGL", "AMZN", "META", "NVDA", "AVGO", "ORCL", "CRM", "ADBE",
    # Cloud / SaaS
    "NOW", "SNOW", "PLTR", "DDOG", "NET", "CRWD", "ZS", "PANW", "FTNT", "MDB",
    "CFLT", "ESTC", "GTLB", "DOCN", "MNDY", "S", "PATH", "DT", "HUBS", "TEAM",
    "WDAY", "VEEV", "ANSS", "CDNS", "SNPS", "TTD", "DKNG", "DASH", "ABNB", "UBER",
    # Semis
    "AMD", "INTC", "QCOM", "TXN", "MRVL", "MU", "LRCX", "AMAT", "KLAC", "ASML",
    "ON", "SWKS", "MCHP", "ADI", "NXPI",
    # Enterprise / legacy tech
    "IBM", "HPE", "DELL", "HPQ", "CSCO", "JNPR",
    # AI / emerging
    "AI", "BBAI", "SOUN", "IREN", "BTDR",
    # Fintech
    "SQ", "PYPL", "COIN", "HOOD", "AFRM", "SOFI", "FIS", "FISV", "GPN",
    # Cybersecurity
    "OKTA", "RPD", "QLYS", "TENB", "VRNS",
    # Dev tools / testing (thesis-relevant)
    "DDOG", "ESTC", "GTLB", "CFLT", "DOCN", "FROG",
    # Data / analytics
    "SPLK", "AYX", "DOMO", "PLAN", "NEWR",
    # Other large tech
    "NFLX", "ROKU", "SPOT", "ZM", "TWLO", "DBX", "BOX", "DOCU",
    # Crypto-adjacent
    "MSTR", "CRWV", "GLXY", "CLSK", "MARA", "RIOT",
]

# Deduplicate
NASDAQ_SOFTWARE = list(dict.fromkeys(NASDAQ_SOFTWARE))


def load_progress():
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {"resolved": {}, "downloaded": [], "errors": []}


def save_progress(progress):
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2, default=str))


class KoyfinClient:
    def __init__(self, token: str):
        self.token = token
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0",
        })

    def resolve_ticker(self, ticker: str) -> dict | None:
        """Resolve a ticker symbol to Koyfin KID."""
        try:
            resp = self.session.post(
                f"{BASE_URL}/v1/bfc/tickers/search",
                json={
                    "searchString": ticker,
                    "categories": ["Equity"],
                    "domains": ["NONE"],
                    "primaryOnly": True,
                },
                timeout=15,
            )
            if resp.status_code == 401:
                print("❌ Auth token expired! Get a fresh one from the browser.")
                sys.exit(1)
            data = resp.json()
            if data.get("data"):
                # Find US equity match
                for item in data["data"]:
                    if item.get("country") == "US" and item.get("ticker", "").upper() == ticker.upper():
                        return {
                            "kid": item["KID"],
                            "ticker": item["ticker"],
                            "name": item.get("name", ""),
                            "exchange": item.get("exchange", ""),
                        }
                # Fall back to first result
                item = data["data"][0]
                return {
                    "kid": item["KID"],
                    "ticker": item["ticker"],
                    "name": item.get("name", ""),
                    "exchange": item.get("exchange", ""),
                }
            return None
        except Exception as e:
            print(f"  Error resolving {ticker}: {e}")
            return None

    def list_transcripts(self, kid: str) -> list[dict]:
        """List all transcripts for a company."""
        try:
            resp = self.session.get(
                f"{BASE_URL}/v1/pubhub/transcript/list/{kid}?limit=1000",
                timeout=15,
            )
            if resp.status_code == 401:
                print("❌ Auth token expired!")
                sys.exit(1)
            if resp.status_code == 200:
                return resp.json()
            return []
        except Exception as e:
            print(f"  Error listing transcripts for {kid}: {e}")
            return []

    def get_transcript(self, key_dev_id: int) -> dict | None:
        """Download a single transcript."""
        try:
            resp = self.session.get(
                f"{BASE_URL}/v1/pubhub/v2/transcript/{key_dev_id}",
                timeout=30,
            )
            if resp.status_code == 401:
                print("❌ Auth token expired!")
                sys.exit(1)
            if resp.status_code == 200:
                return resp.json()
            return None
        except Exception as e:
            print(f"  Error downloading transcript {key_dev_id}: {e}")
            return None


def filter_earnings_calls(transcripts: list[dict], quarters: list[str] = None) -> list[dict]:
    """Filter transcript list to earnings calls for specific quarters.
    
    quarters: list like ["Q3 2026", "Q4 2025"] or None for all earnings calls
    """
    earnings = []
    for t in transcripts:
        # Only earnings calls (not conference presentations, special calls, etc.)
        if t.get("eventType") not in ("Earnings Calls",):
            continue
        
        title = t.get("formattedTitle", "") or t.get("transcriptTitle", "")
        
        if quarters:
            # Match quarter in title (e.g. "Q3 2026 Earnings Call")
            matched = any(q.lower() in title.lower() for q in quarters)
            if not matched:
                # Also try fiscal quarter/year fields
                fq = t.get("fiscalQuarter")
                fy = t.get("fiscalYear")
                if fq and fy:
                    matched = any(f"Q{fq}" in q and str(fy) in q for q in quarters)
            if not matched:
                continue
        
        earnings.append(t)
    
    return earnings


def transcript_to_text(transcript_data: dict) -> str:
    """Convert structured transcript to plain text."""
    if not transcript_data:
        return ""
    
    header = transcript_data.get("header", {})
    components = transcript_data.get("components", [])
    
    lines = []
    lines.append(f"# {header.get('title', header.get('companyName', ''))}")
    lines.append(f"Date: {header.get('eventDateTime', '')}")
    lines.append(f"Type: {header.get('eventType', '')}")
    lines.append("")
    
    for comp in components:
        speaker = comp.get("speakerName", "Unknown")
        role = comp.get("speakerType", "")
        text = comp.get("text", "").replace("\\r\\n", "\n").replace("\\n", "\n")
        
        lines.append(f"## {speaker} ({role})")
        lines.append(text)
        lines.append("")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Koyfin Transcript Downloader")
    parser.add_argument("--token", help="Koyfin auth_token JWT")
    parser.add_argument("--tickers", help="Comma-separated ticker list")
    parser.add_argument("--file", help="File with one ticker per line")
    parser.add_argument("--nasdaq-software", action="store_true", help="Use built-in Nasdaq software list")
    parser.add_argument("--quarters", default="Q3,Q4", help="Quarters to download (default: Q3,Q4)")
    parser.add_argument("--years", default="2025,2026", help="Fiscal years (default: 2025,2026)")
    parser.add_argument("--delay", type=float, default=1.5, help="Delay between requests (seconds)")
    parser.add_argument("--resume", action="store_true", help="Resume from progress file")
    args = parser.parse_args()

    token = args.token or os.environ.get("KOYFIN_TOKEN")
    if not token:
        print("Error: Provide --token or set KOYFIN_TOKEN env var")
        sys.exit(1)

    # Build ticker list
    tickers = []
    if args.nasdaq_software:
        tickers = NASDAQ_SOFTWARE
    elif args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",")]
    elif args.file:
        with open(args.file) as f:
            tickers = [line.strip().upper() for line in f if line.strip()]
    else:
        print("Provide --tickers, --file, or --nasdaq-software")
        sys.exit(1)

    # Build quarter filter
    quarters_raw = [q.strip() for q in args.quarters.split(",")]
    years_raw = [y.strip() for y in args.years.split(",")]
    quarter_filters = [f"{q} {y}" for q in quarters_raw for y in years_raw]
    
    print(f"📋 {len(tickers)} tickers to process")
    print(f"📅 Looking for: {', '.join(quarter_filters)}")
    print(f"⏱️  Delay: {args.delay}s between requests")
    print()

    client = KoyfinClient(token)
    progress = load_progress() if args.resume else {"resolved": {}, "downloaded": [], "errors": []}

    # Phase 1: Resolve tickers
    print("=== Phase 1: Resolving tickers ===")
    unresolved = [t for t in tickers if t not in progress["resolved"]]
    for i, ticker in enumerate(unresolved):
        result = client.resolve_ticker(ticker)
        if result:
            progress["resolved"][ticker] = result
            print(f"  ✅ {ticker} → {result['kid']} ({result['name']})")
        else:
            progress["resolved"][ticker] = None
            print(f"  ❌ {ticker} — not found")
        
        if (i + 1) % 10 == 0:
            save_progress(progress)
        time.sleep(args.delay * 0.5)  # Lighter calls, shorter delay

    save_progress(progress)
    resolved_count = sum(1 for v in progress["resolved"].values() if v)
    print(f"\n✅ Resolved {resolved_count}/{len(tickers)} tickers\n")

    # Phase 2: Download transcripts
    print("=== Phase 2: Downloading transcripts ===")
    total_downloaded = 0
    total_skipped = 0

    for ticker in tickers:
        info = progress["resolved"].get(ticker)
        if not info:
            continue
        
        kid = info["kid"]
        name = info["name"]
        
        # Check if already fully downloaded
        ticker_dir = DATA_DIR / ticker
        if ticker in progress["downloaded"]:
            total_skipped += 1
            continue
        
        print(f"\n📥 {ticker} ({name})")
        
        # List transcripts
        all_transcripts = client.list_transcripts(kid)
        time.sleep(args.delay * 0.5)
        
        if not all_transcripts:
            print(f"  No transcripts available")
            progress["downloaded"].append(ticker)
            continue
        
        # Filter to earnings calls for target quarters
        earnings = filter_earnings_calls(all_transcripts, quarter_filters)
        
        if not earnings:
            print(f"  No matching earnings calls found (had {len(all_transcripts)} total transcripts)")
            progress["downloaded"].append(ticker)
            save_progress(progress)
            continue
        
        print(f"  Found {len(earnings)} matching earnings calls")
        ticker_dir.mkdir(exist_ok=True)
        
        for ec in earnings:
            key_dev_id = ec.get("keyDevId") or ec.get("transcriptKeyDevId")
            title = ec.get("formattedTitle", "unknown")
            event_date = ec.get("eventDateTime", "")[:10]
            
            # Check if already downloaded
            filename = f"{ticker}_{title.replace(' ', '_')}_{event_date}.json"
            filepath = ticker_dir / filename
            if filepath.exists():
                print(f"  ⏭️  {title} — already exists")
                continue
            
            # Download
            transcript = client.get_transcript(key_dev_id)
            time.sleep(args.delay)
            
            if transcript and "components" in transcript:
                # Save structured JSON
                save_data = {
                    "ticker": ticker,
                    "company": name,
                    "title": title,
                    "event_date": event_date,
                    "header": transcript.get("header", {}),
                    "components": transcript.get("components", []),
                    "text": transcript_to_text(transcript),
                    "downloaded_at": datetime.now().isoformat(),
                }
                filepath.write_text(json.dumps(save_data, indent=2, default=str))
                
                text_len = len(save_data["text"])
                speakers = len(transcript.get("components", []))
                print(f"  ✅ {title} — {text_len:,} chars, {speakers} segments")
                total_downloaded += 1
            else:
                print(f"  ❌ {title} — failed to download")
                progress["errors"].append({"ticker": ticker, "title": title, "key_dev_id": key_dev_id})
        
        progress["downloaded"].append(ticker)
        save_progress(progress)

    save_progress(progress)
    
    print(f"\n{'='*50}")
    print(f"✅ Done! Downloaded {total_downloaded} transcripts")
    print(f"⏭️  Skipped {total_skipped} (already complete)")
    print(f"❌ Errors: {len(progress['errors'])}")
    print(f"📁 Saved to: {DATA_DIR}")


if __name__ == "__main__":
    main()
