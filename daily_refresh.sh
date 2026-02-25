#!/bin/bash
# Daily data refresh for Software Spend Shift dashboard
# Runs scanners, scorers, then pushes to GitHub so Streamlit Cloud updates
set -e

cd "$(dirname "$0")"
source .venv/bin/activate

echo "=== Software Dashboard Daily Refresh — $(date) ==="

# 1. Scan for new articles
echo "[1/4] Scanning for new articles..."
python3 scanner_v2.py 2>&1 || echo "WARNING: scanner_v2 failed"

# 2. Score any new/unscored articles (fast scorer)
echo "[2/4] Scoring new articles..."
python3 score_fast3.py 2>&1 || python3 score_articles.py 2>&1 || echo "WARNING: scoring failed"

# 3. Translate any new non-English titles
if [ -f scripts/translate_titles.py ]; then
    echo "[3/4] Translating titles..."
    python3 scripts/translate_titles.py 2>&1 || echo "WARNING: translation failed"
else
    echo "[3/4] No translate script, skipping"
fi

# 4. Push to GitHub
echo "[4/4] Pushing to GitHub..."
git add -A
if git diff --cached --quiet; then
    echo "No changes to push"
else
    git commit -m "Daily data refresh — $(date +%Y-%m-%d)"
    git push origin main
    echo "Pushed to GitHub ✅"
fi

echo "=== Refresh complete — $(date) ==="
