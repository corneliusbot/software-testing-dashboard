#!/bin/bash
cd "$(dirname "$0")"

# Ensure venv exists
if [ ! -d .venv ]; then
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -q streamlit plotly pandas yfinance pytrends feedparser
else
    source .venv/bin/activate
fi

# Kill existing instance
pkill -f "streamlit run app.py.*8501" 2>/dev/null
sleep 1

# Run a news scan first
echo "Scanning for new articles..."
python3 scanner.py

# Launch dashboard
echo "Starting dashboard on http://localhost:8501"
nohup streamlit run app.py --server.headless true --server.port 8501 > /tmp/streamlit-shift.log 2>&1 &
echo "PID: $! — dashboard running in background"
