"""
Market Desk data fetcher
------------------------
Pulls three datasets for elroytoh.github.io/invest.html:

  1. Mag 7 trailing/forward P/E with a simple valuation signal
  2. A broader "quality" watchlist P/E (edit QUALITY_WATCHLIST below)
  3. A covered call screener across the quality watchlist, ranked by
     annualized return

No API key required (yfinance wraps public Yahoo Finance endpoints).
Intended to run on a schedule via GitHub Actions -- see
.github/workflows/market-data.yml -- which commits the resulting JSON
in /data back to the repo.

Run locally to test:
    pip install yfinance
    python scripts/fetch_market_data.py
"""

import json
import os
import time
from datetime import datetime, timezone

import yfinance as yf

MAG7 = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]

# Edit this list freely -- it drives both the watchlist P/E table
# and the covered call screener. Keep it to names you'd actually be
# comfortable holding, since covered calls assume you own (or are
# happy to own) the underlying.
QUALITY_WATCHLIST = [
    "AAPL", "MSFT", "V", "MA", "COST", "JNJ", "PG", "KO", "PEP",
    "HD", "WMT", "JPM", "UNH", "ABBV", "MCD", "XOM", "CVX", "DIS",
    "NKE", "BRK-B",
]

BENCHMARKS = {"S&P 500 (SPY)": "SPY", "Nasdaq 100 (QQQ)": "QQQ"}

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

MAX_EXPIRIES = 3          # nearest N option expiries to screen per ticker
MIN_OPEN_INTEREST = 10    # liquidity floor so illiquid strikes get filtered out
REQUEST_PAUSE = 0.4       # be polite to Yahoo's endpoints


def safe_get(info, key, default=None):
    return info.get(key, default) if info else default


def fetch_pe_row(ticker):
    t = yf.Ticker(ticker)
    info = t.info or {}

    price = safe_get(info, "currentPrice") or safe_get(info, "regularMarketPrice")
    trailing_pe = safe_get(info, "trailingPE")
    forward_pe = safe_get(info, "forwardPE")

    # Crude 5y valuation anchor: reconstruct an implied trailing PE series
    # from monthly close price / current trailing EPS, then average it.
    # This is a simplification (EPS isn't held constant historically in
    # reality) but is good enough as a "rich vs cheap vs its own history" cue.
    avg_pe = None
    try:
        eps = safe_get(info, "trailingEps")
        hist = t.history(period="5y", interval="1mo")["Close"]
        if eps and eps > 0 and not hist.empty:
            avg_pe = round(float((hist / eps).mean()), 2)
    except Exception:
        pass

    signal = "n/a"
    if trailing_pe and avg_pe:
        if trailing_pe < avg_pe * 0.9:
            signal = "below avg"
        elif trailing_pe > avg_pe * 1.1:
            signal = "above avg"
        else:
            signal = "in line"

    return {
        "ticker": ticker,
        "name": safe_get(info, "shortName", ticker),
        "sector": safe_get(info, "sector"),
        "price": price,
        "trailingPE": round(trailing_pe, 2) if trailing_pe else None,
        "forwardPE": round(forward_pe, 2) if forward_pe else None,
        "avgPE5y": avg_pe,
        "signal": signal,
    }


def fetch_covered_calls(ticker):
    t = yf.Ticker(ticker)
    info = t.info or {}
    price = safe_get(info, "currentPrice") or safe_get(info, "regularMarketPrice")
    if not price:
        return []

    try:
        expiries = t.options[:MAX_EXPIRIES]
    except Exception:
        return []

    today = datetime.now(timezone.utc).date()
    rows = []

    for exp in expiries:
        try:
            calls = t.option_chain(exp).calls
        except Exception:
            continue

        exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
        dte = (exp_date - today).days
        if dte <= 0:
            continue

        otm_calls = calls[calls["strike"] > price]
        for _, row in otm_calls.iterrows():
            bid, ask = row.get("bid", 0) or 0, row.get("ask", 0) or 0
            mid = (bid + ask) / 2 if (bid and ask) else (row.get("lastPrice") or 0)
            if not mid or mid <= 0:
                continue

            oi = row.get("openInterest") or 0
            if oi < MIN_OPEN_INTEREST:
                continue

            strike = float(row["strike"])
            premium_yield = round(mid / price * 100, 2)
            annualized = round(premium_yield * (365 / dte), 2)

            rows.append({
                "ticker": ticker,
                "expiry": exp,
                "dte": dte,
                "strike": strike,
                "stockPrice": round(price, 2),
                "premium": round(mid, 2),
                "pctOTM": round((strike - price) / price * 100, 2),
                "premiumYieldPct": premium_yield,
                "annualizedReturnPct": annualized,
                "breakeven": round(price - mid, 2),
                "openInterest": int(oi),
            })

    return rows


def build_rows(tickers, label_overrides=None):
    rows = []
    for tk in tickers:
        try:
            row = fetch_pe_row(tk)
            if label_overrides and tk in label_overrides:
                row["name"] = label_overrides[tk]
            rows.append(row)
        except Exception as e:
            rows.append({"ticker": tk, "error": str(e)})
        time.sleep(REQUEST_PAUSE)
    return rows


def main():
    now = datetime.now(timezone.utc).isoformat()
    os.makedirs(OUT_DIR, exist_ok=True)

    mag7_rows = build_rows(MAG7)
    watchlist_rows = build_rows(QUALITY_WATCHLIST)
    benchmark_rows = build_rows(list(BENCHMARKS.values()), {v: k for k, v in BENCHMARKS.items()})

    covered_calls = []
    for tk in QUALITY_WATCHLIST:
        try:
            covered_calls.extend(fetch_covered_calls(tk))
        except Exception:
            pass
        time.sleep(REQUEST_PAUSE)

    covered_calls.sort(key=lambda r: r.get("annualizedReturnPct", 0), reverse=True)

    with open(os.path.join(OUT_DIR, "mag7_pe.json"), "w") as f:
        json.dump({"asOf": now, "rows": mag7_rows}, f, indent=2)

    with open(os.path.join(OUT_DIR, "watchlist_pe.json"), "w") as f:
        json.dump({"asOf": now, "benchmarks": benchmark_rows, "rows": watchlist_rows}, f, indent=2)

    with open(os.path.join(OUT_DIR, "covered_calls.json"), "w") as f:
        json.dump({"asOf": now, "rows": covered_calls[:150]}, f, indent=2)

    print(f"Mag7: {len(mag7_rows)} | Watchlist: {len(watchlist_rows)} | Covered calls: {len(covered_calls)}")


if __name__ == "__main__":
    main()
