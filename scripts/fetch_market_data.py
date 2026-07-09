"""
Market Desk data fetcher (v2)
------------------------------
Pulls four datasets for elroytoh.github.io/invest.html:

  1. mag7_pe.json         -- Mag 7 trailing/forward P/E + valuation signal
  2. watchlist_pe.json    -- full S&P 500 P/E, sorted smallest -> largest
  3. sector_heatmap.json  -- avg P/E and 1-month return per GICS sector
                             (shows rotation, e.g. tech -> healthcare)
  4. covered_calls.json   -- OTM call screener over a curated quality
                             watchlist, with next-earnings-date awareness
                             so you can avoid writing calls into earnings

Design note: covered call chains are pulled from a smaller curated list
(COVERED_CALL_WATCHLIST), not the full S&P 500 -- pulling option chains
for 500 tickers would make the Action run far too long and risks getting
rate-limited by Yahoo. The P/E watchlist and sector heatmap DO cover the
full S&P 500, since that's just one .info + one history call per ticker.

No API key required (yfinance wraps public Yahoo Finance endpoints).
Run locally to test:
    pip install yfinance pandas lxml requests
    python scripts/fetch_market_data.py
"""

import json
import os
import time
from datetime import datetime, timezone

import requests
import pandas as pd
import yfinance as yf

MAG7 = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]

# Curated "quality" list used ONLY for the covered call screener -- swap
# in whatever names you'd actually want to hold and write calls against.
COVERED_CALL_WATCHLIST = [
    "AAPL", "MSFT", "V", "MA", "COST", "JNJ", "PG", "KO", "PEP",
    "HD", "WMT", "JPM", "UNH", "ABBV", "MCD", "XOM", "CVX", "DIS",
    "NKE", "BRK-B",
]

BENCHMARKS = {"S&P 500 (SPY)": "SPY", "Nasdaq 100 (QQQ)": "QQQ"}

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

MIN_DTE = 30   # covered call sweet spot: enough premium/theta decay...
MAX_DTE = 45   # ...without locking up shares for too long
MIN_OPEN_INTEREST = 10
REQUEST_PAUSE = 0.35
WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def safe_get(info, key, default=None):
    return info.get(key, default) if info else default


def get_sp500_constituents():
    """Scrapes the current S&P 500 list (ticker, name, GICS sector) from
    Wikipedia. Falls back to the Mag7 + covered-call list if it fails,
    so the pipeline never hard-crashes just because Wikipedia's table
    layout shifted."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; MarketDeskBot/1.0)"}
        resp = requests.get(WIKI_SP500_URL, headers=headers, timeout=20)
        resp.raise_for_status()
        tables = pd.read_html(resp.text)
        df = tables[0]
        rows = []
        for _, r in df.iterrows():
            ticker = str(r["Symbol"]).strip().replace(".", "-")
            rows.append({
                "ticker": ticker,
                "name": str(r.get("Security", ticker)),
                "sector": str(r.get("GICS Sector", "Unknown")),
            })
        return rows
    except Exception as e:
        print(f"WARNING: couldn't fetch S&P 500 list from Wikipedia ({e}); "
              f"falling back to a small default list.")
        fallback = MAG7 + COVERED_CALL_WATCHLIST
        return [{"ticker": t, "name": t, "sector": "Unknown"} for t in set(fallback)]


def fetch_pe_row(ticker, sector=None, name=None, retries=2):
    """Returns price/PE fields plus a 1-month return, computed from the
    same 5y monthly history pull used for the valuation anchor (so we
    don't need a second network call per ticker)."""
    for attempt in range(retries + 1):
        try:
            t = yf.Ticker(ticker)
            info = t.info or {}
            price = safe_get(info, "currentPrice") or safe_get(info, "regularMarketPrice")
            trailing_pe = safe_get(info, "trailingPE")
            forward_pe = safe_get(info, "forwardPE")

            avg_pe = None
            return_1m = None
            try:
                eps = safe_get(info, "trailingEps")
                hist = t.history(period="5y", interval="1mo")["Close"]
                if eps and eps > 0 and not hist.empty:
                    avg_pe = round(float((hist / eps).mean()), 2)
                if len(hist) >= 2:
                    return_1m = round(float((hist.iloc[-1] / hist.iloc[-2] - 1) * 100), 2)
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
                "name": name or safe_get(info, "shortName", ticker),
                "sector": sector or safe_get(info, "sector") or "Unknown",
                "price": price,
                "trailingPE": round(trailing_pe, 2) if trailing_pe else None,
                "forwardPE": round(forward_pe, 2) if forward_pe else None,
                "avgPE5y": avg_pe,
                "return1M": return_1m,
                "signal": signal,
            }
        except Exception as e:
            if attempt < retries:
                time.sleep(1.0)
                continue
            return {"ticker": ticker, "name": name or ticker, "sector": sector or "Unknown", "error": str(e)}


def next_earnings_date(ticker_obj):
    """Best-effort lookup of the next upcoming earnings date. Returns
    an ISO date string or None if unavailable (yfinance's earnings
    calendar coverage varies by ticker)."""
    try:
        df = ticker_obj.get_earnings_dates(limit=6)
        if df is None or df.empty:
            return None
        today = pd.Timestamp.now(tz=df.index.tz) if df.index.tz else pd.Timestamp.now()
        future = df.index[df.index >= today]
        if len(future) == 0:
            return None
        return future.min().strftime("%Y-%m-%d")
    except Exception:
        return None


def fetch_covered_calls(ticker):
    t = yf.Ticker(ticker)
    info = t.info or {}
    price = safe_get(info, "currentPrice") or safe_get(info, "regularMarketPrice")
    if not price:
        return []

    earnings_date = next_earnings_date(t)
    earnings_dt = datetime.strptime(earnings_date, "%Y-%m-%d").date() if earnings_date else None

    try:
        all_expiries = t.options
    except Exception:
        return []

    today = datetime.now(timezone.utc).date()

    # Only keep expiries landing in the 30-45 DTE sweet spot
    target_expiries = []
    for exp in all_expiries:
        exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
        dte = (exp_date - today).days
        if MIN_DTE <= dte <= MAX_DTE:
            target_expiries.append((exp, exp_date, dte))

    rows = []
    for exp, exp_date, dte in target_expiries:
        try:
            calls = t.option_chain(exp).calls
        except Exception:
            continue

        earnings_before_expiry = bool(earnings_dt and earnings_dt <= exp_date)

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
                "nextEarningsDate": earnings_date,
                "earningsBeforeExpiry": earnings_before_expiry,
            })

    return rows


def main():
    now = datetime.now(timezone.utc).isoformat()
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- Mag 7 (own tab) ---
    mag7_rows = []
    for tk in MAG7:
        mag7_rows.append(fetch_pe_row(tk))
        time.sleep(REQUEST_PAUSE)

    with open(os.path.join(OUT_DIR, "mag7_pe.json"), "w") as f:
        json.dump({"asOf": now, "rows": mag7_rows}, f, indent=2)

    # --- Benchmarks ---
    benchmark_rows = []
    for name, tk in BENCHMARKS.items():
        row = fetch_pe_row(tk, name=name)
        benchmark_rows.append(row)
        time.sleep(REQUEST_PAUSE)

    # --- Full S&P 500 watchlist ---
    constituents = get_sp500_constituents()
    print(f"Fetching P/E for {len(constituents)} S&P 500 constituents...")

    sp500_rows = []
    for c in constituents:
        row = fetch_pe_row(c["ticker"], sector=c["sector"], name=c["name"])
        sp500_rows.append(row)
        time.sleep(REQUEST_PAUSE)

    # sort ascending by trailing P/E, missing values pushed to the end
    def pe_sort_key(r):
        pe = r.get("trailingPE")
        return pe if pe is not None else float("inf")

    sp500_rows.sort(key=pe_sort_key)

    with open(os.path.join(OUT_DIR, "watchlist_pe.json"), "w") as f:
        json.dump({"asOf": now, "benchmarks": benchmark_rows, "rows": sp500_rows}, f, indent=2)

    # --- Sector heatmap (avg P/E + avg 1-month return per sector) ---
    sector_agg = {}
    for r in sp500_rows:
        sec = r.get("sector") or "Unknown"
        if sec not in sector_agg:
            sector_agg[sec] = {"peSum": 0.0, "peCount": 0, "retSum": 0.0, "retCount": 0}
        if r.get("trailingPE") is not None:
            sector_agg[sec]["peSum"] += r["trailingPE"]
            sector_agg[sec]["peCount"] += 1
        if r.get("return1M") is not None:
            sector_agg[sec]["retSum"] += r["return1M"]
            sector_agg[sec]["retCount"] += 1

    sector_rows = []
    for sec, agg in sector_agg.items():
        sector_rows.append({
            "sector": sec,
            "avgPE": round(agg["peSum"] / agg["peCount"], 2) if agg["peCount"] else None,
            "avgReturn1M": round(agg["retSum"] / agg["retCount"], 2) if agg["retCount"] else None,
            "count": agg["peCount"],
        })
    sector_rows.sort(key=lambda r: (r["avgReturn1M"] is None, -(r["avgReturn1M"] or 0)))

    with open(os.path.join(OUT_DIR, "sector_heatmap.json"), "w") as f:
        json.dump({"asOf": now, "rows": sector_rows}, f, indent=2)

    # --- Covered call screener (curated quality list only) ---
    covered_calls = []
    for tk in COVERED_CALL_WATCHLIST:
        try:
            covered_calls.extend(fetch_covered_calls(tk))
        except Exception:
            pass
        time.sleep(REQUEST_PAUSE)

    covered_calls.sort(key=lambda r: r.get("annualizedReturnPct", 0), reverse=True)

    with open(os.path.join(OUT_DIR, "covered_calls.json"), "w") as f:
        json.dump({"asOf": now, "rows": covered_calls[:200]}, f, indent=2)

    print(f"Mag7: {len(mag7_rows)} | S&P500 watchlist: {len(sp500_rows)} | "
          f"Sectors: {len(sector_rows)} | Covered calls: {len(covered_calls)}")


if __name__ == "__main__":
    main()
