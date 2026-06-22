#!/usr/bin/env python3
"""
options_scanner.py  —  Directional options screener (calls + puts)

WHAT THIS DOES
  Pulls live option chains via yfinance and ranks contracts by a blended score
  built from several factors. It surfaces candidates worth a closer look. It is
  a RESEARCH AID, not a prediction. High score = high leverage + decent liquidity,
  NOT high probability of profit. You can lose 100% of an option's premium fast.

USAGE
  python options_scanner.py AAPL MSFT NVDA
  python options_scanner.py AAPL --side call --max-dte 45 --top 15
  python options_scanner.py SPY  --side put  --min-dte 7 --max-dte 30

  Requires:  pip install yfinance pandas numpy scipy
"""

import argparse
import sys
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except ImportError:
    sys.exit("Missing dependency. Run: pip install yfinance pandas numpy scipy")

from scipy.stats import norm


# ----------------------------------------------------------------------
# Black-Scholes greeks (used to estimate delta/gamma when scoring leverage)
# ----------------------------------------------------------------------
def bs_greeks(S, K, T, r, sigma, kind):
    """Return (delta, gamma) for a European option. T in years, sigma annualized."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return np.nan, np.nan
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    delta = norm.cdf(d1) if kind == "call" else norm.cdf(d1) - 1.0
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return delta, gamma


# ----------------------------------------------------------------------
# Per-ticker scan
# ----------------------------------------------------------------------
def scan_ticker(symbol, side, min_dte, max_dte, r=0.045):
    tk = yf.Ticker(symbol)

    # current price + recent realized volatility (for IV-vs-RV comparison).
    # Wrapped because yfinance can raise (not just return empty) when Yahoo
    # blocks or rate-limits the request — common on shared CI IPs.
    try:
        hist = tk.history(period="3mo")
    except Exception as e:
        print(f"  [skip] {symbol}: price fetch failed ({type(e).__name__})")
        return pd.DataFrame()
    if hist is None or hist.empty:
        print(f"  [skip] {symbol}: no price history")
        return pd.DataFrame()
    S = float(hist["Close"].iloc[-1])
    log_ret = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
    realized_vol = float(log_ret.std() * np.sqrt(252)) if len(log_ret) > 5 else np.nan

    try:
        expirations = tk.options
    except Exception as e:
        print(f"  [skip] {symbol}: options fetch failed ({type(e).__name__})")
        return pd.DataFrame()
    if not expirations:
        print(f"  [skip] {symbol}: no options listed")
        return pd.DataFrame()

    rows = []
    now = datetime.now(timezone.utc)

    for exp in expirations:
        exp_dt = datetime.strptime(exp, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        dte = (exp_dt - now).days
        if dte < min_dte or dte > max_dte:
            continue
        T = max(dte, 1) / 365.0

        try:
            chain = tk.option_chain(exp)
        except Exception:
            continue

        sides = []
        if side in ("call", "both"):
            sides.append(("call", chain.calls))
        if side in ("put", "both"):
            sides.append(("put", chain.puts))

        for kind, df in sides:
            if df is None or df.empty:
                continue
            df = df.copy()

            def num(val, default=0.0):
                """Coerce to float, treating NaN/None/blank as the default."""
                try:
                    f = float(val)
                except (TypeError, ValueError):
                    return default
                return default if np.isnan(f) else f

            for _, opt in df.iterrows():
                K = num(opt.get("strike"), np.nan)
                bid = num(opt.get("bid"))
                ask = num(opt.get("ask"))
                last = num(opt.get("lastPrice"))
                vol = num(opt.get("volume"))
                oi = num(opt.get("openInterest"))
                iv = num(opt.get("impliedVolatility"))

                mid = (bid + ask) / 2 if (bid > 0 and ask > 0) else last
                if mid <= 0 or K <= 0 or np.isnan(K) or np.isnan(mid):
                    continue

                # ---- factor 1: leverage (delta * S / premium) ----
                delta, gamma = bs_greeks(S, K, T, r, iv if iv > 0 else realized_vol, kind)
                if np.isnan(delta):
                    continue
                leverage = abs(delta) * S / (mid * 100) if mid > 0 else 0

                # ---- factor 2: liquidity (volume + open interest, tight spread) ----
                spread_pct = (ask - bid) / mid if (ask > 0 and bid > 0 and mid > 0) else 1.0
                liquidity_raw = np.log1p(vol) + 0.5 * np.log1p(oi)
                tightness = max(0.0, 1.0 - min(spread_pct, 1.0))  # 1 = tight, 0 = wide

                # ---- factor 3: unusual activity (vol relative to OI) ----
                vol_oi = vol / oi if oi > 0 else 0.0

                # ---- factor 4: IV value (IV cheap vs realized = better) ----
                iv_ratio = (iv / realized_vol) if (realized_vol and realized_vol > 0 and iv > 0) else np.nan
                # reward IV below realized, penalize very rich IV
                iv_value = np.clip(1.5 - iv_ratio, -1, 1) if not np.isnan(iv_ratio) else 0.0

                # ---- factor 5: moneyness (favor slightly OTM for convexity) ----
                if kind == "call":
                    otm = (K - S) / S
                else:
                    otm = (S - K) / S
                # peak reward around 2-8% OTM
                moneyness_score = np.exp(-((otm - 0.05) ** 2) / (2 * 0.05 ** 2))

                rows.append({
                    "symbol": symbol, "type": kind, "expiry": exp, "dte": dte,
                    "strike": K, "spot": round(S, 2), "mid": round(mid, 2),
                    "bid": bid, "ask": ask, "volume": int(vol), "open_int": int(oi),
                    "iv": round(iv, 3), "rv": round(realized_vol, 3) if realized_vol else None,
                    "delta": round(delta, 3), "gamma": round(gamma, 4),
                    "leverage": leverage, "spread_pct": round(spread_pct, 3),
                    "vol_oi": round(vol_oi, 2),
                    "_liq": liquidity_raw, "_tight": tightness,
                    "_ivval": iv_value, "_money": moneyness_score, "_unusual": vol_oi,
                })

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Scoring: normalize each factor across the candidate pool, then blend
# ----------------------------------------------------------------------
WEIGHTS = {
    "leverage":  0.30,   # bang for buck
    "liquidity": 0.25,   # can you actually get in/out
    "unusual":   0.15,   # smart-money / flow signal
    "iv_value":  0.15,   # not overpaying for vol
    "moneyness": 0.15,   # convexity sweet spot
}


def zscale(s):
    s = s.astype(float)
    if s.std(ddof=0) == 0 or s.isna().all():
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.mean()) / s.std(ddof=0)


def minmax(s):
    s = s.astype(float)
    lo, hi = s.min(), s.max()
    if hi == lo:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - lo) / (hi - lo)


def score(df):
    if df.empty:
        return df
    # liquidity floor: drop totally dead contracts
    df = df[(df["open_int"] >= 50) | (df["volume"] >= 25)].copy()
    if df.empty:
        return df

    df["f_leverage"]  = minmax(df["leverage"])
    df["f_liquidity"] = minmax(df["_liq"] * (0.5 + 0.5 * df["_tight"]))
    df["f_unusual"]   = minmax(np.log1p(df["_unusual"]))
    df["f_iv_value"]  = minmax(df["_ivval"])
    df["f_moneyness"] = minmax(df["_money"])

    df["score"] = (
        WEIGHTS["leverage"]  * df["f_leverage"] +
        WEIGHTS["liquidity"] * df["f_liquidity"] +
        WEIGHTS["unusual"]   * df["f_unusual"] +
        WEIGHTS["iv_value"]  * df["f_iv_value"] +
        WEIGHTS["moneyness"] * df["f_moneyness"]
    ) * 100

    return df.sort_values("score", ascending=False)


def main():
    p = argparse.ArgumentParser(description="Directional options screener (calls + puts).")
    p.add_argument("tickers", nargs="+", help="Ticker symbols, e.g. AAPL MSFT NVDA")
    p.add_argument("--side", choices=["call", "put", "both"], default="both")
    p.add_argument("--min-dte", type=int, default=7, help="Min days to expiry")
    p.add_argument("--max-dte", type=int, default=45, help="Max days to expiry")
    p.add_argument("--top", type=int, default=10, help="How many to show")
    p.add_argument("--csv", metavar="PATH", default=None,
                   help="Also write the top results to this CSV file")
    args = p.parse_args()

    print(f"\nScanning {len(args.tickers)} ticker(s) | side={args.side} | "
          f"dte {args.min_dte}-{args.max_dte}\n" + "-" * 60)

    frames = []
    for sym in args.tickers:
        print(f"  fetching {sym.upper()} ...")
        try:
            frames.append(scan_ticker(sym.upper(), args.side, args.min_dte, args.max_dte))
        except Exception as e:
            print(f"  [skip] {sym.upper()}: unexpected error ({type(e).__name__}: {e})")
            frames.append(pd.DataFrame())

    allopts = pd.concat([f for f in frames if not f.empty], ignore_index=True) \
        if any(not f.empty for f in frames) else pd.DataFrame()

    cols = ["score", "symbol", "type", "expiry", "dte", "strike", "spot",
            "mid", "delta", "leverage", "iv", "rv", "volume", "open_int",
            "vol_oi", "spread_pct"]

    if allopts.empty:
        print("\nNo qualifying contracts found (data source may be unavailable or "
              "rate-limited). Try widening --max-dte, different tickers, or rerun.")
        if args.csv:
            pd.DataFrame(columns=["scan_utc"] + cols).to_csv(args.csv, index=False)
            print(f"Wrote empty results file to {args.csv}")
        return

    ranked = score(allopts)
    if ranked.empty:
        print("\nContracts found but all failed the liquidity floor.")
        if args.csv:
            pd.DataFrame(columns=["scan_utc"] + cols).to_csv(args.csv, index=False)
            print(f"Wrote empty results file to {args.csv}")
        return

    out = ranked[cols].head(args.top).copy()
    out["score"] = out["score"].round(1)
    out["leverage"] = out["leverage"].round(2)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print("\n" + "=" * 60)
    print(f"TOP {min(args.top, len(out))} CANDIDATES")
    print("=" * 60)
    print(out.to_string(index=False))

    if args.csv:
        out_to_save = out.copy()
        out_to_save.insert(0, "scan_utc", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"))
        out_to_save.to_csv(args.csv, index=False)
        print(f"\nSaved top {len(out_to_save)} results to {args.csv}")

    print("\n" + "-" * 60)
    print("Score blends: leverage, liquidity, unusual volume, IV value, moneyness.")
    print("HIGH SCORE = high leverage + tradeable, NOT high win probability.")
    print("Options can expire worthless. This is a research aid, not advice.")
    print("-" * 60)


if __name__ == "__main__":
    main()
