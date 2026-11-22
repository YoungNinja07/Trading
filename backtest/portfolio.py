"""Combine several robust long-only gold strategies into one portfolio (each trade = 1R risk)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from data import load
from engine import run
from strategies import atr, donchian, macd, rsi_mr, orb

COST = 0.35
STRATS = {
    "A Donchian55 4h":      ("4h",  donchian, dict(n=55), 2.0, 4.0, 0),
    "B MACD+EMA200 4h":     ("4h",  macd, dict(fast=12, slow=26, signal=9, trend=1), 3.0, 3.0, 0),
    "C RSI2 1d":            ("1d",  rsi_mr, dict(n=2, lo=10, trend=0), 3.0, 4.0, 0),
    "D ORB NY 30m":         ("30m", orb, dict(session_hour=14, range_minutes=30), 3.0, 0.0, 16),
    "E Donchian10 4h":      ("4h",  donchian, dict(n=10), 2.0, 3.0, 0),
}


def trades(name):
    tf, fn, p, sl, tp, mh = STRATS[name]
    df = load(tf)
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    r, ei = run(o, h, l, c, atr(df), fn(df, **p), sl, tp, mh, COST, 1)
    return pd.Series(r, index=df.index[ei], name=name)


def summary(s):
    daily = s.groupby(s.index.normalize()).sum()
    eq = daily.cumsum()
    dd = (eq.cummax().clip(lower=0) - eq).max()
    yr = s.groupby(s.index.year).sum()
    return dict(trades_per_month=round(len(s) / 120, 1), win=round((s > 0).mean(), 2),
                R_per_year=round(s.sum() / 10, 1), maxDD_R=round(dd, 1),
                ret_to_dd=round(s.sum() / 10 / dd, 2), worst_year_R=round(yr.min(), 1),
                losing_years=int((yr < 0).sum()))


if __name__ == "__main__":
    T = {k: trades(k) for k in STRATS}
    rows = {k: summary(v) for k, v in T.items()}
    for combo in ["AB", "AC", "ABC", "ABCD", "ABCDE", "ACD", "ABD"]:
        s = pd.concat([T[k] for k in STRATS if k[0] in combo]).sort_index()
        rows["portfolio " + combo] = summary(s)
    print(pd.DataFrame(rows).T.to_string())
    m = pd.DataFrame({k: v.groupby(v.index.to_period("M")).sum() for k, v in T.items()}).fillna(0)
    print("\nmonthly correlation:\n", m.corr().round(2).to_string())
