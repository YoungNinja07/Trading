"""Research round: theory-driven ideas with strict split.
DEV 2016-2023 (choose), VAL 2024-2025 (check), HOLDOUT 2026 (final, look once)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from data import load
from engine import run_trail
from strategies import atr

DEV, VAL = pd.Timestamp("2024-01-01"), pd.Timestamp("2026-01-10")


def split(r, t):
    r, t = np.asarray(r), pd.DatetimeIndex(t)
    parts = {}
    for name, m in (("dev", t < DEV), ("val", (t >= DEV) & (t < VAL)), ("hold", t >= VAL)):
        x = r[m]
        parts[name] = (len(x), x.sum(), x.mean() / x.std(ddof=1) * np.sqrt(len(x)) if len(x) > 2 and x.std() > 0 else np.nan)
    return parts


def fmt(p):
    return " | ".join(f"{k} n={v[0]:4d} {v[1]:+7.1f} t={v[2]:+.1f}" for k, v in p.items())


def idea_turtle():
    print("\n### A) Turtle breakout with trailing exit (4h, 1d)")
    for tf in ("4h", "1d"):
        df = load(tf)
        o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
        a = atr(df)
        for en, ex in ((20, 10), (55, 20), (55, 10), (100, 20), (100, 50)):
            for sh in (0, 1):
                r, ei = run_trail(o, h, l, c, a, en, ex, 2.0, 0.35, sh)
                print(f"{tf} in{en:3d}/out{ex:2d} {'L+S' if sh else 'L  '}: {fmt(split(r, df.index[ei]))}")


def idea_tsmom():
    print("\n### B) Time-series momentum, daily, long/short by sign of N-day return, 1-day holding, vol-scaled (R = daily P&L / ATR)")
    df = load("1d")
    c = df.close
    ret = c.pct_change().shift(-1)  # next day return
    a = pd.Series(atr(df), index=df.index)
    for n in (20, 60, 120, 250):
        sig = np.sign(c - c.shift(n))
        pnl = sig * (c.shift(-1) - c) / a - 0.35 / a * (sig.diff().abs() > 0)
        pnl = pnl.dropna()
        print(f"lookback {n:3d}: {fmt(split(pnl.values, pnl.index))}")
        lo = (sig.clip(lower=0) * (c.shift(-1) - c) / a).dropna()
        print(f"   long-only : {fmt(split(lo.values, lo.index))}")


def idea_hours():
    print("\n### C) Intraday seasonality: gold return by UTC hour (choose on DEV, check VAL/HOLD)")
    df = load("1h")
    r = (df.close - df.open)
    a = pd.Series(atr(df), index=df.index).shift(1)
    z = (r / a).dropna()
    hr = z.index.hour
    tab = pd.DataFrame({"dev": z[z.index < DEV].groupby(hr[z.index < DEV]).mean(),
                        "val": z[(z.index >= DEV) & (z.index < VAL)].groupby(hr[(z.index >= DEV) & (z.index < VAL)]).mean(),
                        "hold": z[z.index >= VAL].groupby(hr[z.index >= VAL]).mean()})
    print((tab * 100).round(1).to_string())
    best = tab.dev.sort_values()
    lo, hi = list(best.index[:3]), list(best.index[-3:])
    print("DEV best long hours:", hi, " best short hours:", lo)
    for name, hours, s in (("long best-3", hi, 1), ("short worst-3", lo, -1)):
        m = np.isin(hr, hours)
        x = s * z[m] - 0.35 / a.reindex(z.index)[m]
        print(f"{name}: {fmt(split(x.values, x.index))}")


if __name__ == "__main__":
    idea_turtle()
    idea_tsmom()
    idea_hours()
