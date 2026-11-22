"""Donchian breakout on many markets, 4h bars. Parameters chosen on DEV 2016-2023,
checked on 2024-2025 (VAL) and 2026 (HOLD). Costs as % of price (spread+commission+slippage)."""
import itertools
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from data import load
from engine import run_pct
from strategies import atr, donchian

ROOT = Path(__file__).resolve().parent.parent
MK = ROOT / "data" / "raw" / "markets"
DEV, VAL = pd.Timestamp("2024-01-01"), pd.Timestamp("2026-01-10")
MARKETS = {  # name: (folder, round-trip cost as fraction of price)
    "Gold": (None, 0.00012),
    "Silver": ("XAGUSD_Silver", 0.0008),
    "Nasdaq 100": ("NAS100USD_US-Nas-100", 0.00012),
    "WTI oil": ("WTICOUSD_West-Texas-Oil", 0.0006),
    "S&P 500": ("SPX500USD_US-SPX-500", 0.00012),
    "DAX": ("DE30EUR_Germany-30", 0.00012),
    "Copper": ("XCUUSD_Copper", 0.0008),
    "Bitcoin": ("BTCUSD_Bitcoin", 0.0015),
}
NS = [20, 40, 55, 80, 100, 150, 200]
SLS = [1.5, 2.0, 3.0]
TPS = [2.0, 3.0, 4.0, 6.0]


def load_market(folder):
    if folder is None:
        return load("4h")
    pq = ROOT / "data" / "cache" / f"{folder}_4h.parquet"
    if pq.exists():
        return pd.read_parquet(pq)
    d = pd.concat([pd.read_parquet(f) for f in sorted((MK / folder).glob("*.parquet"))])
    d.index = pd.to_datetime(d["time"], unit="s")
    d = d[["open", "high", "low", "close"]].astype(float)
    d = d[~d.index.duplicated()].sort_index()
    d = d[d.index >= "2016-01-01"]
    d = d.resample("4h", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    d.to_parquet(pq)
    return d


def st(r):
    if len(r) < 3:
        return len(r), r.sum(), np.nan
    return len(r), r.sum(), r.mean() / r.std(ddof=1) * np.sqrt(len(r))


def main():
    rows = []
    for name, (folder, cost) in MARKETS.items():
        df = load_market(folder)
        o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
        a = atr(df)
        yrs = (df.index[-1] - df.index[0]).days / 365.25
        for n in NS:
            sig = donchian(df, n)
            for sl, tp, lo in itertools.product(SLS, TPS, (1, 0)):
                r, ei = run_pct(o, h, l, c, a, sig, sl, tp, cost, lo)
                t = df.index[ei]
                eq = np.cumsum(r)
                dd = (np.maximum.accumulate(np.maximum(eq, 0)) - eq).max() if len(r) else 0
                row = dict(market=name, n=n, sl=sl, tp=tp, dir="long" if lo else "long+short",
                           trades_per_month=len(r) / yrs / 12, R_per_year=r.sum() / yrs, maxDD=dd)
                for k, m in (("dev", t < DEV), ("val", (t >= DEV) & (t < VAL)), ("hold", t >= VAL)):
                    nn, s, tt = st(r[m])
                    row.update({f"{k}_n": nn, f"{k}_R": s, f"{k}_t": tt})
                rows.append(row)
        print(name, "done", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "results" / "multimarket_runs.csv", index=False)
    return res


if __name__ == "__main__":
    main()
