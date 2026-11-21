"""Brute-force every strategy x parameter x exit x timeframe on XAUUSD 2016-2026.

In-sample (IS): 2016-2021 -> used to pick parameters.
Out-of-sample (OOS): 2022-2026 -> never seen during selection, the honest check.
"""
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from data import load, TIMEFRAMES  # noqa: E402
from engine import run  # noqa: E402
from strategies import atr, grid  # noqa: E402

COST = 0.35          # USD/oz round trip: spread + commission + slippage
SPLIT = pd.Timestamp("2022-01-01")
SL = [1.0, 1.5, 2.0, 3.0]
TP = [0.0, 1.0, 2.0, 3.0, 4.0]   # 0 = no TP, exit on SL / opposite signal / time
OUT = Path(__file__).resolve().parent.parent / "results"


def stats(r):
    n = len(r)
    if n == 0:
        return dict(n=0, win=np.nan, avgR=np.nan, pf=np.nan, sumR=0.0, ddR=0.0, t=np.nan)
    eq = np.cumsum(r)
    dd = (np.maximum.accumulate(np.maximum(eq, 0)) - eq).max()
    pos, neg = r[r > 0].sum(), -r[r < 0].sum()
    sd = r.std(ddof=1) if n > 1 else np.nan
    return dict(n=n, win=(r > 0).mean(), avgR=r.mean(), pf=pos / neg if neg > 0 else np.inf,
                sumR=r.sum(), ddR=dd, t=r.mean() / sd * np.sqrt(n) if sd and sd > 0 else np.nan)


def main(tfs):
    OUT.mkdir(exist_ok=True)
    rows = []
    for tf in tfs:
        df = load(tf)
        o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
        a = atr(df)
        years = df.index.year.values
        is_mask_bar = (df.index < SPLIT)
        max_hold_orb = int(8 * 60 / {"1m": 1, "5m": 5, "15m": 15, "30m": 30}.get(tf, 60))
        t0 = time.time()
        for fam, params, fn in grid(tf):
            sig = fn(df, **params)
            mh = max_hold_orb if fam == "Opening range breakout" else 0
            for sl, tp, lo in itertools.product(SL, TP, (0, 1)):
                r, ei = run(o, h, l, c, a, sig, sl, tp, mh, COST, lo)
                isr = r[is_mask_bar[ei]]
                oos = r[~is_mask_bar[ei]]
                yr = pd.Series(r).groupby(years[ei]).sum()
                row = dict(tf=tf, family=fam, params=str(params), sl=sl, tp=tp,
                           direction="long only" if lo else "long+short",
                           yrs_pos=int((yr > 0).sum()), yrs=len(yr))
                row.update({f"is_{k}": v for k, v in stats(isr).items()})
                row.update({f"oos_{k}": v for k, v in stats(oos).items()})
                row.update({f"all_{k}": v for k, v in stats(r).items()})
                rows.append(row)
        print(f"{tf}: {len(rows)} runs total, {time.time() - t0:.0f}s", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "all_runs.csv", index=False)
    return res


if __name__ == "__main__":
    main(sys.argv[1:] or list(TIMEFRAMES))
