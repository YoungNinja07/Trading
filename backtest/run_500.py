"""Run ~500 strategy definitions x 4 exits x 2 directions with honest validation.

For every run we also simulate RANDOM entries with the same number of signals, same exits and
direction (5 seeds). 'edge' = strategy avgR - random avgR. On gold 2016-2026 random long entries
already make money (gold went ~4x), so only the edge above random is real skill.
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from data import load  # noqa: E402
from engine import run  # noqa: E402
from library import definitions, signal  # noqa: E402
from strategies import atr  # noqa: E402

COST = 0.35
SPLIT = pd.Timestamp("2022-01-01")
EXITS = [(1.5, 3.0), (2.0, 2.0), (2.0, 4.0), (3.0, 3.0)]
SEEDS = 5
OUT = Path(__file__).resolve().parent.parent / "results"


def st(r):
    n = len(r)
    if n < 2:
        return n, np.nan, np.nan, np.nan
    sd = r.std(ddof=1)
    return n, r.mean(), (r > 0).mean(), r.mean() / sd * np.sqrt(n) if sd > 0 else np.nan


def work(tf):
    df = load(tf)
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    a = atr(df)
    isbar = df.index < SPLIT
    years = df.index.year.values
    rng = np.random.default_rng(0)
    rows = []
    for d in [x for x in definitions() if x["tf"] == tf]:
        sig = signal(df, d)
        for lo in (0, 1):
            s_use = np.where(sig > 0, sig, 0).astype(np.int8) if lo else sig
            nl, ns = int((s_use == 1).sum()), int((s_use == -1).sum())
            if nl + ns < 20:
                continue
            for sl, tp in EXITS:
                r, ei = run(o, h, l, c, a, s_use, sl, tp, 0, COST, 0)
                if len(r) < 20:
                    continue
                isr, oosr = r[isbar[ei]], r[~isbar[ei]]
                # random baseline with same signal counts
                rb_is, rb_oos = [], []
                for _ in range(SEEDS):
                    rs = np.zeros(len(c), dtype=np.int8)
                    rs[rng.choice(len(c) - 1, nl, replace=False)] = 1
                    if ns:
                        free = np.flatnonzero(rs == 0)
                        rs[rng.choice(free, ns, replace=False)] = -1
                    rr, re = run(o, h, l, c, a, rs, sl, tp, 0, COST, 0)
                    rb_is.append(rr[isbar[re]].mean())
                    rb_oos.append(rr[~isbar[re]].mean())
                yr = pd.Series(r).groupby(years[ei]).sum()
                eq = np.cumsum(r)
                n_is, a_is, w_is, t_is = st(isr)
                n_oos, a_oos, w_oos, t_oos = st(oosr)
                n_all, a_all, w_all, t_all = st(r)
                rows.append(dict(
                    tf=tf, entry=d["entry"], params=str(d["params"]), filter=d["filter"],
                    direction="long only" if lo else "long+short", sl=sl, tp=tp,
                    trades_per_month=n_all / 120, win=w_all, avgR=a_all, R_per_year=r.sum() / 10,
                    maxDD_R=(np.maximum.accumulate(np.maximum(eq, 0)) - eq).max(),
                    yrs_pos=int((yr > 0).sum()),
                    is_n=n_is, is_avgR=a_is, is_t=t_is, oos_n=n_oos, oos_avgR=a_oos, oos_t=t_oos,
                    rand_is=np.nanmean(rb_is), rand_oos=np.nanmean(rb_oos),
                    edge_is=a_is - np.nanmean(rb_is), edge_oos=a_oos - np.nanmean(rb_oos)))
    print(tf, "done", len(rows), flush=True)
    return rows


if __name__ == "__main__":
    with Pool(4) as p:
        parts = p.map(work, ["30m", "1h", "4h", "1d"])
    res = pd.DataFrame([r for part in parts for r in part])
    OUT.mkdir(exist_ok=True)
    res.to_csv(OUT / "runs_500.csv", index=False)
    print("total runs:", len(res))
