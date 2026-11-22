"""Final holdout: strategies selected earlier (on data up to Jan 2026) tested on 2026-01-10 .. 2026-11-20."""
import ast
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from data import load
from engine import run
from strategies import atr
import library as L

HOLD = pd.Timestamp("2026-01-10")
ROOT = Path(__file__).resolve().parent.parent
ENT = {name: fn for name, fn, _ in L.ENTRIES + L.INTRADAY + L.DAILY}
FLT = dict(L.FILTERS)
rng = np.random.default_rng(0)


def test(tf, entry, params, flt, direction, sl, tp):
    df = load(tf)
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    a = atr(df)
    sig = L.signal(df, dict(fn=ENT[entry], params=params, ff=FLT[flt]))
    lo = direction == "long only"
    if lo:
        sig = np.where(sig > 0, sig, 0).astype(np.int8)
    r, ei = run(o, h, l, c, a, sig, sl, tp, 0, 0.35, 0)
    m = df.index[ei] >= HOLD
    rh = r[m]
    # random baseline in 2026 with the same number of signals
    hb = np.flatnonzero(df.index >= HOLD)
    n_sig = int((sig[hb] != 0).sum())
    rand = []
    for _ in range(20):
        rs = np.zeros(len(c), dtype=np.int8)
        if n_sig:
            pick = rng.choice(hb[:-1], n_sig, replace=False)
            rs[pick] = 1 if lo else rng.choice([-1, 1], n_sig)
        rr, re_ = run(o, h, l, c, a, rs, sl, tp, 0, 0.35, 0)
        rand.append(rr[df.index[re_] >= HOLD].sum())
    return len(rh), rh.sum(), float(np.mean(rand))


if __name__ == "__main__":
    p = pd.read_csv(ROOT / "results" / "passed_500.csv")
    rows = []
    for _, s in p.iterrows():
        n, R, rnd = test(s.tf, s.entry, ast.literal_eval(s.params), s["filter"], s.direction, s.sl, s.tp)
        rows.append(dict(tf=s.tf, entry=s.entry, params=s.params, filter=s["filter"], direction=s.direction,
                         sl=s.sl, tp=s.tp, trades_2026=n, R_2026=round(R, 1), random_R_2026=round(rnd, 1)))
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(out.to_string(index=False))
    print(f"\nprofitable in 2026: {(out.R_2026 > 0).sum()}/{len(out)}  | beat random: {(out.R_2026 > out.random_R_2026).sum()}/{len(out)}"
          f"  | sum R: {out.R_2026.sum():+.1f} vs random {out.random_R_2026.sum():+.1f}")
    out.to_csv(ROOT / "results" / "holdout_2026.csv", index=False)
