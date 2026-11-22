"""'Shooting' test: 1 random-direction trade per day at US session open, 4h ATR stops,
then Monte Carlo of a 2-phase prop challenge (phase 1 +10%, phase 2 +5%, max loss -10%, daily -5%)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit
sys.path.insert(0, str(Path(__file__).parent))
from data import load
from strategies import atr

COST = 0.35
ENTRY_UTC, EXIT_UTC = (14, 30), (20, 30)   # NY 9:30 / 15:30 EST


@njit(cache=True)
def day_trade(h, l, c, i0, i1, entry, d, sl_d, tp_d):
    sl = entry - d * sl_d
    tp = entry + d * tp_d
    for i in range(i0, i1):
        if d == 1:
            if l[i] <= sl:
                return -sl_d
            if h[i] >= tp:
                return tp_d
        else:
            if h[i] >= sl:
                return -sl_d
            if l[i] <= tp:
                return tp_d
    return (c[i1 - 1] - entry) * d


def daily_R(rr, direction="random", seed=0):
    m1 = load("1m")
    h4 = load("4h")
    a4 = pd.Series(atr(h4), index=h4.index).shift(1)   # ATR known before the bar
    h, l, c, o = (m1[k].values for k in ("high", "low", "close", "open"))
    t = m1.index
    mins = t.hour * 60 + t.minute
    rng = np.random.default_rng(seed)
    out = []
    days = t.normalize()
    start = np.flatnonzero(mins == ENTRY_UTC[0] * 60 + ENTRY_UTC[1])
    for i0 in start:
        if t[i0].dayofweek > 4:
            continue
        day = days[i0]
        end_t = day + pd.Timedelta(hours=EXIT_UTC[0], minutes=EXIT_UTC[1])
        i1 = np.searchsorted(t, end_t)
        if i1 <= i0 + 1:
            continue
        atr4 = a4.asof(t[i0])
        if not atr4 > 0:
            continue
        d = rng.choice([-1, 1]) if direction == "random" else (1 if direction == "long" else -1)
        sl_d, tp_d = 1.0 * atr4, rr * atr4
        pnl = day_trade(h, l, c, i0, i1, o[i0], d, sl_d, tp_d) - COST
        out.append((day, pnl / sl_d))
    return pd.Series([x[1] for x in out], index=[x[0] for x in out])


def challenge(R, risk, n_sims=20000, seed=1):
    rng = np.random.default_rng(seed)
    r = R.values
    passed = np.zeros(n_sims, bool)
    days = np.full(n_sims, np.nan)
    p1 = np.zeros(n_sims, bool)
    for k in range(n_sims):
        seq = rng.choice(r, 1500) * risk
        eq, phase, target, day = 0.0, 1, 10.0, 0
        for x in seq:
            day += 1
            if x <= -5.0:          # daily loss limit (one trade per day)
                break
            eq += x
            if eq <= -10.0:
                break
            if eq >= target:
                if phase == 1:
                    p1[k] = True
                    phase, eq, target = 2, 0.0, 5.0
                else:
                    passed[k] = True
                    days[k] = day
                    break
    return p1.mean(), passed.mean(), np.nanmedian(days)


if __name__ == "__main__":
    for rr in (1.0, 2.0, 3.0):
        R = daily_R(rr)
        print(f"\nRR 1:{rr:g}  random direction: {len(R)} days, win {np.mean(R > 0):.2f}, avgR {R.mean():+.3f}")
        for risk in (0.5, 1, 2, 3, 4.9):
            a, b, d = challenge(R, risk)
            print(f"  risk {risk}%: phase1 {a*100:4.1f}%  BOTH phases {b*100:4.1f}%  median {d/21:4.1f} months")
