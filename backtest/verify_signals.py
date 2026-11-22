"""Check published gold trade signals against what price actually did (5-minute GC=F futures).

Signals are quoted in spot XAUUSD; futures trade at a premium (basis), estimated from the
spot prices the authors quoted at publication (~15-25 USD, median ~18). Results are shown
for several basis values so the conclusion does not hinge on one number.

Conservative rules: if SL and TP are both touched inside one 5-minute bar -> counted as SL.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
G = pd.read_parquet(ROOT / "data" / "raw" / "GC_5m.parquet")

# RoboForex weekly trade ideas (all are stop orders). Times converted to UTC.
ROBOFOREX = [
    # published UTC,    dir,  entry, sl,   tp,   valid until UTC
    ("2026-09-21 04:07", "buy", 4370, 4285, 4554, "2026-09-25 05:00"),
    ("2026-09-28 02:05", "sell", 4230, 4270, 4168, "2026-10-02 21:00"),
    ("2026-10-05 04:07", "sell", 4105, 4150, 4027, "2026-10-09 05:00"),
    ("2026-10-09 12:07", "buy", 4255, 4185, 4399, "2026-10-16 05:00"),
    ("2026-10-16 13:56", "buy", 4220, 4100, 4750, "2026-10-23 05:00"),
    ("2026-10-23 13:40", "buy", 4110, 4055, 4355, "2026-11-02 06:00"),
    ("2026-10-30 09:21", "sell", 3985, 4040, 3890, "2026-11-09 06:00"),
    ("2026-11-09 04:55", "sell", 3980, 4020, 3800, "2026-11-23 06:00"),
    ("2026-11-13 10:03", "sell", 3790, 3820, 3600, "2026-11-23 06:00"),
]

# FX.co ideas: no stop loss given. dir, entry type (limit = wait for pullback, market = now), entry, first target,
# entry window end, target window end ("relevance until").
FXCO = [
    ("2026-10-09 04:52", "buy", "limit", 4140, 4218, "2026-10-12 23:59", "2026-10-23 23:59"),
    ("2026-11-02 15:01", "buy", "market", 3947, 3974, "2026-11-05 23:59", "2026-11-16 23:59"),
    ("2026-11-04 16:23", "sell", "limit", 4062, 4000, "2026-11-06 23:59", "2026-11-18 23:59"),
    ("2026-11-06 12:11", "buy", "stop", 4062, 4120, "2026-11-09 23:59", "2026-11-20 23:59"),
    ("2026-11-13 13:09", "buy", "limit", 3880, 3990, "2026-11-16 23:59", "2026-11-27 23:59"),
    ("2026-11-19 03:53", "buy", "limit", 3750, None, "2026-11-21 23:59", "2026-12-03 23:59"),
]


def roboforex(basis):
    out = []
    for pub, d, e, sl, tp, until in ROBOFOREX:
        e, sl, tp = e + basis, sl + basis, tp + basis
        w = G[(G.index > pd.Timestamp(pub)) & (G.index <= pd.Timestamp(until))]
        hit = w[w.high >= e] if d == "buy" else w[w.low <= e]
        if hit.empty:
            out.append(dict(published=pub[:10], dir=d, result="not triggered", R=0.0))
            continue
        t0 = hit.index[0]
        after = G[G.index >= t0]
        res, t1 = "open", None
        for t, b in after.iterrows():
            if d == "buy":
                s_hit, t_hit = b.low <= sl, b.high >= tp
            else:
                s_hit, t_hit = b.high >= sl, b.low <= tp
            if s_hit:
                res, t1 = "SL", t
                break
            if t_hit:
                res, t1 = "TP", t
                break
        risk, reward = abs(e - sl), abs(tp - e)
        r = reward / risk if res == "TP" else (-1.0 if res == "SL" else
                                                ((G.close.iloc[-1] - e) * (1 if d == "buy" else -1)) / risk)
        out.append(dict(published=pub[:10], dir=d, triggered=str(t0)[:16], result=res,
                        closed=str(t1)[:16] if t1 is not None else "", RR=round(reward / risk, 1), R=round(r, 2)))
    return pd.DataFrame(out)


def fxco(basis):
    out = []
    for pub, d, kind, e, t1, ewin, twin in FXCO:
        e2 = e + basis
        w = G[(G.index > pd.Timestamp(pub)) & (G.index <= pd.Timestamp(ewin))]
        if kind == "market":
            hit_t = w.index[0]
        elif (kind == "limit") == (d == "buy"):
            h = w[w.low <= e2]
            hit_t = h.index[0] if not h.empty else None
        else:
            h = w[w.high >= e2]
            hit_t = h.index[0] if not h.empty else None
        if hit_t is None:
            out.append(dict(published=pub[:10], dir=d, entry=e, target=t1, result="entry never reached"))
            continue
        after = G[(G.index >= hit_t) & (G.index <= pd.Timestamp(twin))]
        sgn = 1 if d == "buy" else -1
        res, mae = "target not reached", 0.0
        for t, b in after.iterrows():
            adverse = (e2 - b.low) if d == "buy" else (b.high - e2)
            mae = max(mae, adverse)
            if t1 is not None and ((b.high >= t1 + basis) if d == "buy" else (b.low <= t1 + basis)):
                res = f"TARGET hit {str(t)[:16]}"
                break
        best = (after.high.max() - e2) if d == "buy" else (e2 - after.low.min())
        out.append(dict(published=pub[:10], dir=d, entry=e, target=t1, result=res,
                        target_dist=abs(t1 - e) if t1 else None, worst_against=round(mae, 0),
                        best_for=round(best, 0), last_vs_entry=round(sgn * (after.close.iloc[-1] - e2), 0)))
    return pd.DataFrame(out)


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    for b in (18, 10, 25):
        rf = roboforex(b)
        trig = rf[rf.result != "not triggered"]
        print(f"\n=== RoboForex, basis {b} USD ===")
        print(rf.to_string(index=False))
        print(f"triggered {len(trig)}/{len(rf)}, TP {int((trig.result == 'TP').sum())}, SL {int((trig.result == 'SL').sum())},"
              f" total {trig.R.sum():+.2f}R")
    print("\n=== FX.co, basis 18 USD (no stop loss published) ===")
    print(fxco(18).to_string(index=False))
