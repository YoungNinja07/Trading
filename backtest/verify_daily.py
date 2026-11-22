"""Verify RoboForex DAILY gold trade scenarios (Jan-Nov 2026) against GC=F futures prices.

Rules: stop order active from publication until 05:00 UTC next day (8:00 GMT+3, as stated on the pages).
After the fill: held until SL or TP, max 10 days (then closed at market). 5-minute bars where
available (from 14 Sep 2026), otherwise 1-hour bars. SL and TP inside one bar -> SL (conservative).
Spot->futures basis added to all levels; tested at several values. Cost 0.35 USD per trade.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
M5 = pd.read_parquet(ROOT / "data" / "raw" / "GC_5m.parquet")
H1 = pd.read_parquet(ROOT / "data" / "raw" / "GC_1h.parquet")
PX = pd.concat([H1[H1.index < M5.index[0]], M5]).sort_index()
SIG = pd.read_csv(ROOT / "results" / "roboforex_daily_signals.csv")
SIG = SIG[SIG.valid_struct].copy()
SIG["pub_utc"] = pd.to_datetime(SIG.published, utc=True).dt.tz_localize(None)
COST = 0.35


def evaluate(basis, flip=False):
    out = []
    for _, s in SIG.iterrows():
        d = s.dir if not flip else ("sell" if s.dir == "buy" else "buy")
        e, sl, tp = s.entry + basis, s.sl + basis, s.tp + basis
        if flip:
            sl, tp = tp, sl  # opposite trade: same levels, swapped roles (for comparison only)
        start = s.pub_utc
        until = (start + pd.Timedelta(days=1)).normalize() + pd.Timedelta(hours=5)
        w = PX[(PX.index > start) & (PX.index <= until)]
        if w.empty:
            continue
        if flip:
            fill_t, fill = w.index[0], w.open.iloc[0]
        else:
            trig = w[w.high >= e] if d == "buy" else w[w.low <= e]
            if trig.empty:
                out.append(dict(day=s.day, role=s.role, dir=d, res="not triggered", R=np.nan))
                continue
            fill_t = trig.index[0]
            o = trig.open.iloc[0]
            fill = max(e, o) if d == "buy" else min(e, o)  # gap through the level -> worse fill
        risk = abs(fill - sl)
        if risk <= 0 or (d == "buy" and (fill >= tp or fill <= sl)) or (d == "sell" and (fill <= tp or fill >= sl)):
            out.append(dict(day=s.day, role=s.role, dir=d, res="skipped", R=np.nan))
            continue
        after = PX[(PX.index >= fill_t) & (PX.index <= fill_t + pd.Timedelta(days=10))]
        res, px = "time", after.close.iloc[-1]
        for _, b in after.iterrows():
            if d == "buy":
                if b.low <= sl:
                    res, px = "SL", sl
                    break
                if b.high >= tp:
                    res, px = "TP", tp
                    break
            else:
                if b.high >= sl:
                    res, px = "SL", sl
                    break
                if b.low <= tp:
                    res, px = "TP", tp
                    break
        r = ((px - fill) * (1 if d == "buy" else -1) - COST) / risk
        out.append(dict(day=s.day, role=s.role, dir=d, res=res, R=r, rr=abs(tp - fill) / risk))
    return pd.DataFrame(out)


def summary(df, label):
    t = df.dropna(subset=["R"])
    if t.empty:
        return
    eq = t.R.cumsum()
    dd = (eq.cummax().clip(lower=0) - eq).max()
    tstat = t.R.mean() / t.R.std(ddof=1) * np.sqrt(len(t))
    print(f"{label:34s} signals {len(df):3d} | trades {len(t):3d} | win {np.mean(t.R > 0):.0%} | "
          f"avg RR {t.rr.mean():.1f} | avg {t.R.mean():+.2f}R | total {t.R.sum():+6.1f}R | maxDD {dd:4.1f}R | t={tstat:+.2f}")


if __name__ == "__main__":
    for b in (18, 5, 30):
        r = evaluate(b)
        print(f"\n=== basis {b} USD ===")
        summary(r[r.role == "main"], "MAIN scenario")
        summary(r[r.role == "alt"], "ALTERNATIVE scenario")
        summary(r, "both")
    r = evaluate(18)
    m = r[r.role == "main"].dropna(subset=["R"])
    print("\nMAIN by month (R):", m.groupby(m.day.str[:7]).R.sum().round(1).to_dict())
    print("MAIN by direction:", m.groupby("dir").R.agg(["count", "sum", "mean"]).round(2).to_dict("index"))
    print("MAIN results:", m.res.value_counts().to_dict())
    r.to_csv(ROOT / "results" / "roboforex_daily_check.csv", index=False)
