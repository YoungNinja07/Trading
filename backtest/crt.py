"""CRT (Candle Range Theory) + TBS (Turtle Body Soup) backtest on XAUUSD.

Candle 1 = range (high H1, low L1).
Candle 2 sweeps H1 with its wick but its BODY closes back inside the range (TBS) -> short,
or sweeps L1 and closes back inside -> long.
Entry: open of candle 3. SL: beyond candle 2 wick (+ buffer). TP: fraction of the way to the
opposite side of candle 1 (1.0 = full CRT target) or a fixed R multiple.
"""
import itertools
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from data import load
from engine import run_levels
from strategies import atr, ema

COST = 0.35
SPLIT = pd.Timestamp("2022-01-01")


def crt_signals(df, target, buf_atr, min_range_atr, trend, hours=None):
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    a = atr(df)
    H1, L1 = np.roll(h, 1), np.roll(l, 1)
    rng = H1 - L1
    body_hi, body_lo = np.maximum(o, c), np.minimum(o, c)
    ok = rng >= min_range_atr * np.roll(a, 1)
    ok[0] = False
    # bearish CRT: sweep above H1, body back inside
    short = ok & (h > H1) & (body_hi < H1) & (body_lo > L1)
    long_ = ok & (l < L1) & (body_lo > L1) & (body_hi < H1)
    if trend:
        e = ema(c, 200)
        long_ &= c > e
        short &= c < e
    if hours is not None:
        hm = np.isin(np.roll(df.index.hour.values, 1), hours)  # candle 1 open hour
        long_ &= hm
        short &= hm
    sig = np.zeros(len(c), dtype=np.int8)
    sig[long_] = 1
    sig[short] = -1
    slp = np.where(long_, l - buf_atr * a, h + buf_atr * a)
    if target > 0:   # fraction of the way to opposite side of candle 1, measured from candle-2 close
        tpp = np.where(long_, c + target * (H1 - c), c - target * (c - L1))
    else:            # fixed R multiple
        R = -target
        tpp = np.where(long_, c + R * (c - slp), c - R * (slp - c))
    return sig, slp, tpp


def stats(r):
    if len(r) < 2:
        return dict(n=len(r), win=np.nan, avgR=np.nan, t=np.nan, sumR=r.sum())
    return dict(n=len(r), win=(r > 0).mean(), avgR=r.mean(),
                t=r.mean() / r.std(ddof=1) * np.sqrt(len(r)), sumR=r.sum())


def main():
    rows = []
    for tf in ["15m", "30m", "1h", "4h", "1d"]:
        df = load(tf)
        o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
        hold = {"15m": 32, "30m": 16, "1h": 12, "4h": 6, "1d": 5}[tf]
        hour_sets = [None]
        if tf == "4h":   # popular CRT 4h candles: 1am/5am/9am New York = 06/10/14 UTC (winter) ~ our 4h grid
            hour_sets += [[4, 8, 12]]
        if tf == "1h":
            hour_sets += [[7, 8], [12, 13, 14]]
        for target, buf, mr, trend, hrs, dirn in itertools.product(
                [0.5, 1.0, -1.0, -2.0], [0.0, 0.1], [0.0, 0.5, 1.0], [0, 1], hour_sets, ["both", "long"]):
            sig, slp, tpp = crt_signals(df, target, buf, mr, trend, hrs)
            if dirn == "long":
                sig = np.where(sig > 0, sig, 0).astype(np.int8)
            r, ei = run_levels(o, h, l, c, sig, slp, tpp, COST, hold)
            isn = df.index[ei] < SPLIT
            yr = pd.Series(r).groupby(df.index.year[ei]).sum()
            row = dict(tf=tf, target=target, buf=buf, min_range=mr, trend=trend,
                       hours=str(hrs), direction=dirn, yrs_pos=int((yr > 0).sum()))
            row.update({f"is_{k}": v for k, v in stats(r[isn]).items()})
            row.update({f"oos_{k}": v for k, v in stats(r[~isn]).items()})
            row.update({f"all_{k}": v for k, v in stats(r).items()})
            rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(Path(__file__).resolve().parent.parent / "results" / "crt_runs.csv", index=False)
    return res


if __name__ == "__main__":
    res = main()
    pd.set_option("display.width", 250)
    res["tpm"] = res.all_n / 120
    print("runs:", len(res), " profitable after costs:", round((res.all_avgR > 0).mean(), 3))
    print(res.groupby("tf").apply(lambda x: pd.Series({
        "share_profitable": round((x.all_avgR > 0).mean(), 2),
        "median_avgR": round(x.all_avgR.median(), 3)})).to_string())
    cols = ["tf", "target", "buf", "min_range", "trend", "hours", "direction", "tpm", "all_win",
            "all_avgR", "is_avgR", "is_t", "oos_avgR", "oos_t", "yrs_pos"]
    print("\nbest by in-sample t (then look at OOS):")
    print(res[res.is_n >= 60].sort_values("is_t", ascending=False)[cols].head(15).to_string())
