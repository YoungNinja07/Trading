"""Load XAUUSD M1: HistData CSVs (EST no DST = UTC-5) + OANDA spot extension; resample to higher TFs."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
CACHE = ROOT / "data" / "cache"

TIMEFRAMES = {"1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min",
              "1h": "1h", "4h": "4h", "1d": "24h"}


def load_m1() -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    pq = CACHE / "XAUUSD_1m.parquet"
    if pq.exists():
        return pd.read_parquet(pq)
    frames = []
    for f in sorted(RAW.glob("DAT_MT_XAUUSD_M1_*.csv")):
        df = pd.read_csv(f, header=None,
                         names=["date", "time", "open", "high", "low", "close", "vol"])
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    ts = pd.to_datetime(df["date"] + " " + df["time"], format="%Y.%m.%d %H:%M")
    df.index = ts + pd.Timedelta(hours=5)  # EST (fixed) -> UTC
    df.index.name = "time"
    df = df[["open", "high", "low", "close"]].astype("float64")
    df = df[~df.index.duplicated()].sort_index()
    # extend with OANDA spot M1 (HF dataset bunqi/OANDA_XAUUSD_Gold, UTC) after HistData ends
    oanda = sorted((RAW / "oanda").glob("*.parquet"))
    if oanda:
        o = pd.concat([pd.read_parquet(f) for f in oanda])
        o.index = pd.to_datetime(o["time"], unit="s")
        o.index.name = "time"
        o = o[["open", "high", "low", "close"]].astype("float64")
        df = pd.concat([df, o[o.index > df.index[-1]]])
        df = df[~df.index.duplicated()].sort_index()
    df.to_parquet(pq)
    return df


def load(tf: str) -> pd.DataFrame:
    pq = CACHE / f"XAUUSD_{tf}.parquet"
    if pq.exists():
        return pd.read_parquet(pq)
    m1 = load_m1()
    if tf == "1m":
        return m1
    # daily bars follow the FX day (rollover 22:00 UTC) so Sunday is not a separate bar
    kw = {"offset": "22h"} if tf == "1d" else {}
    df = m1.resample(TIMEFRAMES[tf], label="left", closed="left", **kw).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    df.to_parquet(pq)
    return df


if __name__ == "__main__":
    for tf in TIMEFRAMES:
        d = load(tf)
        print(tf, len(d), d.index[0], d.index[-1])
