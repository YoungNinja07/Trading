"""Gold 4h Donchian breakout signal (55 and 100), long only. SL = 2*ATR(14), TP = 4*ATR(14).

Usage:  python3 signal_now.py --balance 10000 --risk 0.5
Data: Yahoo GC=F hourly (gold futures). Breakout logic uses distances (ATR), so apply the
SL/TP distances to YOUR fill price on your broker (spot XAUUSD differs by a few dollars).
Signals are valid only right after a 4h candle closes (00,04,08,12,16,20 UTC).
"""
import argparse
import json
import urllib.request

import numpy as np
import pandas as pd

URL = "https://query2.finance.yahoo.com/v8/finance/chart/GC=F?interval=1h&range=60d"


def fetch():
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    r = json.load(urllib.request.urlopen(req, timeout=30))["chart"]["result"][0]
    q = r["indicators"]["quote"][0]
    df = pd.DataFrame({k: q[k] for k in ("open", "high", "low", "close")},
                      index=pd.to_datetime(r["timestamp"], unit="s")).dropna()
    h4 = df.resample("4h", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    return h4[h4.index + pd.Timedelta(hours=4) <= now]  # closed candles only


def atr(df, n=14):
    pc = df.close.shift(1).fillna(df.close)
    tr = np.maximum(df.high - df.low, np.maximum((df.high - pc).abs(), (df.low - pc).abs()))
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--balance", type=float, default=10000)
    ap.add_argument("--risk", type=float, default=0.5, help="%% of balance per signal")
    a = ap.parse_args()
    df = fetch()
    df["atr"] = atr(df)
    last = df.iloc[-1]
    print(f"Last closed 4h candle: {df.index[-1]} UTC  close {last.close:.2f}  ATR {last.atr:.2f}")
    any_sig = False
    for n in (55, 100):
        hh = df.high.rolling(n).max().shift(1)
        brk = df.close > hh
        fresh = bool(brk.iloc[-1] and not brk.iloc[-2])
        dist = hh.iloc[-1] - last.close
        if fresh:
            any_sig = True
            sl_d, tp_d = 2 * last.atr, 4 * last.atr
            risk_usd = a.balance * a.risk / 100
            lots = risk_usd / (sl_d * 100)  # 1 lot = 100 oz
            print(f"\n>>> BUY signal Donchian {n}: buy at market at the next candle open")
            print(f"    SL = fill price - {sl_d:.2f} USD   TP = fill price + {tp_d:.2f} USD")
            print(f"    risk {risk_usd:.0f} USD -> position {lots:.2f} lots ({lots * 100:.0f} oz)")
        else:
            print(f"Donchian {n}: no signal. Breakout level {hh.iloc[-1]:.2f} ({dist:+.2f} USD from close)")
    if not any_sig:
        print("\nNo trade now. Check again after the next 4h close.")


if __name__ == "__main__":
    main()
