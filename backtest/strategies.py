"""Strategy signal generators. Each returns int8 array: +1 long entry, -1 short entry, 0 nothing."""
import numpy as np
import pandas as pd


def ema(x, n):
    return pd.Series(x).ewm(span=n, adjust=False).mean().values


def atr(df, n=14):
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    pc = np.roll(c, 1)
    pc[0] = c[0]
    tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))
    return pd.Series(tr).ewm(alpha=1 / n, adjust=False).mean().values


def rsi(c, n):
    d = np.diff(c, prepend=c[0])
    up = pd.Series(np.clip(d, 0, None)).ewm(alpha=1 / n, adjust=False).mean().values
    dn = pd.Series(np.clip(-d, 0, None)).ewm(alpha=1 / n, adjust=False).mean().values
    return 100 - 100 / (1 + up / np.where(dn == 0, 1e-12, dn))


def cross_up(a, b):
    out = np.zeros(len(a), dtype=bool)
    out[1:] = (a[1:] > b[1:]) & (a[:-1] <= b[:-1])
    return out


def to_sig(longs, shorts):
    s = np.zeros(len(longs), dtype=np.int8)
    s[longs] = 1
    s[shorts] = -1
    return s


def ema_cross(df, fast, slow):
    c = df["close"].values
    f, s = ema(c, fast), ema(c, slow)
    return to_sig(cross_up(f, s), cross_up(s, f))


def donchian(df, n):
    c = df["close"].values
    hh = df["high"].rolling(n).max().shift(1).values
    ll = df["low"].rolling(n).min().shift(1).values
    up = c > hh
    dn = c < ll
    first_up = up & ~np.roll(up, 1)
    first_dn = dn & ~np.roll(dn, 1)
    return to_sig(first_up, first_dn)


def rsi_mr(df, n, lo, trend):
    """Buy when RSI crosses back up through lo (sell mirrored). trend=1 -> only with EMA200 direction."""
    c = df["close"].values
    r = rsi(c, n)
    longs = cross_up(r, np.full(len(r), lo))
    shorts = cross_up(np.full(len(r), 100 - lo), r)
    if trend:
        e = ema(c, 200)
        longs &= c > e
        shorts &= c < e
    return to_sig(longs, shorts)


def bollinger_mr(df, n, k, trend):
    c = df["close"].values
    m = df["close"].rolling(n).mean().values
    sd = df["close"].rolling(n).std().values
    lower, upper = m - k * sd, m + k * sd
    longs = cross_up(c, lower)
    shorts = cross_up(upper, c)
    if trend:
        e = ema(c, 200)
        longs &= c > e
        shorts &= c < e
    return to_sig(longs, shorts)


def macd(df, fast, slow, signal, trend):
    c = df["close"].values
    m = ema(c, fast) - ema(c, slow)
    s = ema(m, signal)
    longs, shorts = cross_up(m, s), cross_up(s, m)
    if trend:
        e = ema(c, 200)
        longs &= c > e
        shorts &= c < e
    return to_sig(longs, shorts)


def orb(df, session_hour, range_minutes):
    """Opening-range breakout: range = first `range_minutes` after session_hour (UTC);
    first close above/below range that day -> entry. Intraday timeframes only."""
    idx = df.index
    day = idx.normalize()
    mins = (idx.hour - session_hour) * 60 + idx.minute
    in_range = (mins >= 0) & (mins < range_minutes)
    after = mins >= range_minutes
    g = pd.DataFrame({"h": np.where(in_range, df["high"], np.nan),
                      "l": np.where(in_range, df["low"], np.nan)}, index=idx)
    rh = g["h"].groupby(day).transform("max").values
    rl = g["l"].groupby(day).transform("min").values
    c = df["close"].values
    window = after & (mins < range_minutes + 6 * 60)
    up = window & (c > rh)
    dn = window & (c < rl)
    sig = to_sig(up, dn)
    # only first breakout per day
    s = pd.Series(sig, index=idx)
    nz = s != 0
    first = nz & (nz.groupby(day).cumsum() == 1)
    return np.where(first.values, sig, 0).astype(np.int8)


def grid(tf):
    """Parameter grid per strategy family."""
    g = []
    for f, s in [(5, 20), (9, 21), (10, 50), (20, 50), (20, 100), (50, 200)]:
        g.append(("EMA cross", dict(fast=f, slow=s), ema_cross))
    for n in [10, 20, 40, 55, 100]:
        g.append(("Donchian breakout", dict(n=n), donchian))
    for n, lo in [(2, 10), (2, 5), (7, 25), (14, 30), (14, 20)]:
        for t in (0, 1):
            g.append(("RSI mean reversion", dict(n=n, lo=lo, trend=t), rsi_mr))
    for n, k in [(20, 2.0), (20, 2.5), (50, 2.0)]:
        for t in (0, 1):
            g.append(("Bollinger mean reversion", dict(n=n, k=k, trend=t), bollinger_mr))
    for t in (0, 1):
        g.append(("MACD", dict(fast=12, slow=26, signal=9, trend=t), macd))
    if tf in ("1m", "5m", "15m", "30m"):
        for sh in (0, 7, 8, 13, 14):  # Asia, London (7/8 UTC), New York (13/14 UTC)
            for rm in (15, 30, 60):
                if rm >= {"1m": 1, "5m": 5, "15m": 15, "30m": 30}[tf]:
                    g.append(("Opening range breakout", dict(session_hour=sh, range_minutes=rm), orb))
    return g
