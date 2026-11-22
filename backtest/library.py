"""Library of ~30 entry ideas x parameters x filters x timeframes = ~500 strategy definitions.

Every entry returns (longs, shorts) boolean arrays evaluated on bar close.
"""
import numpy as np
import pandas as pd
from numba import njit

from strategies import atr, cross_up, ema, rsi


# ---------- indicators ----------
def sma(x, n):
    return pd.Series(x).rolling(n).mean().values


def hhv(x, n):
    return pd.Series(x).rolling(n).max().shift(1).values


def llv(x, n):
    return pd.Series(x).rolling(n).min().shift(1).values


def adx(df, n=14):
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    up = np.diff(h, prepend=h[0])
    dn = -np.diff(l, prepend=l[0])
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    ndm = np.where((dn > up) & (dn > 0), dn, 0.0)
    a = atr(df, n)
    sm = lambda x: pd.Series(x).ewm(alpha=1 / n, adjust=False).mean().values  # noqa: E731
    pdi = 100 * sm(pdm) / np.where(a == 0, 1e-12, a)
    ndi = 100 * sm(ndm) / np.where(a == 0, 1e-12, a)
    dx = 100 * np.abs(pdi - ndi) / np.where(pdi + ndi == 0, 1e-12, pdi + ndi)
    return sm(dx), pdi, ndi


def stoch(df, n=14, d=3):
    h, l, c = df["high"], df["low"], df["close"]
    k = 100 * (c - l.rolling(n).min()) / (h.rolling(n).max() - l.rolling(n).min()).replace(0, np.nan)
    k = k.rolling(d).mean()
    return k.values, k.rolling(d).mean().values


@njit(cache=True)
def _supertrend(h, l, c, a, mult):
    n = len(c)
    d = np.ones(n, dtype=np.int8)
    ub = np.zeros(n)
    lb = np.zeros(n)
    for i in range(n):
        mid = (h[i] + l[i]) / 2
        bu, bl = mid + mult * a[i], mid - mult * a[i]
        if i == 0:
            ub[i], lb[i] = bu, bl
            continue
        ub[i] = bu if (bu < ub[i - 1] or c[i - 1] > ub[i - 1]) else ub[i - 1]
        lb[i] = bl if (bl > lb[i - 1] or c[i - 1] < lb[i - 1]) else lb[i - 1]
        if d[i - 1] == 1:
            d[i] = -1 if c[i] < lb[i] else 1
        else:
            d[i] = 1 if c[i] > ub[i] else -1
    return d


def flip(state):
    """state +1/-1 array -> (became +1, became -1)."""
    prev = np.roll(state, 1)
    prev[0] = state[0]
    return (state == 1) & (prev != 1), (state == -1) & (prev != -1)


def first(x):
    p = np.roll(x, 1)
    p[0] = False
    return x & ~p


# ---------- entry ideas ----------
def e_donchian(df, n):
    c = df.close.values
    return first(c > hhv(df.high.values, n)), first(c < llv(df.low.values, n))


def e_ema_cross(df, f, s):
    c = df.close.values
    a, b = ema(c, f), ema(c, s)
    return cross_up(a, b), cross_up(b, a)


def e_price_ema(df, n):
    c = df.close.values
    e = ema(c, n)
    return cross_up(c, e), cross_up(e, c)


def e_macd(df, zero):
    c = df.close.values
    m = ema(c, 12) - ema(c, 26)
    s = ema(m, 9)
    if zero:
        z = np.zeros(len(m))
        return cross_up(m, z), cross_up(z, m)
    return cross_up(m, s), cross_up(s, m)


def e_rsi_mom(df, n):
    r = rsi(df.close.values, n)
    f = np.full(len(r), 50.0)
    return cross_up(r, f), cross_up(f, r)


def e_rsi_mr(df, n, lo):
    r = rsi(df.close.values, n)
    return cross_up(r, np.full(len(r), lo)), cross_up(np.full(len(r), 100 - lo), r)


def e_bb_break(df, n, k):
    c = df.close
    m, sd = c.rolling(n).mean(), c.rolling(n).std()
    up, lo = (m + k * sd).values, (m - k * sd).values
    return cross_up(c.values, up), cross_up(lo, c.values)


def e_bb_mr(df, n, k):
    c = df.close
    m, sd = c.rolling(n).mean(), c.rolling(n).std()
    up, lo = (m + k * sd).values, (m - k * sd).values
    return cross_up(c.values, lo), cross_up(up, c.values)


def e_keltner(df, n, k):
    c = df.close.values
    m = ema(c, n)
    a = atr(df, n)
    return cross_up(c, m + k * a), cross_up(m - k * a, c)


def e_inside_bar(df):
    h, l, c = df.high.values, df.low.values, df.close.values
    ib = (h < np.roll(h, 1)) & (l > np.roll(l, 1))
    ibh, ibl = np.roll(np.where(ib, h, np.nan), 1), np.roll(np.where(ib, l, np.nan), 1)
    return c > ibh, c < ibl


def e_nr7(df):
    h, l, c = df.high.values, df.low.values, df.close.values
    r = h - l
    nr = r <= pd.Series(r).rolling(7).min().values
    nh, nl = np.roll(np.where(nr, h, np.nan), 1), np.roll(np.where(nr, l, np.nan), 1)
    return c > nh, c < nl


def e_engulf(df):
    o, c = df.open.values, df.close.values
    po, pc = np.roll(o, 1), np.roll(c, 1)
    bull = (pc < po) & (c > o) & (c > po) & (o < pc)
    bear = (pc > po) & (c < o) & (c < po) & (o > pc)
    return bull, bear


def e_pinbar(df):
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    rng = np.where(h - l == 0, 1e-12, h - l)
    body = np.abs(c - o)
    lw = np.minimum(o, c) - l
    uw = h - np.maximum(o, c)
    return (lw / rng > 0.6) & (body / rng < 0.3), (uw / rng > 0.6) & (body / rng < 0.3)


def e_streak(df, n):
    """n consecutive down closes -> buy (mean reversion); n up -> sell."""
    c = df.close.values
    d = np.sign(np.diff(c, prepend=c[0]))
    dn = pd.Series(d < 0).rolling(n).sum().values == n
    up = pd.Series(d > 0).rolling(n).sum().values == n
    return first(dn), first(up)


def e_roc(df, n):
    c = df.close.values
    roc = c - np.roll(c, n)
    z = np.zeros(len(c))
    return cross_up(roc, z), cross_up(z, roc)


def e_stoch(df, lo):
    k, d = stoch(df)
    return cross_up(k, d) & (k < lo), cross_up(d, k) & (k > 100 - lo)


def e_adx_di(df, th):
    a, p, m = adx(df)
    return cross_up(p, m) & (a > th), cross_up(m, p) & (a > th)


def e_supertrend(df, mult):
    d = _supertrend(df.high.values, df.low.values, df.close.values, atr(df, 10), mult)
    return flip(d)


def e_pullback(df, n):
    """Uptrend (EMA50>EMA200): buy when low touches EMA(n) and close stays above. Mirror for shorts."""
    c, l, h = df.close.values, df.low.values, df.high.values
    e50, e200, en = ema(c, 50), ema(c, 200), ema(c, n)
    return (e50 > e200) & (l <= en) & (c > en), (e50 < e200) & (h >= en) & (c < en)


def e_squeeze(df, n):
    c = df.close
    m, sd = c.rolling(20).mean(), c.rolling(20).std()
    w = (4 * sd / m)
    sq = (w <= w.rolling(n).min() * 1.1).values
    sq_recent = pd.Series(sq).rolling(5).max().values > 0
    up, lo = (m + 2 * sd).values, (m - 2 * sd).values
    return cross_up(c.values, up) & sq_recent, cross_up(lo, c.values) & sq_recent


def e_heikin(df, n):
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    hc = (o + h + l + c) / 4
    ho = np.empty(len(c))
    ho[0] = (o[0] + c[0]) / 2
    for i in range(1, len(c)):
        ho[i] = (ho[i - 1] + hc[i - 1]) / 2
    green = (hc > ho).astype(int)
    g = pd.Series(green).rolling(n).sum().values
    return first(g == n), first(g == 0)


def e_ichimoku(df):
    h, l, c = df.high, df.low, df.close
    tenkan = (h.rolling(9).max() + l.rolling(9).min()) / 2
    kijun = (h.rolling(26).max() + l.rolling(26).min()) / 2
    sa = ((tenkan + kijun) / 2).shift(26)
    sb = ((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(26)
    top, bot = np.maximum(sa, sb).values, np.minimum(sa, sb).values
    tk_up, tk_dn = cross_up(tenkan.values, kijun.values), cross_up(kijun.values, tenkan.values)
    cv = c.values
    return tk_up & (cv > top), tk_dn & (cv < bot)


def e_hh_momentum(df, n):
    """Close above close n bars ago AND new n-bar closing high."""
    c = df.close.values
    ch = pd.Series(c).rolling(n).max().shift(1).values
    cl = pd.Series(c).rolling(n).min().shift(1).values
    return first(c > ch), first(c < cl)


def e_prev_day_break(df):
    """Intraday: break of previous day's high / low (first per day)."""
    day = df.index.normalize()
    d = df.groupby(day).agg(h=("high", "max"), l=("low", "min"))
    ph = d.h.shift(1).reindex(day).values
    pl = d.l.shift(1).reindex(day).values
    c = df.close.values
    up, dn = c > ph, c < pl
    s = pd.Series(np.where(up, 1, np.where(dn, -1, 0)), index=df.index)
    nz = s != 0
    f = nz & (nz.groupby(day).cumsum() == 1)
    return (f & (s == 1)).values, (f & (s == -1)).values


def e_session(df, hour):
    """Time-of-day: buy at a fixed UTC hour every day (pure seasonality test)."""
    hit = (df.index.hour == hour) & (df.index.minute == 0)
    return hit, np.zeros(len(df), dtype=bool)


def e_dow(df, dow):
    """Daily: buy at close of a given weekday."""
    hit = df.index.dayofweek == dow
    return hit, np.zeros(len(df), dtype=bool)


# ---------- filters ----------
def f_none(df):
    return None


def f_trend(df):
    c = df.close.values
    e = ema(c, 200)
    return c > e, c < e


def f_adx(df):
    a, _, _ = adx(df)
    return a > 20, a > 20


def f_vol(df):
    a = atr(df)
    hi = a > pd.Series(a).rolling(200).median().values
    return hi, hi


ENTRIES = [
    ("Donchian", e_donchian, [dict(n=n) for n in (10, 20, 55, 100)]),
    ("EMA cross", e_ema_cross, [dict(f=f, s=s) for f, s in ((9, 21), (20, 50), (50, 200))]),
    ("Price x EMA", e_price_ema, [dict(n=n) for n in (50, 200)]),
    ("MACD", e_macd, [dict(zero=0), dict(zero=1)]),
    ("RSI 50 momentum", e_rsi_mom, [dict(n=14)]),
    ("RSI reversion", e_rsi_mr, [dict(n=2, lo=10), dict(n=14, lo=30)]),
    ("BB breakout", e_bb_break, [dict(n=20, k=2.0)]),
    ("BB reversion", e_bb_mr, [dict(n=20, k=2.0)]),
    ("Keltner breakout", e_keltner, [dict(n=20, k=2.0)]),
    ("Inside bar", e_inside_bar, [dict()]),
    ("NR7", e_nr7, [dict()]),
    ("Engulfing", e_engulf, [dict()]),
    ("Pin bar", e_pinbar, [dict()]),
    ("Streak reversal", e_streak, [dict(n=3)]),
    ("ROC momentum", e_roc, [dict(n=20)]),
    ("Stochastic", e_stoch, [dict(lo=20)]),
    ("ADX/DI", e_adx_di, [dict(th=25)]),
    ("Supertrend", e_supertrend, [dict(mult=3.0)]),
    ("Pullback EMA20", e_pullback, [dict(n=20)]),
    ("BB squeeze", e_squeeze, [dict(n=100)]),
    ("Heikin-Ashi", e_heikin, [dict(n=3)]),
    ("Ichimoku", e_ichimoku, [dict()]),
    ("Closing-high momentum", e_hh_momentum, [dict(n=20)]),
]
INTRADAY = [
    ("Prev-day break", e_prev_day_break, [dict()]),
    ("Time of day", e_session, [dict(hour=h) for h in (0, 7, 13)]),
]
DAILY = [("Day of week", e_dow, [dict(dow=d) for d in (0, 4)])]
FILTERS = [("none", f_none), ("EMA200 trend", f_trend), ("ADX>20", f_adx), ("high vol", f_vol)]
TFS = ["30m", "1h", "4h", "1d"]


def definitions():
    defs = []
    for tf in TFS:
        ent = ENTRIES + (INTRADAY if tf in ("30m", "1h") else []) + (DAILY if tf == "1d" else [])
        for name, fn, plist in ent:
            for p in plist:
                for fname, ff in FILTERS:
                    defs.append(dict(tf=tf, entry=name, params=p, filter=fname, fn=fn, ff=ff))
    return defs


def signal(df, d):
    longs, shorts = d["fn"](df, **d["params"])
    longs = np.nan_to_num(longs).astype(bool)
    shorts = np.nan_to_num(shorts).astype(bool)
    flt = d["ff"](df)
    if flt is not None:
        longs &= flt[0]
        shorts &= flt[1]
    s = np.zeros(len(df), dtype=np.int8)
    s[longs] = 1
    s[shorts & ~longs] = -1
    return s
