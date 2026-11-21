"""Bar-by-bar backtest engine (numba).

Rules (conservative):
- signal computed on bar close i -> entry at open of bar i+1
- stop loss = sl_mult * ATR(14) at signal time, take profit = tp_mult * ATR (0 = no TP)
- if SL and TP are both touched in one bar -> assume SL (worst case)
- opposite signal closes the trade at next open (and opens the reverse)
- max_hold bars (0 = unlimited) closes at bar close
- round-trip cost (spread + commission + slippage) in USD per ounce
- result of every trade expressed in R (multiples of initial risk)
"""
import numpy as np
from numba import njit


@njit(cache=True)
def run(o, h, l, c, atr, sig, sl_mult, tp_mult, max_hold, cost, long_only):
    n = len(c)
    out_r = np.empty(n)
    out_i = np.empty(n, dtype=np.int64)
    k = 0
    pos = 0
    entry = 0.0
    sl = 0.0
    tp = 0.0
    risk = 0.0
    ebar = 0
    for i in range(1, n):
        s = sig[i - 1]
        if long_only and s < 0:
            s = 0
        # 1) signal from previous close -> act at this open
        if s != 0 and atr[i - 1] > 0:
            if pos != 0 and s == -pos:
                r = ((o[i] - entry) * pos - cost) / risk
                out_r[k] = r
                out_i[k] = ebar
                k += 1
                pos = 0
            if pos == 0:
                pos = s
                entry = o[i]
                risk = sl_mult * atr[i - 1]
                sl = entry - pos * risk
                tp = entry + pos * tp_mult * atr[i - 1] if tp_mult > 0 else 0.0
                ebar = i
        # 2) manage open position inside bar i
        if pos != 0:
            exitp = 0.0
            done = False
            if pos == 1:
                if l[i] <= sl:
                    exitp = min(sl, o[i]) if i != ebar else sl
                    done = True
                elif tp_mult > 0 and h[i] >= tp:
                    exitp = max(tp, o[i]) if i != ebar else tp
                    done = True
            else:
                if h[i] >= sl:
                    exitp = max(sl, o[i]) if i != ebar else sl
                    done = True
                elif tp_mult > 0 and l[i] <= tp:
                    exitp = min(tp, o[i]) if i != ebar else tp
                    done = True
            if not done and max_hold > 0 and i - ebar + 1 >= max_hold:
                exitp = c[i]
                done = True
            if done:
                out_r[k] = ((exitp - entry) * pos - cost) / risk
                out_i[k] = ebar
                k += 1
                pos = 0
    return out_r[:k], out_i[:k]
