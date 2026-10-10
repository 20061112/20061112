"""第二十七部分：候選版本 OW（IB 整個在前一天價值區外、順交易方向）的完整研究，M5 / M15 新資料。

共用函數：
  prev_profiles(days, ARR, va_pct)   前一天 30 分字母 TPO 的 POC / VAH / VAL
  label_ow(S, PREV, mode)            mode = ib（IB 整個在價值區外）/ mid（IB 中點在外）/ lvl（進場價在外）
  run(path, bar_min, **kw)           產生訊號 → 出場 → 標記 OW → 回傳全部交易（未篩）
"""
import numpy as np
import pandas as pd
from oos_engine import load_bars, signals, simulate
from tpo import poc_va

PER = [("2023", "2023-03-01", "2023-12-31"), ("24H1", "2024-01-01", "2024-06-30"), ("24H2", "2024-07-01", "2024-12-31"),
       ("25H1", "2025-01-01", "2025-06-30"), ("25H2", "2025-07-01", "2025-12-31"), ("2026", "2026-01-01", "2026-12-31")]
_CACHE = {}


def profile(H, L, mins, va_pct=0.70):
    per = mins // 30
    lo0 = np.floor(L.min()); n = int(np.floor(H.max()) - lo0) + 1
    diff = np.zeros(n + 1)
    for k in np.unique(per):
        m = per == k
        diff[int(np.floor(L[m].min()) - lo0)] += 1; diff[int(np.floor(H[m].max()) - lo0) + 1] -= 1
    cnt = np.cumsum(diff)[:n].astype(int)
    poc, vah, val = poc_va(lo0 + np.arange(n), cnt, va_pct)
    return poc + 0.5, vah + 1, val


def prev_profiles(days, ARR, va_pct=0.70):
    out = {}
    for di in range(1, len(days)):
        t, O, H, L, C, S, mins = ARR[days[di - 1]]
        out[days[di]] = profile(H, L, mins, va_pct)
    return out


def label_ow(S, PREV, mode="ib"):
    lab = []
    for r in S.itertuples():
        if r.day not in PREV:
            lab.append(False); continue
        poc, vah, val = PREV[r.day]
        if mode == "ib":
            ok = (r.ibl > vah) if r.side == 1 else (r.ibh < val)
        elif mode == "mid":
            m = (r.ibh + r.ibl) / 2; ok = (m > vah) if r.side == 1 else (m < val)
        else:
            ok = (r.lvl > vah) if r.side == 1 else (r.lvl < val)
        lab.append(bool(ok))
    return np.array(lab)


def data(path, bar_min):
    if path not in _CACHE:
        _CACHE[path] = load_bars(path, bar_min)
    return _CACHE[path]


def run(path, bar_min, mode="ib", va_pct=0.70, monday=True, stop_mode="opp", **kw):
    days, D, ARR = data(path, bar_min)
    S = signals(days, D, ARR, bar_min, monday=monday, **kw)
    X = simulate(S, ARR, bar_min, stop_mode=stop_mode)
    PREV = prev_profiles(days, ARR, va_pct)
    X["ow"] = label_ow(X, PREV, mode)
    X = X.sort_values(["t_in", "h"]).drop_duplicates(["t_in", "side"]).reset_index(drop=True)
    return X, days


def pf(p):
    return p[p > 0].sum() / max(-p[p <= 0].sum(), 1e-9)


def per_cells(T, days, per=PER, col="R"):
    cells = []
    for n, a, b in per:
        x = T[(T.day >= a) & (T.day <= b)]
        if not len(x):
            cells.append(f"{n}   -   "); continue
        td = sum(1 for d in days if pd.Timestamp(a) <= d <= pd.Timestamp(b))
        cells.append(f"{n} {x.R.mean():+.2f}R/PF{pf(x.R):.2f}/勝{(x.pnl > 0).mean() * 100:.0f}%/{len(x) / max(td, 1):.1f}")
    return "  ".join(cells)
