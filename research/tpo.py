"""第十部分：標準 TPO（市場剖面）與 Initial Balance。

時間：broker 時間（= 紐約 + 7），交易日 = broker 01:00 ~ 23:59。
TPO：每 30 分鐘一個字母（01:00 = A, 01:30 = B, ... 共 46 個，A~Z 之後接 a~t），
     該時段 K 棒的 high~low 碰到的每個價格格子各記一個 TPO。
價格格子：tick = 1 美元（黃金日波幅約 40~150 美元，每天約 40~150 列）。
POC：TPO 最多的格子；同票時取離區間中心最近的。
價值區（VA）：標準 CBOT 做法，從 POC 往上下各看兩格，取 TPO 較多的一邊加入，直到 ≥ 70%。
IB：交易日的前 60 分鐘（A+B 期）。另外提供倫敦 / 紐約 IB（任意起點 60 分鐘）。
"""
import numpy as np
import pandas as pd
from load import load

TICK = 1.0
PERIOD = 30
DAY_START_H = 1
LETTERS = [chr(c) for c in range(ord("A"), ord("Z") + 1)] + [chr(c) for c in range(ord("a"), ord("z") + 1)]


def load_m1(path="data/XAUUSD_M1_2026.csv"):
    d = load(path)
    d["day"] = (d.index - pd.Timedelta(hours=DAY_START_H)).normalize()
    mins = (d.index.hour - DAY_START_H) * 60 + d.index.minute
    d["period"] = mins // PERIOD
    return d


def tpo_profile(g, tick=TICK):
    """g：單日 M1。回傳 (prices, counts, letters_by_row)。prices 為格子下緣。"""
    lo = np.floor(g.low.min() / tick) * tick
    hi = np.floor(g.high.max() / tick) * tick
    prices = np.arange(lo, hi + tick / 2, tick)
    counts = np.zeros(len(prices), int)
    rows = [[] for _ in prices]
    for p, gp in g.groupby("period"):
        a = int(round((np.floor(gp.low.min() / tick) * tick - lo) / tick))
        b = int(round((np.floor(gp.high.max() / tick) * tick - lo) / tick))
        counts[a:b + 1] += 1
        for i in range(a, b + 1):
            rows[i].append(LETTERS[p] if p < len(LETTERS) else "?")
    return prices, counts, rows


def poc_va(prices, counts, va_pct=0.70):
    mx = counts.max()
    cand = np.where(counts == mx)[0]
    mid = (len(prices) - 1) / 2
    poc = cand[np.argmin(np.abs(cand - mid))]
    total, target = counts.sum(), va_pct * counts.sum()
    lo = hi = poc
    acc = counts[poc]
    while acc < target and (lo > 0 or hi < len(counts) - 1):
        up = counts[hi + 1:hi + 3].sum() if hi < len(counts) - 1 else -1
        dn = counts[max(lo - 2, 0):lo].sum() if lo > 0 else -1
        if up >= dn:
            step = min(2, len(counts) - 1 - hi)
            acc += counts[hi + 1:hi + 1 + step].sum(); hi += step
        else:
            step = min(2, lo)
            acc += counts[lo - step:lo].sum(); lo -= step
    return prices[poc], prices[hi], prices[lo]          # POC, VAH, VAL（格子下緣）


def profile_stats(prices, counts):
    """把剖面當成分佈：TPO 加權平均、標準差、偏態；以及每列 TPO 數的 z-score（HVN/LVN）。"""
    c = prices + TICK / 2
    w = counts / counts.sum()
    mu = (w * c).sum()
    sd = np.sqrt((w * (c - mu) ** 2).sum())
    skew = (w * ((c - mu) / sd) ** 3).sum() if sd > 0 else 0.0
    cz = (counts - counts.mean()) / (counts.std() + 1e-9)
    return mu, sd, skew, cz


def build_days(d, min_bars=1000):
    """每日剖面摘要。"""
    out, prof = {}, {}
    for day, g in d.groupby("day"):
        if len(g) < min_bars:
            continue
        prices, counts, rows = tpo_profile(g)
        poc, vah, val = poc_va(prices, counts)
        mu, sd, skew, cz = profile_stats(prices, counts)
        ib = g[g.period < 2]
        singles = int(((counts == 1)).sum())
        out[day] = dict(open=g.open.iloc[0], high=g.high.max(), low=g.low.min(), close=g.close.iloc[-1],
                        poc=poc + TICK / 2, vah=vah + TICK, val=val, mu=mu, sd=sd, skew=skew,
                        ibh=ib.high.max(), ibl=ib.low.min(), ntpo=int(counts.sum()), singles=singles,
                        nper=int(g.period.nunique()))
        prof[day] = (prices, counts, rows, cz)
    D = pd.DataFrame(out).T.astype(float)
    D.index.name = "day"
    D["range"] = D.high - D.low
    D["tr"] = np.maximum(D.range, np.maximum((D.high - D.close.shift()).abs(), (D.low - D.close.shift()).abs()))
    D["atr10"] = D.tr.rolling(10).mean().shift(1)
    return D, prof


def print_profile(prices, counts, rows, poc, vah, val, ibh, ibl, step=1):
    """文字版 TPO 字母圖（由上到下）。"""
    lines = []
    for i in range(len(prices) - 1, -1, -step):
        p = prices[i]
        tag = ""
        if abs(p + TICK / 2 - poc) < TICK / 2: tag += " <POC"
        if abs(p + TICK - vah) < TICK / 2: tag += " <VAH"
        if abs(p - val) < TICK / 2: tag += " <VAL"
        ib = "|" if ibl <= p + TICK / 2 <= ibh else " "
        lines.append(f"{p:8.0f} {ib} {''.join(rows[i]):<46}{tag}")
    return "\n".join(lines)
