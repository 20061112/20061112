"""第二十四部分 B：Sperandeo（Trader Vic）1-2-3 法則與 2B 法則，多種定義 / 演算法一起測。

波段點（swing pivot）演算法 — 全部只在「確認」之後才使用，不偷看：
  frac2 / frac3 / frac5 : 碎形，左右各 N 根；第 i 根為高點，要等到 i+N 收盤才確認
  zz2 / zz3 / zz5       : ZigZag，自極值回撤 k × ATR14 才確認該極值
  連續同類型的波段點只保留較極端的那個（在確認當下替換，仍然是因果的）

1-2-3（以做多為例；做空鏡像）
  波段序列 … L1（低）→ H2（高）→ L3（低），L3 > L1（不再破底）
  進場
    break : L3 確認後掛停損買單於 H2，有效 20 根；若先跌破 L3 則取消（經典 1-2-3 / Sperandeo 第 3 條件）
    early : L3 確認後下一根開盤直接進（積極版）
  停損：s3 = L3 下方 0.1 ATR；s1 = L1 下方 0.1 ATR
  停利：1R / 2R / 3R / mm（等幅測量：H2 + (H2 − L1)）/ time（只靠時間平倉）
  篩選（事後套用的特徵）
    ctx    : none；low50 = L1 為過去 50 根最低；low50+drop4 = 另外 L1 之前 50 根內跌幅 ≥ 4 ATR
    sper   : Sperandeo 第 1 條件 — 連接 L1 之前兩個遞降高點的下降趨勢線，在訊號前已被收盤突破
    trend  : none / with / against（日線收盤相對日線 SMA50，用前一天以前的資料）

2B（以做空為例；做多鏡像）
  已確認的波段高點 P，之後 100 根內第一次被突破，且在 w 根內（含突破那根）收盤跌回 P 之下
  進場
    close : 訊號 K 棒收盤後下一根開盤進場
    stop  : 掛停損賣單於訊號 K 棒低點，有效 3 根；若先漲破突破後的新高則取消
  停損：突破後新高 + 0.1 ATR
  停利：1R / 2R / 3R / time
  篩選：w ≤ 1 / 3 / 5；突破幅度 pen ≤ 1 ATR 或不限；sig = P 為過去 50 根最高；trend 同上

共同設定
  週期 M15 / H1 / H4；進場、停損、停利全部用 M1 逐根模擬；同一根 M1 同時碰到停損與停利算停損；
  成交那根 M1 只檢查停損；扣進場點差；持倉上限 M15 96 根 / H1 72 根 / H4 30 根
  每個參數組合各自做「同一時間只持有一筆」的篩選（依成交時間貪婪挑選）
資料：data/XAUUSD_M1_2023_2026.csv；樣本內 2023/6~2024/12，樣本外 2025/1~2026/10
"""
import itertools
import multiprocessing as mp
import sys

import numpy as np
import pandas as pd

from backtest import resample
from load import load

IS_END = pd.Timestamp("2025-01-01")
RULE = {"M15": "15min", "H1": "1h", "H4": "4h"}
HOLD = {"M15": 96, "H1": 72, "H4": 30}
SWINGS = ("frac2", "frac3", "frac5", "zz2", "zz3", "zz5")
BUF = 0.1

M1 = None


class TF:
    def __init__(self, m1, tf):
        df = resample(m1, RULE[tf])
        self.tf = tf
        self.o, self.h, self.l, self.c = (df[x].to_numpy(float) for x in ("open", "high", "low", "close"))
        pc = np.r_[self.c[0], self.c[:-1]]
        tr = np.maximum(self.h - self.l, np.maximum(abs(self.h - pc), abs(self.l - pc)))
        self.atr = pd.Series(tr).rolling(14).mean().to_numpy()
        self.time = df.index
        self.n = len(df)
        self.mo, self.mh, self.ml, self.mc = (m1[x].to_numpy(float) for x in ("open", "high", "low", "close"))
        self.mspr = m1.spread.to_numpy(float) * 0.01
        self.start = np.searchsorted(m1.index.values, df.index.values)
        self.end = np.r_[self.start[1:], len(m1)]
        self.tf_of_m1 = np.searchsorted(df.index.values, m1.index.values, side="right") - 1
        # 日線趨勢：前一天收盤 vs 前一天的 SMA50
        daily = m1.close.resample("1D").last().dropna()
        up = (daily > daily.rolling(50).mean()).shift(1)
        day = df.index.normalize()
        self.trend = np.where(up.reindex(day).to_numpy() == True, 1,  # noqa: E712
                              np.where(up.reindex(day).to_numpy() == False, -1, 0))  # noqa: E712


# ---------------------------------------------------------------- 波段點

def pivots_fractal(d, N):
    h, l, n = d.h, d.l, d.n
    out = []
    for i in range(N, n - N):
        if h[i] > h[i - N:i].max() and h[i] >= h[i + 1:i + N + 1].max():
            out.append((i + N, i, h[i], 1))
        if l[i] < l[i - N:i].min() and l[i] <= l[i + 1:i + N + 1].min():
            out.append((i + N, i, l[i], -1))
    out.sort()
    return out  # (確認根, 極值根, 價格, 1=高 / -1=低)


def pivots_zigzag(d, k):
    h, l, atr, n = d.h, d.l, d.atr, d.n
    out = []
    t0 = 15
    direc, ext_i = 1, t0
    for t in range(t0 + 1, n):
        a = atr[t]
        if not a > 0:
            continue
        if direc == 1:
            if h[t] >= h[ext_i]:
                ext_i = t
            elif l[t] <= h[ext_i] - k * a:
                out.append((t, ext_i, h[ext_i], 1))
                direc = -1
                ext_i = ext_i + 1 + int(np.argmin(l[ext_i + 1:t + 1]))
        else:
            if l[t] <= l[ext_i]:
                ext_i = t
            elif h[t] >= l[ext_i] + k * a:
                out.append((t, ext_i, l[ext_i], -1))
                direc = 1
                ext_i = ext_i + 1 + int(np.argmax(h[ext_i + 1:t + 1]))
    return out


def get_pivots(d, swing):
    if swing.startswith("frac"):
        return pivots_fractal(d, int(swing[4:]))
    return pivots_zigzag(d, float(swing[2:]))


# ---------------------------------------------------------------- 進出場模擬（M1）

def fill_stop(d, t0, bars, side, level, inval):
    """TF 第 t0 根起 bars 根內，掛停損單於 level；先碰到 inval 則取消。回傳 (M1 索引, 成交價) 或 None。"""
    if t0 >= d.n:
        return None
    b0, b1 = d.start[t0], d.end[min(t0 + bars - 1, d.n - 1)]
    hi, lo = d.mh[b0:b1], d.ml[b0:b1]
    hit = (hi >= level) if side == 1 else (lo <= level)
    bad = (lo <= inval) if side == 1 else (hi >= inval)
    if not hit.any():
        return None
    i = int(np.argmax(hit))
    if bad.any() and int(np.argmax(bad)) <= i:
        return None
    a = b0 + i
    px = max(d.mo[a], level) if side == 1 else min(d.mo[a], level)
    return a, px


def sim_exit(d, a, t, side, px, stop, tp, hold):
    """從 M1 索引 a 成交開始；回傳 (出場價, 原因, 出場 M1 索引)。"""
    if (d.ml[a] <= stop) if side == 1 else (d.mh[a] >= stop):
        return stop, "sl", a
    e = d.end[min(t + hold, d.n - 1)]
    lo, hi, op = d.ml[a + 1:e], d.mh[a + 1:e], d.mo[a + 1:e]
    if len(lo) == 0:
        return d.mc[a], "time", a
    hs = (lo <= stop) if side == 1 else (hi >= stop)
    i_s = int(np.argmax(hs)) if hs.any() else None
    i_t = None
    if tp is not None:
        ht = (hi >= tp) if side == 1 else (lo <= tp)
        i_t = int(np.argmax(ht)) if ht.any() else None
    if i_s is not None and (i_t is None or i_s <= i_t):
        return (min(op[i_s], stop) if side == 1 else max(op[i_s], stop)), "sl", a + 1 + i_s
    if i_t is not None:
        return (max(op[i_t], tp) if side == 1 else min(op[i_t], tp)), "tp", a + 1 + i_t
    return d.mc[e - 1], "time", e - 1


def trade(d, sig_t, a, side, px, stop, tp_spec, hold, mm=None):
    risk = side * (px - stop)
    if not risk > 0:
        return None
    if tp_spec == "time":
        tp = None
    elif tp_spec == "mm":
        tp = mm
        if tp is None or side * (tp - px) <= 0:
            return None
    else:
        tp = px + side * float(tp_spec) * risk
    t_fill = d.tf_of_m1[a]
    xp, why, xa = sim_exit(d, a, t_fill, side, px, stop, tp, hold)
    pnl = side * (xp - px) - d.mspr[a]
    return dict(time=d.time[sig_t], side=side, fill_m1=a, exit_m1=xa, why=why,
                pnl_r=pnl / risk, pnl_atr=pnl / d.atr[sig_t], risk_atr=risk / d.atr[sig_t],
                tp_r=side * (tp - px) / risk if tp is not None else np.nan)


# ---------------------------------------------------------------- 1-2-3

def signals_123(d, piv):
    """以確認順序處理波段點，回傳 1-2-3 訊號（含特徵）。"""
    seq, sigs, used = [], [], set()
    h, l, c, atr = d.h, d.l, d.c, d.atr
    for conf, idx, price, typ in piv:
        if seq and seq[-1][3] == typ:
            if (typ == 1 and price > seq[-1][2]) or (typ == -1 and price < seq[-1][2]):
                seq[-1] = (conf, idx, price, typ)
            else:
                continue
        else:
            seq.append((conf, idx, price, typ))
        if len(seq) < 3:
            continue
        p1, p2, p3 = seq[-3], seq[-2], seq[-1]
        side = -p3[3]          # 新確認的是低點 → 做多
        if side * (p3[2] - p1[2]) <= 0:    # 做多需 L3 > L1；做空需 H3 < H1
            continue
        key = (p2[1], side)
        if key in used:
            continue
        used.add(key)
        i1 = p1[1]
        if i1 < 60 or not atr[i1] > 0:
            continue
        w0 = i1 - 50
        if side == 1:
            ext50 = l[i1] <= l[w0:i1].min()
            move = (h[w0:i1].max() - p1[2]) / atr[i1]
        else:
            ext50 = h[i1] >= h[w0:i1].max()
            move = (p1[2] - l[w0:i1].min()) / atr[i1]
        # Sperandeo 第 1 條件：L1 之前兩個同向（做多看高點）波段點連成的趨勢線被收盤突破
        sper = False
        if len(seq) >= 6:
            pa, pb = seq[-6], seq[-4]
            if side * (pa[2] - pb[2]) > 0 and pb[1] > pa[1]:
                slope = (pb[2] - pa[2]) / (pb[1] - pa[1])
                xs = np.arange(i1, p3[0] + 1)
                line = pb[2] + (xs - pb[1]) * slope
                sper = bool((side * (c[i1:p3[0] + 1] - line) > 0).any())
        sigs.append(dict(t=p3[0], side=side, p1=p1[2], p2=p2[2], p3=p3[2], ext50=ext50, move=move, sper=sper,
                         trend=d.trend[p3[0]]))
    return sigs


def run_123(d, swing):
    piv = get_pivots(d, swing)
    sigs = signals_123(d, piv)
    hold = HOLD[d.tf]
    rows = []
    for s in sigs:
        t, side, A = s["t"], s["side"], d.atr[s["t"]]
        if t + 1 >= d.n or not A > 0:
            continue
        mm = s["p2"] + (s["p2"] - s["p1"])
        for entry in ("break", "early"):
            if entry == "break":
                f = fill_stop(d, t + 1, 20, side, s["p2"], s["p3"])
            else:
                a = d.start[t + 1]
                f = (a, d.mo[a])
            if f is None:
                continue
            a, px = f
            for stop_kind in ("s3", "s1"):
                base = s["p3"] if stop_kind == "s3" else s["p1"]
                stop = base - side * BUF * A
                for tp in ("1", "2", "3", "mm", "time"):
                    r = trade(d, t, a, side, px, stop, tp, hold, mm=mm)
                    if r is None:
                        continue
                    r.update(entry=entry, stop=stop_kind, tp=tp, ext50=s["ext50"], move=s["move"],
                             sper=s["sper"], trend=s["trend"])
                    rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 2B

def run_2b(d, swing):
    piv = get_pivots(d, swing)
    h, l, c, atr = d.h, d.l, d.c, d.atr
    hold = HOLD[d.tf]
    rows = []
    for conf, idx, price, typ in piv:
        side = -typ        # 假突破高點 → 做空
        if idx < 60 or conf + 1 >= d.n:
            continue
        sig50 = (h[idx] >= h[idx - 50:idx].max()) if typ == 1 else (l[idx] <= l[idx - 50:idx].min())
        seg_end = min(conf + 101, d.n)
        seg = h[conf + 1:seg_end] if typ == 1 else l[conf + 1:seg_end]
        br = (seg > price) if typ == 1 else (seg < price)
        if not br.any():
            continue
        tb = conf + 1 + int(np.argmax(br))
        s = None
        for j in range(tb, min(tb + 5, d.n - 1)):
            if (c[j] < price) if typ == 1 else (c[j] > price):
                s = j
                break
        if s is None:
            continue
        A = atr[s]
        if not A > 0:
            continue
        ext = h[tb:s + 1].max() if typ == 1 else l[tb:s + 1].min()
        pen = abs(ext - price) / A
        stop = ext - side * BUF * A
        feat = dict(w=s - tb + 1, pen=pen, sig50=bool(sig50), trend=d.trend[s])
        for entry in ("close", "stop"):
            if entry == "close":
                a = d.start[s + 1]
                f = (a, d.mo[a])
            else:
                lvl = l[s] if side == -1 else h[s]
                f = fill_stop(d, s + 1, 3, side, lvl, ext)
            if f is None:
                continue
            a, px = f
            for tp in ("1", "2", "3", "time"):
                r = trade(d, s, a, side, px, stop, tp, hold)
                if r is None:
                    continue
                r.update(entry=entry, tp=tp, **feat)
                rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 彙整

def no_overlap(tr):
    tr = tr.sort_values("fill_m1")
    keep, last = [], -1
    for i, a, x in zip(tr.index, tr.fill_m1.to_numpy(), tr.exit_m1.to_numpy()):
        if a > last:
            keep.append(i)
            last = x
    return tr.loc[keep]


def tstat(x):
    x = np.asarray(x, float)
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2 else np.nan


def summarize(tr, cfg):
    r = dict(cfg)
    tr = no_overlap(tr)
    for tag, part in (("IS", tr[tr.time < IS_END]), ("OOS", tr[tr.time >= IS_END])):
        p = part.pnl_r.to_numpy()
        r[f"{tag}_n"] = len(p)
        r[f"{tag}_win"] = (p > 0).mean() if len(p) else np.nan
        r[f"{tag}_R"] = p.mean() if len(p) else np.nan
        r[f"{tag}_t"] = tstat(p)
    r["long_R"] = tr[tr.side == 1].pnl_r.mean()
    r["short_R"] = tr[tr.side == -1].pnl_r.mean()
    for y, part in tr.groupby(tr.time.dt.year):
        r[f"y{y}"] = part.pnl_r.mean()
    return r


TREND = {"none": None, "with": "with", "against": "against"}


def apply_trend(df, mode):
    if mode == "with":
        return df[df.side == df.trend]
    if mode == "against":
        return df[df.side == -df.trend]
    return df


def job(args):
    tf, swing = args
    d = TF(M1, tf)
    out = []
    t123 = run_123(d, swing)
    if len(t123):
        for (entry, stop, tp), g in t123.groupby(["entry", "stop", "tp"]):
            for ctx, sper, trend in itertools.product(("none", "low50", "low50+drop4"), (False, True),
                                                      ("none", "with", "against")):
                x = g
                if ctx != "none":
                    x = x[x.ext50]
                if ctx == "low50+drop4":
                    x = x[x.move >= 4]
                if sper:
                    x = x[x.sper]
                x = apply_trend(x, trend)
                if len(x) < 30:
                    continue
                out.append(summarize(x, dict(pattern="123", tf=tf, swing=swing, entry=entry, stop=stop, tp=tp,
                                             f1=ctx, f2="sper" if sper else "-", trend=trend)))
    t2b = run_2b(d, swing)
    if len(t2b):
        for (entry, tp), g in t2b.groupby(["entry", "tp"]):
            for w, pen, sig, trend in itertools.product((1, 3, 5), (np.inf, 1.0), (False, True),
                                                        ("none", "with", "against")):
                x = g[(g.w <= w) & (g.pen <= pen)]
                if sig:
                    x = x[x.sig50]
                x = apply_trend(x, trend)
                if len(x) < 30:
                    continue
                out.append(summarize(x, dict(pattern="2B", tf=tf, swing=swing, entry=entry, stop=f"w{w}", tp=tp,
                                             f1="pen<=1" if pen < np.inf else "-", f2="sig50" if sig else "-",
                                             trend=trend)))
    print(tf, swing, len(t123), len(t2b), file=sys.stderr, flush=True)
    return out


def main():
    global M1
    M1 = load("data/XAUUSD_M1_2023_2026.csv")
    jobs = [(tf, s) for tf in ("H4", "H1", "M15") for s in SWINGS]
    with mp.get_context("fork").Pool(4) as pool:
        res = pool.map(job, jobs, chunksize=1)
    df = pd.DataFrame([r for part in res for r in part])
    df.to_csv("part24_scan.csv", index=False)
    print(len(df), "組", file=sys.stderr)


if __name__ == "__main__":
    main()
