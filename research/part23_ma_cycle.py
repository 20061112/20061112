"""第二十三部分：均線「發散 → 密集 → 突破」循環（SMA vs EMA、各種長度）。

和第六部分的差別：第六部分只要求「最近有壓縮」；這裡要求完整循環 ——
先有一段行情（三條均線依序排列且距離很大 = 發散），之後均線收斂（密集），
在密集區間被突破時進場，目的是抓「新的一段」。

定義（全部只用到 K 棒 t 收盤以前的資料，下一根開盤進場）
  W_t   = (三條均線最大 − 最小) / ATR100；rank_t = W_t 在過去 500 根中的百分位
  發散  : rank ≥ div_pct 且三條均線依序排列（快>中>慢 → 前段方向 +1，反之 −1）
  密集  : rank ≤ sq_pct；且距離上一次發散不超過 gap × 慢線長度 → 開始一個「密集區間」
  區間  : 從密集開始那根起算的最高 / 最低（至少 m = 快線長度 根，最長 3 × 慢線長度 根，逾時取消）
  突破  : 收盤 > 區間高（做多）/ < 區間低（做空）；一次發散只做一次
  方向  : cont = 突破方向與前段相同（續行），rev = 相反（反轉）
出場
  停損 = 區間另一側（box）或區間中點（mid）
  tp2 / tp3 = 2R / 3R 停利；trail = 收盤反向穿越慢線出場；最長持有 10 × 慢線長度 根
  同一根同時碰停損與停利算停損；跳空穿越停損以開盤價計；每筆扣進場那根第一分鐘的點差
對照組 base：同樣的密集 + 突破，但不要求先發散（每個密集區間做一次）。
資料：data/XAUUSD_M1_2026.csv + 上傳檔補 4/3~4/6；拆半 1/2~5/31 vs 6/1~10/8。
"""
import itertools
import sys
import numpy as np
import pandas as pd
from load import load

TFS = ["1min", "5min", "15min", "30min", "1h"]
SETS = [(5, 10, 20), (10, 20, 40), (8, 21, 55), (20, 40, 80), (20, 50, 100), (50, 100, 200)]
TYPES = ["sma", "ema"]
DIV = [0.8, 0.9]
SQ = [0.1, 0.25]
GAP = [1, 3]
EXITS = ["tp2", "tp3", "trail"]
STOPS = ["box", "mid"]
SPLIT = pd.Timestamp("2026-06-01")
RANK_LEN = 500
EXTRA = "/root/.claude/uploads/756e03f6-1d52-567b-8e69-9ce4bee9faa3/a5bf7bbb-XAUUSD_M1_202602020100_202604061706.csv"


def load_m1():
    m1 = load("data/XAUUSD_M1_2026.csv")
    try:
        ex = load(EXTRA)
        m1 = pd.concat([m1, ex[~ex.index.isin(m1.index)]]).sort_index()
    except FileNotFoundError:
        pass
    return m1


def resample(m1, rule):
    if rule == "1min":
        return m1[["open", "high", "low", "close", "spread"]].copy()
    return m1.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "spread": "first"}).dropna()


def atr(df, n):
    pc = df.close.shift()
    tr = pd.concat([df.high - df.low, (df.high - pc).abs(), (df.low - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def ma_block(df, typ, lens):
    c = df.close
    mas = [c.ewm(span=n, adjust=False).mean() if typ == "ema" else c.rolling(n).mean() for n in lens]
    M = np.column_stack([m.to_numpy() for m in mas])
    a = atr(df, 100).to_numpy()
    W = (M.max(1) - M.min(1)) / a
    rank = pd.Series(W).rolling(RANK_LEN).rank(pct=True).to_numpy()
    order = np.where((M[:, 0] > M[:, 1]) & (M[:, 1] > M[:, 2]), 1,
                     np.where((M[:, 0] < M[:, 1]) & (M[:, 1] < M[:, 2]), -1, 0))
    return dict(slow=M[:, 2], rank=rank, order=order)


def events(df, B, lens, div_pct, sq_pct, gap, need_div=True):
    """回傳訊號 list：(t, dir, box_hi, box_lo, prior_dir)。"""
    h, l, c = df.high.to_numpy(), df.low.to_numpy(), df.close.to_numpy()
    rank, order = B["rank"], B["order"]
    fast, slow = lens[0], lens[2]
    m, maxlen, G = max(fast, 3), 3 * slow, gap * slow
    last_div, div_dir, used = -10 ** 9, 0, True
    s = -1
    out = []
    for t in range(len(c)):
        r = rank[t]
        if np.isnan(r):
            continue
        if s >= 0:                                   # 在密集區間中：先檢查突破（區間 = s .. t-1）
            if t - s >= m:
                hi, lo = h[s:t].max(), l[s:t].min()
                d = 1 if c[t] > hi else -1 if c[t] < lo else 0
                if d:
                    out.append((t, d, hi, lo, div_dir))
                    s = -1
                    used = True
                    continue
            if t - s > maxlen:
                s = -1
        if r >= div_pct and order[t] != 0:
            last_div, div_dir, used = t, order[t], False
            s = -1 if need_div else s
            continue
        if s < 0 and r <= sq_pct:
            if not need_div:
                s = t
            elif not used and t - last_div <= G:
                s = t
    return out


def simulate(df, sig, slow, exit_, stop_mode, hold):
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    spr = df.spread.to_numpy(float) * 0.01
    n = len(c)
    res = []
    for t, d, hi, lo, _ in sig:
        e = t + 1
        if e >= n:
            res.append((np.nan, 0))
            continue
        ep = o[e]
        stop = (lo if d == 1 else hi) if stop_mode == "box" else (hi + lo) / 2
        R = d * (ep - stop)
        if not R > 0:
            res.append((np.nan, 0))
            continue
        end = min(e + hold, n)
        hh, ll, cc, oo = h[e:end], l[e:end], c[e:end], o[e:end]
        hit_s = (ll <= stop) if d == 1 else (hh >= stop)
        k_s = hit_s.argmax() if hit_s.any() else 10 ** 9
        if exit_ == "trail":
            ss = slow[e:end]
            hit_x = (cc < ss) if d == 1 else (cc > ss)
        else:
            tgt = ep + d * (2 if exit_ == "tp2" else 3) * R
            hit_x = (hh >= tgt) if d == 1 else (ll <= tgt)
        k_x = hit_x.argmax() if hit_x.any() else 10 ** 9
        if k_s == 10 ** 9 and k_x == 10 ** 9:
            k, xp = end - e - 1, cc[-1]
        elif k_s <= k_x:
            k = k_s
            xp = stop if k == 0 else (min(stop, oo[k]) if d == 1 else max(stop, oo[k]))
        else:
            k = k_x
            xp = cc[k] if exit_ == "trail" else tgt
        res.append(((d * (xp - ep) - spr[e]) / R, k + 1))
    return res


def fwd(df, sig, A, k):
    c = df.close.to_numpy()
    return np.array([d * (c[t + k] - c[t]) / A[t] if t + k < len(c) else np.nan for t, d, *_ in sig])


def st(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if len(v) < 3:
        return len(v), np.nan, np.nan, np.nan, np.nan
    g, b = v[v > 0].sum(), -v[v < 0].sum()
    return len(v), v.mean(), v.mean() / v.std(ddof=1) * np.sqrt(len(v)), (v > 0).mean(), g / b if b else np.nan


def main():
    m1 = load_m1()
    rows, fw = [], []
    for tf in TFS:
        df = resample(m1, tf)
        A14 = atr(df, 14).to_numpy()
        idx = df.index
        ndays = len(np.unique(idx.date))
        for typ, lens in itertools.product(TYPES, SETS):
            B = ma_block(df, typ, lens)
            hold = 10 * lens[2]
            variants = [("cycle", dv, sq, g) for dv in DIV for sq in SQ for g in GAP] + \
                       [("base", np.nan, sq, np.nan) for sq in SQ]
            for kind, dv, sq, g in variants:
                sig = events(df, B, lens, dv, sq, g, need_div=(kind == "cycle"))
                if len(sig) < 5:
                    continue
                t_idx = np.array([s[0] for s in sig])
                half = np.where(idx[t_idx] < SPLIT, "H1", "H2")
                dirs = np.array([s[1] for s in sig])
                rel = np.where(dirs == np.array([s[4] for s in sig]), "cont", "rev")
                for k in (5, 20, 60):
                    v = fwd(df, sig, A14, k)
                    for grp in ("all", "cont", "rev"):
                        mk = np.ones(len(v), bool) if grp == "all" else rel == grp
                        n_, mu, tt, *_ = st(v[mk])
                        fw.append(dict(tf=tf, type=typ, set=str(lens), kind=kind, div=dv, sq=sq, gap=g,
                                       k=k, dir=grp, n=n_, fwd=mu, t=tt))
                for ex, sm in itertools.product(EXITS, STOPS):
                    res = simulate(df, sig, B["slow"], ex, sm, hold)
                    R = np.array([r[0] for r in res])
                    bars = np.array([r[1] for r in res])
                    for grp in ("all", "cont", "rev"):
                        mk = np.ones(len(R), bool) if grp == "all" else rel == grp
                        n_, mu, tt, win, pf = st(R[mk])
                        _, m1_, t1, *_ = st(R[mk & (half == "H1")])
                        _, m2_, t2, *_ = st(R[mk & (half == "H2")])
                        rows.append(dict(tf=tf, type=typ, set=str(lens), kind=kind, div=dv, sq=sq, gap=g,
                                         exit=ex, stop=sm, dir=grp, n=n_, per_day=n_ / ndays, avgR=mu, t=tt,
                                         win=win, pf=pf, h1=m1_, t1=t1, h2=m2_, t2=t2,
                                         hold=np.nanmean(bars[mk]) if mk.any() else np.nan))
            print(tf, typ, lens, file=sys.stderr, flush=True)
    G = pd.DataFrame(rows)
    F = pd.DataFrame(fw)
    G.to_csv("part23_grid.csv", index=False)
    F.to_csv("part23_fwd.csv", index=False)


if __name__ == "__main__":
    main()
