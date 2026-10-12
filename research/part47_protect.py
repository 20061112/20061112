"""第四十七部分：RR 3 / 4 加上「保護獲利」的出場方式。

最終版流程（A+B+斜率 → 上限 8 → 影子），停損 2 ATR15，0.02 口，點差 + 0.3 滑價。
出場方式（觸發價在進場後的 K 棒才生效；觸發那一根不改停損，從下一根開始）：
  原始        固定停損 / 停利 / 收盤
  保本 X      浮盈達 X·R → 停損移到進場價
  鎖利 X→Y    浮盈達 X·R → 停損移到 +Y·R
  移動 X/D    浮盈達 X·R 之後，停損 = 目前最高浮盈 − D·R（用到前一根為止的最高，不偷看），且不低於保本
  分批 X      浮盈達 X·R 平一半；另一半停損移到進場價，照原停利 / 收盤
同時列出：原始版本中「曾經浮盈 ≥ 2R、最後卻虧錢出場」的比例（獲利回吐有多嚴重）。
"""
import numpy as np
import pandas as pd
import part29_rr_grid as G
from part30_max_positions import cap_total

W, IDX = G.W, G.IDX
C, H, Lo, SP, dayv, day_end = W.C, W.H, W.Lo, W.SP, W.dayv, W.day_end
SLIP = 0.3
dl = W.B.daily
sma50 = dl.C.rolling(50).mean()
atrd = (dl.H - dl.L).rolling(10).mean()
ds = pd.DataFrame({"pc": dl.C.shift(1), "ma": sma50.shift(1), "ma10": sma50.shift(11), "atr": atrd.shift(1), "o": dl.O})
EV = []
for i, L, side, a in G.ev:
    s = ds.loc[dayv[i]]
    if not (np.isfinite(s.ma) and np.isfinite(s.ma10) and np.isfinite(s.atr)):
        continue
    if (s.pc - s.ma) * side > 0 and (L - s.o) * side / s.atr > -0.17 and (s.ma - s.ma10) * side > 0 and IDX[i] >= pd.Timestamp("2023-08-24"):
        EV.append((i, L, side, a))


def first(mask):
    return int(np.argmax(mask)) if mask.any() else 10 ** 9


def simulate(i, L, side, risk, t, rule):
    """回傳 (出場索引, 每盎司損益（未扣成本）, 最大浮盈 R)。以多單的「有利方向」計算：f = (價 − L)·side。"""
    base = G.sim_detail(i, L, side, L - side * risk, L + side * t * risk)
    if base is None:
        return None
    j0, px0, why0 = base
    last = day_end[i]
    fav = (H[i + 1:last + 1] - L) if side > 0 else (L - Lo[i + 1:last + 1])   # 每根最有利
    adv = (Lo[i + 1:last + 1] - L) if side > 0 else (L - H[i + 1:last + 1])   # 每根最不利（負 = 虧）
    cl = (C[i + 1:last + 1] - L) * side
    mfe = max(fav[:max(j0 - i, 0)].max() if j0 > i else 0.0, 0.0) / risk
    base_pnl = (px0 - L) * side
    if rule[0] == "none" or j0 <= i:
        return j0, base_pnl, mfe
    kind = rule[0]
    X = rule[1]
    jb = j0 - i - 1  # 原始出場在 fav 陣列中的位置
    ja = first(fav >= X * risk)
    if ja >= jb or ja == 10 ** 9:  # 原始出場前沒到觸發價（或同一根）→ 結果不變
        return j0, base_pnl, mfe
    tgt = t * risk
    seg_f, seg_a, seg_c = fav[ja + 1:], adv[ja + 1:], cl[ja + 1:]
    if kind in ("be", "lock", "partial"):
        lockv = 0.0 if kind in ("be", "partial") else rule[2] * risk
        js = first(seg_a <= lockv)
        jt = first(seg_f >= tgt)
        if js == jt == 10 ** 9:
            rest = seg_c[-1] if len(seg_c) else cl[ja]; je = len(cl) - 1
        elif js <= jt:
            rest = lockv; je = ja + 1 + js
        else:
            rest = tgt; je = ja + 1 + jt
        if kind == "partial":
            return i + 1 + je, 0.5 * X * risk + 0.5 * rest, mfe
        return i + 1 + je, rest, mfe
    if kind == "trail":
        D = rule[2] * risk
        run = np.maximum.accumulate(np.r_[fav[ja], seg_f])[:-1]  # 到前一根為止的最高浮盈
        stop = np.maximum(run - D, 0.0)
        js = first(seg_a <= stop)
        jt = first(seg_f >= tgt)
        if js == jt == 10 ** 9:
            return last, seg_c[-1] if len(seg_c) else cl[ja], mfe
        if js <= jt:
            return i + 1 + ja + 1 + js, stop[js], mfe
        return i + 1 + ja + 1 + jt, tgt, mfe
    raise ValueError(kind)


def pipeline(t, rule):
    rows = []
    for i, L, side, a in EV:
        risk = 2.0 * a
        r = simulate(i, L, side, risk, t, rule)
        if r is None:
            continue
        j, pnl, mfe = r
        net = pnl - SP[i] - SLIP
        rows.append((IDX[i], IDX[min(j, len(IDX) - 1)], i, min(j, len(IDX) - 1), side, L, net, net / risk, mfe))
    x = pd.DataFrame(rows, columns=["進場時間", "出場時間", "i", "j", "dir", "entry", "net", "R", "mfe"])
    x = cap_total(x.sort_values("進場時間").reset_index(drop=True), 8).reset_index(drop=True)
    ex = x.sort_values("出場時間"); cum = ex.R.cumsum().values
    ma = pd.Series(cum).rolling(50, min_periods=50).mean().values; et = ex.出場時間.values
    keep = []
    for tt in x.進場時間.values:
        k = np.searchsorted(et, tt, side="right") - 1
        keep.append(not (k >= 50 and cum[k] < ma[k]))
    return x[np.array(keep)]


def minute_mdd(df):
    n = len(IDX)
    D, E, R = np.zeros(n + 1), np.zeros(n + 1), np.zeros(n + 1)
    for r in df.itertuples():
        D[r.i] += r.dir * 2; D[r.j] -= r.dir * 2
        E[r.i] += r.dir * r.entry * 2; E[r.j] -= r.dir * r.entry * 2
        R[r.j] += r.net * 2
    D, E, R = np.cumsum(D)[:n], np.cumsum(E)[:n], np.cumsum(R)[:n]
    eq = R + C * D - E
    return (np.maximum.accumulate(eq) - eq).max()


RULES = [("原始", ("none", 0)), ("保本 1R", ("be", 1.0)), ("保本 1.5R", ("be", 1.5)), ("保本 2R", ("be", 2.0)), ("保本 2.5R", ("be", 2.5)),
         ("鎖利 2R→+1R", ("lock", 2.0, 1.0)), ("鎖利 2.5R→+1R", ("lock", 2.5, 1.0)), ("鎖利 3R→+1.5R", ("lock", 3.0, 1.5)),
         ("移動 2R / 距離 1R", ("trail", 2.0, 1.0)), ("移動 2R / 距離 1.5R", ("trail", 2.0, 1.5)), ("移動 1.5R / 距離 1.5R", ("trail", 1.5, 1.5)),
         ("分批 1.5R 平一半", ("partial", 1.5)), ("分批 2R 平一半", ("partial", 2.0))]
lines = []


def p(s=""):
    print(s, flush=True)
    lines.append(s)


p("第四十七部分：RR 3 / 4 + 保護獲利（最終版流程，0.02 口，點差 + 0.3 滑價，2023/8 ~ 2026/10）")
for t in (3, 4):
    base = pipeline(t, RULES[0][1])
    giveback = base[(base.mfe >= 2) & (base.R < 0)]
    p(f"\n=== RR {t} ===   原始版本：曾浮盈 ≥ 2R 的單 {(base.mfe >= 2).sum()} 筆，其中最後虧錢出場 {len(giveback)} 筆（{len(giveback) / max((base.mfe >= 2).sum(), 1):.0%}）")
    p(f"{'出場方式':18s} {'筆/天':>5s} {'勝率':>5s} {'每筆R':>7s} {'總損益':>8s} {'MDD':>6s} {'最差月':>7s} {'年獲利/MDD':>9s} | {'訓練':>4s} {'驗證':>4s} | 逐年")
    for lab, rule in RULES:
        x = base if rule[0] == "none" else pipeline(t, rule)
        pnl = x.net * 2
        yrs = (x.進場時間.max() - x.進場時間.min()).days / 365
        m = minute_mdd(x)
        mon = pnl.groupby(x.進場時間.dt.to_period("M")).sum()
        tr = (x.進場時間 < "2025-01-01").values

        def rat(z, pz):
            d = pz.groupby(z.進場時間.dt.date).sum(); e = d.cumsum(); mm = (e.cummax() - e).max()
            return pz.sum() / mm
        yr = " / ".join(f"{pnl[(x.進場時間.dt.year == y).values].sum():+,.0f}" for y in (2023, 2024, 2025, 2026))
        p(f"{lab:18s} {len(x) / x.進場時間.dt.date.nunique():5.1f} {(x.R > 0).mean():5.0%} {x.R.mean():+7.3f} {pnl.sum():+8,.0f} {m:6,.0f} "
          f"{mon.min():+7,.0f} {pnl.sum() / yrs / m:9.2f} | {rat(x[tr], pnl[tr]):4.1f} {rat(x[~tr], pnl[~tr]):4.1f} | {yr}")
open("results_part47.txt", "w").write("\n".join(lines) + "\n")
