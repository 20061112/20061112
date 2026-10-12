"""第六十部分：未來函數 / 不切實際成交的稽核。

A. 指標截斷測試：把資料截在隨機時間點 T，重算指標；T 以前所有用到的值必須和用全部資料算的一模一樣
   - M15 事件的 ATR15（前 20 根平均波幅）、前一根高低、ATR 熵變化（前一根已收完的 M15）
   - 日線：前日收盤、SMA50、SMA50 10 日前、ATR10、ATR 比（volrel）
   - IB 核心：ATR10、前日效率
B. 成交假設（不是未來函數，但可能太樂觀）：
   B1 進場跳空：原本「價格碰到點位」才進場、以點位成交；若某根 M1 開盤就已經跳過點位（之前沒碰到），
      真實的停損單會在那根開盤價成交（更差），而且原本的寫法會等價格「回來」才進場，甚至漏掉跑掉的單
      → 改成：期間內第一根最高（做多）≥ 點位的 M1 進場，成交價 = max(開盤, 點位)
   B2 停損跳空：原本停損一律以停損價成交 → 改成 M1 開盤已經跳過停損時以開盤價成交（停利跳空仍以停利價）
   B3 同一分鐘：權益濾網 / 上限 8 筆 / 單日 −15R 原本把「同一分鐘出場」的單當成已知 → 改成只用嚴格早於進場分鐘的出場
C. 全部修正後重跑 M15 最終版與 IB(2R 保本) 1 + M15 2 組合（只扣點差 / 扣 0.3 滑價）。
"""
import numpy as np
import pandas as pd
import part58_robust_opt as R8

Q = R8.Q
P, G, W, B = Q.P, Q.G, Q.W, Q.B
IDX, H, Lo, C, SP, mins, dayv, day_end = Q.IDX, W.H, W.Lo, W.C, W.SP, W.mins, W.dayv, W.day_end
O1 = W.O1
rng = np.random.default_rng(60)
lines = []
Q.lines = lines


def p(s=""):
    print(s, flush=True)
    lines.append(s)


p("第六十部分：未來函數與成交假設稽核")

# ---------------- A. 截斷測試 ----------------
p("\n=== A. 指標截斷測試（截在隨機 10 個時間點；比較截斷點以前用到的值）===")


def m15_indicators(n):
    """只用前 n 根 M1 重算：每根 M15 的前一根高低、ATR15（不含當根）、熵變化。"""
    pid = B.period_ids(15)[:n]
    g = pd.DataFrame({"pid": pid, "H": H[:n], "L": Lo[:n], "C": C[:n]})
    b = g.groupby("pid").agg(H=("H", "max"), L=("L", "min"), C=("C", "last"))
    atr = (b.H - b.L).rolling(20).mean().shift(1).values
    pH, pL = b.H.shift(1).values, b.L.shift(1).values
    tr = np.nan_to_num(np.maximum(b.H - b.L, np.maximum((b.H - b.C.shift(1)).abs(), (b.L - b.C.shift(1)).abs())).values)
    v = np.lib.stride_tricks.sliding_window_view(tr, 20)
    pp = v / v.sum(1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = np.r_[np.full(19, np.nan), -np.nansum(np.where(pp > 0, pp * np.log(pp), 0), 1) / np.log(20)]
    dent = ent - np.r_[np.full(10, np.nan), ent[:-10]]
    return pid, atr, pH, pL, dent


def daily_ind(days_upto):
    dl = W.B.daily.loc[:days_upto]
    sma50 = dl.C.rolling(50).mean(); atrd = (dl.H - dl.L).rolling(10).mean()
    return pd.DataFrame({"pc": dl.C.shift(1), "ma": sma50.shift(1), "ma10": sma50.shift(11), "atr": atrd.shift(1),
                         "volrel": (atrd / atrd.rolling(60).mean()).shift(1)})


full = m15_indicators(len(H))
fd = daily_ind(W.B.daily.index[-1])
ev_i = np.array([e[0] for e in P.EV])
bad = 0
for n in sorted(rng.integers(len(H) // 5, len(H) - 1000, 10)):
    t = m15_indicators(n)
    used = ev_i[ev_i < n]                      # 截斷點以前的進場分鐘
    k = full[0][used]                          # 該分鐘所在的 M15
    # 進場那根 M15 的 ATR / 前高低（不含當根）+ 前一根的熵變化
    for name, a, b_ in (("ATR15", full[1][k], t[1][k]), ("前高", full[2][k], t[2][k]), ("前低", full[3][k], t[3][k]),
                        ("熵變化", full[4][k - 1], t[4][k - 1])):
        diff = ~np.isclose(a, b_, equal_nan=True)
        bad += diff.sum()
        if diff.any():
            p(f"  ✗ 截斷 {IDX[n]}：{name} 有 {diff.sum()} 筆不同")
    day_cut = dayv[n]
    td = daily_ind(day_cut)
    cmp = fd.loc[td.index]
    for c_ in td.columns:
        diff = ~np.isclose(cmp[c_].values, td[c_].values, equal_nan=True)
        bad += diff.sum()
        if diff.any():
            p(f"  ✗ 截斷 {day_cut}：日線 {c_} 有 {diff.sum()} 天不同")
p(f"  M15 與日線指標：{'全部一致（沒有用到未來資料）' if bad == 0 else f'{bad} 個值不同'}")
# IB 核心
D = Q.CI.D
for cut in pd.DatetimeIndex(D.index)[[200, 400, 600, 800]]:
    sub = D.loc[:cut, ["o", "h", "l", "c"]].copy()
    pc = sub.c.shift()
    tr = np.maximum(sub.h - sub.l, np.maximum((sub.h - pc).abs(), (sub.l - pc).abs()))
    a10 = tr.rolling(10).mean().shift()
    peff = ((sub.c - sub.o).abs() / (sub.h - sub.l)).shift()
    ok = np.isclose(a10, D.loc[:cut, "atr10"], equal_nan=True).all() and np.isclose(peff, D.loc[:cut, "peff"], equal_nan=True).all()
    bad += 0 if ok else 1
p(f"  IB 核心 ATR10 / 前日效率：{'截斷前後一致' if bad == 0 else '有差異'}")
p("  程式檢查（人工）：進場價 = 前一根已收完 K 棒的高低；日線條件全部 shift(1)，只有「今天開盤價」是當天值（進場時已知）；"
  "IB 用完整收完的 IB、M5 收盤確認後下一根才開始算停損；熵用前一根已收完的 M15；熵門檻 / 單日上限分位只用 2023/8~2024/12。")

# ---------------- B. 成交假設 ----------------
pid15 = B.period_ids(15)
se = pd.Series(np.arange(len(H))).groupby(pid15).agg(["first", "last"])
first_of = se["first"].values


def real_events():
    """重新產生 M15 事件：期間內第一根「越過」點位的 M1（不要求碰到），成交 = max(開, 點位)。"""
    g = pd.DataFrame({"pid": pid15, "H": H, "L": Lo, "O": O1})
    agg = g.groupby("pid").agg(H=("H", "max"), L=("L", "min"), O=("O", "first"))
    atr = (agg.H - agg.L).rolling(20).mean().shift(1).values
    pH, pL, O = agg.H.shift(1).values, agg.L.shift(1).values, agg.O.values
    f_, l_ = se["first"].values, se["last"].values
    _, dent = Q.entropy_change(15)
    rows = []
    for k in range(1, len(O)):
        a = atr[k]
        if not np.isfinite(a) or a <= 0 or mins[f_[k]] >= 21 * 60:
            continue
        for side, L in ((1, pH[k]), (-1, pL[k])):
            if (O[k] - L) * side >= 0:
                continue
            hit = ((H[f_[k]:l_[k] + 1] >= L) if side > 0 else (Lo[f_[k]:l_[k] + 1] <= L)) & (mins[f_[k]:l_[k] + 1] < 22 * 60)
            if not hit.any():
                continue
            i = f_[k] + int(np.argmax(hit))
            if i == f_[k] and i > 0 and (C[i - 1] - L) * side > 0:
                continue
            if IDX[i] < pd.Timestamp("2023-08-24"):
                continue
            s_ = P.ds.loc[dayv[i]]
            if not (np.isfinite(s_.ma) and np.isfinite(s_.ma10) and np.isfinite(s_.atr)):
                continue
            fill = max(O1[i], L) if side > 0 else min(O1[i], L)
            rows.append((i, L, fill, side, a, s_.volrel, (s_.pc - s_.ma) * side > 0, (L - s_.o) * side / s_.atr > -0.17,
                         (s_.ma - s_.ma10) * side > 0, dent[max(pid15[i] - 1, 0)]))
    return pd.DataFrame(rows, columns=["i", "L", "fill", "side", "a", "volrel", "A", "B", "S", "dent"])


def sim_real(i, entry, side, stop, tgt, gap_stop):
    r = G.sim_detail(i, entry, side, stop, tgt)
    if r is None:
        return None
    j, px, why = r
    if gap_stop and why == "停損" and j > i:
        o = O1[j]
        if (o < stop) if side > 0 else (o > stop):
            px = o
    return j, px, why


def m15_real(slip, fix_entry, fix_stop, strict):
    E = REAL if fix_entry else Q.EVS[15]
    E = E[E.A & E.B & E.S]
    rows, n_gap, n_moved = [], 0, 0
    for r in E.itertuples():
        risk = 2.0 * r.a
        i, L = r.i, r.L
        stop, tgt = L - r.side * risk, L + r.side * 3 * risk
        entry = L
        if fix_entry:
            entry = r.fill
            n_gap += entry != L
        x = sim_real(i, entry, r.side, stop, tgt, fix_stop)
        if x is None:
            continue
        j, px, why = x
        net = (px - entry) * r.side - SP[i] - slip
        rows.append((IDX[i], IDX[j], i, j, r.side, entry, net, net / risk, risk, r.volrel, r.dent))
    x = pd.DataFrame(rows, columns=["進場時間", "出場時間", "i", "j", "dir", "entry", "net", "R", "risk", "volrel", "dent"])
    x = x.sort_values("進場時間").reset_index(drop=True)
    # 上限 8：嚴格版把同一分鐘出場的單仍算在場上
    keep, open_ = [], []
    for rr in x.itertuples():
        open_ = [e for e in open_ if (e >= rr.進場時間 if strict else e > rr.進場時間)]
        if len(open_) < 8:
            open_.append(rr.出場時間); keep.append(rr.Index)
    x = x.loc[keep].reset_index(drop=True)
    ex = x.sort_values("出場時間"); cum = ex.R.cumsum().values
    ma = pd.Series(cum).rolling(50, min_periods=50).mean().values; et = ex.出場時間.values
    side_ = "left" if strict else "right"
    kk = [not (k >= 50 and cum[k] < ma[k]) for k in (np.searchsorted(et, tt, side=side_) - 1 for tt in x.進場時間.values)]
    x = x[np.array(kk)].reset_index(drop=True)
    x = x[~(x.volrel > 2.2).values & ~(x.dent > Q.CUT[15]).values].reset_index(drop=True)
    keep, real, cur = [], [], None
    for rr in x.itertuples():
        day = rr.進場時間.normalize()
        if day != cur:
            cur, real = day, []
        lost = sum(v for (te, v) in real if (te < rr.進場時間 if strict else te <= rr.進場時間))
        if lost <= -15:
            keep.append(False); continue
        keep.append(True); real.append((rr.出場時間, rr.R))
    return x[np.array(keep)].reset_index(drop=True), n_gap, n_moved


REAL = real_events()
orig = Q.EVS[15]; o2 = orig[orig.A & orig.B & orig.S]; r2 = REAL[REAL.A & REAL.B & REAL.S]
p(f"\nM15 事件（A+B+S 後）：原本 {len(o2)}、修正後 {len(r2)}（多出 {len(r2) - len(o2)} 筆＝跳過點位後沒再回來、原本漏掉的單；"
  f"以開盤跳空成交 {(r2.fill != r2.L).sum()} 筆，平均差 {(np.abs(r2.fill - r2.L)[r2.fill != r2.L]).mean():.2f} 美元）")
p("\n=== B / C. 成交假設修正後重跑（M15 2 份單獨、IB(2R保本) 1 + M15 2）===")
for slip in (0.0, 0.3):
    I = R8.ib(slip)
    p(f"-- 滑價 {slip} --")
    for lab, fe, fs, st in (("原本", False, False, False), ("B1 進場跳空", True, False, False), ("B2 停損跳空", False, True, False),
                            ("B3 同一分鐘嚴格", False, False, True), ("全部修正", True, True, True)):
        M, ng, nm = m15_real(slip, fe, fs, st)
        p(f"  [{lab}]")
        R8.show(f"M15 2｜{lab}", [(M, 2)])
        R8.show(f"IB 1 + M15 2｜{lab}", [(I, 1), (M, 2)])
open("results_part60.txt", "w").write("\n".join(lines) + "\n")
