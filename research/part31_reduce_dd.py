"""第三十一部分：降低回撤、提高「總獲利 / MDD」。

基準：A+B 過濾的 M15 突破，停損 2 ATR15、RR 3，持倉上限 8，每筆固定風險 1R（R 計，不受金價水準影響）。
指標：每筆 R、每天筆數、總 R、MDD（日結，R）、總 / MDD；訓練 2023/6~2024/12、驗證 2025/1~2026/10 分開算（各自的 總/MDD）。
測試：
  1. 更嚴格的趨勢狀態（全部用前一日為止的日線資料）：
       er10   前 10 日效率比（|C−C10| / Σ|ΔC|）—— 盤整期回撤最大
       ma20   前日收盤也在 20 日均線同側
       slope  50 日均線 10 日斜率與方向一致
       dist   前日收盤離 50 日均線（日 ATR，乘方向）
  2. 出場：浮盈 1R / 1.5R 後停損移到保本
  3. 部位控制（依序模擬，只用已出場的資訊）：
       當日停損上限：當天已實現虧損達 −X R 後不再開新單
       權益曲線濾網：策略本身的累積 R 低於其 N 筆均線時，新單只下一半（或不下）
需要先跑 part28_m15_filters.py。
"""
import numpy as np
import pandas as pd
import part29_rr_grid as G
from part30_max_positions import cap_total

W, IDX = G.W, G.IDX
H, Lo, C, SP, day_end = W.H, W.Lo, W.C, W.SP, W.day_end
lines = []


def p(s=""):
    print(s)
    lines.append(s)


# ---------- 事件 + 日線狀態 ----------
evs = [e for e, k in zip(G.ev, G.mA & G.mB) if k]
dl = W.B.daily.copy()
dC = dl.C
er10 = ((dC - dC.shift(10)).abs() / dC.diff().abs().rolling(10).sum())
sma20, sma50 = dC.rolling(20).mean(), dC.rolling(50).mean()
atrd = (dl.H - dl.L).rolling(10).mean()
st_ = pd.DataFrame({"er10": er10, "up20": dC > sma20, "slope": np.sign(sma50 - sma50.shift(10)),
                    "dist": (dC - sma50) / atrd}).shift(1)  # 前一日為止


def sim_be(i, L, side, risk, t, be):
    """be = None：一般；be = k：浮盈 k·R 後停損移到進場價。回傳 (出場索引, 價格損益)。"""
    stop, tgt = L - side * risk, L + side * t * risk
    r = G.sim_detail(i, L, side, stop, tgt)
    if r is None:
        return None
    j, px, why = r
    if be is None:
        return j, (px - L) * side
    trig = L + side * be * risk
    last = day_end[i]
    hh, ll = H[i + 1:last + 1], Lo[i + 1:last + 1]
    reach = (hh >= trig) if side > 0 else (ll <= trig)
    if not reach.any():
        return j, (px - L) * side
    jr = i + 1 + int(np.argmax(reach))
    if j < jr:  # 先停損/停利/收盤才到觸發價
        return j, (px - L) * side
    # 觸發後：停損 = 進場價
    hh2, ll2 = H[jr + 1:last + 1], Lo[jr + 1:last + 1]
    hs = (ll2 <= L) if side > 0 else (hh2 >= L)
    ht = (hh2 >= tgt) if side > 0 else (ll2 <= tgt)
    if (H[jr] >= tgt) if side > 0 else (Lo[jr] <= tgt):
        return jr, (tgt - L) * side
    js = int(np.argmax(hs)) if hs.any() else 10 ** 9
    jt = int(np.argmax(ht)) if ht.any() else 10 ** 9
    if js == jt == 10 ** 9:
        return last, (C[last] - L) * side
    if jt < js:
        return jr + 1 + jt, (tgt - L) * side
    return jr + 1 + js, 0.0


def build(be=None, s=2.0, t=3.0):
    rows = []
    for i, L, side, a in evs:
        risk = s * a
        r = sim_be(i, L, side, risk, t, be)
        if r is None:
            continue
        j, pnl = r
        d0 = W.dayv[i]
        rows.append((IDX[i], IDX[j], side, (pnl - SP[i]) / risk, d0))
    T = pd.DataFrame(rows, columns=["進場時間", "出場時間", "dir", "R", "day"])
    T = T.join(st_, on="day")
    T["ma20_al"] = np.where(T.dir > 0, T.up20 == True, T.up20 == False)
    T["slope_al"] = T.slope == T.dir
    T["dist_al"] = T.dist * T.dir
    T["train"] = T.進場時間 < "2025-01-01"
    T["w"] = 1.0
    return T.sort_values("進場時間").reset_index(drop=True)


def mdd(x):
    d = (x.R * x.w).groupby(x.進場時間.dt.date).sum()
    e = d.cumsum()
    return (e.cummax() - e).max(), d


def row(x, lab):
    out = []
    for part in (x[x.train], x[~x.train], x):
        m, d = mdd(part)
        tot = (part.R * part.w).sum()
        out.append((tot, m, tot / m if m > 0 else np.nan))
    nday = len(x) / x.進場時間.dt.date.nunique()
    ry = (x.R * x.w).sum() / (x.進場時間.max() - x.進場時間.min()).days * 365
    return (f"{lab:34s} {nday:5.1f} {(x.R * x.w).sum() / x.w.sum():+.3f} | "
            f"{out[0][0]:+6.0f} {out[0][1]:5.0f} {out[0][2]:5.1f} | {out[1][0]:+6.0f} {out[1][1]:5.0f} {out[1][2]:5.1f} | "
            f"{out[2][0]:+6.0f} {out[2][1]:5.0f} {out[2][2]:5.1f}  年化R/MDD {ry / out[2][1]:4.1f}")


HDR = (f"{'設定':34s} {'筆/天':>5s} {'每筆R':>6s} | {'訓練總R':>6s} {'MDD':>5s} {'比值':>5s} | {'驗證總R':>6s} {'MDD':>5s} {'比值':>5s} | "
       f"{'全期總R':>6s} {'MDD':>5s} {'比值':>5s}")

# ---------- 基準 ----------
T = build()
base = cap_total(T, 8)
p("第三十一部分：降低回撤、提高 總獲利/MDD（A+B，停損 2 ATR15，RR 3，上限 8，每筆 1R）")
p("比值 = 總 R / MDD（R，日結）；訓練 / 驗證 各自計算\n")
p(HDR)
p(row(base, "基準"))

# ---------- 1. 趨勢狀態：訓練期分組 ----------
p("\n=== 1. 日線趨勢狀態（先看訓練期每筆 R 分組；驗證期同切點）===")
for f in ("er10", "dist_al"):
    tr = T[T.train]
    qs = np.nanquantile(tr[f], [0.2, 0.4, 0.6, 0.8])
    bins = [-np.inf, *qs, np.inf]
    a = tr.groupby(pd.cut(tr[f], bins), observed=True).R.mean()
    b = T[~T.train].groupby(pd.cut(T[~T.train][f], bins), observed=True).R.mean()
    p(f"  {f:8s} 切點 {np.round(qs, 2)}  訓練 " + " ".join(f"{v:+.3f}" for v in a) + "  | 驗證 " + " ".join(f"{v:+.3f}" for v in b))
for f, lab in (("ma20_al", "也在 20 日均線同側"), ("slope_al", "50 日均線斜率同向")):
    for v in (True, False):
        a, b = T[T.train & (T[f] == v)].R, T[~T.train & (T[f] == v)].R
        p(f"  {lab}={'是' if v else '否'}：訓練 {a.mean():+.3f}（{len(a)}） 驗證 {b.mean():+.3f}（{len(b)}）")

p("\n" + HDR)
er_cut = np.nanquantile(T[T.train].er10, 0.2)
dist_hi = np.nanquantile(T[T.train].dist_al, 0.8)
for lab, m in [("er10 > 訓練期 20% 分位", T.er10 > er_cut),
               ("也在 20 日均線同側", T.ma20_al),
               ("50 日均線斜率同向", T.slope_al),
               ("離 50MA 不太遠（< 訓練期 80% 分位）", T.dist_al < dist_hi),
               ("20MA 同側 + 斜率同向", T.ma20_al & T.slope_al)]:
    p(row(cap_total(T[m.values], 8), lab))

# ---------- 2. 保本停損 ----------
p("\n=== 2. 保本停損 ===")
for be in (1.0, 1.5, 2.0):
    p(row(cap_total(build(be=be), 8), f"浮盈 {be}R 後移到保本"))


# ---------- 3. 部位控制 ----------
def daily_stop(df, lim, n=8):
    keep, open_ = [], []
    real = {}
    for r in df.itertuples():
        open_ = [(x, rr) for x, rr in open_ if x > r.進場時間]
        dloss = sum(v for (x, v, d) in real.get(r.day, []) if x <= r.進場時間)
        if dloss <= -lim or len(open_) >= n:
            continue
        open_.append((r.出場時間, r.R))
        real.setdefault(r.day, []).append((r.出場時間, r.R, r.day))
        keep.append(r.Index)
    return df.loc[keep]


def eq_filter(df, look, mode, n=8):
    """策略本身（上限 n 下的影子交易）的已實現累積 R 低於其最近 look 筆的均值 → 新單權重 0.5 或 0。"""
    sh = cap_total(df, n).copy()
    ex = sh.sort_values("出場時間")
    cum = ex.R.cumsum().values
    ma = pd.Series(cum).rolling(look, min_periods=look).mean().values
    ex_t = ex.出場時間.values
    w = []
    for r in sh.itertuples():
        k = np.searchsorted(ex_t, np.datetime64(r.進場時間), side="right") - 1
        below = k >= look and cum[k] < ma[k]
        w.append((0.5 if mode == "half" else 0.0) if below else 1.0)
    sh["w"] = w
    return sh[sh.w > 0] if mode == "skip" else sh


p("\n=== 3. 部位控制 ===")
for lim in (2, 3, 5):
    p(row(daily_stop(T, lim), f"當日已實現虧損 −{lim}R 後停止開新單"))
for look in (20, 50, 100):
    for mode in ("half", "skip"):
        p(row(eq_filter(T, look, mode), f"權益 < {look} 筆均線時{'減半' if mode == 'half' else '不下'}"))
p("\n=== 4. 組合 + 參數敏感度 ===")
p(HDR)
TS = T[T.slope_al.values]
p(row(cap_total(TS, 8), "斜率同向"))
for look in (30, 40, 50, 60, 75):
    p(row(eq_filter(T, look, "skip"), f"權益 < {look} 筆均線不下"))
    p(row(eq_filter(TS, look, "skip"), f"斜率同向 + 權益 < {look} 筆均線不下"))
for n in (3, 5, 12):
    p(row(eq_filter(TS, 50, "skip", n=n), f"斜率同向 + 權益<50 不下，上限 {n}"))
open("results_part31.txt", "w").write("\n".join(lines) + "\n")
