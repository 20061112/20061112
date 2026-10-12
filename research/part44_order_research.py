"""第四十四部分：(1) 篩選順序 (2) 從 L0 重新研究篩選依據。

(1) A / B / 斜率 是純條件（互換順序結果相同）；會受順序影響的是「上限 8」與「權益濾網（影子）」：
    影子權益曲線用哪一個訊號流算（全部 L0、套完條件後、套完上限後），結果不同。
    → 列出 8 種條件組合 × 4 種影子位置（不用 / 用 L0 全流 / 條件後、上限前 / 上限後）。
(2) 從 L0（全部突破，多空）出發，只用訓練期（2023/8~2024/12）做「逐步加入」：
    每一步從 ~25 個因子 × 9 個切點（保留 ≥ 或 ≤）中挑出讓訓練期「日 Sharpe」最高的一條，
    直到改善 < 0.05 或已 6 條；每一步同時列出驗證期（2025~2026/10）的結果。最後套上限 8 + 影子 與最終版比較。
R 一律扣點差 + 每筆 0.3 美元滑價；美元以 0.02 口計；MDD = 逐分鐘含浮動。
"""
import numpy as np
import pandas as pd
import part29_rr_grid as G
from part30_max_positions import cap_total
import part24_prev_bar_breakout as B

W, IDX = G.W, G.IDX
C, H, Lo, SP, dayv = W.C, W.H, W.Lo, W.SP, W.dayv
O1 = W.O1
SLIP = 0.3
d = B.d
dl = W.B.daily
sma50 = dl.C.rolling(50).mean()
atrd = (dl.H - dl.L).rolling(10).mean()
dd = pd.DataFrame({"pc": dl.C, "ma": sma50, "ma10": sma50.shift(10), "atr": atrd,
                   "er10": (dl.C - dl.C.shift(10)).abs() / dl.C.diff().abs().rolling(10).sum(),
                   "pret": dl.C - dl.O, "prng": dl.H - dl.L, "run20": dl.C - dl.C.shift(20),
                   "atr_pct": atrd.rolling(250, min_periods=60).rank(pct=True)}).shift(1)
dd["o"] = dl.O

# ---------- 事件（L0 全部）----------
rows = []
for i, L, side, a in G.ev:
    s = dd.loc[dayv[i]]
    if not (np.isfinite(s.ma) and np.isfinite(s.ma10) and np.isfinite(s.atr)):
        continue
    risk = 2.0 * a
    r = G.sim_detail(i, L, side, L - side * risk, L + side * 3 * risk)
    if r is None:
        continue
    j, px, why = r
    rows.append((IDX[i], IDX[j], i, j, side, L, (px - L) * side - SP[i], risk, a))
E = pd.DataFrame(rows, columns=["進場時間", "出場時間", "i", "j", "dir", "entry", "net", "risk", "atr15"])
E = E[E.進場時間 >= "2023-08-24"].sort_values("進場時間").reset_index(drop=True)
E["R"] = (E.net - SLIP) / E.risk
E["train"] = E.進場時間 < "2025-01-01"
E["date"] = E.進場時間.dt.normalize()
i, dirv, px = E.i.values, E.dir.values, E.entry.values
S_ = dd.reindex(E.date).reset_index(drop=True)

# ---------- 因子 ----------
pid = B.period_ids(15)
g = pd.DataFrame({"pid": pid, "H": H, "L": Lo, "O": O1, "C": C, "V": d.tickvol.values.astype(float)})
b = g.groupby("pid").agg(H=("H", "max"), L=("L", "min"), O=("O", "first"), C=("C", "last"), V=("V", "sum"))
rng = b.H - b.L
f15 = pd.DataFrame({"sma20": b.C.rolling(20).mean(), "sd20": b.C.rolling(20).std(), "atr20": rng.rolling(20).mean(),
                    "atr5": rng.rolling(5).mean(), "bbw": (4 * b.C.rolling(20).std() / b.C.rolling(20).mean()).rolling(500, min_periods=100).rank(pct=True),
                    "vr": b.V / b.V.rolling(20).mean(), "prng": rng, "pbody": b.C - b.O, "ret4": b.C - b.C.shift(4), "ret16": b.C - b.C.shift(16)}).shift(1)
F15 = f15.iloc[pid[i]].reset_index(drop=True)
first = pd.Series(np.arange(len(d))).groupby(dayv).transform("first").values
f0 = first[i]
cum_path = np.concatenate([[0], np.cumsum(np.abs(np.diff(C)))])
hi_s = pd.Series(H).groupby(dayv).cummax().values
lo_s = pd.Series(Lo).groupby(dayv).cummin().values
im1 = np.maximum(i - 1, f0)
pos = (px - lo_s[im1]) / np.maximum(hi_s[im1] - lo_s[im1], 1e-9)
FE = pd.DataFrame({
    "A 順50MA": ((S_.pc - S_.ma).values * dirv > 0).astype(float),
    "斜率同向": ((S_.ma - S_.ma10).values * dirv > 0).astype(float),
    "離50MA/ATR(dir)": (S_.pc - S_.ma).values / S_.atr.values * dirv,
    "斜率強度(dir)": (S_.ma - S_.ma10).values / S_.atr.values * dirv,
    "今日漲跌/ATR(dir)": (px - S_.o.values) / S_.atr.values * dirv,
    "前20日漲/ATR(dir)": S_.run20.values / S_.atr.values * dirv,
    "前日漲跌/ATR(dir)": S_.pret.values / S_.atr.values * dirv,
    "前日波幅/ATR": S_.prng.values / S_.atr.values,
    "日ER10": S_.er10.values,
    "日ATR百分位": S_.atr_pct.values,
    "z20_m15(dir)": (px - F15.sma20.values) / F15.sd20.values * dirv,
    "ATR15/ATR日": F15.atr20.values / S_.atr.values,
    "ATR15 5/20": F15.atr5.values / F15.atr20.values,
    "BB寬度百分位": F15.bbw.values,
    "量比": F15.vr.values,
    "前根波幅/ATR15": F15.prng.values / F15.atr20.values,
    "前根實體(dir)": F15.pbody.values / F15.atr20.values * dirv,
    "1h動能(dir)": F15.ret4.values / F15.atr20.values * dirv,
    "4h動能(dir)": F15.ret16.values / F15.atr20.values * dirv,
    "今日ER": np.abs(C[im1] - O1[f0]) / (cum_path[i] - cum_path[f0] + 1e-9),
    "今日區間位置(dir)": np.where(dirv > 0, pos, 1 - pos),
    "今日波幅/ATR": (hi_s[im1] - lo_s[im1]) / S_.atr.values,
    "小時": E.進場時間.dt.hour.values.astype(float),
    "星期": E.進場時間.dt.dayofweek.values.astype(float),
    "點差/停損": SP[i] / E.risk.values,
})
E["A"] = FE["A 順50MA"].values == 1
E["slope"] = FE["斜率同向"].values == 1
E["B"] = FE["今日漲跌/ATR(dir)"].values > -0.17
E["ext"] = FE["離50MA/ATR(dir)"].values

lines = []


def p(s=""):
    print(s)
    lines.append(s)


def eq_filter(df, look=50):
    ex = df.sort_values("出場時間")
    cum = ex.R.cumsum().values
    ma = pd.Series(cum).rolling(look, min_periods=look).mean().values
    et = ex.出場時間.values
    keep = np.ones(len(df), bool)
    for k_, t in enumerate(df.進場時間.values):
        k = np.searchsorted(et, t, side="right") - 1
        keep[k_] = not (k >= look and cum[k] < ma[k])
    return keep


def minute_mdd(df):
    n = len(IDX)
    D, Ee, R = np.zeros(n + 1), np.zeros(n + 1), np.zeros(n + 1)
    for r in df.itertuples():
        D[r.i] += r.dir * 2; D[r.j] -= r.dir * 2
        Ee[r.i] += r.dir * r.entry * 2; Ee[r.j] -= r.dir * r.entry * 2
        R[r.j] += (r.net - SLIP) * 2
    D, Ee, R = np.cumsum(D)[:n], np.cumsum(Ee)[:n], np.cumsum(R)[:n]
    eq = R + C * D - Ee
    return (np.maximum.accumulate(eq) - eq).max()


yrs = (E.進場時間.max() - E.進場時間.min()).days / 365


def usd_line(x, lab):
    pnl = (x.net - SLIP) * 2
    m = minute_mdd(x)
    tr = x.train.values

    def rat(z, pz):
        dz = pz.groupby(z.date).sum(); ez = dz.cumsum(); mz = (ez.cummax() - ez).max()
        return pz.sum() / mz if mz > 0 else np.nan
    p(f"  {lab:40s} {len(x) / x.date.nunique():5.1f} 筆/天 每筆 {x.R.mean():+.3f}R 總 {pnl.sum():+8,.0f} MDD {m:6,.0f} "
      f"年獲利/MDD {pnl.sum() / yrs / m:5.2f} | 訓練 {rat(x[tr], pnl[tr]):4.1f} 驗證 {rat(x[~tr], pnl[~tr]):4.1f}")


# ---------- (1) 順序 ----------
p("第四十四部分：篩選順序 + 從 L0 重新研究（0.02 口；點差 + 每筆 0.3 美元滑價）")
p("\n=== (1) 條件組合 × 影子（權益濾網）位置；全部最後都套上限 8（「無上限」另列）===")
SHADOW_L0 = eq_filter(E)  # 用 L0 全流算的影子狀態
for name, m in [("無條件", np.ones(len(E), bool)), ("A", E.A.values), ("B", E.B.values), ("斜率", E.slope.values),
                ("A+B", (E.A & E.B).values), ("A+斜率", (E.A & E.slope).values), ("B+斜率", (E.B & E.slope).values),
                ("A+B+斜率", (E.A & E.B & E.slope).values)]:
    x = E[m].reset_index(drop=True)
    usd_line(cap_total(x, 8), f"{name} → 上限8")
    usd_line(cap_total(E[m & SHADOW_L0].reset_index(drop=True), 8), f"{name} → 影子(L0全流) → 上限8")
    k1 = eq_filter(x)
    usd_line(cap_total(x[k1].reset_index(drop=True), 8), f"{name} → 影子(條件後) → 上限8")
    xc = cap_total(x, 8).reset_index(drop=True)
    usd_line(xc[eq_filter(xc)], f"{name} → 上限8 → 影子（現行順序）")

# ---------- (2) 從 L0 逐步挑篩選 ----------
p("\n=== (2) 從 L0 重新研究：只用訓練期逐步挑條件（目標 = 訓練期日 Sharpe）===")


def sharpe(x):
    dr = x.groupby("date").R.sum()
    allday = pd.Series(0.0, index=E.date[E.train].unique() if x.train.all() else E.date[~E.train].unique())
    dr = allday.add(dr, fill_value=0)
    return dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0


tr_mask = E.train.values


def greedy(exclude=()):
    keep = np.ones(len(E), bool)
    chosen = []
    cands = []
    for f in FE.columns:
        if f in exclude:
            continue
        v = FE[f].values
        vt = v[tr_mask & np.isfinite(v)]
        if f in ("A 順50MA", "斜率同向"):
            cands.append((f"{f} = 是", v == 1))
            continue
        for q in np.arange(0.1, 0.95, 0.1):
            c = np.quantile(vt, q)
            cands.append((f"{f} ≥ {c:.3g}", v >= c))
            cands.append((f"{f} ≤ {c:.3g}", v <= c))
    base = sharpe(E[tr_mask & keep])
    p(f"  第 0 步 L0：訓練 Sharpe {base:.2f} / 驗證 {sharpe(E[~tr_mask & keep]):.2f}，每筆 {E.R[tr_mask].mean():+.3f} / {E.R[~tr_mask].mean():+.3f}R")
    for step in range(1, 7):
        best = None
        for lab, m in cands:
            k2 = keep & m
            if k2[tr_mask].sum() < 0.15 * tr_mask.sum():
                continue
            s_ = sharpe(E[tr_mask & k2])
            if best is None or s_ > best[0]:
                best = (s_, lab, m)
        if best is None or best[0] < base + 0.05:
            break
        base = best[0]
        keep &= best[2]
        chosen.append(best[1])
        p(f"  第 {step} 步 + {best[1]:28s} 訓練 Sharpe {best[0]:.2f} / 驗證 {sharpe(E[~tr_mask & keep]):.2f}，"
          f"每筆 {E.R[tr_mask & keep].mean():+.3f} / {E.R[~tr_mask & keep].mean():+.3f}R，"
          f"每天 {keep[tr_mask].sum() / E.date[tr_mask].nunique():.1f} / {keep[~tr_mask].sum() / E.date[~tr_mask].nunique():.1f} 筆")
        xs = cap_total(E[keep].reset_index(drop=True), 8).reset_index(drop=True)
        usd_line(xs[eq_filter(xs)], f"    （到第 {step} 步為止 → 上限8 → 影子）")
    return keep


p("  [a] 全部候選因子")
keep = greedy()
p("  [b] 排除 星期 / 小時 / 點差（避免挑到行事曆巧合）")
keep_b = greedy(exclude=("星期", "小時", "點差/停損"))

p("\n=== 重新研究的條件 vs 最終版（都套上限 8 → 影子）===")
for nm, kk in (("[a]", keep), ("[b]", keep_b)):
    xn = cap_total(E[kk].reset_index(drop=True), 8).reset_index(drop=True)
    usd_line(xn, f"重新研究 {nm} → 上限8")
    usd_line(xn[eq_filter(xn)], f"重新研究 {nm} → 上限8 → 影子")
xf = cap_total(E[(E.A & E.B & E.slope).values].reset_index(drop=True), 8).reset_index(drop=True)
usd_line(xf[eq_filter(xf)], "最終版（A+B+斜率 → 上限8 → 影子）")
open("results_part44.txt", "w").write("\n".join(lines) + "\n")
