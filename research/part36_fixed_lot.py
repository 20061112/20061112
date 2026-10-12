"""第三十六部分：以 0.01 口（= 1 盎司）為單位的最終版 M15 突破績效（美元）。

A. 每筆固定 0.01 口
B. 加權版（第三十五部分）：基準 0.02 口；影子權益高於均線較多 → 0.03 口；已持倉 ≥ 4 → 0.01 口；兩者同時 → 0.02 口
   （0.01 口是最小單位，所以加權版的基準要用 0.02 口）
滑價：0 / 每筆 0.3 美元（每盎司）兩種。
MDD：日結、以及逐分鐘含浮動損益。
資料：final_m15_trades.csv（實際下單，已扣點差）；加權版需要 part34_X.pkl。
"""
import numpy as np
import pandas as pd
import part24_prev_bar_breakout as B

IDX = B.d.index
C = B.C
S = pd.read_csv("final_m15_signals.csv", parse_dates=["進場時間", "出場時間"])
X = pd.read_pickle("part34_X.pkl")
T = S[S.實際下單].copy()
XT = X[S.實際下單.values]
eqv = XT["權益濾網(累積R−均線)"].values
eq_cut = np.nanquantile(X[X.train]["權益濾網(累積R−均線)"], 0.5)
nopen = XT["進場時已持倉數"].values
w = np.where(eqv > eq_cut, 1.5, 1.0) * np.where(nopen >= 4, 0.5, 1.0)
T["lotB"] = np.round(0.02 * w / 0.01 + 1e-9).clip(1) * 0.01  # 0.01 / 0.02 / 0.03
T["i"] = IDX.get_indexer(T.進場時間)
T["j"] = IDX.get_indexer(T.出場時間)
T["dir"] = np.where(T.方向 == "多", 1, -1)
T["entry"] = T["進場價(=點位)"]
lines = []


def p(s=""):
    print(s)
    lines.append(s)


def minute_mdd(T, oz, slip):
    n = len(IDX)
    D, E, R = np.zeros(n + 1), np.zeros(n + 1), np.zeros(n + 1)
    for r, q in zip(T.itertuples(), oz):
        D[r.i] += r.dir * q; D[r.j] -= r.dir * q
        E[r.i] += r.dir * r.entry * q; E[r.j] -= r.dir * r.entry * q
        R[r.j] += (r.淨利_每盎司 - slip) * q
    D, E, R = np.cumsum(D)[:n], np.cumsum(E)[:n], np.cumsum(R)[:n]
    eq = R + C * D - E
    return (np.maximum.accumulate(eq) - eq).max()


def report(lab, oz, slip):
    pnl = (T.淨利_每盎司 - slip) * oz
    d = pnl.groupby(T.進場時間.dt.date).sum()
    e = d.cumsum()
    mdd_d = (e.cummax() - e).max()
    mdd_m = minute_mdd(T, oz, slip)
    yrs = (T.進場時間.max() - T.進場時間.min()).days / 365
    mon = pnl.groupby(T.進場時間.dt.to_period("M")).sum()
    lots = oz * 0.01
    conc = []
    ev = pd.concat([pd.Series(-lots, T.出場時間.values), pd.Series(lots, T.進場時間.values)]).sort_index(kind="stable")
    p(f"\n=== {lab}，滑價 {slip} 美元/盎司 ===")
    p(f"  筆數 {len(T)}，總損益 {pnl.sum():+,.0f} 美元，每年約 {pnl.sum() / yrs:+,.0f}，每月平均 {mon.mean():+,.0f}（正報酬 {(mon > 0).sum()}/{len(mon)} 個月）")
    p(f"  每筆平均 {pnl.mean():+.2f}，賺的平均 {pnl[pnl > 0].mean():+.2f}、賠的平均 {pnl[pnl <= 0].mean():+.2f}，單筆最大虧損 {pnl.min():+.2f}")
    p(f"  MDD 日結 {mdd_d:,.0f} / 逐分鐘含浮動 {mdd_m:,.0f} 美元；最差一天 {d.min():+,.0f}、最差一月 {mon.min():+,.0f}（{mon.idxmin()}）")
    p(f"  同時持倉最多 {ev.cumsum().max():.2f} 口")
    for y in (2023, 2024, 2025, 2026):
        m = (T.進場時間.dt.year == y).values
        dy = pnl[m].groupby(T.進場時間[m].dt.date).sum(); ey = dy.cumsum()
        p(f"    {y}: {m.sum():5d} 筆  損益 {pnl[m].sum():+7,.0f}  每筆 {pnl[m].mean():+.2f}  當年 MDD {(ey.cummax() - ey).max():6,.0f}")
    return mdd_m, pnl.sum() / yrs


p("第三十六部分：以 0.01 口（1 盎司）為單位 —— 最終版 M15 突破（2023/8 ~ 2026/10，美元）")
res = {}
for slip in (0.0, 0.3):
    res[("A", slip)] = report("A. 每筆固定 0.01 口", np.ones(len(T)), slip)
    res[("B", slip)] = report("B. 加權版（基準 0.02 口，0.01~0.03）", T.lotB.values / 0.01, slip)

p("\n=== 需要多少資金（用逐分鐘 MDD、滑價 0.3）===")
for k in ("A", "B"):
    mdd, ann = res[(k, 0.3)]
    p(f"  {k}: MDD {mdd:,.0f} 美元、年獲利約 {ann:+,.0f} → 想讓 MDD ≤ 20%：本金 ≥ {mdd / 0.2:,.0f}；≤ 30%：≥ {mdd / 0.3:,.0f}"
      f"（對應年報酬約 {ann / (mdd / 0.2):.0%} / {ann / (mdd / 0.3):.0%}，未複利）")
p("\n注意：固定口數時，停損的美元金額隨金價與波動變大（停損距離中位數 2023 3.7 → 2026 18.6 美元/盎司），"
  "損益與 MDD 大多來自 2025~2026；金價 4,000 時 0.01 口名目約 4,000 美元，保證金依券商槓桿而定（1:100 約 40 美元/0.01 口）。")
open("results_part36.txt", "w").write("\n".join(lines) + "\n")

# 固定風險時，0.01 口的最低資金需求
out = ["", "=== 改用固定風險時：0.01 口夠不夠細？ ===",
       "口數 = 每筆風險金額 ÷ (停損距離 × 100)；最小 0.01 口 → 每筆風險金額至少要 = 停損距離（美元/盎司）"]
for y in (2024, 2025, 2026):
    sd = T.停損距離[T.進場時間.dt.year == y]
    out.append(f"  {y}: 停損距離 中位數 {sd.median():.1f}、90% 分位 {sd.quantile(.9):.1f} 美元 → "
               f"每筆 0.1% 風險時，帳戶需 ≥ {sd.median() / 0.001:,.0f}（中位數）/ {sd.quantile(.9) / 0.001:,.0f}（90%）才不會被 0.01 口卡住")
print("\n".join(out))
open("results_part36.txt", "a").write("\n".join(out) + "\n")
