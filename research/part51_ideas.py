"""第五十一部分：還有哪裡可以優化？在 M1 2023/6~2026/10 上檢查（定案版為基準）。

（以下沿用第五十部分的資料處理）
第五十部分：M1 2023/6 ~ 2026/10 全期（新上傳 2023/6~2024/9，與既有 M1 合併；重疊 28,681 根完全相同）。

2023/6 ~ 2024/8 是 M1 第一次看到的期間（M15 顯示這段是區間盤、策略不賺）。
A. M1 定案版（OW + IB 寬度 + 2R 保本 + 分倉 3/2/1 + 最多 5 筆）逐期
B. 對照：無寬度篩選、全部 0.01 手、不保本
C. 候選 F（只加碼在賺錢部位）、G（前 20 天日內單邊比例中位 < 0.4 暫停）
D. 依突破快慢分組（驗證分倉）
"""
import numpy as np
import pandas as pd
from oos_engine import signals, simulate
from part27_ow import data, pf, prev_profiles

out = open("results_part51.txt", "w")
def P(s=""): print(s); out.write(s + "\n"); out.flush()
PER = [("23H2", "2023-06-01", "2023-12-31"), ("24H1", "2024-01-01", "2024-06-30"), ("24H2", "2024-07-01", "2024-12-31"),
       ("25H1", "2025-01-01", "2025-06-30"), ("25H2", "2025-07-01", "2025-12-31"), ("26H1", "2026-01-01", "2026-06-30"),
       ("26H2", "2026-07-01", "2026-12-31"), ("新 23/6~24/8", "2023-06-01", "2024-08-31"), ("全部", "2023-06-01", "2026-12-31")]
days, D, ARR = data("data/XAUUSD_M1_2023_2026.csv", 1)
PV = prev_profiles(days, ARR)


def manage(r, be=True):
    t, O, H, L, C, S, mins = ARR[r.day]
    s, e = r.side, r.lvl
    st = r.ibl if s == 1 else r.ibh
    R = s * (e - st); best = e; done = False
    for k in range(r.i, len(C)):
        if (L[k] <= st) if s == 1 else (H[k] >= st):
            return (min(O[k], st) if s == 1 else max(O[k], st)), t[k] + pd.Timedelta(minutes=1)
        best = max(best, H[k]) if s == 1 else min(best, L[k])
        if be and not done and s * (best - e) >= 2 * R:
            done = True; st = e
    return C[-1], t[-1] + pd.Timedelta(minutes=1)


def cap(T, n=5):
    keep, open_ = [], []
    for r in T.sort_values("t_in").itertuples():
        open_ = [x for x in open_ if x > r.t_in]
        if len(open_) < n:
            keep.append(r.Index); open_.append(r.exit_time)
    return T.loc[keep]


W = {}
for d in days:
    t, O, H, L, C, Sp, mins = ARR[d]
    row = {}
    for h in np.arange(2, 11, 0.5):
        m = (mins >= h * 60) & (mins < h * 60 + 60)
        if m.sum() >= 48:
            row[h] = H[m].max() - L[m].min()
    W[d] = row
REL = pd.DataFrame(W).T.sort_index().rolling(20, min_periods=10).median().shift(1)
X = simulate(signals(days, D, ARR, 1, monday=True), ARR, 1)
X["ow"] = [(r.day in PV) and ((r.ibl > PV[r.day][1]) if r.side == 1 else (r.ibh < PV[r.day][2])) for r in X.itertuples()]
X["w_atr"] = (X.ibh - X.ibl) / X.atr
X["w_rel"] = [(r.ibh - r.ibl) / REL.at[r.day, r.h] if (r.day in REL.index and r.h in REL.columns and REL.at[r.day, r.h] > 0) else np.nan for r in X.itertuples()]
X["wok"] = (X.w_atr <= 0.4) & (X.w_rel >= 0.6)
TR = ((D.close - D.open).abs() / (D.high - D.low)).rolling(20).median().shift(1)


def build(mask, be=True):
    T = X[mask].sort_values(["t_in", "h"]).drop_duplicates(["t_in", "side"]).copy()
    ex = [manage(r, be) for r in T.itertuples()]
    T["exit_px"] = [a for a, b in ex]; T["exit_time"] = [b for a, b in ex]
    T["pnl"] = T.side * (T.exit_px - T.lvl) - T.spread; T["R"] = T.pnl / T.risk
    T["delay"] = (T.t_in - T.day - pd.to_timedelta(T.h + 1, unit="h")).dt.total_seconds() / 3600
    T["lots"] = np.where(T.delay <= 0.5, 3.0, np.where(T.delay <= 1, 2.0, 1.0))
    return T.sort_values("t_in").reset_index(drop=True)


alldays = [d for d in days if d >= pd.Timestamp("2023-06-02")]


def line(name, T, w=None):
    C = cap(T)
    lots = C.lots if w is None else w(C)
    C = C.assign(v=C.pnl * lots)
    cells = []
    for n, a, b in PER:
        x = C[(C.day >= a) & (C.day <= b)]
        if not len(x):
            continue
        xs = x.sort_values("exit_time"); eq = xs.v.cumsum(); dd = (eq - eq.cummax()).min()
        dl = x.groupby("day").v.sum().reindex([d for d in alldays if pd.Timestamp(a) <= d <= pd.Timestamp(b)], fill_value=0)
        cells.append(f"{n} {x.R.mean():+.2f}R/{x.v.sum():+,.0f}$/DD{dd:,.0f}/S{dl.mean() / dl.std() * np.sqrt(250):.1f}")
    P(f"  {name:26s} " + "  ".join(cells))
    return C



fin = build(X.ow & X.wok)
P("第五十一部分：優化方向掃描（M1 2023/6~2026/10；每格 = 每筆R / 美元 / 該期回撤 / 日Sharpe）")
P("\n1. 基準")
line("定案", fin)
P("\n2. 多空")
line("只做多", fin[fin.side == 1]); line("只做空", fin[fin.side == -1])
P("\n3. IB 起點時段")
for a, b in [(2, 4), (4, 6), (6, 8), (8, 11)]:
    line(f"IB {a:02d}~{b:02d} 點", fin[(fin.h >= a) & (fin.h < b)])
line("不做 08 點後的 IB", fin[fin.h < 8])
P("\n4. 同時持倉上限")
for n in (3, 8, 99):
    C = cap(fin, n)
    line(f"上限 {n} 筆", C) if False else None
    v = (C.pnl * C.lots); e = C.assign(v=v).sort_values("exit_time").v.cumsum()
    dl = C.assign(v=v).groupby("day").v.sum().reindex(alldays, fill_value=0)
    P(f"  上限 {n:2d} 筆  {len(C)}筆 {C.R.mean():+.3f}R {v.sum():+,.0f}$ 回撤{(e - e.cummax()).min():+,.0f} Sharpe{dl.mean() / dl.std() * np.sqrt(250):.2f}")
P("\n5. 大方向（只用過去資料）：做的方向與「前一日收盤 vs 過去 N 日收盤均線」一致才做")
for n in (10, 20, 50):
    ma = D.close.rolling(n).mean()
    trend = np.sign(D.close - ma).shift(1)
    line(f"順 {n} 日均線", fin[fin.day.map(trend) == fin.side])
    line(f"逆 {n} 日均線", fin[fin.day.map(trend) == -fin.side])
P("\n6. 星期")
for k, nm in enumerate(["一", "二", "三", "四", "五"]):
    line(f"星期{nm}", fin[fin.day.dt.dayofweek == k])
P("\n7. 候選 E（反向先突破也做，含寬度篩選）")
q = np.array([(1 if r.ibl > PV[r.day][1] else (-1 if r.ibh < PV[r.day][2] else 0)) if r.day in PV else 0 for r in X.itertuples()])
X["qual"] = q
opp = build((X.qual != 0) & (X.side == -X.qual) & X.wok)
line("反向單本身", opp)
line("定案 + 反向單", build(((X.qual != 0) & X.wok)))
out.close()
