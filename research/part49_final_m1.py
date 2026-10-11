"""第四十九部分：M1 定案版（OW + IB 寬度兩端篩選 + 2R 保本 + 依突破快慢分倉 + 最多 5 筆），M1 2024/9 ~ 2026/10。

IB 寬度篩選（候選 B 採用）：
  IB 寬 ≤ 0.4 × ATR10，且 IB 寬 ≥ 0.6 × 前 20 個交易日同一起點 IB 寬的中位數（至少 10 天資料）。
先篩（OW + 寬度）再去重（同一根 K 棒同方向一筆，取起點最早的 IB）。
輸出：results_part49.txt、part49_m1_final_trades.csv、part49_m1_final.png
"""
import numpy as np
import pandas as pd
from oos_engine import signals, simulate
from part27_ow import data, pf, prev_profiles

out = open("results_part49.txt", "w")
def P(s=""): print(s); out.write(s + "\n"); out.flush()
PER = [("24H2", "2024-09-01", "2024-12-31"), ("25H1", "2025-01-01", "2025-06-30"), ("25H2", "2025-07-01", "2025-12-31"),
       ("26H1", "2026-01-01", "2026-06-30"), ("26H2", "2026-07-01", "2026-12-31")]
days, D, ARR = data("data/XAUUSD_M1_2024_2026.csv", 1)
PV = prev_profiles(days, ARR)


def manage(r):
    t, O, H, L, C, S, mins = ARR[r.day]
    s, e = r.side, r.lvl
    st = r.ibl if s == 1 else r.ibh
    R = s * (e - st); best = e; be = False
    for k in range(r.i, len(C)):
        if (L[k] <= st) if s == 1 else (H[k] >= st):
            return (min(O[k], st) if s == 1 else max(O[k], st)), t[k] + pd.Timedelta(minutes=1), ("保本" if be else "停損"), s * (best - e) / R
        best = max(best, H[k]) if s == 1 else min(best, L[k])
        if not be and s * (best - e) >= 2 * R:
            be = True; st = e
    return C[-1], t[-1] + pd.Timedelta(minutes=1), "收盤", s * (best - e) / R


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

S = signals(days, D, ARR, 1, monday=True)
X = simulate(S, ARR, 1)
X["ow"] = [(r.day in PV) and ((r.ibl > PV[r.day][1]) if r.side == 1 else (r.ibh < PV[r.day][2])) for r in X.itertuples()]
X["w_atr"] = (X.ibh - X.ibl) / X.atr
X["w_rel"] = [(r.ibh - r.ibl) / REL.at[r.day, r.h] if (r.day in REL.index and r.h in REL.columns and REL.at[r.day, r.h] > 0) else np.nan for r in X.itertuples()]
X["wok"] = (X.w_atr <= 0.4) & (X.w_rel >= 0.6)

res = {}
for lab, mask in [("定案（OW + 寬度）", X.ow & X.wok), ("不含寬度篩選（前版）", X.ow)]:
    T = X[mask].sort_values(["t_in", "h"]).drop_duplicates(["t_in", "side"]).copy()
    ex = [manage(r) for r in T.itertuples()]
    T["exit_px"] = [a[0] for a in ex]; T["exit_time"] = [a[1] for a in ex]; T["exit_reason"] = [a[2] for a in ex]; T["mfe_R"] = [a[3] for a in ex]
    T["pnl"] = T.side * (T.exit_px - T.lvl) - T.spread; T["R"] = T.pnl / T.risk
    T["delay"] = (T.t_in - T.day - pd.to_timedelta(T.h + 1, unit="h")).dt.total_seconds() / 3600
    T["lots"] = np.where(T.delay <= 0.5, 0.03, np.where(T.delay <= 1, 0.02, 0.01))
    T["usd"] = T.pnl * T.lots * 100
    res[lab] = cap(T).sort_values("exit_time").reset_index(drop=True)

alldays = [d for d in days if d >= pd.Timestamp("2024-09-03")]
for lab, C in res.items():
    eq = C.usd.cumsum(); dd = eq - eq.cummax()
    dl = C.groupby("day").usd.sum().reindex(alldays, fill_value=0)
    i1 = int(dd.idxmin()); i0 = int(eq[:i1 + 1].idxmax())
    P("\n" + "=" * 120 + f"\n[M1 {lab}]  2024/9/3 ~ 2026/10/8，{len(alldays)} 個交易日")
    P(f"  交易 {len(C)} 筆（多 {(C.side == 1).sum()} / 空 {(C.side == -1).sum()}）、有交易的日子 {C.day.nunique()} 天；手數 0.03×{(C.lots == 0.03).sum()}、0.02×{(C.lots == 0.02).sum()}、0.01×{(C.lots == 0.01).sum()}")
    P(f"  勝率 {(C.R > 0.05).mean() * 100:.1f}%  每筆 {C.R.mean():+.3f}R  PF {pf(C.R):.2f}  出場 停損 {(C.exit_reason == '停損').mean() * 100:.0f}% / 保本 {(C.exit_reason == '保本').mean() * 100:.0f}% / 收盤 {(C.exit_reason == '收盤').mean() * 100:.0f}%")
    P(f"  總 {C.usd.sum():+,.0f} 美元  平倉最大回撤 {dd.min():+,.0f}（{C.exit_time[i0]:%Y-%m-%d} → {C.exit_time[i1]:%Y-%m-%d}）  報酬÷回撤 {C.usd.sum() / -dd.min():.1f}")
    P(f"  最差單日 {dl.min():+,.0f}  最好單日 {dl.max():+,.0f}  日 Sharpe {dl.mean() / dl.std() * np.sqrt(250):.2f}  正報酬月 {(C.groupby(C.day.dt.to_period('M')).usd.sum() > 0).mean() * 100:.0f}%")
    lose = (C.R <= 0.05).astype(int)
    l2 = (C.R < -0.05).astype(int)
    P(f"  最長連續沒賺 {lose.groupby((lose != lose.shift()).cumsum()).sum().max()} 筆（不算保本的最長連虧 {l2.groupby((l2 != l2.shift()).cumsum()).sum().max()} 筆）；平均持倉 {((C.exit_time - C.t_in).dt.total_seconds() / 3600).mean():.1f} 小時")
    P("  各期：" + "  ".join(f"{n} {len(x)}筆 {x.R.mean():+.2f}R PF{pf(x.R):.2f} {x.usd.sum():+,.0f}$" for n, a, b in PER for x in [C[(C.day >= a) & (C.day <= b)]] if len(x)))
    P("  各組：" + "  ".join(f"{l}手 {len(g)}筆 {g.R.mean():+.2f}R {g.usd.sum():+,.0f}$" for l, g in C.groupby("lots")))
    P("  逐月：" + "  ".join(f"{p.strftime('%y/%m')} {g.usd.sum():+,.0f}" for p, g in C.groupby(C.day.dt.to_period("M"))))
res["定案（OW + 寬度）"].to_csv("part49_m1_final_trades.csv", index=False)

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(12, 4.5))
for lab, col in [("不含寬度篩選（前版）", "#999"), ("定案（OW + 寬度）", "#2e8b57")]:
    C = res[lab]; ax.plot(C.exit_time, C.usd.cumsum(), color=col, lw=1.5, label="final (OW + IB width)" if col != "#999" else "previous (no width filter)")
ax.set_title("M1 OW, 2R BE, 0.03/0.02/0.01 lots by breakout speed, max 5 open (USD)"); ax.legend(); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig("part49_m1_final.png", dpi=110)
out.close()
