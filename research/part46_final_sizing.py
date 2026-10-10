"""第四十六部分：最終規則加入「依突破快慢分倉」後重跑交易內容。

分倉：IB 結束到突破進場 ≤ 0.5h → 0.03 手；0.5~1h → 0.02 手；1~3h → 0.01 手。
交易 = part37（OW + 2R 保本）；最多同時 5 筆（以筆數計，與手數無關）。
輸出：results_part46.txt、part46_trades_{M1,M5,M15}.csv、part46_sizing.png
"""
import numpy as np
import pandas as pd
from part27_ow import data

out = open("results_part46.txt", "w")
def P(s=""): print(s); out.write(s + "\n"); out.flush()
SRC = {"M1": "data/XAUUSD_M1_2026.csv", "M5": "data/XAUUSD_M5_2025_2026.csv", "M15": "data/XAUUSD_M15_2023_2026.csv"}


def cap(T, n=5):
    keep, open_ = [], []
    for r in T.sort_values("t_in").itertuples():
        open_ = [x for x in open_ if x > r.t_in]
        if len(open_) < n:
            keep.append(r.Index); open_.append(r.exit_time)
    return T.loc[keep]


def streak(x):
    return int(x.groupby((x != x.shift()).cumsum()).sum().max())


ALL = {}
for tf, path in SRC.items():
    T = pd.read_csv(f"part37_trades_{tf}.csv", parse_dates=["day", "t_in", "exit_time"]).sort_values("t_in").reset_index(drop=True)
    T["delay"] = (T.t_in - T.day - pd.to_timedelta(T.h + 1, unit="h")).dt.total_seconds() / 3600
    T["lots"] = np.where(T.delay <= 0.5, 0.03, np.where(T.delay <= 1, 0.02, 0.01))
    T["usd"] = T.pnl * T.lots * 100                                  # 0.01 手 = 1 盎司
    C = cap(T).sort_values("exit_time").copy()
    days = data(path, {"M1": 1, "M5": 5, "M15": 15}[tf])[0]
    alldays = [d for d in days if d >= C.day.min()]
    ALL[tf] = C
    eq = C.usd.cumsum(); eq1 = C.pnl.cumsum()
    dl = C.groupby("day").usd.sum().reindex(alldays, fill_value=0); dl1 = C.groupby("day").pnl.sum().reindex(alldays, fill_value=0)
    ev = sorted([(a, l) for a, l in zip(C.t_in, C.lots)] + [(b, -l) for b, l in zip(C.exit_time, C.lots)], key=lambda z: (z[0], z[1])); cur = mx = 0
    for _, l in ev:
        cur += l; mx = max(mx, cur)
    P("\n" + "=" * 120 + f"\n[{tf} 最終版：OW + 2R 保本 + 依突破快慢分倉、最多 5 筆]  {C.day.min():%Y-%m-%d} ~ {C.day.max():%Y-%m-%d}")
    P(f"  交易 {len(C)} 筆；手數分佈 0.03 × {(C.lots == 0.03).sum()}、0.02 × {(C.lots == 0.02).sum()}、0.01 × {(C.lots == 0.01).sum()}；平均 {C.lots.mean():.3f} 手；同時最多 {mx:.2f} 手")
    for l, lab in [(0.03, "≤0.5h"), (0.02, "0.5~1h"), (0.01, "1~3h")]:
        g = C[C.lots == l]
        P(f"   {lab:7s}（{l} 手）{len(g):4d} 筆 勝率 {(g.R > 0.05).mean() * 100:3.0f}% 每筆 {g.R.mean():+.3f}R 合計 {g.usd.sum():+7,.0f} 美元")
    P(f"  勝率 {(C.R > 0.05).mean() * 100:.1f}%  每筆 {C.R.mean():+.3f}R  最長連虧 {streak((C.R <= 0.05).astype(int))} 筆")
    P(f"  分倉後：總 {C.usd.sum():+,.0f} 美元  平倉最大回撤 {(eq - eq.cummax()).min():+,.0f}  報酬÷回撤 {C.usd.sum() / -(eq - eq.cummax()).min():.1f}  "
      f"最差單日 {dl.min():+,.0f}  最好單日 {dl.max():+,.0f}  日 Sharpe {dl.mean() / dl.std() * np.sqrt(250):.2f}")
    P(f"  對照全部 0.01：總 {C.pnl.sum():+,.0f}  回撤 {(eq1 - eq1.cummax()).min():+,.0f}  報酬÷回撤 {C.pnl.sum() / -(eq1 - eq1.cummax()).min():.1f}  "
      f"最差單日 {dl1.min():+,.0f}  日 Sharpe {dl1.mean() / dl1.std() * np.sqrt(250):.2f}")
    if tf == "M15":
        P("  逐年（分倉 / 全 0.01）：" + "  ".join(f"{y} {g.usd.sum():+,.0f} / {g.pnl.sum():+,.0f}" for y, g in C.groupby(C.day.dt.year)))
    else:
        P("  逐月美元：" + "  ".join(f"{p.month}月 {g.usd.sum():+,.0f}" for p, g in C.groupby(C.day.dt.to_period("M"))))
    C.to_csv(f"part46_trades_{tf}.csv", index=False)

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
for a, tf in zip(ax, ["M1", "M5", "M15"]):
    C = ALL[tf]
    a.plot(C.exit_time, C.pnl.cumsum(), color="#999", lw=1.1, label="all 0.01 lot")
    a.plot(C.exit_time, C.usd.cumsum(), color="#2e8b57", lw=1.5, label="0.03 / 0.02 / 0.01 by breakout speed")
    a.set_title(f"{tf} OW + 2R BE, max 5 open (USD)"); a.legend(); a.grid(alpha=.3); a.tick_params(axis="x", rotation=30)
plt.tight_layout(); plt.savefig("part46_sizing.png", dpi=110)
out.close()
