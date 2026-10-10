"""第三十三部分：M1 版 OW 的交易曲線（2026/1~10/8）。
現行（收盤出場）與「3R 啟動、距 2R 移動停損」；帳戶 1 萬美元、每筆 0.25% 風險（每 R = 25 美元）、不複利。
含逐分鐘浮動權益（持倉中的未實現損益，用 K 棒最差價）。"""
import numpy as np, pandas as pd
from part27_ow import data
from part32_constraints import manage

X = pd.read_csv("part31_m1_trades.csv", parse_dates=["day", "t_in", "exit_time"]).sort_values("t_in").reset_index(drop=True)
days, D, ARR = data("data/XAUUSD_M1_2026.csv", 1)
X["R_trail"] = [manage(r, ARR, 1, trail=(3, 2))[0] for r in X.itertuples()]

# 逐分鐘浮動（現行版）
rows = []
for n, r in enumerate(X.itertuples()):
    t, O, H, L, C, S, mins = ARR[r.day]
    j = int(np.searchsorted(t, r.exit_time - pd.Timedelta(minutes=1)))
    if j > r.i:
        worst = r.side * ((L if r.side == 1 else H)[r.i:j] - r.lvl) / r.risk
        rows.append(pd.Series(worst, index=t[r.i:j], name=n))
U = pd.concat(rows, axis=1).sort_index()
openR = U.sum(axis=1, min_count=1).fillna(0)
real = X.groupby("exit_time").R.sum().sort_index()
ix = openR.index.union(real.index)
mtm = (real.reindex(ix, fill_value=0).cumsum() + openR.reindex(ix, fill_value=0)) * 25 + 10000
closed = real.cumsum() * 25 + 10000
trail = X.groupby("exit_time").R_trail.sum().sort_index().cumsum() * 25 + 10000

def dd(s): return ((s - s.cummax()) / s.cummax() * 100)
lines = [f"M1 OW 2026/1~10/8：{len(X)} 筆",
         f"  現行：期末 {closed.iloc[-1]:,.0f}（{(closed.iloc[-1] / 1e4 - 1) * 100:+.1f}%），平倉最大回撤 {dd(closed).min():.1f}%，含浮動最大回撤 {dd(mtm).min():.1f}%",
         f"  3R/2R 移動停損：期末 {trail.iloc[-1]:,.0f}（{(trail.iloc[-1] / 1e4 - 1) * 100:+.1f}%），平倉最大回撤 {dd(trail).min():.1f}%",
         f"  最長水下期（平倉，現行）：{(lambda s: (s.index.to_series().groupby((s >= s.cummax()).cumsum()).agg(lambda x: (x.max() - x.min()).days)).max())(closed)} 天"]
print("\n".join(lines)); open("results_part33.txt", "w").write("\n".join(lines) + "\n")

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"axes.grid": True, "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(3, 1, figsize=(13, 11), sharex=True, gridspec_kw=dict(height_ratios=[2.4, 1, 1]))
m = mtm.resample("30min").last().dropna()
ax[0].plot(m.index, m.values, color="#9bb7d6", lw=.8, label="equity incl. open positions (minute, worst price)")
ax[0].plot(closed.index, closed.values, color="#3b6ea8", lw=1.6, drawstyle="steps-post", label="closed trades — current rules")
ax[0].plot(trail.index, trail.values, color="#2e8b57", lw=1.3, drawstyle="steps-post", label="closed trades — with 3R/2R trailing stop")
ax[0].axhline(10000, color="#888", lw=.8)
ax[0].set_ylabel("account (USD, 0.25% risk per trade)"); ax[0].legend(loc="upper left")
ax[0].set_title("XAUUSD OW strategy on M1 — 2026-01 to 2026-10 (no data 4/3-4/14)")
d = dd(mtm).resample("30min").min().dropna()
ax[1].fill_between(d.index, d.values, 0, color="#c0392b", alpha=.45, label="drawdown incl. open")
ax[1].plot(dd(closed).index, dd(closed).values, color="#7a1f15", lw=1, drawstyle="steps-post", label="drawdown closed")
ax[1].set_ylabel("drawdown %"); ax[1].legend(loc="lower left")
dr = X.groupby("day").R.sum() * 0.25
ax[2].bar(dr.index, dr.values, width=0.8, color=["#2e8b57" if v > 0 else "#c0392b" for v in dr.values])
ax[2].set_ylabel("daily P&L %")
plt.tight_layout(); plt.savefig("part33_m1_curve.png", dpi=110)
