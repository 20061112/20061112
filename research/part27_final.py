"""第二十七部分 D：OW 定案統計與圖表（M15 2023~2026、M5 2025/5~2026/10）。"""
import numpy as np
import pandas as pd
from part27_ow import pf, PER

out = open("results_part27.txt", "a")
def P(s=""): print(s); out.write(s + "\n"); out.flush()
T15 = pd.read_csv("part27_ow_M15.csv", parse_dates=["day", "t_in", "exit_time"]).sort_values("t_in")
T5 = pd.read_csv("part27_ow_M5.csv", parse_dates=["day", "t_in", "exit_time"]).sort_values("t_in")
T15 = T15[T15.day >= "2023-03-01"]; T5 = T5[T5.day >= "2025-07-01"]

P("\n第二十七部分 D：OW 詳細統計")
for lab, T in [("M15 2023/3~2026/10", T15), ("M15 2024/7~2026/10", T15[T15.day >= "2024-07-01"]), ("M5 2025/7~2026/10", T5)]:
    r = T.R; lose = (r <= 0).astype(int)
    eq = (T.sort_values("exit_time").R * 0.25).cumsum()
    ev = sorted([(a, 1) for a in T.t_in] + [(b, -1) for b in T.exit_time], key=lambda x: (x[0], x[1])); cur = mx = 0
    for _, e in ev:
        cur += e; mx = max(mx, cur)
    hold = (T.exit_time - T.t_in).dt.total_seconds() / 3600
    mon = T.groupby(T.day.dt.to_period("M")).R.sum()
    P(f"  [{lab}] {len(T)} 筆，勝率 {(T.pnl > 0).mean() * 100:.1f}%，每筆 {r.mean():+.3f}R（中位 {r.median():+.2f}R），PF {pf(r):.2f}，"
      f"總 {r.sum():+.0f}R；最長連虧 {lose.groupby((lose != lose.shift()).cumsum()).sum().max()}；同時持倉最多 {mx}；"
      f"平均持倉 {hold.mean():.1f}h；正報酬月 {(mon > 0).mean() * 100:.0f}%（{len(mon)} 個月）；"
      f"每筆 0.25% 風險：總 {eq.iloc[-1]:+.0f}%、最大回撤 {(eq - eq.cummax()).min():.1f}%（平倉、不複利）")

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"axes.grid": True, "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(3, 1, figsize=(12, 11), gridspec_kw=dict(height_ratios=[2.2, 1, 1]))
e15 = T15.sort_values("exit_time"); e5 = T5.sort_values("exit_time")
ax[0].plot(e15.exit_time, e15.R.cumsum(), color="#3b6ea8", lw=1.3, label="M15 (2023-2026)")
base5 = e15[e15.exit_time < e5.exit_time.iloc[0]].R.sum()
ax[0].plot(e5.exit_time, base5 + e5.R.cumsum(), color="#c9824a", lw=1.1, label="M5 (2025/7-2026/10, same start level)")
for n, a, b in PER:
    if n in ("2023", "24H1"):
        ax[0].axvspan(pd.Timestamp(a), pd.Timestamp(b), color="#c0392b", alpha=.06)
ax[0].set_ylabel("cumulative R"); ax[0].legend(loc="upper left")
ax[0].set_title("OW: IB breakout only when the whole IB is outside prior-day value area (in trade direction)\nred shading = periods where it did not work (2023, 2024H1)")
mon = T15.groupby(T15.day.dt.to_period("M")).R.sum()
ax[1].bar(mon.index.to_timestamp(), mon.values, width=20, color=["#2e8b57" if v > 0 else "#c0392b" for v in mon.values])
ax[1].set_ylabel("R per month (M15)")
cells = []
for n, a, b in PER:
    x = T15[(T15.day >= a) & (T15.day <= b)]
    cells.append((n, x.R.mean(), (x.pnl > 0).mean() * 100, pf(x.R)))
xs = np.arange(len(cells))
ax[2].bar(xs - .2, [c[1] for c in cells], width=.4, color="#3b6ea8", label="avg R per trade")
ax2 = ax[2].twinx(); ax2.plot(xs, [c[2] for c in cells], "o-", color="#c9824a", label="win rate %"); ax2.set_ylabel("win %")
ax[2].set_xticks(xs); ax[2].set_xticklabels([c[0] for c in cells]); ax[2].set_ylabel("avg R"); ax[2].axhline(0, color="k", lw=.8)
ax[2].legend(loc="upper left"); ax2.legend(loc="upper right")
plt.tight_layout(); plt.savefig("part27_ow.png", dpi=110)
out.close()
