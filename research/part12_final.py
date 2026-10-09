"""第十二部分：最終兩個版本的全期統計、逐月損益、成本壓力測試、權益曲線。"""
import numpy as np, pandas as pd
from part12_strategy import backtest, stats, SPLIT

H = tuple(range(2, 11))
FINAL = {
    "A 少量（不反手, IB60, 停損IB中點, 2h窗）": dict(hours=H, ib_min=60, win=2, stop="mid", be=None, reverse=False),
    "B 多量（反手, IB30, 停損IB另一側, 1倍IB保本, 5h窗）": dict(hours=H, ib_min=30, win=5, stop="opp", be=1.0, reverse=True),
}
out = open("results_part12.txt", "a")
def P(s=""): print(s); out.write(s + "\n")

P("\n" + "=" * 100 + "\n最終版本（全期 2026/1/2 ~ 10/8；單位 美元/盎司 = 0.01 手美元）")
curves = {}
for name, cfg in FINAL.items():
    T = backtest(**cfg)
    T.to_csv(f"part12_trades_{name[0]}.csv", index=False)
    P(f"\n[{name}]")
    for lab, x in [("全期", T), ("1~5 月（選參數）", T[T.day < SPLIT]), ("6~10 月（驗證）", T[T.day >= SPLIT])]:
        s = stats(x)
        P(f"  {lab:14s} {s['n']:4d} 筆  {s['per_day']:.1f}/天  勝率 {s['win']:.0f}%  每筆 {s['avg']:+.2f}"
          f"  均賺 {s['avg_w']:.1f} / 均賠 {s['avg_l']:.1f}  PF {s['pf']:.2f}  總計 {s['total']:+.0f}  最大回撤 {s['maxdd']:.0f}")
    for extra in (0.2, 0.5):
        p = T.pnl - extra
        P(f"  成本壓力：每筆再扣 {extra} 美元滑價 → PF {p[p > 0].sum() / -p[p <= 0].sum():.2f}  總計 {p.sum():+.0f}")
    m = T.groupby(pd.to_datetime(T.day).dt.to_period("M")).pnl.agg(["count", "sum"])
    P("  逐月損益：" + "  ".join(f"{k.strftime('%m')}月 {v['sum']:+.0f}({int(v['count'])})" for k, v in m.iterrows()))
    P("  出場原因：" + str(T.why.value_counts().to_dict()))
    curves[name] = T.set_index("t_out").pnl.cumsum()
out.close()

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(11, 4.8))
for (name, c), col in zip(curves.items(), ["#3b6ea8", "#c9824a"]):
    ax.plot(c.index, c.values, label=("A: no reverse, IB60, stop=IB mid" if name[0] == "A" else "B: reverse, IB30, stop=IB opp, BE 1xIB"), color=col, lw=1.6)
ax.axvline(SPLIT, color="gray", ls="--", lw=1); ax.text(SPLIT, ax.get_ylim()[1] * 0.92, "  out-of-sample →", color="gray")
ax.set_ylabel("cumulative PnL (USD / oz = USD per 0.01 lot)"); ax.grid(alpha=.3); ax.legend(loc="upper left")
ax.set_title("IB breakout, single net position — XAUUSD 2026")
plt.tight_layout(); plt.savefig("part12_equity.png", dpi=110)
