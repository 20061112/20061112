"""第十七部分 C：平衡版（每 30 分起一個 IB、篩選分位數 0.9）+ 浮盈 2R 移保本，逐月與資金模擬。"""
import numpy as np, pandas as pd
from part17_balance import gen, wf, score
import part17_risk as RK

out = open("results_part17.txt", "a")
def P(s=""): print(s); out.write(s + "\n"); out.flush()
P("\n" + "=" * 120 + "\n第十七部分 C：平衡版")
cands = {
    "現行（整點、0.8/0.8）": (gen("close5", 0.05, (2, 10), 60), 0.8, 0.8),
    "平衡 A（每 30 分、0.9/0.9）": (gen("close5", 0.05, (2, 10), 30), 0.9, 0.9),
    "平衡 B（01~12 每 30 分、0.9/0.9）": (gen("close5", 0.05, (1, 12), 30), 0.9, 0.9),
}
for name, (X, qt, qe) in cands.items():
    T = wf(X, qt, qe).sort_values("t_in").reset_index(drop=True)
    Y = RK.protect(T, 2.0, 0.0)
    for lab, Z in [("", T), (" + 2R保本", Y)]:
        s = score(Z)
        tday = len(Z) / Z.day.nunique()
        P(f"  {name + lab:34s} {s['per_day']:.1f}/交易日（有單的日子 {tday:.1f}） 每筆 {s['avg']:+6.2f} PF {s['pf']:.2f} 勝率 {s['win']:.0f}% "
          f"總 {s['total']:+6.0f} DD {s['dd']:5.0f} 總/DD {s['ret_dd']:4.1f} Sharpe {s['sharpe']:.2f} 去前10天 {s['ex10']:+.0f}")
    m = Y.groupby(Y.day.dt.to_period("M")).pnl.agg(["count", "sum"])
    P("     逐月（+2R保本）：" + "  ".join(f"{k.strftime('%m')}月 {v['sum']:+.0f}({int(v['count'])})" for k, v in m.iterrows()))
    for r in (0.25, 0.5):
        eq, mdd, n = RK.sizing(Y, r)
        P(f"     資金 每筆 {r}%：期末 {eq:,.0f}（{(eq / 1e4 - 1) * 100:+.0f}%） 最大回撤 {mdd * 100:.1f}%")
    if name.startswith("平衡 A"):
        Y.to_csv("part17_balanced_trades.csv", index=False)
out.close()
