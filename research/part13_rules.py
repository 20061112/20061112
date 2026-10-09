"""第十三部分：篩選規則組合（1~5 月挑、6~10 月驗證）。讀 part13_trades.csv。
avg = 每筆美元/盎司；avgATR = 每筆 /ATR（不受停損大小影響）；R = 每筆 / IB 區間（= 每單位風險）。"""
import numpy as np, pandas as pd
T = pd.read_csv("part13_trades.csv", parse_dates=["day", "t_in", "t_out"])
from part13_filters import D
T["atr"] = D.atr10.reindex(T.day).values
T["pnl_atr"] = T.pnl / T.atr
out = open("results_part13.txt", "a")
def P(s=""): print(s); out.write(s + "\n")


def line(x, nd):
    p = x.pnl
    pf = p[p > 0].sum() / -p[p <= 0].sum()
    return (f"{len(x) / nd:4.1f}/天 avg {p.mean():+6.2f}  avgATR {x.pnl_atr.mean():+.3f}  R {x.R.mean():+.2f}"
            f"  PF {pf:.2f}  win {(p > 0).mean() * 100:.0f}%")


ND_IS, ND_OOS = T[T["is"]].day.nunique(), T[~T["is"]].day.nunique()
RULES = {
    "原始（不篩）": lambda t: t.h > 0,
    "a 去掉 01 點": lambda t: t.h != 1,
    "b IB/ATR ≥ 0.13（去掉最小 40% IB）": lambda t: t.ib_atr >= 0.13,
    "c IB z ≥ -0.44（去掉最小 40%）": lambda t: t.ib_z >= -0.44,
    "d mom60 ≥ 0.06（突破前 1 小時有走）": lambda t: t.mom60 >= 0.06,
    "e 逆前日方向（prev_dir = -1）": lambda t: t.prev_dir == -1,
    "f 不做順前日方向且 IB 小的": lambda t: ~((t.prev_dir == 1) & (t.ib_atr < 0.13)),
    "a+b": lambda t: (t.h != 1) & (t.ib_atr >= 0.13),
    "a+b+d": lambda t: (t.h != 1) & (t.ib_atr >= 0.13) & (t.mom60 >= 0.06),
    "a+b+e": lambda t: (t.h != 1) & (t.ib_atr >= 0.13) & (t.prev_dir == -1),
    "a+f": lambda t: (t.h != 1) & ~((t.prev_dir == 1) & (t.ib_atr < 0.13)),
    "a+c": lambda t: (t.h != 1) & (t.ib_z >= -0.44),
}
P("\n" + "=" * 110 + "\n篩選規則：1~5 月（挑規則用） → 6~10 月（驗證）")
for name, f in RULES.items():
    m = f(T).fillna(False).astype(bool)
    P(f"  {name:34s} IS: {line(T[m & T['is']], ND_IS)}\n  {'':34s} OOS:{line(T[m & ~T['is']], ND_OOS)}")
out.close()
