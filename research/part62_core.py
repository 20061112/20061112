"""第六十二部分：IB 突破核心（L1）+「前一天效率高不做」—— 分段績效 → results_part62.txt

L1 / 版本 A 交易取自第六十一部分（data/pseudo/part61_*.pkl，需先跑 part61_regime.py）。
濾網：前一天日效率 |收−開|/(高−低) > 0.633 不做（0.633 = 研究期 2024/11~2025/12 日子的 2/3 分位；
      這個方向是第二十部分在 2026 年先發現的，第六十一部分在 2024/11~2025/12 獨立確認）。
另外列：只做每天第一筆（同方向不疊單）的版本，看疊單是否必要。
"""
import numpy as np
import pandas as pd
from part61_regime import daily_features

PER = [("2024/11~12", "2024-11-01", "2025-01-01"), ("2025H1", "2025-01-01", "2025-07-01"),
       ("2025H2", "2025-07-01", "2026-01-01"), ("研究期合計", "2024-11-01", "2026-01-01"), ("2026（驗證）", "2026-01-01", "2026-11-01")]
F = daily_features()


def stat(T):
    p = T.pnl
    if len(p) < 3:
        return dict(筆數=len(p))
    days = T.day.nunique()
    d = T.groupby("day").pnl.sum()
    eq = T.sort_values("exit_time").pnl.cumsum()
    return dict(筆數=len(p), 每天=round(len(p) / days, 1), 勝率=round((p > 0).mean(), 2),
                PF=round(p[p > 0].sum() / -p[p <= 0].sum(), 2), 每筆=round(p.mean(), 2), 每筆R=round(T.R.mean(), 3),
                t值=round(p.mean() / p.std(ddof=1) * np.sqrt(len(p)), 1), 日t值=round(d.mean() / d.std(ddof=1) * np.sqrt(len(d)), 1),
                總=round(p.sum()), 回撤=round((eq - eq.cummax()).min()))


out = []
for name in ("L1", "A"):
    T = pd.read_pickle(f"data/pseudo/part61_{name}.pkl")
    T = T[T.day >= "2024-11-01"].join(F[["peff"]], on="day")
    first = T.sort_values("t_in").drop_duplicates(["day"])
    for lab, X in ((f"{name} 原版", T), (f"{name} + 前一天效率 ≤ 0.633", T[T.peff <= 0.633]),
                   (f"{name} + 濾網 + 每天只做第一筆", first[first.peff <= 0.633])):
        rows = [dict(期間=p, **stat(X[(X.day >= a) & (X.day < b)])) for p, a, b in PER]
        out.append(f"\n=== {lab} ===\n" + pd.DataFrame(rows).set_index("期間").to_string())
    X = T[T.peff <= 0.633]
    mo = X.groupby(X.day.dt.to_period("M")).pnl.sum()
    out.append(f"{name} + 濾網 逐月正報酬：{(mo > 0).sum()}/{len(mo)}；" + " ".join(f"{k.strftime('%y%m')}:{v:+.0f}" for k, v in mo.items()))
open("results_part62.txt", "w").write("\n".join(out) + "\n")
print("\n".join(out))
