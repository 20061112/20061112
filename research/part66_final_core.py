"""第六十六部分：核心 + 兩個前後一致的延伸（疊單上限、12 點後不進場）的最終比較 → results_part66.txt

版本
  0 核心（第六十二部分）：IB 突破 + M5 收盤確認 + 前一天效率 > 0.633 不做
  1 + 每天每方向最多 3 筆（第六十三、六十四部分：研究期與 2026 的逐日 t 值都提高）
  2 + 12:00（broker）以後不進新單（第六十五部分：12~15 點進場的單四段 PF 都 < 1.15）
"""
import numpy as np
import pandas as pd
from core_ib import events, simulate, PEFF_MAX, SUBS, RESEARCH, VALID


def cap_stack(X, n):
    X = X.sort_values("t_in")
    return X[X.groupby(["day", "side"]).cumcount() < n]


def stat(X):
    p = X.pnl
    d = X.groupby("day").pnl.sum()
    eq = X.sort_values("t_in").pnl.cumsum()
    mo = X.groupby(X.day.dt.to_period("M")).pnl.sum()
    return dict(筆數=len(p), 每天=round(len(p) / X.day.nunique(), 1), 勝率=round((p > 0).mean(), 2),
                PF=round(p[p > 0].sum() / -p[p <= 0].sum(), 2), 每筆=round(p.mean(), 2), 每筆R=round(X.R.mean(), 3),
                日t=round(d.mean() / d.std(ddof=1) * np.sqrt(len(d)), 2), 總=round(p.sum()),
                回撤=round((eq - eq.cummax()).min()), P_MDD=round(p.sum() / -(eq - eq.cummax()).min(), 2),
                月正=f"{(mo > 0).sum()}/{len(mo)}", 最差日=round(d.min()))


X = simulate(events())
X = X[X.peff <= PEFF_MAX]
V = {"0 核心": X, "1 + 每方向 ≤ 3": cap_stack(X, 3), "2 + 12 點後不進": cap_stack(X[X.min_in < 720], 3)}
out = []
for name, Y in V.items():
    rows = [dict(期間=lab, **stat(Y[(Y.day >= a) & (Y.day < b)])) for lab, a, b in SUBS + [("研究合計", *RESEARCH), ("2026 驗證", *VALID)]]
    out.append(f"\n=== {name} ===\n" + pd.DataFrame(rows).set_index("期間").to_string())
Y = V["2 + 12 點後不進"]
mo = Y.groupby(Y.day.dt.to_period("M")).pnl.sum()
out.append("\n版本 2 逐月：" + " ".join(f"{k.strftime('%y%m')}:{v:+.0f}" for k, v in mo.items()))
Y.to_csv("part66_trades.csv", index=False)
txt = "第六十六部分：核心 + 一致的延伸（每筆 1 盎司、美元、已扣點差）" + "\n".join(out)
open("results_part66.txt", "w").write(txt + "\n")
print(txt)
