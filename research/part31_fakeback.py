"""第三十一部分 F：「進場後 N 分鐘內收盤回到 IB 內就出場」—— 即時可執行的出場規則，跨週期、跨年份驗證。
M1：N = 15 分（15 根）；M5：15 分（3 根）；M15：15 分（1 根）、30 分（2 根）。出場價 = 回到 IB 內那根的收盤。"""
import numpy as np, pandas as pd
from part31_m1 import ow, in26, M1, M5, M15, PER, pf, data

out = open("results_part31.txt", "a")
def P(s=""): print(s); out.write(s + "\n"); out.flush()


def apply(X, src, n_bars):
    days, D, ARR = data(*src)
    R, why = [], []
    for r in X.itertuples():
        t, O, H, L, C, S, mins = ARR[r.day]
        k_exit = None
        for k in range(r.i, min(r.i + n_bars, len(C))):
            if t[k] >= r.exit_time:            # 已經先停損或收盤
                break
            if (C[k] < r.ibh) if r.side == 1 else (C[k] > r.ibl):
                k_exit = k; break
        if k_exit is None:
            R.append(r.R); why.append(r.exit_reason)
        else:
            R.append((r.side * (C[k_exit] - r.lvl) - r.spread) / r.risk); why.append("收回IB")
    Y = X.copy(); Y["R"] = R; Y["why"] = why; Y["pnl"] = Y.R * Y.risk
    return Y


P("\n第三十一部分 F：進場後 15 分鐘內收盤回到 IB 內就出場（每格 = 每筆 R / PF / 勝率）")
for lab, src, nb in [("M1 15分", M1, 15), ("M5 15分", M5, 3), ("M15 15分", M15, 1), ("M15 30分", M15, 2)]:
    X, _ = ow(src)
    Y = apply(X, src, nb)
    for tag, Z in [("原本", X), ("加規則", Y)]:
        cells = [f"{n} {x.R.mean():+.2f}/PF{pf(x.R):.2f}/勝{(x.R > 0).mean() * 100:.0f}%" for n, a, b in PER for x in [Z[(Z.day >= a) & (Z.day <= b)]] if len(x)]
        if src is M1:
            z = in26(Z); cells = [f"2026 {z.R.mean():+.3f}/PF{pf(z.R):.2f}/勝{(z.R > 0).mean() * 100:.0f}%"]
        P(f"  {lab:8s} {tag:4s} " + "  ".join(cells))
    if src is not M1:
        P(f"           觸發比例 {(Y.why == '收回IB').mean() * 100:.0f}%，觸發的單原本平均 {X[Y.why == '收回IB'].R.mean():+.2f}R → 提早出場平均 {Y[Y.why == '收回IB'].R.mean():+.2f}R")
    else:
        z, x0 = in26(Y), in26(X)
        P(f"           觸發比例 {(z.why == '收回IB').mean() * 100:.0f}%，觸發的單原本平均 {x0[z.why.values == '收回IB'].R.mean():+.2f}R → 提早出場平均 {z[z.why == '收回IB'].R.mean():+.2f}R")
out.close()
