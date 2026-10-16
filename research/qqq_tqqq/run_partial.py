"""檢查 3：震盪指標只「降檔」不下車 —— 上升趨勢且平靜時，若 ER / 排列熵 / 變異數比判定為震盪，
用 2 倍（TQQQ 權重 0.5）代替 3 倍；其餘照原規則。輸出 results_partial.csv。"""
import numpy as np
import pandas as pd
import indicators as ind
from run_tests import q, OK0, c, r, w_rule, evaluate

w0 = w_rule(OK0)
b, broll, bcal = evaluate(w0)
b, broll, bcal = evaluate(w0, broll, bcal)
rows = [dict(name="BASE") | b]
conds = {}
for n in [20, 60, 120]:
    for a in [0.0, 0.1, 0.2]:
        conds[f"ER{n}>{a}"] = (ind.er(c, n) > a).fillna(True)
for win in [20, 60, 120]:
    conds[f"PE{win} z<0"] = (ind.rz(ind.perm_entropy(c, 3, win)) < 0).fillna(True)
for qn in [5, 10]:
    conds[f"VR{qn}/250>1"] = (ind.var_ratio(r, qn, 250) > 1).fillna(True)
for k, cond in conds.items():
    full = pd.Series(w_rule(OK0), index=q.index)
    trend_ok = pd.Series(cond.shift(1).fillna(True).values, index=q.index)
    w = np.where(full > 0, np.where(trend_ok, 1.0, 0.5), 0.0)
    o, _, _ = evaluate(w, broll, bcal)
    rows.append(dict(name=k) | o)
    print(f"{k:12s} irr99 {o['irr99']:.3f} irr10 {o['irr10']:.3f} win10 {o['win10']}/17 dIRR10y {o['d_irr10y_med']:+.3f} "
          f"worst {o['worst']:.2f} g10min {o['g10_min']:.2f} worstyr {o['worst_year']:.2f} sh {o['sharpe']:.2f} sw {o['sw']:.1f}")
pd.DataFrame(rows).to_csv("results_partial.csv", index=False, float_format="%.4f")
