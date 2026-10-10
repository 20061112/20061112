"""第三十六部分：浮盈到 kR 就把停損移到進場價（保本），只求大單不變虧。另列 2026/3/23 每筆最大浮盈。"""
import sys; pass
import io, contextlib, pandas as pd, numpy as np
src = open("part35_lock.py").read().split("RULES = [")[0].replace('out = open("results_part35.txt", "w")', 'out = io.StringIO()')
exec(src)
T = pd.read_csv("part31_m1_trades.csv", parse_dates=["day","t_in","exit_time"]).sort_values("t_in").reset_index(drop=True)
days, D, ARR = data("data/XAUUSD_M1_2026.csv", 1)
Y = run_set(T, ARR)
d = Y[Y.day == "2026-03-23"]
print(d[["t_in","side","lvl","risk","mfe","r"]].assign(usd=d.r*d.risk).to_string(float_format="%.2f"))
for lab, csv, src_, in [("M1", "part31_m1_trades.csv", ("data/XAUUSD_M1_2026.csv",1)), ("M15", "part27_ow_M15.csv", ("data/XAUUSD_M15_2023_2026.csv",15))]:
    T = pd.read_csv(csv, parse_dates=["day","t_in","exit_time"]).sort_values("t_in").reset_index(drop=True); T = T[T.day>="2023-03-01"]
    days, D, ARR = data(*src_)
    print("\n", lab)
    for trig in [None, 1, 1.5, 2, 3, 4, 5]:
        Y = run_set(T, ARR, ladder=[(trig, 0.0)] if trig else None)
        bad = Y[(Y.mfe >= 2) & (Y.r < -0.05)]
        print(f"  {'現行' if trig is None else f'{trig}R→保本':8s} 每筆{Y.r.mean():+.3f}R PF{pf(Y.r):.2f} 勝{(Y.r>0).mean()*100:.0f}% 總{Y.r.sum():+.0f}R 浮盈≥2R仍虧{len(bad)}筆 0.01手{(Y.r*Y.risk).sum():+.0f}$")
