"""第三十七部分 第一階段（續）：把單一起點組成 XAU 核心那樣的「時段窗」（每 30 分一個 IB，同一天可疊單），掃窗的起訖與 IB 長度，看高原。
A = 順勢（歐洲盤 ~ 美股盤前的 IB）；B = 反向（亞洲盤後段的 IB）。研究期 only。→ part37_stage1_windows.csv"""
import numpy as np, pandas as pd
from nas_core import run_many, period, HALVES, RESEARCH, hm


def summ(X):
    X = X.copy()
    X["A"] = X.pnl / X.atr
    out = dict(n=len(X), per_day=len(X) / X.day.nunique())
    for col in ("R", "A"):
        d = X.groupby("day")[col].sum()
        out[f"m{col}"] = X[col].mean()
        out[f"dt{col}"] = d.mean() / d.std(ddof=1) * np.sqrt(len(d))
        h = [period(X, a, b).groupby("day")[col].sum().mean() for _, a, b in HALVES]   # 每天平均
        out[f"hpos{col}"] = int(np.sum(np.array(h) > 0))
        out[f"hmin{col}"] = min(h)
    out["longA"] = X.A[X.side == 1].mean()
    out["shortA"] = X.A[X.side == -1].mean()
    out["PF"] = X.R[X.R > 0].sum() / -X.R[X.R <= 0].sum()
    return out


rows = []
for w0 in range(600, 781, 30):
    for w1 in range(870, 991, 30):
        for ib in (30, 45, 60):
            X = period(run_many(range(w0, w1 + 1, 30), ib=ib, mode=1), *RESEARCH)
            rows.append(dict(set="A", w0=hm(w0), w1=hm(w1), ib=ib, **summ(X)))
for w0 in range(330, 451, 30):
    for w1 in range(480, 601, 30):
        for ib in (45, 60, 90):
            X = period(run_many(range(w0, w1 + 1, 30), ib=ib, mode=-1), *RESEARCH)
            rows.append(dict(set="B", w0=hm(w0), w1=hm(w1), ib=ib, **summ(X)))
W = pd.DataFrame(rows)
W.to_csv("part37_stage1_windows.csv", index=False)
pd.set_option("display.width", 250)
for s in "AB":
    for ib in sorted(W[W.set == s].ib.unique()):
        P = W[(W.set == s) & (W.ib == ib)].pivot(index="w0", columns="w1", values="dtA").round(2)
        Q = W[(W.set == s) & (W.ib == ib)].pivot(index="w0", columns="w1", values="hposA")
        print(f"\n{s} IB{ib}: daily t (ATR units) | halves positive")
        print(pd.concat([P, Q.add_prefix("h")], axis=1).to_string())
