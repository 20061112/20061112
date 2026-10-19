"""第三十七部分 第一階段：時框 —— IB 起點（全天每 30 分）× IB 長度 × 用法（順勢 / 反向）
其餘照 XAU 核心：M5 收盤 + 0.05 ATR10 確認、突破有效 3 小時、IB 另一側停損、當天收盤出場、不加濾網。
只用研究期 2023/10 ~ 2025/09；每個 (起點, 長度, 用法) 是一個獨立的「每天最多一筆」策略。→ part37_stage1_grid.csv
兩種單位：R（固定風險）與 ATR（固定名目，不受停損大小影響 —— 小 IB 的大 R 單不會灌水）。
"""
import numpy as np, pandas as pd
from nas_core import run, period, HALVES, RESEARCH, hm


def summ(X, col):
    v = X[col]
    d = X.groupby("day")[col].sum()
    h = [period(X, a, b)[col].mean() for _, a, b in HALVES]
    return dict(m=v.mean(), dt=d.mean() / d.std(ddof=1) * np.sqrt(len(d)), hpos=int(np.sum(np.array(h) > 0)), hmin=min(h),
                h=h, long=v[X.side == 1].mean(), short=v[X.side == -1].mean())


rows = []
for mode in (1, -1):
    for ib in (15, 30, 45, 60, 90, 120):
        for s0 in range(60, 22 * 60 + 1, 30):
            if s0 + ib + 30 > 23 * 60 + 50:
                continue
            X = period(run(s0, ib=ib, mode=mode), *RESEARCH)
            if len(X) < 30:
                continue
            X["A"] = X.pnl / X.atr
            r, a = summ(X, "R"), summ(X, "A")
            rows.append(dict(mode=mode, ib=ib, s0=s0, t=hm(s0), n=len(X), PF=X.R[X.R > 0].sum() / -X.R[X.R <= 0].sum(),
                             mR=r["m"], dtR=r["dt"], hposR=r["hpos"], longR=r["long"], shortR=r["short"],
                             mA=a["m"], dtA=a["dt"], hposA=a["hpos"], hminA=a["hmin"], longA=a["long"], shortA=a["short"],
                             **{f"hA{i+1}": a["h"][i] for i in range(4)},
                             top5share=X.R.nlargest(5).sum() / X.R.sum() if X.R.sum() > 0 else np.nan))
G = pd.DataFrame(rows)
IBS = [15, 30, 45, 60, 90, 120]
G["ibi"] = G.ib.map({v: i for i, v in enumerate(IBS)})


def nb(r):
    m = G[(G["mode"] == r["mode"]) & ((G.s0 - r.s0).abs() <= 30) & ((G.ibi - r.ibi).abs() <= 1)]
    return pd.Series(dict(nb_mA=m.mA.mean(), nb_dtA=m.dtA.mean(), nb_hposA_min=m.hposA.min(), nb_posfrac=(m.mA > 0).mean(), nb_n=len(m)))


G = pd.concat([G, G.apply(nb, axis=1)], axis=1)
G.to_csv("part37_stage1_grid.csv", index=False)
