"""第四十一部分（續）：進場時段窗（起點 h0、長度 1~6 小時）× K 線週期，順日線方向；看高原，並減去對照（同時段每根開盤直接順方向進場）。→ part41_piv_windows.csv"""
import numpy as np, pandas as pd
from piv_core import run, summ, direction

DIRS = direction()
KS = [5, 10, 15, 30, 60]
rows = []
for k in KS:
    for h0 in range(1, 22):
        for ln in range(1, 7):
            h1 = min(h0 + ln, 23)
            if h1 - h0 < ln:
                continue
            s = summ(run(k=k, mode=1, hours=(h0, h1), dirs=DIRS))
            c = summ(run(k=k, mode=1, hours=(h0, h1), dirs=DIRS, ctrl=True))
            rows.append(dict(k=k, h0=h0, ln=ln, n=s["n"], mR=s["mR"], dt=s["dt"], hpos=s["hpos"], L=s["L"], S=s["S"],
                             ctrl_mR=c["mR"], ctrl_dt=c["dt"], excess=s["mR"] - c["mR"]))
W = pd.DataFrame(rows)
W["ki"] = W.k.map({k: i for i, k in enumerate(KS)})


def nb(r):
    m = W[((W.h0 - r.h0).abs() <= 1) & ((W.ln - r.ln).abs() <= 1) & ((W.ki - r.ki).abs() <= 1)]
    return pd.Series(dict(nb_dt=m.dt.mean(), nb_min_dt=m.dt.min(), nb_ex=m.excess.mean(), nb_pos=(m.mR > 0).mean(), nb_hpos=m.hpos.mean()))


W = pd.concat([W, W.apply(nb, axis=1)], axis=1)
W.to_csv("part41_piv_windows.csv", index=False)
pd.set_option("display.width", 250)
print("鄰格平均日 t 前 25 名（鄰格 = 起點 ±1 小時、長度 ±1 小時、K 線 ±1 檔）")
print(W.sort_values("nb_dt", ascending=False).head(25).round(3).to_string(index=False))
print("\n鄰格平均『突破 − 對照』前 15 名")
print(W[W.n >= 300].sort_values("nb_ex", ascending=False).head(15).round(3).to_string(index=False))
