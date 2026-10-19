"""第四十五部分（續）：時段窗（起點 h0、長度 ln 小時）× K 線，兩種訊號源；鄰格平滑找高原。研究期 only。→ part45_z_windows.csv"""
import numpy as np, pandas as pd
from z_core import run, summ

KS = [10, 15, 20, 30]
rows = []
for src in ("pattern", "zcross"):
    for k in KS:
        for h0 in range(1, 20):
            for ln in (3, 5, 7, 9, 11, 13):
                h1 = h0 + ln
                if h1 > 24:
                    continue
                s = summ(run(k=k, src=src, hours=(h0, h1)))
                if s["n"] < 30:
                    continue
                rows.append(dict(src=src, k=k, h0=h0, ln=ln, **s))
W = pd.DataFrame(rows)
W["ki"] = W.k.map({k: i for i, k in enumerate(KS)})
W["li"] = W.ln.map({v: i for i, v in enumerate((3, 5, 7, 9, 11, 13))})


def nb(r):
    m = W[(W.src == r.src) & ((W.h0 - r.h0).abs() <= 1) & ((W.li - r.li).abs() <= 1) & ((W.ki - r.ki).abs() <= 1)]
    return pd.Series(dict(nb_dt=m.dt.mean(), nb_min=m.dt.min(), nb_pos=(m.mR > 0).mean(), nb_hpos=m.hpos.mean()))


W = pd.concat([W, W.apply(nb, axis=1)], axis=1)
W.to_csv("part45_z_windows.csv", index=False)
pd.set_option("display.width", 250)
cols = ["src", "k", "h0", "ln", "n", "per_day", "mR", "PF", "dt", "hpos", "h1", "h2", "h3", "h4", "L", "S", "nb_dt", "nb_min", "nb_pos", "nb_hpos"]
for src in ("pattern", "zcross"):
    print(f"\n=== {src}：鄰格平均日 t 前 15（鄰格 = 起點 ±1h、長度 ±1 檔、K 線 ±1 檔）===")
    print(W[W.src == src].sort_values("nb_dt", ascending=False)[cols].head(15).round(3).to_string(index=False))
