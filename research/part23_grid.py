"""第二十三部分穩健性：所有 27 個 ln(pmax/p邊緣) 變體 × 4 時框，同一套規則（K ≥ 前半 90 百分位、停損 1S、停利 2R）。
看『整個區域』是不是正的，而不是只看前半最佳的那一個。輸出 results_part23_grid.txt
"""
import numpy as np
import pandas as pd
import part23_trade as T

rows = []
for tf in T.TFS:
    E = pd.read_pickle(f"part23_events_M{tf}.pkl")
    E["year"] = E.time_close.dt.year
    E["half"] = np.where(E.time_close < "2025-01-01", "A", "B")
    A = (E.half == "A").to_numpy()
    for occ in T.OCCS:
        for c in T.CS:
            for h in T.HS:
                x = E[f"{occ}_c{c}_h{h}_sh"].to_numpy(float)
                for q, b in ((0.9, 1.0), (0.9, 0.5), (0.7, 1.0)):
                    tr = T.simulate(E, x >= np.nanquantile(x[A], q), b=b, rr=2.0)
                    r = dict(tf=tf, occ=occ, c=c, h=h, q=q, b=b)
                    for hv in ("A", "B"):
                        y = tr[tr.half == hv]
                        r[f"n_{hv}"], r[f"R_{hv}"] = len(y), y.R.mean()
                        r[f"t_{hv}"] = y.R.mean() / T.cluster_se(y.R, y.day)
                    r.update({f"y{y}": v for y, v in tr.groupby("year").R.mean().items()})
                    rows.append(r)
G = pd.DataFrame(rows)
G.to_csv("part23_grid.csv", index=False)
out = []
for (tf, q, b), g in G.groupby(["tf", "q", "b"]):
    out.append(dict(tf=tf, q=q, b=b, 前半中位R=g.R_A.median(), 後半中位R=g.R_B.median(),
                    前半為正=(g.R_A > 0).mean(), 後半為正=(g.R_B > 0).mean(), 兩半都正=((g.R_A > 0) & (g.R_B > 0)).mean(),
                    四年都正=(g[["y2023", "y2024", "y2025", "y2026"]] > 0).all(axis=1).mean(),
                    前後半R相關=g.R_A.corr(g.R_B)))
S = pd.DataFrame(out)
txt = ("27 個變體 × 規則的分佈\n" + S.round(3).to_string(index=False) +
       "\n\nM1、K≥90%、1S/2R 全部 27 個變體：\n" +
       G[(G.tf == 1) & (G.q == 0.9) & (G.b == 1.0)].round(3).to_string(index=False))
print(txt)
open("results_part23_grid.txt", "w").write(txt + "\n")
