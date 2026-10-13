"""第二十四部分 B：唯一扣掉隨機基準後仍顯著的族群（M15、1-2-3、突破 H2 進場、停損在 L1、L1 為 50 根極值、順日線趨勢）
的鄰近參數、多空、逐年、點差加倍檢查。"""
import itertools

import numpy as np
import pandas as pd

import part24_123_2b as P
from load import load
from part24_baseline import baseline

P.M1 = load("data/XAUUSD_M1_2023_2026.csv")
rows = []
for tf in ("M15", "H1"):
    d = P.TF(P.M1, tf)
    for swing in ("frac2", "frac3", "frac5", "zz2", "zz3"):
        t = P.run_123(d, swing)
        t = P.apply_trend(t[(t.entry == "break") & (t.stop == "s1") & t.ext50], "with")
        for tp in ("1", "2", "mm"):
            tr = P.no_overlap(t[t.tp == tp])
            b = baseline(d, tr)
            spr = d.mspr[tr.fill_m1.to_numpy()] / (tr.risk_atr.to_numpy() * d.atr[np.searchsorted(d.time, tr.time)])
            r = dict(tf=tf, swing=swing, tp=tp, n=len(tr), R=tr.pnl_r.mean(), rand=b.pnl_r.mean())
            r["edge"] = r["R"] - r["rand"]
            r["edge_t"] = r["edge"] / np.sqrt(tr.pnl_r.var() / len(tr) + b.pnl_r.var() / len(b))
            r["R_2xspr"] = (tr.pnl_r - spr).mean()
            for tag, m in (("IS", tr.time < P.IS_END), ("OOS", tr.time >= P.IS_END)):
                bm = b.time < P.IS_END if tag == "IS" else b.time >= P.IS_END
                r[f"{tag}_edge"] = tr[m].pnl_r.mean() - b[bm].pnl_r.mean()
            for s, nm in ((1, "long"), (-1, "short")):
                r[f"{nm}_n"] = int((tr.side == s).sum())
                r[f"{nm}_edge"] = tr[tr.side == s].pnl_r.mean() - b[b.side == s].pnl_r.mean()
            for y in (2023, 2024, 2025, 2026):
                r[f"y{y}_edge"] = tr[tr.time.dt.year == y].pnl_r.mean() - b[b.time.dt.year == y].pnl_r.mean()
            rows.append(r)
res = pd.DataFrame(rows)
res.to_csv("part24_cluster.csv", index=False)
pd.set_option("display.width", 250, "display.max_columns", 30)
print(res.round(3).to_string(index=False))
for tf in ("M15", "H1"):
    x = res[res.tf == tf]
    print(f"\n{tf}：edge > 0 的比例 {(x.edge > 0).mean():.2f}，IS 與 OOS 都 > 0 的比例 "
          f"{((x.IS_edge > 0) & (x.OOS_edge > 0)).mean():.2f}，平均 edge t {x.edge_t.mean():.2f}")
