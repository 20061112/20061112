"""第二十三部分 補充：H1 k=2、停損 2σ、收盤跌回中軌出場 的鄰近參數、多空拆解、與隨機基準的差距 t 值、點差加倍。"""
import itertools

import numpy as np
import pandas as pd

from load import load
from part23_bb_breakout import Data, simulate, random_baseline, IS_END, HOLD

m1 = load("data/XAUUSD_M1_2023_2026.csv")


def edge_t(tr, base):
    """訊號 − 隨機基準（多空分開）；回傳差距與 t 值。"""
    parts = []
    for s in (1, -1):
        a, b = tr[tr.side == s].pnl_sd, base[base.side == s].pnl_sd
        if len(a) > 1:
            parts.append((len(a), a.mean() - b.mean(), a.var() / len(a) + b.var() / len(b)))
    n = sum(p[0] for p in parts)
    diff = sum(p[0] * p[1] for p in parts) / n
    var = sum((p[0] / n) ** 2 * p[2] for p in parts)
    return diff, diff / np.sqrt(var)


rows = []
for bb_n in (14, 20, 30):
    d = Data(m1, "H1", n=bb_n)
    for k, sl, ex in itertools.product((1.5, 2, 2.5), (1.5, 2, 2.5), ("trail_mid", "trail_1")):
        tr = simulate(d, k, sl, ex, HOLD["H1"])
        base = random_baseline(d, tr, sl, ex, HOLD["H1"], reps=10)
        r = dict(n=bb_n, k=k, sl=sl, ex=ex, trades=len(tr))
        for tag, part, bpart in (("IS", tr[tr.time < IS_END], base[base.time < IS_END]),
                                 ("OOS", tr[tr.time >= IS_END], base[base.time >= IS_END])):
            r[f"{tag}_sd"] = part.pnl_sd.mean()
            r[f"{tag}_edge"], r[f"{tag}_edge_t"] = edge_t(part, bpart)
        r["long"] = tr[tr.side == 1].pnl_sd.mean()
        r["short"] = tr[tr.side == -1].pnl_sd.mean()
        # 點差加倍：再扣一次進場點差（以 σ 為單位）
        spr = (m1.spread * 0.01).resample("1h", label="left", closed="left").max().reindex(tr.time).to_numpy()
        r["2xspr_sd"] = (tr.pnl_sd - spr / (tr.pnl / tr.pnl_sd)).mean()
        for y, part in tr.groupby(tr.time.dt.year):
            r[f"y{y}"] = part.pnl_sd.mean()
        rows.append(r)
res = pd.DataFrame(rows)
res.to_csv("part23_robust.csv", index=False)
pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 200)
print(res.round(3).to_string(index=False))
print("\n鄰近參數中 IS/OOS edge > 0 的比例：",
      round((res.IS_edge > 0).mean(), 2), round((res.OOS_edge > 0).mean(), 2))
print("IS/OOS edge 平均 t：", round(res.IS_edge_t.mean(), 2), round(res.OOS_edge_t.mean(), 2))
