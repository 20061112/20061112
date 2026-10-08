import sys
import numpy as np
import pandas as pd

FEATS = ["z20", "z50", "z100", "bbw_pct", "bbw_chg5", "close_outside", "er5", "er10", "er20", "er40", "er80",
         "er5_20", "er10_40", "er20_80", "er_short_long", "trend_atr", "mid_dist_R", "tv_e"]


def folds_for(T, k):
    q = T.entry_time.sort_values().iloc[np.linspace(0, len(T) - 1, k + 1).astype(int)].to_list()
    q[-1] = q[-1] + pd.Timedelta(minutes=1)
    return list(zip(q[:-1], q[1:]))


def stability(T, y, k):
    F = folds_for(T, k)
    rows = []
    for f in FEATS:
        v = T[f]
        b = v.astype(int) if v.nunique() <= 2 else pd.qcut(v.rank(method="first"), 3, labels=False)
        lo_l, hi_l = b.min(), b.max()
        d = []
        for a, z in F:
            m = (T.entry_time >= a) & (T.entry_time < z)
            g = T[m].groupby(b[m])[y].mean()
            d.append(g.get(hi_l, np.nan) - g.get(lo_l, np.nan))
        g = T.groupby(b)[y].mean()
        tot = g[hi_l] - g[lo_l]
        rows.append(dict(feat=f, 低=round(g[lo_l], 3), 中=round(g.get(1, np.nan), 3) if v.nunique() > 2 else None,
                         高=round(g[hi_l], 3), 高減低=round(tot, 3), 各段=" ".join(f"{x:+.2f}" for x in d),
                         一致=int(all(np.sign(x) == np.sign(tot) for x in d))))
    return pd.DataFrame(rows).sort_values(["一致", "高減低"], ascending=[False, False])


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    for tf, k in (("m15", 4), ("m5", 3)):
        T = pd.read_pickle(f"cand4_{tf}.pkl")
        for y in ("pnl_R", "pnl_mid"):
            print(f"\n=== {tf.upper()}  結果={y}  （{k} 段）===")
            print(stability(T, y, k).to_string(index=False))
