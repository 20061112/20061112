"""每個特徵分三等分（低/中/高），看各時間區段的平均 R 是否方向一致。"""
import numpy as np
import pandas as pd

T = pd.read_pickle("cand_m15.pkl")
T = T.loc[:, ~T.columns.duplicated()]
FOLDS = [("2/2-3/31", "2026-02-01", "2026-04-01"), ("4/1-5/31", "2026-04-01", "2026-06-01"),
         ("6/1-7/31", "2026-06-01", "2026-08-01"), ("8/1-10/8", "2026-08-01", "2026-10-09")]
FEATS = ["bbw_pct", "bbw_chg3", "bbw_chg10", "squeeze_recent", "band_pierce", "close_outside", "pctb_t",
         "rsi2_e", "rsi14_e", "rsi_div", "hma_turn", "hma_slope", "roc_decel", "er20", "trend_atr",
         "wick_e", "rng_e", "conf_rng", "tv_e", "tv_t", "tv_push", "atr_regime", "dist_pd", "hour"]


def table(T, y="pnl_R"):
    rows = []
    for f in FEATS:
        v = T[f]
        if v.nunique() <= 2:
            b = v.astype(int)
            labels = sorted(b.dropna().unique())
        else:
            b = pd.qcut(v.rank(method="first"), 3, labels=False)
            labels = [0, 1, 2]
        r = {"feat": f}
        diffs = []
        for name, a, z in FOLDS:
            m = (T.entry_time >= a) & (T.entry_time < z)
            g = T[m].groupby(b[m])[y].mean().reindex(labels)
            diffs.append(g.iloc[-1] - g.iloc[0])
            r[name] = round(diffs[-1], 2)
        g = T.groupby(b)[y].mean().reindex(labels)
        r["低"] , r["高"] = round(g.iloc[0], 3), round(g.iloc[-1], 3)
        r["高-低"] = round(g.iloc[-1] - g.iloc[0], 3)
        r["一致"] = int(np.all(np.sign(diffs) == np.sign(r["高-低"])))
        rows.append(r)
    return pd.DataFrame(rows).sort_values(["一致", "高-低"], ascending=[False, False])


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    print("各區段筆數", [int(((T.entry_time >= a) & (T.entry_time < z)).sum()) for _, a, z in FOLDS])
    print("平均 R 標準誤（三等分、單一區段）≈", round(T.pnl_R.std() / np.sqrt(len(T) / 12), 2))
    print(table(T).to_string(index=False))
