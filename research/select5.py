"""嚴格流程：只用研究期 (6/29~8/15) 挑特徵，驗證期 (8/15~10/8) 只看結果。"""
import numpy as np
import pandas as pd

T = pd.read_pickle("cand5_m1.pkl")
SPLIT = pd.Timestamp("2026-08-15")
DEV, TEST = T[T.entry_time < SPLIT], T[T.entry_time >= SPLIT]
DEV_H = [DEV[DEV.entry_time < "2026-07-23"], DEV[DEV.entry_time >= "2026-07-23"]]
FEATS = [c for c in T.columns if c not in ("setup", "t", "entry_time", "exit_time", "side", "kind", "R_usd", "pnl_R")]


def q_effect(D, f, q1, q4):
    return D[D[f] > q4].pnl_R.mean() - D[D[f] <= q1].pnl_R.mean()


rows = []
for f in FEATS:
    v = DEV[f]
    if v.nunique() <= 3:
        lo, hi = v.min(), v.max()
        eff = [D[D[f] == hi].pnl_R.mean() - D[D[f] == lo].pnl_R.mean() for D in DEV_H]
        te = TEST[TEST[f] == hi].pnl_R.mean() - TEST[TEST[f] == lo].pnl_R.mean()
        thr = ("==", hi, lo)
    else:
        q1, q4 = v.quantile([0.2, 0.8])
        eff = [q_effect(D, f, q1, q4) for D in DEV_H]
        te = q_effect(TEST, f, q1, q4)
        thr = (">", q4, q1)
    sel = np.sign(eff[0]) == np.sign(eff[1]) and min(abs(eff[0]), abs(eff[1])) >= 0.10
    rows.append(dict(feat=f, 研究前半=round(eff[0], 3), 研究後半=round(eff[1], 3), 入選=int(sel), 驗證期=round(te, 3),
                     驗證同向=int(np.sign(te) == np.sign(eff[0])) if sel else None, q_low=round(thr[2], 3), q_high=round(thr[1], 3)))
R = pd.DataFrame(rows).sort_values(["入選", "研究後半"], ascending=False)

if __name__ == "__main__":
    pd.set_option("display.width", 220)
    print(f"研究期 {len(DEV)} 筆（平均 {DEV.pnl_R.mean():+.3f}R），驗證期 {len(TEST)} 筆（平均 {TEST.pnl_R.mean():+.3f}R）")
    print("效果 = 上 20% 減下 20% 的平均 R（二元特徵為 1 減 0 / +1 減 -1）")
    print(R.to_string(index=False))
