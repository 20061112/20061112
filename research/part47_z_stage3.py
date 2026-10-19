"""第四十七部分 第三階段：濾網 —— 固定 Z1（純 z20 回到 ±2 帶內、M15、09:00~20:00 訊號、突破訊號 K 進場、停損近 3 根極值外 0.1 ATR、2R、16 根）。研究期 only。→ part47_z_stage3.csv
三分位（研究期門檻）→ 單調 + 四段一致的才做門檻掃描 → 套在鄰近設定上看是否一致改善。"""
import numpy as np, pandas as pd
from z_core import run, summ, bars, period, RESEARCH, HALVES, D
from piv_core import direction

DIR30 = pd.Series(direction(30, 10), index=D.index)
DIR50 = pd.Series(direction(50, 10), index=D.index)


def feats(X, k=15):
    B = bars(k)
    X = X.copy()
    for f in ("bbw_pct", "er5_20"):
        X[f] = B[f].to_numpy()[X.t.values]
    X = X.join(D[["peff", "volr", "pret_atr"]], on="day")
    X["dir30_al"] = DIR30.reindex(X.day).to_numpy() * X.side      # +1 = 順日線趨勢的反轉、−1 = 逆
    X["dir50_al"] = DIR50.reindex(X.day).to_numpy() * X.side
    X["pret_al"] = X.pret_atr * X.side
    return X


FEATS = ["bbw_pct", "er5_20", "peff", "volr", "pret_al", "hour"]
if __name__ == "__main__":
    pd.set_option("display.width", 250)
    X = feats(run(k=15, src="zcross", hours=(9, 20)))
    R = period(X, *RESEARCH)
    rows = []
    for f in FEATS:
        q = R[f].quantile([1 / 3, 2 / 3]).to_numpy()
        b = np.digitize(R[f], q)
        for kk in range(3):
            Y = R[b == kk]
            h = [period(Y, a, bb).R.mean() for _, a, bb in HALVES]
            rows.append(dict(feat=f, bucket=["低", "中", "高"][kk], hi=(q[kk] if kk < 2 else np.inf), n=len(Y), mR=Y.R.mean(),
                             h1=h[0], h2=h[1], h3=h[2], h4=h[3], hpos=int(np.sum(np.array(h) > 0))))
    for f in ("dir30_al", "dir50_al"):
        for v, lab in ((1, "順"), (0, "不做日"), (-1, "逆")):
            Y = R[R[f] == v]
            h = [period(Y, a, bb).R.mean() for _, a, bb in HALVES]
            rows.append(dict(feat=f, bucket=lab, n=len(Y), mR=Y.R.mean(), h1=h[0], h2=h[1], h3=h[2], h4=h[3], hpos=int(np.sum(np.array(h) > 0))))
    T = pd.DataFrame(rows)
    T.to_csv("part47_z_stage3.csv", index=False)
    print("三分位（研究期）\n" + T.round(3).to_string(index=False))
