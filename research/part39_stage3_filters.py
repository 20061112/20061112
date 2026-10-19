"""第三十九部分 第三階段：濾網 —— 只用事前可知的狀態，三分位門檻只用研究期決定。研究期 only。→ part39_stage3.csv
A* = 第二階段定案：IB 起點 10:30~16:00 每 30 分、IB 45 分、M5 收盤 + 0.05 ATR、有效 3 小時、IB 另一側停損、當天收盤出場、每天每方向 ≤ 2 筆
C* = 01:00 開盤 IB 30 分（其餘同基準）
"""
import numpy as np, pandas as pd
from nas_core import run_many, period, HALVES, RESEARCH, D
from part38_stage2_signal import cap, summ


def trades(name):
    if name == "A":
        X = cap(run_many(range(630, 961, 30), ib=45, mode=1), 2)
    else:
        X = run_many([60], ib=30, mode=1)
    X = X.join(D[["o", "peff", "pdir", "pret_atr", "trend20", "volr"]], on="day")
    X["A"] = X.pnl / X.atr
    X["ib_atr"] = X.ibw / X.atr
    X["ext"] = (X.lvl - X.o) * X.side / X.atr          # 進場時已經往交易方向走了幾個 ATR（從 01:00 開盤算）
    X["pret_al"] = X.pret_atr * X.side                 # 前一天漲跌是否與交易同向
    X["trend_al"] = X.trend20 * X.side
    X["hour"] = X.ent // 60
    return X


FEATS = ["peff", "pret_al", "trend_al", "volr", "ib_atr", "ext", "hour"]
if __name__ == "__main__":
    rows = []
    for name in ("A", "C"):
        X = trades(name)
        R = period(X, *RESEARCH)
        for f in FEATS:
            if name == "C" and f in ("ext", "hour"):
                continue
            q = R[f].quantile([1 / 3, 2 / 3]).to_numpy()
            b = np.digitize(R[f], q)
            for k in range(3):
                Y = R[b == k]
                h = [period(Y, a, bb).A.mean() for _, a, bb in HALVES]
                rows.append(dict(set=name, feat=f, bucket=["低", "中", "高"][k], lo=(-np.inf if k == 0 else q[k - 1]),
                                 hi=(np.inf if k == 2 else q[k]), n=len(Y), mA=Y.A.mean(), mR=Y.R.mean(),
                                 h1=h[0], h2=h[1], h3=h[2], h4=h[3], hpos=int(np.sum(np.array(h) > 0))))
    T = pd.DataFrame(rows)
    T.to_csv("part39_stage3.csv", index=False)
    pd.set_option("display.width", 250)
    print(T.round(3).to_string(index=False))

    # ---- 門檻掃描（看高原）：volr、peff 只保留 ≤ 研究期分位數 q 的日子 ----
    from nas_core import run_many, hm
    RDAYS = D.index[(D.index >= RESEARCH[0]) & (D.index < RESEARCH[1])]

    def s(X):
        X = period(X, *RESEARCH)
        d = (X.pnl / X.atr).groupby(X.day).sum().reindex(RDAYS, fill_value=0)
        h = [d[(d.index >= a) & (d.index < b)].mean() for _, a, b in HALVES]
        return d.mean() / d.std() * np.sqrt(len(d)), int(np.sum(np.array(h) > 0)), h, len(X)

    print("\n門檻掃描（日 t，四個半年為正個數）")
    for name in "AC":
        X = trades(name)
        R = period(X, *RESEARCH)
        t, hp, h, n = s(X)
        print(f"\n== {name} 基準 n={n} 日t={t:.2f} hpos={hp}")
        for f in ("volr", "peff"):
            for q in (0.5, 0.6, 0.67, 0.75, 0.8, 0.9):
                thr = R[f].quantile(q)
                t, hp, h, n = s(X[X[f] <= thr])
                print(f"  {f} <= q{q:.2f} ({thr:.3f}) n={n:5d} 日t={t:.2f} hpos={hp} " + " ".join(f"{v:+.3f}" for v in h))
    print("\npeff ≤ 0.75 套在 A 的鄰近設定（窗口起訖 × IB 長度 × 疊單上限）是否都改善")
    out = []
    for w0 in (600, 630, 660):
        for w1 in (930, 960, 990):
            for ib in (30, 45, 60):
                for n in (1, 2, 3):
                    X = cap(run_many(range(w0, w1 + 1, 30), ib=ib, mode=1), n).join(D[["peff"]], on="day")
                    a, b = s(X), s(X[X.peff <= 0.75])
                    out.append((hm(w0), hm(w1), ib, n, round(a[0], 2), a[1], round(b[0], 2), b[1]))
    O = pd.DataFrame(out, columns="w0 w1 ib cap dt hpos dt_peff hpos_peff".split())
    print(O.to_string(index=False))
    print(f"改善比例 {(O.dt_peff > O.dt).mean():.0%}，日 t 中位數提高 {(O.dt_peff - O.dt).median():+.2f}")
