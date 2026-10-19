"""第三十八部分 第二階段：訊號源 —— 固定第一階段高原中心，一次改一個面向。研究期 only。→ part38_stage2.csv
A（順勢）：IB 起點 10:30~16:00 每 30 分、IB 45 分
B（反向）：IB 起點 06:30~08:30 每 30 分、IB 60 分
C（順勢、觀察）：01:00 開盤 IB 30 分
基準其餘：M5 收盤 + 0.05 ATR 確認、突破有效 3 小時、IB 另一側停損、當天收盤出場。
"""
import numpy as np, pandas as pd
from nas_core import run_many, period, HALVES, RESEARCH, D

BASE = {"A": dict(starts=range(630, 961, 30), ib=45, mode=1),
        "B": dict(starts=range(390, 511, 30), ib=60, mode=-1),
        "C": dict(starts=[60], ib=30, mode=1)}
RDAYS = D.index[(D.index >= RESEARCH[0]) & (D.index < RESEARCH[1])]


def cap(X, n):
    if n is None:
        return X
    X = X.sort_values(["day", "ent"])
    return X[X.groupby(["day", "side"]).cumcount() < n]


def summ(X):
    X = period(X, *RESEARCH).copy()
    X["A"] = X.pnl / X.atr
    d = X.groupby("day").A.sum().reindex(RDAYS, fill_value=0)
    h = [d[(d.index >= a) & (d.index < b)].mean() for _, a, b in HALVES]
    return dict(n=len(X), mA=X.A.mean(), dayA=d.mean(), dtA=d.mean() / d.std() * np.sqrt(len(d)), hpos=int(np.sum(np.array(h) > 0)),
                h1=h[0], h2=h[1], h3=h[2], h4=h[3], mR=X.R.mean(), PF=X.R[X.R > 0].sum() / -X.R[X.R <= 0].sum(),
                L=X.A[X.side == 1].mean(), S=X.A[X.side == -1].mean(), stop_pct=(X.why == 1).mean())


if __name__ == "__main__":
    VARS = [("基準", {})]
    VARS += [(f"確認 {lab}", kw) for lab, kw in (("觸價", dict(touch=True)), ("M1 收盤", dict(k=1)), ("M15 收盤", dict(k=15)))]
    VARS += [(f"緩衝 {b}", dict(buf=b)) for b in (0.0, 0.025, 0.1, 0.15, 0.2)]
    VARS += [(f"有效 {w}分", dict(win=w)) for w in (60, 120, 300, 600)]
    VARS += [(f"停損 {lab}", dict(stop=st)) for lab, st in (("中點", "mid"), ("cap0.3", ("cap", 0.3)), ("cap0.5", ("cap", 0.5)),
                                                             ("cap1.0", ("cap", 1.0)), ("atr0.2", ("atr", 0.2)), ("atr0.3", ("atr", 0.3)), ("atr0.5", ("atr", 0.5)))]
    VARS += [(f"出場 {h}:00", dict(exit_min=h * 60)) for h in (18, 20, 22, 23)]
    VARS += [(f"持有 {h}分", dict(hold=h)) for h in (60, 120, 240, 360, 600)]
    VARS += [(f"停利 {t}R", dict(tp=t)) for t in (1.0, 2.0, 3.0, 5.0)]
    VARS += [("區間 VA70", dict(rng=1)), ("區間 VA70+IB停損", dict(rng=2))]
    VARS += [(f"滑價 {s}點", dict(slip=s)) for s in (1.0, 3.0)]

    rows = []
    for name, base in BASE.items():
        for lab, kw in VARS:
            X = run_many(base["starts"], ib=base["ib"], mode=base["mode"], **kw)
            rows.append(dict(set=name, var=lab, **summ(X)))
        X = run_many(base["starts"], ib=base["ib"], mode=base["mode"])
        for n in (1, 2, 3):
            rows.append(dict(set=name, var=f"每天每方向 ≤{n}", **summ(cap(X, n))))
    R = pd.DataFrame(rows)
    R.to_csv("part38_stage2.csv", index=False)
    pd.set_option("display.width", 250)
    for name in BASE:
        print(f"\n===== {name} =====")
        print(R[R.set == name].drop(columns="set").round(3).to_string(index=False))
