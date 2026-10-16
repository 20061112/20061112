"""新想法的區塊拼貼蒙地卡羅（3 個月區塊、10 年、300 條，與 run_checks.py 同法）。
除了 IRR，也看「最慘相對本金」—— OR 類想法多開槓桿，風險可能是代價。輸出附加到 results_ideas.txt。"""
import numpy as np
import pandas as pd
import indicators as ind
import engine as E
from run_sigma import har_rel_daily

rng = np.random.default_rng(11)
q0 = E.load()
lr = np.log(q0.close).diff().fillna(0).values
ho = np.log(q0.high / q0.close).values; lo = np.log(q0.low / q0.close).values; oo = np.log(q0.open / q0.close).values
rf = q0.rf.values
B = 63; WARM = 1000; N = WARM + 2520            # σ̂ 需要 750 天暖機
idx = pd.bdate_range("1995-01-02", periods=N)


def path():
    pos = []
    while len(pos) < N:
        s = rng.integers(1, len(lr) - B); pos.extend(range(s, s + B))
    pos = np.array(pos[:N])
    cl = 100 * np.exp(np.cumsum(lr[pos]))
    d = pd.DataFrame(dict(close=cl, high=cl * np.exp(ho[pos]), low=cl * np.exp(lo[pos]), open=cl * np.exp(oo[pos])), index=idx)
    d["r"] = d.close.pct_change().fillna(0); d["rf"] = rf[pos]
    d["tq_syn"] = (1 + 3 * d.r - E.FEE / 252 - 2 * (d.rf * 252 + E.SPREAD) / 252).cumprod()
    p = d.index.to_period("W").to_series(index=d.index); d["week_start"] = p.ne(p.shift(1)).values
    return d


def wr(d, ok, stay=None):
    stay = ok if stay is None else stay
    return E.weights_from_rules(d, ok, ~stay)


res = []
for k in range(300):
    d = path(); cc = d.close; r = d.r
    z = E.atr_z(d); above = cc > cc.rolling(200).mean()
    sig = har_rel_daily(d, 5, "PK"); zs = ind.rz(np.log(sig))
    W = {"BASE": wr(d, above & (z < 0.5))}
    for h in [4.0, 6.0, 8.0]:
        W[f"CUSUM_OR h{h:.0f}"] = wr(d, (above | (ind.cusum_state(r, 0.25, h) > 0)) & (z < 0.5))
    W["TSMOM63_OR"] = wr(d, (above | (ind.tsmom(cc, 63) > 0).fillna(False)) & (z < 0.5))
    W["ASYM σ̂ 0.75"] = wr(d, above & (zs.fillna(z) < 0.5) & (z < 0.75), above & (z < 0.75))
    s0 = d.index[WARM]
    row = {}
    for name, w in W.items():
        st = E.stats(E.run(d, w, start=s0, z=z)); row[name] = st["irr"]; row[name + "|worst"] = st["worst"]
    trig = (z > 1.5) & (sig < r.rolling(20).std())
    zf = z.where(z <= 2, 1.9).where(~trig, 3.0)
    st = E.stats(E.run(d, W["BASE"], start=s0, z=zf)); row["PANIC σ̂"] = st["irr"]; row["PANIC σ̂|worst"] = st["worst"]
    res.append(row)
R = pd.DataFrame(res)
out = ["\n區塊拼貼蒙地卡羅（300 條 10 年路徑）：相對 BASE"]
for k in [c for c in R.columns if "|" not in c and c != "BASE"]:
    d = R[k] - R.BASE; dw = R[k + "|worst"] - R["BASE|worst"]
    out.append(f"  {k:14s} IRR 差 中位 {d.median():+.4f} 平均 {d.mean():+.4f} 勝率 {(d > 0).mean():4.0%} 5% {d.quantile(.05):+.4f} | "
               f"最慘相對本金 差 中位 {dw.median():+.3f} 5% {dw.quantile(.05):+.3f}（BASE 最慘中位 {R['BASE|worst'].median():.2f}、5% {R['BASE|worst'].quantile(.05):.2f}）")
print("\n".join(out))
open("results_ideas.txt", "a").write("\n".join(out) + "\n")
