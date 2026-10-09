"""第二十四部分 (3)：順勢版的門檻敏感度、多空分開、逐月、K 棒起點偏移 → results_part24_detail.txt"""
import numpy as np
import pandas as pd
from part23_ma_cycle import load_m1, st, SPLIT
from part24_er_cascade import ser, atr, signals
from part24_continuation import sim

CFG = [("15min", (20, 10, 3)), ("15min", (40, 20, 5)), ("30min", (20, 10, 3)), ("5min", (20, 10, 3))]
OFFS = {"5min": [0, 2], "15min": [0, 5, 10], "30min": [0, 10, 20]}


def rs(m1, rule, off):
    return m1.resample(rule, label="left", closed="left", offset=f"{off}min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "spread": "first"}).dropna()


m1 = load_m1()
out = []
P = out.append
for tf, (L, M, S) in CFG:
    P(f"\n===== {tf}  ER{L}/ER{M}/ER{S}  full  tp2 =====")
    rows, trades = [], []
    for off in OFFS[tf]:
        df = rs(m1, tf, off)
        o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        spr = df.spread.to_numpy(float) * 0.01
        A = atr(h, l, c)
        E = {n: ser(c, n) for n in (L, M, S)}
        for a in (0.2, 0.3, 0.4, 0.5, 0.6):
            for cc in (0.3, 0.0, -0.3, -0.6):
                t, rev = signals(E, L, M, S, a, 0.0, cc, "full")
                keep = (t > 30) & (t < len(c) - 2)
                t, side = t[keep], -rev[keep]
                R = sim(o, h, l, c, spr, A, t, side, M, 2 * L, 2)
                half = np.where(df.index[t] < SPLIT, 1, 2)
                n_, mu, tt, win, pf = st(R)
                rows.append(dict(off=off, a=a, c=cc, n=n_, R=mu, t=tt, h1=st(R[half == 1])[1],
                                 h2=st(R[half == 2])[1], long=st(R[side == 1])[1], short=st(R[side == -1])[1]))
                if off == 0 and a == 0.3 and cc == 0.0:
                    trades = pd.DataFrame(dict(time=df.index[t], side=side, R=R))
    D = pd.DataFrame(rows)
    P("門檻（K 棒偏移平均）：平均 R")
    P(D.groupby(["a", "c"]).R.mean().unstack().round(3).to_string())
    P("筆數")
    P(D.groupby(["a", "c"]).n.mean().unstack().round(0).to_string())
    P("偏移（所有門檻平均）")
    P(D.groupby("off")[["n", "R", "h1", "h2", "long", "short"]].mean().round(3).to_string())
    T = trades
    P(f"中心 a=0.3 c=0 偏移 0：n={T.R.notna().sum()} avg={T.R.mean():+.3f} win={(T.R > 0).mean():.0%} "
      f"PF={T.R[T.R > 0].sum() / -T.R[T.R < 0].sum():.2f} 多={T.R[T.side == 1].mean():+.3f}({(T.side == 1).sum()}) "
      f"空={T.R[T.side == -1].mean():+.3f}({(T.side == -1).sum()})")
    T["m"] = T.time.dt.to_period("M")
    P("逐月 R 合計 / 筆數")
    P(T.groupby("m").R.agg(["sum", "count"]).round(1).T.to_string())
open("results_part24_detail.txt", "w").write("\n".join(out) + "\n")
print("\n".join(out))
