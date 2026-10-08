"""M1 型態 + 較大週期布林外軌（多時框共振）。較大週期只用已收完的 K 棒計算帶，再和當下 M1 極值比較。"""
import numpy as np
import pandas as pd
from load import load
from backtest import resample, stats
from m1_momentum import htf_to_m1
from select5 import T, DEV, TEST

m1 = load("data/XAUUSD_M1_full.csv")
spread = m1.spread.to_numpy() * 0.01
lo, hi = m1.low.to_numpy(), m1.high.to_numpy()


def band_flags(rule, mins, n=20, k=2.0):
    h = resample(m1, rule)
    ma, sd = h.close.rolling(n).mean(), h.close.rolling(n).std()
    up, dn = htf_to_m1(m1.index, ma + k * sd, mins).to_numpy(), htf_to_m1(m1.index, ma - k * sd, mins).to_numpy()
    out = []
    for r in T.itertuples():
        seg = slice(r.t - 2, r.t + 1)
        out.append(int(lo[seg].min() < dn[r.t]) if r.side == 1 else int(hi[seg].max() > up[r.t]))
    return np.array(out)


for rule, mins in (("5min", 5), ("15min", 15), ("60min", 60)):
    T[f"out_{rule}"] = band_flags(rule, mins)

# 成本分析：毛期望 = 加回點差
T["gross_R"] = T.pnl_R + spread[T.t.to_numpy()] / T.R_usd

if __name__ == "__main__":
    split = T.entry_time < pd.Timestamp("2026-08-15")
    print(f"全部 M1 候選：淨 {T.pnl_R.mean():+.3f}R，未扣點差 {T.gross_R.mean():+.3f}R，中位 R=${T.R_usd.median():.2f}")
    days = {True: DEV.setup.dt.normalize().nunique(), False: TEST.setup.dt.normalize().nunique()}
    for nm, m in [("M5 外軌外", T.out_5min == 1), ("M15 外軌外", T["out_15min"] == 1), ("H1 外軌外", T.out_60min == 1),
                  ("M5 且 M15 外軌外", (T.out_5min == 1) & (T["out_15min"] == 1)),
                  ("M15 外軌外 + R>=$2", (T["out_15min"] == 1) & (T.R_usd >= 2))]:
        a, b = stats(T[m & split]), stats(T[m & ~split])
        print(f"{nm:18s} 研究: n={a['n']:4d} ({a['n'] / days[True]:.1f}/日) {a['expR']:+.3f}R t={a['t']:+.2f} | "
              f"驗證: n={b['n']:4d} ({b['n'] / days[False]:.1f}/日) {b['expR']:+.3f}R t={b['t']:+.2f} PF={b['pf']:.2f}")
    T.to_pickle("cand5_m1_mtf.pkl")
