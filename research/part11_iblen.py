"""第十一部分補充：IB 長度 15 / 30 / 60 分（5 分字母的 3 / 6 / 12 格）的突破延續，broker 01~10 點每個整點。
規則同 part10_freq（3 小時內首次突破、IB 另一側停損、收盤出場、扣一個點差、不反手）。"""
import numpy as np, pandas as pd
from tpo import load_m1, build_days

d = load_m1(); D, _ = build_days(d)
G = {k: g for k, g in d.groupby("day") if k in D.index}
SPLIT = pd.Timestamp("2026-06-01")


def run(h, mins, win=3):
    out = []
    for day in D.index:
        g = G[day]; atr = D.atr10.get(day)
        if not np.isfinite(atr): continue
        t0 = day + pd.Timedelta(hours=h); t1 = t0 + pd.Timedelta(minutes=mins)
        ib = g[(g.index >= t0) & (g.index < t1)]; aft = g[g.index >= t1]
        if len(ib) < mins * 0.8 or len(aft) < 30: continue
        ibh, ibl = ib.high.max(), ib.low.min()
        H, L, C, S = aft.high.values, aft.low.values, aft.close.values, aft.spread.values * 0.01
        inw = aft.index < t1 + pd.Timedelta(hours=win)
        up = np.where((H > ibh) & inw)[0]; dn = np.where((L < ibl) & inw)[0]
        if not len(up) and not len(dn): continue
        side, i0 = (1, up[0]) if len(up) and (not len(dn) or up[0] < dn[0]) else (-1, dn[0])
        lvl, stop = (ibh, ibl) if side == 1 else (ibl, ibh)
        hit = np.where((L[i0:] <= stop) if side == 1 else (H[i0:] >= stop))[0]
        px = stop if len(hit) else C[-1]
        out.append(dict(day=day, h=h, ret=(side * (px - lvl) - S[i0]) / atr))
    return pd.DataFrame(out).set_index("day")


for mins in (15, 30, 60):
    E = pd.concat([run(h, mins) for h in range(1, 11)])
    day = E.groupby(level=0).ret.sum()
    t = lambda x: x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    per_h = E.groupby("h").ret.mean().round(3).to_dict()
    print(f"IB {mins:2d} 分：{len(E)} 筆（{len(E) / len(day):.1f}/天）每筆 {E.ret.mean():+.3f} ATR t(筆)={t(E.ret):+.2f} t(日)={t(day):+.2f}"
          f"  H1 {E.ret[E.index < SPLIT].mean():+.3f} H2 {E.ret[E.index >= SPLIT].mean():+.3f}")
    print("        各整點每筆：", per_h)
