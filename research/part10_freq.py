"""第十部分補充：增加交易次數的 IB 突破掃描。
任一 broker 整點 h 起 60 分鐘當 IB；IB 後 win 小時內第一次突破進場（掛單於 IB 邊），IB 另一側停損，
當天收盤（或 hold 小時後）出場；成本 = 進場當根點差（約一個點差）。
可選：被停損後反手做另一邊突破（stop-and-reverse，一次）。
IB z = log(IB 區間) 對過去 20 天同時段 IB 的 z（只用過去）。
"""
import numpy as np, pandas as pd
from tpo import load_m1, build_days

d = load_m1()
D, _ = build_days(d)
G = {k: g for k, g in d.groupby("day") if k in D.index}
SPLIT = pd.Timestamp("2026-06-01")


def trades(h, win=4, rev=False, hold=None):
    out = []
    for day in D.index:
        g = G[day]; atr = D.atr10.get(day)
        t = g.index
        ib = g[(t.hour == h)]
        if len(ib) < 50 or not np.isfinite(atr):
            continue
        ibh, ibl = ib.high.max(), ib.low.min(); rng = ibh - ibl
        aft = g[t >= ib.index[-1] + pd.Timedelta(minutes=1)]
        if len(aft) < 30:
            continue
        H, L, C, S = aft.high.values, aft.low.values, aft.close.values, aft.spread.values * 0.01
        tm = aft.index
        wend = ib.index[-1] + pd.Timedelta(hours=win)
        inwin = tm <= wend
        up = np.where((H > ibh) & inwin)[0]; dn = np.where((L < ibl) & inwin)[0]
        if not len(up) and not len(dn):
            out.append(dict(day=day, h=h, rng=rng, ret=np.nan)); continue
        legs = []
        if len(up) and (not len(dn) or up[0] < dn[0]): legs.append((1, up[0]))
        else: legs.append((-1, dn[0]))
        res = []
        k = 0
        while k < len(legs):
            side, i0 = legs[k]
            lvl = ibh if side == 1 else ibl; stop = ibl if side == 1 else ibh
            end = len(C) - 1 if hold is None else min(len(C) - 1, i0 + hold * 60)
            seg = slice(i0, end + 1)
            hit = np.where((L[seg] <= stop) if side == 1 else (H[seg] >= stop))[0]
            if len(hit):
                px = stop; j = i0 + hit[0]
            else:
                px = C[end]; j = None
            res.append((side * (px - lvl) - S[i0]) / atr)
            if rev and j is not None and k == 0 and tm[j] <= wend:
                legs.append((-side, j))
            k += 1
        out.append(dict(day=day, h=h, rng=rng, ret=sum(res), n=len(res)))
    E = pd.DataFrame(out).set_index("day")
    lr = np.log(E.rng)
    E["z"] = (lr - lr.rolling(20).mean().shift(1)) / lr.rolling(20).std().shift(1)
    return E


def st(x):
    x = x.dropna()
    if len(x) < 5: return None
    t = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    h1 = x[x.index < SPLIT]; h2 = x[x.index >= SPLIT]
    return len(x), x.mean(), t, h1.mean(), h2.mean()


rows = []
for h in range(1, 21):
    for win in [3, 6]:
        for rev in [False, True]:
            E = trades(h, win, rev)
            for zlab, m in [("all", E.z > -99), ("z>0", E.z > 0), ("z>0.5", E.z > 0.5)]:
                r = st(E.ret[m & E.z.notna()])
                if r:
                    ntr = E.n[m & E.z.notna()].sum()
                    rows.append(dict(h=h, win=win, rev=rev, z=zlab, days=r[0], trades=int(ntr), mean=r[1], t=r[2], H1=r[3], H2=r[4]))
R = pd.DataFrame(rows)
R.to_csv("part10_freq_scan.csv", index=False)
pd.set_option("display.width", 200)
print(R[(R.t > 1.5) & (R.H1 > 0) & (R.H2 > 0)].sort_values("t", ascending=False).round(3).to_string(index=False))
print("\n每個整點（win=6, 無反手, 全部日子）")
print(R[(R.win == 6) & (~R.rev) & (R.z == "all")][["h", "days", "mean", "t", "H1", "H2"]].round(3).to_string(index=False))
