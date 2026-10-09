"""第十三部分：原始疊單版（每個 IB 各自一筆，約 9.7 筆/天）+ 進場當下特徵 → 找能提高每筆平均損益的篩選。

交易規則（同 part11 的 60 分 IB）：broker 01~10 點每個整點起 60 分鐘當 IB；IB 結束後 3 小時內第一次突破，
在 IB 邊進場；停損 IB 另一側；當天收盤出場；扣進場 K 棒點差。單位：美元/盎司（= 0.01 手美元）。
特徵全部只用進場那一刻（含）之前的資料。篩選只用 1~5 月挑，6~10 月驗證。
"""
import numpy as np
import pandas as pd
from tpo import load_m1, build_days, TICK

d = load_m1()
D, PROF = build_days(d)
G = {k: g for k, g in d.groupby("day") if k in D.index}
DAYS = list(D.index)
SPLIT = pd.Timestamp("2026-06-01")


def build(hours=range(1, 11), ib_min=60, win=3):
    rows = []
    ibr = {h: [] for h in hours}
    for di, day in enumerate(DAYS):
        g = G[day]; atr = D.atr10.iloc[di]
        prev = D.iloc[di - 1] if di > 0 else None
        t = g.index; O, H, L, C, S = g.open.values, g.high.values, g.low.values, g.close.values, g.spread.values * 0.01
        for h in hours:
            t0 = day + pd.Timedelta(hours=h); t1 = t0 + pd.Timedelta(minutes=ib_min)
            m = (t >= t0) & (t < t1)
            if m.sum() < ib_min * 0.8:
                continue
            ibh, ibl = H[m].max(), L[m].min(); rng = ibh - ibl
            hist = ibr[h][-20:]
            ib_z = (np.log(rng) - np.mean(np.log(hist))) / np.std(np.log(hist)) if len(hist) >= 10 else np.nan
            ibr[h].append(rng)
            if prev is None or not np.isfinite(atr):
                continue
            w = np.where((t >= t1) & (t < t1 + pd.Timedelta(hours=win)))[0]
            if not len(w):
                continue
            up = w[H[w] > ibh]; dn = w[L[w] < ibl]
            if len(up) and (not len(dn) or up[0] < dn[0]): side, i, lvl, stop = 1, up[0], ibh, ibl
            elif len(dn): side, i, lvl, stop = -1, dn[0], ibl, ibh
            else: continue
            hit = np.where((L[i:] <= stop) if side == 1 else (H[i:] >= stop))[0]
            j = i + hit[0] if len(hit) else len(C) - 1
            px = stop if len(hit) else C[-1]
            pnl = side * (px - lvl) - S[i]
            # ---- 進場當下特徵 ----
            hi_before, lo_before = H[:i].max(), L[:i].min()
            pr, pc, _, cz = PROF[DAYS[di - 1]]
            k = int(np.floor((lvl - pr[0]) / TICK))
            ref60 = C[max(i - 60, 0)]; ref240 = C[max(i - 240, 0)]
            rows.append(dict(
                day=day, h=h, side=side, t_in=t[i], t_out=t[j], pnl=pnl, R=pnl / rng, stopped=len(hit) > 0,
                ib_atr=rng / atr, ib_z=ib_z, delay=int((t[i] - t1).total_seconds() // 60),
                day_trend=side * (lvl - O[0]) / atr,                      # + = 順著今天開盤以來的方向
                new_ext=float((lvl >= hi_before) if side == 1 else (lvl <= lo_before)),   # 突破就是當天新高/新低
                mom60=side * (lvl - ref60) / atr, mom240=side * (lvl - ref240) / atr,
                prev_dir=side * np.sign(prev.close - prev.open),
                va_z=side * (lvl - prev.mu) / prev.sd,                    # + = 往前日價值外面突破
                tree_z=cz[k] if 0 <= k < len(cz) else np.nan,
                spread=S[i]))
    T = pd.DataFrame(rows)
    T = T.sort_values("t_in").reset_index(drop=True)
    # 同一天、進場當下已持有的同向 / 反向單數
    same, opp = [], []
    for r in T.itertuples():
        o = T[(T.day == r.day) & (T.t_in < r.t_in) & (T.t_out > r.t_in)]
        same.append(int((o.side == r.side).sum())); opp.append(int((o.side != r.side).sum()))
    T["open_same"], T["open_opp"] = same, opp
    T["is"] = T.day < SPLIT
    return T


def summ(x):
    p = x.pnl
    if not len(p):
        return "n=0"
    pf = p[p > 0].sum() / -p[p <= 0].sum() if (p <= 0).any() else np.inf
    return f"n={len(p):4d} avg={p.mean():+6.2f} PF={pf:4.2f} win={(p > 0).mean() * 100:3.0f}%"


if __name__ == "__main__":
    out = open("results_part13.txt", "w")
    def P(s=""): print(s); out.write(s + "\n")
    T = build()
    T.to_csv("part13_trades.csv", index=False)
    IS, OOS = T[T["is"]], T[~T["is"]]
    days_is, days_oos = IS.day.nunique(), OOS.day.nunique()
    P("原始疊單版（IB 60 分、01~10 點、3 小時窗、IB 另一側停損、收盤出場）")
    P(f"  全期 {summ(T)}  ({len(T) / T.day.nunique():.1f} 筆/天)")
    P(f"  1~5 月 {summ(IS)}   6~10 月 {summ(OOS)}")

    FEATS = ["h", "ib_atr", "ib_z", "delay", "day_trend", "new_ext", "mom60", "mom240", "prev_dir", "va_z",
             "tree_z", "open_same", "open_opp"]
    P("\n各特徵分組（分界用 1~5 月的五分位），每格：1~5 月 avg | 6~10 月 avg（美元/盎司）")
    for f in FEATS:
        x = T[f]
        if x.nunique() <= 10:
            bins = sorted(x.dropna().unique()); lab = [str(b) for b in bins]
            grp = x
        else:
            edges = np.unique(np.nanquantile(IS[f], [0, .2, .4, .6, .8, 1]))
            edges[0], edges[-1] = -np.inf, np.inf
            grp = pd.cut(x, edges)
            bins = grp.cat.categories; lab = [f"{b.left:.2f}~{b.right:.2f}" for b in bins]
        cells = []
        for b, l in zip(bins, lab):
            a = T[(grp == b) & T["is"]].pnl; o = T[(grp == b) & ~T["is"]].pnl
            cells.append(f"[{l}] {a.mean():+.1f}({len(a)}) | {o.mean():+.1f}({len(o)})")
        P(f"  {f:9s} " + "   ".join(cells))
    out.close()
