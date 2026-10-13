"""第二十四部分 B：候選組合 vs 隨機基準。

基準：隨機挑選「日線趨勢狀態相同」的 K 棒，以開盤價進場，方向、停損距離（ATR 倍數）、停利 R 倍數都與原交易相同。
這樣扣掉的是「黃金多頭 drift」與「順日線趨勢」本身的報酬，剩下的才是 1-2-3 / 2B 型態的貢獻。
"""
import numpy as np
import pandas as pd

import part24_123_2b as P
from load import load

cache = {}


def trades_for(row):
    key = (row.pattern, row.tf, row.swing)
    if key not in cache:
        d = P.TF(P.M1, row.tf)
        cache[key] = (d, P.run_123(d, row.swing) if row.pattern == "123" else P.run_2b(d, row.swing))
    d, t = cache[key]
    if row.pattern == "123":
        x = t[(t.entry == row.entry) & (t.stop == row.stop) & (t.tp == str(row.tp))]
        if row.f1 != "none":
            x = x[x.ext50]
        if row.f1 == "low50+drop4":
            x = x[x.move >= 4]
        if row.f2 == "sper":
            x = x[x.sper]
    else:
        x = t[(t.entry == row.entry) & (t.tp == str(row.tp)) & (t.w <= int(row.stop[1:]))]
        if row.f1 == "pen<=1":
            x = x[x.pen <= 1]
        if row.f2 == "sig50":
            x = x[x.sig50]
    x = P.apply_trend(x, row.trend)
    return d, P.no_overlap(x)


def baseline(d, tr, reps=10, seed=1):
    rng = np.random.default_rng(seed)
    hold = P.HOLD[d.tf]
    ok = np.where(d.atr > 0)[0]
    ok = ok[(ok > 60) & (ok < d.n - 2)]
    pools = {s: ok[d.trend[ok] == s] for s in (-1, 0, 1)}
    out = []
    for _ in range(reps):
        for side, trd, ra, tr_r, tm in zip(tr.side, tr.trend, tr.risk_atr, tr.tp_r, tr.time):
            t = int(rng.choice(pools[trd]))
            a = d.start[t + 1]
            px = d.mo[a]
            risk = ra * d.atr[t]
            stop = px - side * risk
            tp = None if np.isnan(tr_r) else px + side * tr_r * risk
            xp, why, xa = P.sim_exit(d, a, t + 1, side, px, stop, tp, hold)
            out.append((d.time[t], side, (side * (xp - px) - d.mspr[a]) / risk))
    return pd.DataFrame(out, columns=["time", "side", "pnl_r"])



def main():
    P.M1 = load("data/XAUUSD_M1_2023_2026.csv")
    scan = pd.read_csv("part24_scan.csv")
    scan = scan[(scan.IS_n >= 30) & (scan.OOS_n >= 30)]
    both = scan[(scan.IS_t > 2) & (scan.OOS_t > 2)]
    top_is = scan[scan.IS_n >= 100].sort_values("IS_t", ascending=False).groupby(["pattern", "tf"]).head(3)
    cands = pd.concat([both, top_is]).drop_duplicates(
        subset=["pattern", "tf", "swing", "entry", "stop", "tp", "f1", "f2", "trend"])

    rows = []
    for _, row in cands.iterrows():
        d, tr = trades_for(row)
        b = baseline(d, tr)
        r = row[["pattern", "tf", "swing", "entry", "stop", "tp", "f1", "f2", "trend"]].to_dict()
        for tag, a, bb in (("IS", tr[tr.time < P.IS_END], b[b.time < P.IS_END]),
                           ("OOS", tr[tr.time >= P.IS_END], b[b.time >= P.IS_END])):
            r[f"{tag}_n"] = len(a)
            r[f"{tag}_R"] = a.pnl_r.mean()
            r[f"{tag}_rand"] = bb.pnl_r.mean()
            diff = a.pnl_r.mean() - bb.pnl_r.mean()
            r[f"{tag}_edge"] = diff
            r[f"{tag}_edge_t"] = diff / np.sqrt(a.pnl_r.var() / len(a) + bb.pnl_r.var() / len(bb))
        rows.append(r)
    res = pd.DataFrame(rows).sort_values(["pattern", "tf", "OOS_edge_t"], ascending=[True, True, False])
    res.to_csv("part24_baseline.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 100)
    print(res.round(3).to_string(index=False))
    print("\n樣本內、樣本外 edge 都 > 0 的候選：", int(((res.IS_edge > 0) & (res.OOS_edge > 0)).sum()), "/", len(res))
    print("樣本內、樣本外 edge t 都 > 2 的候選：", int(((res.IS_edge_t > 2) & (res.OOS_edge_t > 2)).sum()), "/", len(res))


if __name__ == "__main__":
    main()
