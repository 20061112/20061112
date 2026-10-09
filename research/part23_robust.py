"""第二十三部分 (2)：M15 / M30 穩健性 — 均線長度鄰域、K 棒起點偏移、循環 vs 對照、逐月。

長度鄰域：快 {5,8,10,13} × 中 {15,21,26,34} × 慢 {40,55,80}（快<中<慢）× SMA/EMA
K 棒起點偏移：M15 偏移 0/5/10 分、M30 偏移 0/10/20 分（同一份 M1，只換切法）
其他參數：發散 0.8/0.9 × 密集 0.1/0.25 × 間隔 1/3 × 出場 tp2/tp3/trail，停損 = 區間另一側
→ results_part23_robust.txt、part23_robust.csv、part23_center_trades.csv
"""
import itertools
import numpy as np
import pandas as pd
from part23_ma_cycle import load_m1, ma_block, events, simulate, st, SPLIT

FASTS, MIDS, SLOWS = [5, 8, 10, 13], [15, 21, 26, 34], [40, 55, 80]
OFFS = {"15min": [0, 5, 10], "30min": [0, 10, 20]}


def rs(m1, rule, off):
    return m1.resample(rule, label="left", closed="left", offset=f"{off}min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "spread": "first"}).dropna()


def main():
    m1 = load_m1()
    rows, center = [], []
    for tf, offs in OFFS.items():
        for off in offs:
            df = rs(m1, tf, off)
            for typ, f, m, s in itertools.product(["sma", "ema"], FASTS, MIDS, SLOWS):
                if not f < m < s:
                    continue
                lens = (f, m, s)
                B = ma_block(df, typ, lens)
                var = [("cycle", dv, sq, g) for dv in (0.8, 0.9) for sq in (0.1, 0.25) for g in (1, 3)] + \
                      [("base", np.nan, sq, np.nan) for sq in (0.1, 0.25)]
                for kind, dv, sq, g in var:
                    sig = events(df, B, lens, dv, sq, g, need_div=kind == "cycle")
                    if not sig:
                        continue
                    ti = np.array([x[0] for x in sig])
                    half = np.where(df.index[ti] < SPLIT, 1, 2)
                    rel = np.array([("cont" if x[1] == x[4] else "rev") if kind == "cycle" else "all" for x in sig])
                    for ex in ("tp2", "tp3", "trail"):
                        R = np.array([r[0] for r in simulate(df, sig, B["slow"], ex, "box", 10 * s)])
                        for grp in (["all", "cont", "rev"] if kind == "cycle" else ["all"]):
                            mk = np.ones(len(R), bool) if grp == "all" else rel == grp
                            n, mu, t, win, pf = st(R[mk])
                            rows.append(dict(tf=tf, off=off, type=typ, f=f, m=m, s=s, kind=kind, div=dv, sq=sq,
                                             gap=g, exit=ex, dir=grp, n=n, avgR=mu, t=t, win=win, pf=pf,
                                             h1=st(R[mk & (half == 1)])[1], h2=st(R[mk & (half == 2)])[1]))
                        if (tf, off, f, m, s, kind, dv, sq, g, ex) == ("15min", 0, 8, 21, 55, "cycle", 0.8, 0.25, 1, "trail"):
                            for x, r in zip(sig, R):
                                center.append(dict(type=typ, time=df.index[x[0]], side=x[1], prior=x[4],
                                                   box_hi=x[2], box_lo=x[3], R=r))
    D = pd.DataFrame(rows)
    D.to_csv("part23_robust.csv", index=False)
    pd.DataFrame(center).to_csv("part23_center_trades.csv", index=False)

    out = []
    P = out.append

    def reg(df, by):
        g = df.groupby(by)
        return pd.DataFrame({"組數": g.size(), "平均筆數": g.n.mean().round(0), "全期R": g.avgR.mean().round(3),
                             "H1": g.h1.mean().round(3), "H2": g.h2.mean().round(3),
                             "全期>0%": g.apply(lambda x: (x.avgR > 0).mean() * 100, include_groups=False).round(0),
                             "兩半皆正%": g.apply(lambda x: ((x.h1 > 0) & (x.h2 > 0)).mean() * 100,
                                              include_groups=False).round(0)})

    P("=== A. 循環(依方向) vs 對照，週期 × 均線種類（鄰域內所有長度、所有偏移平均）===")
    P(reg(D, ["tf", "type", "kind", "dir"]).to_string())
    P("\n=== B. K 棒起點偏移（循環 cont）===")
    P(reg(D[D.dir == "cont"], ["tf", "off", "type"]).to_string())
    P("\n=== C. 慢線長度（循環 cont）===")
    P(reg(D[D.dir == "cont"], ["tf", "type", "s"]).to_string())
    P("\n=== D. 快線長度（循環 cont）===")
    P(reg(D[D.dir == "cont"], ["tf", "type", "f"]).to_string())
    P("\n=== E. 中線長度（循環 cont）===")
    P(reg(D[D.dir == "cont"], ["tf", "type", "m"]).to_string())
    P("\n=== F. 出場（循環 cont vs 對照 all）===")
    P(reg(D[D.dir.isin(["cont"]) | (D.kind == "base")], ["tf", "kind", "exit"]).to_string())
    P("\n=== G. 同一組（週期/偏移/種類/長度/出場/密集門檻）配對：循環 cont − 對照 ===")
    key = ["tf", "off", "type", "f", "m", "s", "sq", "exit"]
    b = D[D.kind == "base"].set_index(key)[["avgR"]].rename(columns={"avgR": "base"})
    c = D[(D.kind == "cycle") & (D.dir == "cont")].join(b, on=key)
    c["diff"] = c.avgR - c.base
    P(c.groupby(["tf", "type"]).agg(cont=("avgR", "mean"), base=("base", "mean"), diff=("diff", "mean"),
                                    cont較好比例=("diff", lambda x: (x > 0).mean())).round(3).to_string())

    T = pd.DataFrame(center)
    if len(T):
        T["month"] = pd.to_datetime(T.time).dt.to_period("M")
        T["rel"] = np.where(T.side == T.prior, "cont", "rev")
        P("\n=== H. 中心參數逐月（M15、8/21/55、發散 0.8、密集 0.25、間隔 1、trail、區間停損）R 合計 / 筆數 ===")
        P(T.pivot_table(index="month", columns=["type", "rel"], values="R", aggfunc=["sum", "count"]).round(2).to_string())
    open("results_part23_robust.txt", "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
