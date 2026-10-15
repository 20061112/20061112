"""第二十三部分分析：障礙高 → 順勢追突破，M1 / M3 / M5 / M15。
十分位：esc05（先往外 0.5S）對 ln(p_max/p_邊緣)。交易：M1 逐根、扣點差，門檻只用前半（2023/6~2024）。
輸出 results_part23.txt、part23_deciles.png、part23_variants.csv
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from part22_kramers import load_all
from part23_breakout import TFS, OCCS, CS, HS

lines = []


def cluster_se(y, day):
    """日群聚標準誤（平均值）。"""
    y = np.asarray(y, float); m = ~np.isnan(y)
    y, day = y[m], np.asarray(day)[m]
    r = pd.Series(y - y.mean()).groupby(day).sum()
    return np.sqrt((r ** 2).sum()) / len(y)


def deciles(x, y, q=10):
    m = np.isfinite(x) & np.isfinite(y)
    b = pd.qcut(pd.Series(x[m]).rank(method="first"), q, labels=False)
    return pd.Series(y[m]).groupby(b.values).mean()


def spearman(a, b):
    a, b = pd.Series(np.asarray(a, float)), pd.Series(np.asarray(b, float))
    return a.rank().corr(b.rank())


def mono_stats(rates):
    d = np.diff(rates.values)
    rho = spearman(rates.values, np.arange(len(rates)))
    return rho, int((d > 0).sum()), rates.iloc[-1] - rates.iloc[0]


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    lines.append(s)


d = load_all()
M1 = {k: (g.index.values, g.high.to_numpy(), g.low.to_numpy(), g.close.to_numpy()) for k, g in d.groupby("day")}


def simulate(E, sel, b=0.5, rr=2.0, tmax=240):
    """順勢：方向 = side。停損 = 進場往內 b·S；停利 = rr·R（rr=None 則只有時間出場）。"""
    out, busy = [], {}
    for i in np.nonzero(sel)[0]:
        ev = E.iloc[i]
        day, s = ev.day, int(ev.side)
        tidx, hh, ll, cc = M1[day]
        j0 = int(np.searchsorted(tidx, np.datetime64(ev.time_close)))
        if j0 >= len(cc) or busy.get(day, -1) >= j0:
            continue
        entry = ev.close
        risk = b * ev.S
        stop = entry - s * risk
        tgt = entry + s * rr * risk if rr else None
        end = min(len(cc), j0 + tmax)
        res, jx = None, end - 1
        for j in range(j0, end):
            if (ll[j] <= stop) if s == 1 else (hh[j] >= stop):
                res, jx = -risk, j; break
            if tgt is not None and ((hh[j] >= tgt) if s == 1 else (ll[j] <= tgt)):
                res, jx = rr * risk, j; break
        if res is None:
            res = s * (cc[end - 1] - entry)
        busy[day] = jx
        out.append(dict(day=day, half=ev.half, year=ev.year, R=(res - ev.spread) / risk))
    return pd.DataFrame(out)


def summ(T, label):
    if len(T) == 0:
        P(f"    {label:<40} 無交易"); return {}
    res = {}
    txt = []
    for hv in ("A", "B"):
        x = T[T.half == hv]
        if len(x) < 5:
            continue
        se = cluster_se(x.R, x.day)
        pf = x.R[x.R > 0].sum() / max(1e-9, -x.R[x.R < 0].sum())
        txt.append(f"{hv}: n={len(x):5d} 勝率 {(x.R > 0).mean():.1%} {x.R.mean():+.3f}R t {x.R.mean() / se:+.2f} PF {pf:.2f}")
        res[hv] = x.R.mean()
    P(f"    {label:<40} " + " | ".join(txt))
    P(f"    {'':<40} 逐年 " + "  ".join(f"{y}:{v:+.3f}" for y, v in T.groupby("year").R.mean().items()))
    return res


allrows = []
fig, axes = plt.subplots(1, len(TFS), figsize=(4.3 * len(TFS), 3.8), sharey=True)
for ax, tf in zip(axes, TFS):
    E = pd.read_pickle(f"part23_events_M{tf}.pkl")
    E["year"] = E.time_close.dt.year
    E["half"] = np.where(E.time_close < "2025-01-01", "A", "B")
    A = (E.half == "A").to_numpy()
    tr = (E.Deq / E.D30).replace(np.inf, np.nan)
    P("\n" + "=" * 110)
    P(f"M{tf}：{len(E)} 個邊緣事件，{E.day.nunique()} 天")
    for o in ("esc05", "esc10"):
        P(f"  整體 {o} = {E[o].mean():.4f} ± {cluster_se(E[o], E.day):.4f}（0.5 = 隨機）")
    P(f"  fwd60 往外 {E.fwd60.mean():+.4f}S ± {cluster_se(E.fwd60, E.day):.4f}   fwd120 {E.fwd120.mean():+.4f}S")

    # 變體掃描
    feats = {}
    for occ in OCCS:
        for c in CS:
            for h in HS:
                for kind in ("sh", "harm"):
                    base = E[f"{occ}_c{c}_h{h}_{kind}"]
                    feats[(occ, c, h, kind, "none")] = base.to_numpy(float)
                    feats[(occ, c, h, kind, "cc30")] = (base * tr).to_numpy(float)
    rows = []
    for key, x in feats.items():
        r = dict(tf=tf, occ=key[0], c=key[1], h=key[2], kind=key[3], temp=key[4])
        for o in ("esc05", "esc10"):
            y = E[o].to_numpy()
            r[f"{o}_rho"], r[f"{o}_inc"], r[f"{o}_d10_1"] = mono_stats(deciles(x, y))
            for hv in ("A", "B"):
                m = (E.half == hv).to_numpy()
                r[f"{o}_rho_{hv}"], _, r[f"{o}_d10_1_{hv}"] = mono_stats(deciles(x[m], y[m]))
        rows.append(r)
    V = pd.DataFrame(rows)
    allrows.append(V)
    P("  各『障礙 × 溫度』組（27 個算法變體的中位數），esc05：")
    P(V.groupby(["kind", "temp"])[["esc05_rho", "esc05_d10_1", "esc05_d10_1_A", "esc05_d10_1_B", "esc10_d10_1"]]
      .median().round(3).to_string().replace("\n", "\n    "))
    sh = V[(V.kind == "sh") & (V.temp == "none")]
    P(f"  ln(pmax/p邊緣) 的 27 個變體：esc05 D10−D1 前後半都 > 0 的有 {((sh.esc05_d10_1_A > 0) & (sh.esc05_d10_1_B > 0)).sum()} 個，"
      f"嚴格單調 {(sh.esc05_inc == 9).sum()} 個")

    # 代表變體十分位
    x = feats[("range", 1.0, 1, "sh", "none")]
    tab = pd.DataFrame({o: deciles(x, E[o].to_numpy()) for o in ("esc05", "esc10", "fwd60")})
    for hv in ("A", "B"):
        m = (E.half == hv).to_numpy()
        tab[f"esc05_{hv}"] = deciles(x[m], E.esc05.to_numpy()[m])
    tab.index = [f"D{i + 1}" for i in range(10)]
    P("  代表變體 range c=1 h=1 ln(pmax/p邊緣) 十分位：")
    P("    " + tab.round(3).to_string().replace("\n", "\n    "))
    ax.plot(range(1, 11), tab.esc05.values, "o-", label="esc05")
    ax.plot(range(1, 11), tab.esc10.values, "o-", label="esc10")
    ax.plot(range(1, 11), tab.esc05_A.values, ":", color="C0", label="esc05 2023-24")
    ax.plot(range(1, 11), tab.esc05_B.values, "--", color="C0", label="esc05 2025-26")
    ax.axhline(0.5, color="k", lw=.6); ax.set_title(f"M{tf}  ln(pmax/p_edge)", fontsize=9)
    ax.set_xlabel("decile of barrier")

    # 交易
    P("  交易（順勢追，M1 逐根，扣點差；門檻 = 前半分位數）：")
    summ(simulate(E, np.ones(len(E), bool)), "全部邊緣事件順勢（停損 0.5S、2R）")
    best_A = sh.sort_values("esc05_d10_1_A", ascending=False).iloc[0]
    cands = [(("range", 1.0, 1, "sh", "none"), "range c1 h1"),
             ((best_A.occ, best_A.c, best_A.h, "sh", "none"), f"前半最佳 {best_A.occ} c{best_A.c} h{best_A.h}")]
    for key, lab in cands:
        x = feats[key]
        for q in (0.7, 0.9):
            thr = np.nanquantile(x[A], q)
            sel = x >= thr
            summ(simulate(E, sel), f"{lab} K≥{int(q * 100)}%  0.5S/2R")
            if q == 0.9:
                summ(simulate(E, sel, b=1.0, rr=2.0), f"{lab} K≥90%  1S/2R")
                summ(simulate(E, sel, b=1.0, rr=None, tmax=120), f"{lab} K≥90%  1S/120 分鐘出場")
                summ(simulate(E, sel & (E.newext == 1).to_numpy()), f"{lab} K≥90% 且當根創新高低")
        thr = np.nanquantile(x[A], 0.3)
        summ(simulate(E, x <= thr), f"{lab} K≤30%（對照：障礙低也追）")

axes[0].set_ylabel("escape rate (outward first)"); axes[0].legend(fontsize=7)
plt.tight_layout(); plt.savefig("part23_deciles.png", dpi=110)
pd.concat(allrows).to_csv("part23_variants.csv", index=False)
open("results_part23.txt", "w").write("\n".join(lines) + "\n")
