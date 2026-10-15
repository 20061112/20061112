"""第二十五部分：27 個停留算法變體全部做「停損 × RR」，再對凍結候選輸出詳細交易數據。

變體：occ（close / range / tickvol）× 格寬 c（0.5 / 1 / 2）× 平滑 h（0 / 1 / 3）的 ln(p_max/p_邊緣)。
規則：M1 邊緣事件順勢追；障礙 ≥ 滾動 90 百分位（L = 20 / 60 / 120 / 擴張）；停損 9 種 × RR 6 種（同第二十四部分）。
評估：2023/11/20 起（L=120 有值以後），M1 逐根、扣點差、同時只持有一筆。
詳細交易：第二十四部分事先寫下的候選（range c=1 h=1、L=120、q=90%、ATR5、RR3），
  加上本部分看完全部 27 個變體後的 tickvol 對照（tickvol c=1 h=0、L=60、q=90%、ATR8、RR3）。
輸出 results_part25.txt、part25_grid.csv、part25_trades_*.csv、part25_equity.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from part24_rolling import (paths, outcomes, non_overlap, rolling_pct, cluster_t, STOPS, RRS, LS, H)
from part23_breakout import OCCS, CS, HS

Q = 0.9
YRS = [2023, 2024, 2025, 2026]
lines = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    lines.append(s)


E = pd.read_pickle("part23_events_M1.pkl").reset_index(drop=True)
E["year"] = E.time_close.dt.year
fav, adv, cl, atr, j0 = paths(E)
day = E.day.to_numpy()
sp = E.spread.to_numpy()
yrs = E.year.to_numpy()
VARS = [f"{o}_c{c}_h{h}_sh" for o in OCCS for c in CS for h in HS]
pct = {(v, L): rolling_pct(E, v, L) for v in VARS for L in LS}
common = np.isfinite(pct[(VARS[0], 120)])
P(f"M1 邊緣事件 {len(E)}，評估期間 {E.time_close[common].min():%Y-%m-%d} 起")

# ---- 1. 全部 27 × 4 × 9 × 6 ----
OUTC = {}
for stype, k in STOPS:
    risk = k * (E.S.to_numpy() if stype == "S" else atr)
    ok = np.isfinite(risk) & (risk > 0)
    for rr in RRS:
        OUTC[(stype, k, rr)] = (risk, ok) + outcomes(fav, adv, cl, risk, rr)
rows = []
for v in VARS:
    occ, c, h = v.split("_")[0], float(v.split("_")[1][1:]), int(v.split("_")[2][1:])
    for L in LS:
        sel0 = common & (pct[(v, L)] >= Q)
        for (stype, k, rr), (risk, ok, R0, jx) in OUTC.items():
            keep = non_overlap(sel0 & ok, day, j0, jx)
            R = R0[keep] - sp[keep] / risk[keep]
            r = dict(occ=occ, c=c, h=h, L=L, stop=f"{stype}{k}", rr=rr or 0, n=len(R), R=R.mean(),
                     t=cluster_t(R, day[keep]), pf=R[R > 0].sum() / max(1e-9, -R[R < 0].sum()),
                     R_slip30=(R0[keep] - (sp[keep] + 0.30) / risk[keep]).mean())
            for y in YRS:
                r[f"y{y}"] = R[yrs[keep] == y].mean()
            rows.append(r)
G = pd.DataFrame(rows)
G["pos3"] = (G[["y2024", "y2025", "y2026"]] > 0).all(axis=1)
G.to_csv("part25_grid.csv", index=False)
STOP_ORDER = [f"{a}{b}" for a, b in STOPS]

P("\n" + "=" * 110)
P("1. 每個變體：跨 54 種停損 × RR、4 種 L 的中位 R；『區域』= 停損 ≥ 1S 或 ATR5~8 且 RR 2~4（第二十四部分的甜蜜點）")
G["zone"] = G.stop.isin(["S1.0", "S1.5", "ATR5", "ATR8"]) & G.rr.isin([2.0, 3.0, 4.0])
V = G.groupby(["occ", "c", "h"]).apply(lambda g: pd.Series(dict(
    全部中位R=g.R.median(), 全部為正比例=(g.R > 0).mean(),
    區域中位R=g[g.zone].R.median(), 區域中位t=g[g.zone].t.median(),
    區域2024_26年年正=g[g.zone].pos3.mean(), 區域扣30美分=g[g.zone].R_slip30.median())), include_groups=False)
P(V.round(3).to_string())
P("\n依停留算法彙總（各 9 個變體的中位數）")
P(V.groupby(level="occ").median().round(3).to_string())
P("\n依格寬 / 平滑彙總")
P(V.groupby(level="c").median().round(3).to_string())
P(V.groupby(level="h").median().round(3).to_string())

P("\n" + "=" * 110)
P("2. 停損 × RR 熱圖：各停留算法 9 個變體 × 4 種 L 的中位 R")
for occ in OCCS:
    P(f"\n[{occ}]")
    P(G[G.occ == occ].pivot_table(index="stop", columns="rr", values="R", aggfunc="median").reindex(STOP_ORDER).round(3).to_string())
P("\n[tickvol] t 值中位")
P(G[G.occ == "tickvol"].pivot_table(index="stop", columns="rr", values="t", aggfunc="median").reindex(STOP_ORDER).round(2).to_string())
P("\n[tickvol] 2024~2026 年年為正的比例")
P(G[G.occ == "tickvol"].pivot_table(index="stop", columns="rr", values="pos3", aggfunc="mean").reindex(STOP_ORDER).round(2).to_string())

P("\n" + "=" * 110)
P("3. 多重檢定提醒：27 × 4 × 54 = 5,832 組裡")
P(f"  t > 2 的有 {(G.t > 2).sum()} 組（{(G.t > 2).mean():.1%}），t > 3 的有 {(G.t > 3).sum()} 組；最好的 10 組：")
P(G.sort_values("t", ascending=False).head(10)[["occ", "c", "h", "L", "stop", "rr", "n", "R", "t", "pf"] + [f"y{y}" for y in YRS]].round(3).to_string(index=False))


# ---- 4. 詳細交易 ----
def trade_list(v, L, stype, k, rr, tag):
    risk, ok, R0, jx = OUTC[(stype, k, rr)]
    keep = non_overlap(common & (pct[(v, L)] >= Q) & ok, day, j0, jx)
    idx = np.nonzero(keep)[0]
    T = E.loc[idx, ["time_close", "day", "year", "side", "close", "edge", "S", "spread", "min"]].copy()
    T["barrier"] = E.loc[idx, v].values
    T["barrier_pct"] = pct[(v, L)][idx]
    T["atr60"] = atr[idx]
    T["risk"] = risk[idx]
    T["stop"] = T.close - T.side * T.risk
    T["target"] = T.close + T.side * rr * T.risk if rr else np.nan
    T["R_gross"] = R0[idx]
    T["R"] = R0[idx] - sp[idx] / risk[idx]
    T["exit_type"] = np.where(R0[idx] <= -1 + 1e-9, "stop", np.where(np.isclose(R0[idx], rr or -99), "target", "time"))
    T["hold_min"] = jx[idx] + 1
    T["exit_time"] = T.time_close + pd.to_timedelta(T.hold_min, unit="min")
    T["exit_price"] = T.close + T.side * T.R_gross * T.risk
    T["usd_per_lot"] = (T.R_gross * T.risk - T.spread) * 100           # 1 手 = 100 盎司
    T["dir"] = np.where(T.side == 1, "多", "空")
    T.to_csv(f"part25_trades_{tag}.csv", index=False)
    return T


def max_streak(b):
    m = c = 0
    for x in b:
        c = c + 1 if x else 0
        m = max(m, c)
    return m


def report(T, label):
    P("\n" + "-" * 110)
    P(f"{label}")
    R = T.R.to_numpy()
    eq = np.cumsum(R)
    dd = eq - np.maximum.accumulate(np.r_[0, eq][1:])
    dd = (np.r_[0, eq][1:] - np.maximum.accumulate(np.r_[0, eq])[1:])
    win, loss = R[R > 0], R[R <= 0]
    span_days = (T.time_close.max() - T.time_close.min()).days
    P(f"  期間 {T.time_close.min():%Y-%m-%d} ~ {T.time_close.max():%Y-%m-%d}，{len(T)} 筆，{T.day.nunique()} 個交易日有單"
      f"（約 {len(T) / (span_days / 7):.1f} 筆/週）")
    P(f"  勝率 {len(win) / len(R):.1%}；平均賺 {win.mean():+.2f}R、平均賠 {loss.mean():+.2f}R；期望 {R.mean():+.3f}R（t {cluster_t(R, T.day):.2f}）；"
      f"PF {win.sum() / -loss.sum():.2f}")
    P(f"  累計 {eq[-1]:+.1f}R；最大回撤 {dd.min():.1f}R；最長連虧 {max_streak(R <= 0)} 筆；最長連賺 {max_streak(R > 0)} 筆")
    P(f"  出場：" + "、".join(f"{k} {v:.1%}" for k, v in T.exit_type.value_counts(normalize=True).items())
      + f"；持倉中位 {T.hold_min.median():.0f} 分（停利單 {T[T.exit_type == 'target'].hold_min.median():.0f}、"
      f"停損單 {T[T.exit_type == 'stop'].hold_min.median():.0f}）")
    P(f"  停損金額中位 ${T.risk.median():.2f}（10%~90%：${T.risk.quantile(.1):.2f}~${T.risk.quantile(.9):.2f}）；點差佔 R 中位 {(T.spread / T.risk).median():.1%}；"
      f"毛期望 {T.R_gross.mean():+.3f}R")
    P(f"  每手（100 oz）損益：平均 ${T.usd_per_lot.mean():+.1f}、總計 ${T.usd_per_lot.sum():+,.0f}"
      f"（注意：2023 金價約 2000、2026 約 4000，美元損益後期權重大）")
    for r_pct in (0.25, 0.5, 1.0):
        e = np.cumprod(1 + r_pct / 100 * R)
        m = np.maximum.accumulate(e)
        P(f"  固定風險 {r_pct}%/筆（複利）：總報酬 {e[-1] - 1:+.1%}，最大回撤 {(e / m - 1).min():.1%}")
    by = lambda col: T.groupby(col).R.agg(n="size", 勝率=lambda x: (x > 0).mean(), 平均R="mean", 累計R="sum")
    P("\n  逐年\n" + by("year").round(3).to_string())
    T["ym"] = T.time_close.dt.to_period("M")
    M = T.groupby("ym").R.sum()
    P(f"\n  逐月累計 R：{len(M)} 個月中 {(M > 0).sum()} 個月為正；最好 {M.max():+.1f}R（{M.idxmax()}）、最差 {M.min():+.1f}R（{M.idxmin()}）")
    P("  " + "  ".join(f"{p}:{v:+.1f}" for p, v in M.items()))
    P("\n  多空\n" + by("dir").round(3).to_string())
    T["session"] = pd.cut(T["min"], [0, 420, 720, 900, 1300], labels=["亞洲 03-08", "倫敦 08-13", "紐約前 13-16", "紐約後 16-22"])
    P("\n  時段（broker 時間）\n" + T.groupby("session", observed=True).R.agg(n="size", 勝率=lambda x: (x > 0).mean(), 平均R="mean", 累計R="sum").round(3).to_string())
    T["wd"] = T.time_close.dt.dayofweek.map(dict(enumerate("一二三四五六日")))
    P("\n  星期\n" + by("wd").round(3).to_string())
    T["bq"] = pd.cut(T.barrier_pct, [0.9, 0.95, 0.98, 1.0001], labels=["90~95", "95~98", "98~100"])
    P("\n  障礙百分位\n" + T.groupby("bq", observed=True).R.agg(n="size", 勝率=lambda x: (x > 0).mean(), 平均R="mean").round(3).to_string())
    P("\n  R 分佈：" + "  ".join(f"{q:.0%}:{T.R.quantile(q):+.2f}" for q in (.05, .25, .5, .75, .95)))
    last10 = T.tail(10)[["time_close", "dir", "close", "stop", "target", "exit_time", "exit_price", "exit_type", "R"]]
    P("\n  最近 10 筆\n" + last10.round(2).to_string(index=False))
    return eq


cands = [("range_c1.0_h1_sh", 120, "ATR", 5, 3.0, "A_range_c1h1_L120_ATR5_RR3",
          "候選 A（第二十四部分事先寫下）：range c=1 h=1、L=120、障礙 ≥ 90%、停損 5×ATR60、RR 3"),
         ("tickvol_c1.0_h0_sh", 60, "ATR", 8, 3.0, "B_tickvol_c1h0_L60_ATR8_RR3",
          "候選 B（看過結果後挑的 tickvol 對照）：tickvol c=1 h=0、L=60、障礙 ≥ 90%、停損 8×ATR60、RR 3")]
P("\n" + "=" * 110)
P("4. 詳細交易數據")
fig, ax = plt.subplots(figsize=(10, 4))
for v, L, st, k, rr, tag, label in cands:
    T = trade_list(v, L, st, k, rr, tag)
    eq = report(T, label)
    ax.plot(T.time_close.values, eq, label=tag)
ax.axhline(0, color="k", lw=.6); ax.set_ylabel("cumulative R"); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig("part25_equity.png", dpi=110)
open("results_part25.txt", "w").write("\n".join(lines) + "\n")
