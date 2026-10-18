"""第三十三部分：H1 黃金只做多順勢 + NAS100 波動濾網，對照一直抱多。
資料：XAUUSD M15 合成 H1；NAS100 M1 合成 H1。比較期間 2023/11/1 ~ 2026/10/9（NAS 波動 z 值要先暖機）。
樣本內 2023/11 ~ 2024/12、樣本外 2025/1 ~ 2026/10。

注意：「NAS 波動 z > 1 不做」這個濾網是在第三十二部分看過樣本外之後才提出的，所以這裡的樣本外對濾網而言**不是**乾淨的樣本外；
順勢規則與回看仍然只用樣本內選。

策略（全部只做多或空手，部位 0/1，訊號 t 收盤、賺 t→t+1，部位變動付半個點差）：
  BH            一直抱多
  BH+NAS濾網     一直抱多，但 NAS 24 小時實現波動 z > 門檻時空手 → 看濾網本身是不是就夠了
  順勢多         6 規則 × 3 回看（同第三十二部分），訊號為空時空手
  順勢多+NAS濾網
  順勢多+黃金自己波動濾網  → 看是不是 NAS 特有，還是任何「波動變大」都一樣
濾網門檻 z = 0.5 / 1 / 1.5 / 2 做敏感度。
輸出：results_part33.txt、part33_equity.png
"""
import glob
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from load import load

START, IS_END = pd.Timestamp("2023-11-01"), pd.Timestamp("2025-01-01")
OUT = open("results_part33.txt", "w")


def P(*a):
    print(*a); print(*a, file=OUT)


x15 = load("data/XAUUSD_M15_2023_2026.csv")
XH = pd.DataFrame({"close": x15.close.resample("1h").last(), "spread": x15.spread.resample("1h").last() * 0.01}).dropna()
n1 = pd.concat([load(f) for f in sorted(glob.glob("data/NAS100_M1_*.csv"))])
n1 = n1[~n1.index.duplicated(keep="last")].sort_index()
NH = n1.close.resample("1h").last().dropna()


def volz(close):
    v = np.log(close).diff().rolling(24).std()
    z = (v - v.rolling(480).mean()) / v.rolling(480).std()
    z.index = z.index + pd.Timedelta("1h")          # H1 標在開盤時間，收完才知道
    return z


def at(series, idx):
    return series.reindex(series.index.union(idx)).ffill().reindex(idx)


NVZ, XVZ = at(volz(NH), XH.index), at(volz(XH.close), XH.index)


def position(s, L, rule):
    if rule == "tsmom":
        return np.sign(s - s.shift(L)).fillna(0)
    if rule == "ma":
        return np.sign(s - s.rolling(L).mean()).fillna(0)
    v = s.to_numpy()
    pos = np.zeros(len(v))
    if rule.startswith("zbrk"):
        E = float(rule[4:])
        z = ((s - s.rolling(L).mean()) / s.rolling(L).std()).to_numpy()
        p = 0
        for i in range(len(v)):
            if p and p * z[i] <= 0: p = 0
            if not p and abs(z[i]) > E: p = np.sign(z[i])
            pos[i] = p
    else:
        hi, lo = s.rolling(L).max().shift().to_numpy(), s.rolling(L).min().shift().to_numpy()
        xhi, xlo = s.rolling(L // 2).max().shift().to_numpy(), s.rolling(L // 2).min().shift().to_numpy()
        p = 0
        for i in range(len(v)):
            if p == 1 and v[i] < xlo[i]: p = 0
            if p == -1 and v[i] > xhi[i]: p = 0
            if v[i] > hi[i]: p = 1
            elif v[i] < lo[i]: p = -1
            pos[i] = p
    return pd.Series(pos, s.index)


def run(pos, lag=1, cost_x=1.0):
    D = XH
    pos = pos.shift(lag - 1).fillna(0)
    p = pos.shift().fillna(0)
    dp = pos.diff().abs().fillna(pos.abs())
    R = pd.DataFrame({"gross": p * D.close.pct_change(), "cost": dp * D.spread / D.close / 2 * cost_x}).fillna(0) * 1e4
    R["pnl"] = R.gross - R.cost
    R["pos"], R["dp"] = pos, dp
    return R[R.index >= START]


def stats(R):
    d = R.pnl.groupby(R.index.normalize()).sum()
    eq = d.cumsum()
    yrs = len(d) / 252
    m = R.pnl.groupby(R.index.to_period("M")).sum()
    dd = (eq.cummax() - eq).max()
    return dict(sharpe=d.mean() / d.std() * np.sqrt(252), ann=d.sum() / yrs, dd=dd, calmar=d.sum() / yrs / dd,
                inmkt=R.pos.gt(0).mean(), trades=R.dp.gt(0).sum() / yrs, worst_m=m.min(), worst_d=d.min())


def fmt(s):
    return (f"Sharpe {s['sharpe']:+5.2f} 年化 {s['ann']:+6.0f}bp 最大回撤 {s['dd']:5.0f}bp 年化/回撤 {s['calmar']:4.2f} "
            f"最差月 {s['worst_m']:+5.0f} 最差日 {s['worst_d']:+5.0f} 在場 {s['inmkt']:.0%} 年換手 {s['trades']:4.0f}")


def split(R):
    return R[R.index < IS_END], R[R.index >= IS_END]


def show(name, R):
    i, o = split(R)
    P(f"  {name}")
    P(f"      樣本內 {fmt(stats(i))}")
    P(f"      樣本外 {fmt(stats(o))}")


s = np.log(XH.close)
ones = pd.Series(1.0, XH.index)
P(f"期間 {START.date()} ~ {XH.index[-1]}；樣本內 < {IS_END.date()} ≤ 樣本外；H1 {len(XH[XH.index >= START])} 根")
P(f"NAS 波動 z > 1 的時間比例：{(NVZ[XH.index >= START] > 1).mean():.0%}；黃金自己波動 z > 1：{(XVZ[XH.index >= START] > 1).mean():.0%}；"
  f"兩者同時 > 1：{((NVZ > 1) & (XVZ > 1))[XH.index >= START].mean():.0%}")

P("\n## 1. 基準與濾網本身")
show("BH 一直抱多", run(ones))
for th in [0.5, 1.0, 1.5, 2.0]:
    show(f"BH + NAS 波動 z>{th} 空手", run(ones.where(~(NVZ > th), 0.0)))
show("BH + 黃金自己波動 z>1 空手", run(ones.where(~(XVZ > 1), 0.0)))

P("\n## 2. 順勢只做多：18 組（樣本內 → 樣本外 Sharpe；括號為年化/回撤）")
rows = []
for L, rule in itertools.product([24, 120, 480], ["tsmom", "ma", "zbrk1", "zbrk1.5", "zbrk2", "donch"]):
    base = position(s, L, rule).clip(lower=0)
    for filt, f in [("無", ones.astype(bool)), ("NAS z>1", ~(NVZ > 1)), ("黃金 z>1", ~(XVZ > 1))]:
        R = run(base.where(f, 0.0))
        i, o = split(R)
        si, so = stats(i), stats(o)
        rows.append(dict(L=L, rule=rule, filt=filt, is_sh=si["sharpe"], oos_sh=so["sharpe"], is_cal=si["calmar"], oos_cal=so["calmar"],
                         oos_ann=so["ann"], oos_dd=so["dd"]))
T = pd.DataFrame(rows)
for (L, rule), g in T.groupby(["L", "rule"], sort=False):
    g = g.set_index("filt")
    P(f"  L={L:3d} {rule:7s}: " + "  ".join(
        f"{f} {g.loc[f, 'is_sh']:+.2f}→{g.loc[f, 'oos_sh']:+.2f}({g.loc[f, 'is_cal']:.1f}→{g.loc[f, 'oos_cal']:.1f})" for f in ["無", "NAS z>1", "黃金 z>1"]))
for f in ["無", "NAS z>1", "黃金 z>1"]:
    g = T[T.filt == f]
    P(f"  濾網 {f:7s}：Sharpe 中位數 IS {g.is_sh.median():+.2f} / OOS {g.oos_sh.median():+.2f}；年化/回撤 中位數 IS {g.is_cal.median():.2f} / OOS {g.oos_cal.median():.2f}")
w = T.pivot_table(index=["L", "rule"], columns="filt", values=["is_sh", "oos_sh"])
P(f"  加 NAS 濾網後 Sharpe 變好的組合：樣本內 {(w['is_sh']['NAS z>1'] > w['is_sh']['無']).sum()}/18、樣本外 {(w['oos_sh']['NAS z>1'] > w['oos_sh']['無']).sum()}/18；"
  f"黃金濾網：樣本內 {(w['is_sh']['黃金 z>1'] > w['is_sh']['無']).sum()}/18、樣本外 {(w['oos_sh']['黃金 z>1'] > w['oos_sh']['無']).sum()}/18")

best = T[T.filt == "無"].sort_values("is_sh", ascending=False).iloc[0]
L, rule = int(best.L), best.rule
P(f"\n## 3. 只用樣本內選出的規則：H1 L={L} {rule}（只做多）")
base = position(s, L, rule).clip(lower=0)
FINAL = {"BH": run(ones), "BH+NAS濾網": run(ones.where(~(NVZ > 1), 0.0)), "順勢多": run(base),
         "順勢多+NAS濾網": run(base.where(~(NVZ > 1), 0.0))}
for k, R in FINAL.items():
    show(k, R)
R = FINAL["順勢多+NAS濾網"]
P("  逐年（順勢多+NAS濾網 vs BH）：" + "  ".join(
    f"{y}: {stats(g)['sharpe']:+.2f} 回撤 {stats(g)['dd']:.0f} vs {stats(FINAL['BH'][FINAL['BH'].index.year == y])['sharpe']:+.2f} 回撤 {stats(FINAL['BH'][FINAL['BH'].index.year == y])['dd']:.0f}"
    for y, g in R.groupby(R.index.year)))
for lab, b in [("順勢多", base), ("順勢多+NAS濾網", base.where(~(NVZ > 1), 0.0))]:
    for name, kw in [("晚一根", dict(lag=2)), ("點差×2", dict(cost_x=2.0)), ("兩者", dict(lag=2, cost_x=2.0))]:
        Rx = run(b, **kw)
        P(f"  壓力 {lab} {name}: 樣本外 {fmt(stats(split(Rx)[1]))}")
R = FINAL["順勢多"]
P("  逐年（順勢多，不加濾網）：" + "  ".join(f"{y}: Sharpe {stats(g)['sharpe']:+.2f} 年化 {stats(g)['ann']:+.0f} 回撤 {stats(g)['dd']:.0f}" for y, g in R.groupby(R.index.year)))

P("\n## 4. 黃金最大的幾次下跌，各策略吃了多少（樣本外，BH 日損益最差 10 天）")
bd = FINAL["BH"].pnl.groupby(FINAL["BH"].index.normalize()).sum()
worst = bd[bd.index >= IS_END].nsmallest(10).index
for k, Rk in FINAL.items():
    d = Rk.pnl.groupby(Rk.index.normalize()).sum().reindex(worst).fillna(0)
    P(f"  {k:12s}: 合計 {d.sum():+6.0f}bp  " + " ".join(f"{v:+.0f}" for v in d.values))
P("  日期：" + " ".join(t.strftime("%y/%m/%d") for t in worst))

P(f"\n## 5. 同一條規則（H1 L={L} {rule} 只做多）用在 NAS100，對照 NAS 一直抱多（同期間，NAS 也是多頭）")
NX = pd.DataFrame({"close": NH, "spread": n1.spread.resample("1h").last().reindex(NH.index) * 0.1})
for name, pos in [("NAS 一直抱多", pd.Series(1.0, NX.index)), ("NAS 順勢多", position(np.log(NX.close), L, rule).clip(lower=0))]:
    XH_, XH = XH, NX
    show(name, run(pos))
    XH = XH_

LABEL = {"BH": "buy & hold", "BH+NAS濾網": "B&H + NAS-vol filter", "順勢多": "trend long-only", "順勢多+NAS濾網": "trend long-only + NAS-vol filter"}
fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
for k, Rk in FINAL.items():
    eq = Rk.pnl.groupby(Rk.index.normalize()).sum().cumsum()
    ax[0].plot(eq, lw=1, label=LABEL[k])
    ax[1].plot(eq - eq.cummax(), lw=1, label=LABEL[k])
ax[0].axvline(IS_END, color="k", ls="--"); ax[0].legend(fontsize=8); ax[0].set_title(f"Gold H1 long-only L={L} {rule}, NAS vol filter vs buy & hold (cumulative bp)")
ax[1].set_title("Drawdown (bp)"); ax[1].axvline(IS_END, color="k", ls="--")
plt.tight_layout(); plt.savefig("part33_equity.png", dpi=110)
