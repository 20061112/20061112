"""第三十一部分：NAS100 / XAUUSD 比值順勢（第三十部分的反面）。
資料同第二十八~三十部分（NAS100 M1 合成 × XAUUSD M15，2023/10/2 ~ 2026/10/9），合成 H1、H4、D1。

比值 lr = log(NAS / XAU)。部位 +1 = 多 NAS、空 XAU（同名目），−1 反之；損益以總名目計（每腳 1/2）。
規則（訊號在 t 收盤決定，賺 t → t+1 的報酬）：
  tsmom ：sign(lr_t − lr_{t−L})，一直在場
  ma    ：sign(lr_t − SMA_L(lr))，一直在場
  zbrk  ：z = (lr − 滾動平均)/滾動標準差（L 根）；z > E 做多比值、z < −E 做空，z 回到 0 出場（E = 1、1.5、2）
  donch ：lr 突破過去 L 根最高 → 多；跌破最低 → 空；反向跌破 L/2 根通道出場
成本：部位每變動 1 單位，兩腳各付半個點差（翻倉 = 2 單位）。

檢查：
  1. 樣本內 2023/10 ~ 2024/12 vs 樣本外 2025/1 ~ 2026/10，全部組合的分佈與排名相關
  2. 比值本身的漂移（一直多 / 一直空比值）當基準
  3. 拆腳：賺的是 NAS 腳還是 XAU 腳？同一套規則只做單一商品（NAS 自己順勢、XAU 自己順勢）比一比
  4. 樣本內前幾名逐年
輸出：results_part31.txt、part31_grid.csv、part31_equity.png
"""
import glob
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from load import load

IS_END = pd.Timestamp("2025-01-01")
OUT = open("results_part31.txt", "w")


def P(*a):
    print(*a); print(*a, file=OUT)


n1 = pd.concat([load(f) for f in sorted(glob.glob("data/NAS100_M1_*.csv"))])
n1 = n1[~n1.index.duplicated(keep="last")].sort_index()
n15 = pd.DataFrame({"close": n1.close.resample("15min").last(), "spread": n1.spread.resample("15min").last() * 0.1}).dropna()
x15 = load("data/XAUUSD_M15_2023_2026.csv")[["close", "spread"]].assign(spread=lambda d: d.spread * 0.01)
M15 = pd.concat({"nas": n15, "xau": x15}, axis=1, sort=True).dropna()
M15.columns = ["n", "n_sp", "x", "x_sp"]
DATA = {"M15": M15, "H1": M15.resample("1h").last().dropna(), "H4": M15.resample("4h").last().dropna(),
        "D1": M15.resample("1D").last().dropna()}
LOOKBACK = {"M15": [96, 384, 960], "H1": [24, 120, 480], "H4": [30, 120, 360], "D1": [20, 60, 120]}
RULES = ["tsmom", "ma", "zbrk1", "zbrk1.5", "zbrk2", "donch"]


def position(s, L, rule):
    """s：要做順勢的 log 價格序列；回傳 t 收盤後的部位（−1/0/+1）。"""
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
            if not p and np.isfinite(z[i]) and abs(z[i]) > E: p = np.sign(z[i])
            pos[i] = p
    else:  # donch
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


def run(D, L, rule, what="pair", lag=1, cost_x=1.0):
    """what：pair = 比值；nas / xau = 同一套規則只做單一商品（全名目）。回傳每根損益 bp 的 DataFrame。"""
    rn, rx = D.n.pct_change(), D.x.pct_change()
    cn, cx = D.n_sp / D.n / 2 * cost_x, D.x_sp / D.x / 2 * cost_x   # 半個點差
    if what == "pair":
        pos = position(np.log(D.n / D.x), L, rule).shift(lag - 1).fillna(0)
        p = pos.shift().fillna(0)
        dp = pos.diff().abs().fillna(pos.abs())
        out = pd.DataFrame({"nas": p * rn / 2, "xau": -p * rx / 2, "cost": dp * (cn + cx) / 2})
    else:
        px, r, c = (D.n, rn, cn) if what == "nas" else (D.x, rx, cx)
        pos = position(np.log(px), L, rule).shift(lag - 1).fillna(0)
        p = pos.shift().fillna(0)
        dp = pos.diff().abs().fillna(pos.abs())
        out = pd.DataFrame({"nas": p * r if what == "nas" else 0.0, "xau": p * r if what == "xau" else 0.0, "cost": dp * c})
    out = out.fillna(0) * 1e4
    out["pnl"] = out.nas + out.xau - out.cost
    out["pos"] = pos
    out["dp"] = dp
    return out


def stats(R):
    d = R.pnl.groupby(R.index.normalize()).sum()
    eq = d.cumsum()
    yrs = len(d) / 252
    return dict(sharpe=d.mean() / d.std() * np.sqrt(252) if d.std() > 0 else np.nan, ann_bp=d.sum() / yrs,
                dd_bp=(eq.cummax() - eq).max(), trades=R.dp.gt(0).sum() / yrs, inmkt=R.pos.ne(0).mean(),
                gross=(R.nas + R.xau).sum() / yrs, cost=R.cost.sum() / yrs)


def fmt(s):
    return (f"Sharpe {s['sharpe']:+5.2f} 年化 {s['ann_bp']:+6.0f}bp（扣成本前 {s['gross']:+6.0f}、成本 {s['cost']:4.0f}）"
            f" 回撤 {s['dd_bp']:5.0f}bp 年換手 {s['trades']:5.0f} 在場 {s['inmkt']:.0%}")


def split(R):
    return R[R.index < IS_END], R[R.index >= IS_END]


P(f"資料：共同 M15 {len(M15)} 根，{M15.index[0]} ~ {M15.index[-1]}；樣本內 < {IS_END.date()} ≤ 樣本外")
P("損益單位：bp（總名目，每腳 1/2）；年化 = 每年合計 bp。參考：黃金年化波動約 1,500~2,000bp，這裡每腳半名目。")

P("\n## 0. 基準：比值本身的漂移（一直持有，不換手）")
D1 = DATA["D1"]
for name, sgn in [("一直多比值（多 NAS 空 XAU）", 1), ("一直空比值（空 NAS 多 XAU）", -1)]:
    r = sgn * (D1.n.pct_change() - D1.x.pct_change()) / 2 * 1e4
    for per, m in [("樣本內", r.index < IS_END), ("樣本外", r.index >= IS_END)]:
        g = r[m].dropna()
        P(f"  {name} {per}：年化 {g.mean() * 252:+6.0f}bp Sharpe {g.mean() / g.std() * np.sqrt(252):+.2f}")

rows, ALL = [], {}
for tf, D in DATA.items():
    for L, rule in itertools.product(LOOKBACK[tf], RULES):
        for what in ["pair", "nas", "xau"]:
            R = run(D, L, rule, what)
            ALL[(tf, L, rule, what)] = R
            for per, g in zip(["IS", "OOS"], split(R)):
                rows.append(dict(tf=tf, L=L, rule=rule, what=what, per=per, **stats(g)))
G = pd.DataFrame(rows)
G.to_csv("part31_grid.csv", index=False)
W = G.pivot_table(index=["tf", "L", "rule", "what"], columns="per", values=["sharpe", "ann_bp", "gross", "cost"])
Wp = W.xs("pair", level="what")

P(f"\n## 1. 比值順勢：全部 {len(Wp)} 組（4 週期 × 3 回看 × 6 規則）")
P(f"  樣本內 Sharpe > 0：{(Wp['sharpe']['IS'] > 0).sum()}/{len(Wp)}；樣本外 > 0：{(Wp['sharpe']['OOS'] > 0).sum()}/{len(Wp)}；"
  f"兩段都 > 0：{((Wp['sharpe']['IS'] > 0) & (Wp['sharpe']['OOS'] > 0)).sum()}")
P(f"  Sharpe 中位數：樣本內 {Wp['sharpe']['IS'].median():+.2f}、樣本外 {Wp['sharpe']['OOS'].median():+.2f}；"
  f"扣成本前樣本外 > 0：{(Wp['gross']['OOS'] > 0).sum()}/{len(Wp)}")
P(f"  樣本內 vs 樣本外 Sharpe 排名相關：{Wp['sharpe']['IS'].corr(Wp['sharpe']['OOS'], method='spearman'):+.2f}")
for lvl in ["tf", "rule"]:
    g = Wp["sharpe"].groupby(level=lvl).median()
    P(f"  依 {lvl:4s}（Sharpe 中位數 IS / OOS）：" + "  ".join(f"{k}: {v['IS']:+.2f} / {v['OOS']:+.2f}" for k, v in g.iterrows()))

P("\n## 2. 每一組（Sharpe：樣本內 → 樣本外；年化 bp 樣本外）")
for tf in DATA:
    for L in LOOKBACK[tf]:
        P(f"  {tf:3s} L={L:3d}: " + "  ".join(
            f"{r} {Wp.loc[(tf, L, r), ('sharpe', 'IS')]:+.1f}→{Wp.loc[(tf, L, r), ('sharpe', 'OOS')]:+.1f}({Wp.loc[(tf, L, r), ('ann_bp', 'OOS')]:+.0f})"
            for r in RULES))

P("\n## 3. 拆腳與單一商品對照（樣本外，所有組合中位數）")
for what, label in [("pair", "比值順勢"), ("nas", "只做 NAS 自己順勢（全名目）"), ("xau", "只做 XAU 自己順勢（全名目）")]:
    w = W.xs(what, level="what")
    P(f"  {label:22s}：Sharpe 中位數 IS {w['sharpe']['IS'].median():+.2f} / OOS {w['sharpe']['OOS'].median():+.2f}；"
      f"樣本外 > 0 {(w['sharpe']['OOS'] > 0).sum()}/{len(w)}")
leg = pd.DataFrame({k[:3]: {"nas": split(R)[1].nas.sum(), "xau": split(R)[1].xau.sum()} for k, R in ALL.items() if k[3] == "pair"}).T
P(f"  比值順勢樣本外的兩隻腳：NAS 腳合計 > 0 的組合 {(leg.nas > 0).mean():.0%}（中位數 {leg.nas.median():+.0f}bp），"
  f"XAU 腳 > 0 {(leg.xau > 0).mean():.0%}（中位數 {leg.xau.median():+.0f}bp）")
pr = Wp["sharpe"]["OOS"]
xa = W.xs("xau", level="what")["sharpe"]["OOS"]
P(f"  比值順勢 vs 只做 XAU 順勢（同週期同回看同規則）樣本外 Sharpe 的相關：{pr.corr(xa):+.2f}；比值較好的組合 {(pr > xa).mean():.0%}")

P("\n## 4. 樣本內前 8 名（比值）→ 樣本外與逐年 Sharpe")
top = Wp.sort_values(("sharpe", "IS"), ascending=False).head(8)
for k in top.index:
    R = ALL[(*k, "pair")]
    i, o = split(R)
    yr = "  ".join(f"{y}:{stats(g)['sharpe']:+.1f}" for y, g in R.groupby(R.index.year))
    P(f"  {k[0]} L={k[1]} {k[2]:7s} IS  {fmt(stats(i))}")
    P(f"  {'':18s} OOS {fmt(stats(o))}  | 逐年 {yr}")
    lo, sh = o[o.pos.shift() > 0].pnl.sum(), o[o.pos.shift() < 0].pnl.sum()
    P(f"  {'':18s}     樣本外 多比值 {lo:+.0f}bp / 空比值 {sh:+.0f}bp")

P("\n## 5. 壓力測試：樣本內前 8 名，晚一根進場 / 點差 ×2 / 兩者都有（樣本外 Sharpe、年化 bp）；以及只做 XAU 同規則")
for k in top.index:
    cells = []
    for name, kw in [("原本", {}), ("晚一根", dict(lag=2)), ("點差×2", dict(cost_x=2.0)), ("兩者", dict(lag=2, cost_x=2.0))]:
        st = stats(split(run(DATA[k[0]], k[1], k[2], "pair", **kw))[1])
        cells.append(f"{name} {st['sharpe']:+.2f}({st['ann_bp']:+.0f})")
    sx = stats(split(ALL[(*k, "xau")])[1])
    sp = stats(split(ALL[(*k, "pair")])[1])
    P(f"  {k[0]:3s} L={k[1]:3d} {k[2]:7s}: " + "  ".join(cells) +
      f"  | 只做 XAU Sharpe {sx['sharpe']:+.2f} 回撤/年化 {sx['dd_bp'] / sx['ann_bp']:.2f} vs 比值 {sp['dd_bp'] / sp['ann_bp']:.2f}")

fig, ax = plt.subplots(2, 1, figsize=(11, 7))
for k in top.index[:5]:
    R = ALL[(*k, "pair")]
    ax[0].plot(R.pnl.groupby(R.index.normalize()).sum().cumsum(), lw=1, label=f"{k[0]} L={k[1]} {k[2]}")
lr = np.log(D1.n / D1.x)
ax0b = ax[0].twinx(); ax0b.plot((lr - lr.iloc[0]) * 1e4, color="grey", lw=.8, ls=":", label="log(NAS/XAU)")
ax[0].axvline(IS_END, color="k", ls="--"); ax[0].legend(fontsize=7, loc="upper left"); ax0b.legend(fontsize=7, loc="lower left")
ax[0].set_title("Top-5 in-sample ratio-momentum combos, cumulative bp (dotted = ratio, right axis)")
ax[1].scatter(Wp["sharpe"]["IS"], Wp["sharpe"]["OOS"], s=12)
ax[1].axhline(0, color="k", lw=.5); ax[1].axvline(0, color="k", lw=.5)
ax[1].set_xlabel("in-sample Sharpe"); ax[1].set_ylabel("out-of-sample Sharpe"); ax[1].set_title(f"All {len(Wp)} ratio-momentum combos")
plt.tight_layout(); plt.savefig("part31_equity.png", dpi=110)
