"""第三十二部分：只做黃金順勢（第三十一部分的延伸），含更小週期；以及 NAS100 能扮演什麼角色。
資料：XAUUSD M1（2023/6/1~2026/10/8，缺 2026/4/3~4/14）合成 M1/M5；XAUUSD M15 檔合成 M15/H1/H4（從 2023/6/1 起算）。
NAS100 M1（2023/10/2 起）只用在第三段。樣本內 < 2025/1/1 ≤ 樣本外。

1. 黃金順勢：同第三十一部分的 6 種規則（tsmom / ma / zbrk1 / zbrk1.5 / zbrk2 / donch），M1 / M5 / M15 / H1 / H4 各 3 種回看。
   部位 ±1（全名目），訊號 t 收盤、賺 t→t+1；部位變動付半個點差（當根 spread）。
2. 壓力測試：晚一根進場、點差 ×2 —— 小週期最怕這個。
3. NAS 的角色（2023/10 起，條件都用 H1 收完的值，不偷看）：
   a. 方向濾網：黃金部位與 NAS 自己的順勢方向相同 / 相反
   b. 相關性濾網：H1 24 根相關的 z 值（相對過去 480 根）脫鉤 / 正常 / 同動
   c. 風險事件：NAS 24 小時實現波動 z > 1（股市不安）
   d. 組合：黃金順勢 + NAS 順勢（同規則，各自波動調整到相同風險）是否比單做黃金好
輸出：results_part32.txt、part32_grid.csv、part32_equity.png
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
START = pd.Timestamp("2023-06-01")
OUT = open("results_part32.txt", "w")


def P(*a):
    print(*a); print(*a, file=OUT)


def bars(df, rule):
    if rule is None:
        return df[["close", "spread"]].copy()
    return pd.DataFrame({"close": df.close.resample(rule).last(), "spread": df.spread.resample(rule).last()}).dropna()


xm1 = load("data/XAUUSD_M1_2023_2026.csv")
xm15 = load("data/XAUUSD_M15_2023_2026.csv")
xm15 = xm15[xm15.index >= START]
GOLD = {"M1": bars(xm1, None), "M5": bars(xm1, "5min"), "M15": bars(xm15, None), "H1": bars(xm15, "1h"), "H4": bars(xm15, "4h")}
for d in GOLD.values():
    d["spread"] *= 0.01
LOOKBACK = {"M1": [60, 240, 960], "M5": [48, 144, 576], "M15": [96, 384, 960], "H1": [24, 120, 480], "H4": [30, 120, 360]}
RULES = ["tsmom", "ma", "zbrk1", "zbrk1.5", "zbrk2", "donch"]


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
            if not p and abs(z[i]) > E: p = np.sign(z[i])     # NaN 比較為 False
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


def run(D, pos, lag=1, cost_x=1.0):
    """pos：t 收盤後想要的部位；lag=1 表示 t 收盤成交，2 表示晚一根。回傳每根損益 bp。"""
    pos = pos.shift(lag - 1).fillna(0)
    p = pos.shift().fillna(0)
    dp = pos.diff().abs().fillna(pos.abs())
    R = pd.DataFrame({"gross": p * D.close.pct_change(), "cost": dp * D.spread / D.close / 2 * cost_x}).fillna(0) * 1e4
    R["pnl"] = R.gross - R.cost
    R["pos"], R["dp"] = pos, dp
    return R


def stats(R):
    d = R.pnl.groupby(R.index.normalize()).sum()
    eq = d.cumsum()
    yrs = len(d) / 252
    return dict(sharpe=d.mean() / d.std() * np.sqrt(252) if d.std() > 0 else np.nan, ann_bp=d.sum() / yrs,
                dd_bp=(eq.cummax() - eq).max(), trades=R.dp.gt(0).sum() / yrs, inmkt=R.pos.ne(0).mean(),
                gross=R.gross.sum() / yrs, cost=R.cost.sum() / yrs)


def fmt(s):
    return (f"Sharpe {s['sharpe']:+5.2f} 年化 {s['ann_bp']:+6.0f}bp（成本 {s['cost']:5.0f}） 回撤 {s['dd_bp']:5.0f}bp "
            f"年換手 {s['trades']:5.0f} 在場 {s['inmkt']:.0%}")


def split(R):
    return R[R.index < IS_END], R[R.index >= IS_END]


P(f"黃金：{START.date()} ~ {xm15.index[-1]}；樣本內 < {IS_END.date()} ≤ 樣本外")
P(f"點差中位數：M1 {GOLD['M1'].spread.median():.2f}、M15 {GOLD['M15'].spread.median():.2f} 美元")
bh = GOLD["H1"].close.pct_change().fillna(0) * 1e4
for per, g in zip(["樣本內", "樣本外"], split(pd.DataFrame({"pnl": bh, "gross": bh, "cost": 0.0, "pos": 1.0, "dp": 0.0}))):
    P(f"  基準：一直抱多黃金 {per} {fmt(stats(g))}")

rows, POS, ALL = [], {}, {}
for tf, D in GOLD.items():
    s = np.log(D.close)
    for L, rule in itertools.product(LOOKBACK[tf], RULES):
        pos = position(s, L, rule)
        POS[(tf, L, rule)] = pos
        for stress, kw in [("base", {}), ("lag", dict(lag=2)), ("cost2", dict(cost_x=2.0)), ("both", dict(lag=2, cost_x=2.0))]:
            R = run(D, pos, **kw)
            if stress == "base":
                ALL[(tf, L, rule)] = R
            for per, g in zip(["IS", "OOS"], split(R)):
                rows.append(dict(tf=tf, L=L, rule=rule, stress=stress, per=per, **stats(g)))
G = pd.DataFrame(rows)
G.to_csv("part32_grid.csv", index=False)
W = G.pivot_table(index=["tf", "L", "rule"], columns=["stress", "per"], values="sharpe")

P(f"\n## 1. 黃金順勢：{len(W)} 組（5 週期 × 3 回看 × 6 規則）")
for tf in GOLD:
    w = W.loc[tf]
    P(f"  {tf:3s}: 樣本內 >0 {(w[('base', 'IS')] > 0).sum():2d}/18、樣本外 >0 {(w[('base', 'OOS')] > 0).sum():2d}/18；"
      f"Sharpe 中位數 IS {w[('base', 'IS')].median():+.2f} / OOS {w[('base', 'OOS')].median():+.2f}；"
      f"晚一根 {w[('lag', 'OOS')].median():+.2f}、點差×2 {w[('cost2', 'OOS')].median():+.2f}、兩者 {w[('both', 'OOS')].median():+.2f}")
P(f"  全部：樣本內 vs 樣本外 Sharpe 排名相關 {W[('base', 'IS')].corr(W[('base', 'OOS')], method='spearman'):+.2f}")

P("\n## 2. 每一組（Sharpe 樣本內 → 樣本外 / 樣本外晚一根+點差×2）")
for tf in GOLD:
    for L in LOOKBACK[tf]:
        P(f"  {tf:3s} L={L:3d}: " + "  ".join(
            f"{r} {W.loc[(tf, L, r), ('base', 'IS')]:+.1f}→{W.loc[(tf, L, r), ('base', 'OOS')]:+.1f}/{W.loc[(tf, L, r), ('both', 'OOS')]:+.1f}"
            for r in RULES))

P("\n## 3. 小週期的成本：每個週期樣本外的年化扣成本前 / 成本 / 扣成本後（中位數，bp）")
Gb = G[(G.stress == "base") & (G.per == "OOS")]
for tf in GOLD:
    g = Gb[Gb.tf == tf]
    P(f"  {tf:3s}: 扣成本前 {g.gross.median():+6.0f}  成本 {g.cost.median():5.0f}  扣成本後 {g.ann_bp.median():+6.0f}  年換手中位數 {g.trades.median():5.0f}")

P("\n## 4. 每個週期樣本內第一名 → 樣本外、逐年")
BEST = {}
for tf in GOLD:
    k = W.loc[tf][("base", "IS")].idxmax()
    k = (tf, *k)
    BEST[tf] = k
    R = ALL[k]
    i, o = split(R)
    yr = "  ".join(f"{y}:{stats(g)['sharpe']:+.1f}" for y, g in R.groupby(R.index.year))
    P(f"  {tf:3s} L={k[1]:3d} {k[2]:7s} IS  {fmt(stats(i))}")
    P(f"  {'':17s} OOS {fmt(stats(o))}")
    P(f"  {'':17s} 晚一根+點差×2 OOS Sharpe {W.loc[k, ('both', 'OOS')]:+.2f}；逐年 {yr}；"
      f"樣本外 多 {o[o.pos.shift() > 0].pnl.sum():+.0f}bp / 空 {o[o.pos.shift() < 0].pnl.sum():+.0f}bp")

# ---------------- 3. NAS 的角色 ----------------
n1 = pd.concat([load(f) for f in sorted(glob.glob("data/NAS100_M1_*.csv"))])
n1 = n1[~n1.index.duplicated(keep="last")].sort_index()
NH1 = pd.DataFrame({"close": n1.close.resample("1h").last(), "spread": n1.spread.resample("1h").last() * 0.1}).dropna()
XH1 = GOLD["H1"]
rn, rx = np.log(NH1.close).diff(), np.log(XH1.close).diff()
J = pd.concat({"rn": rn, "rx": rx}, axis=1, sort=True).dropna()
c24 = J.rn.rolling(24).corr(J.rx)
COND = pd.DataFrame({
    "nas_dir": position(np.log(NH1.close), 120, "tsmom"),
    "corr_z": (c24 - c24.rolling(480).mean()) / c24.rolling(480).std(),
    "nas_volz": (lambda v: (v - v.rolling(480).mean()) / v.rolling(480).std())(rn.rolling(24).std()),
})
COND.index = COND.index + pd.Timedelta("1h")        # H1 標在開盤時間，收完才知道 → 往後移一小時才能用
NAS_START = NH1.index[0] + pd.Timedelta(days=30)


def cond_at(idx):
    return COND.reindex(COND.index.union(idx)).ffill().reindex(idx)


P(f"\n## 5. NAS 的角色（{NAS_START.date()} 起；以每個週期樣本內第一名為底，濾網外的時間不持倉）")
P("   每格：樣本內 Sharpe / 樣本外 Sharpe（年化 bp）")
for tf, k in BEST.items():
    D = GOLD[tf]
    pos = POS[k]
    m = D.index >= NAS_START
    D, pos = D[m], pos[m]
    C = cond_at(D.index)
    gold_dir = pos
    filters = {
        "不篩": pd.Series(True, D.index),
        "與 NAS 同向": np.sign(gold_dir) == C.nas_dir,
        "與 NAS 反向": np.sign(gold_dir) == -C.nas_dir,
        "脫鉤 cz<−1": C.corr_z < -1,
        "正常 |cz|≤1": C.corr_z.abs() <= 1,
        "同動 cz>+1": C.corr_z > 1,
        "NAS 波動高 z>1": C.nas_volz > 1,
        "NAS 波動不高": C.nas_volz <= 1,
    }
    cells = []
    for name, f in filters.items():
        R = run(D, pos.where(f.fillna(False), 0.0))
        i, o = split(R)
        si, so = stats(i), stats(o)
        cells.append(f"{name} {si['sharpe']:+.2f}/{so['sharpe']:+.2f}({so['ann_bp']:+.0f})")
    P(f"  {tf:3s} L={k[1]} {k[2]}:")
    P("      " + "  ".join(cells[:3]))
    P("      " + "  ".join(cells[3:6]))
    P("      " + "  ".join(cells[6:]))

P("\n## 6. 組合：黃金順勢 + NAS 順勢（同規則同回看，各自用樣本內日波動調成相同風險）")
for tf in ["M15", "H1", "H4"]:
    k = BEST[tf]
    Dn = {"M15": n1, "H1": n1, "H4": n1}[tf]
    rule = {"M15": "15min", "H1": "1h", "H4": "4h"}[tf]
    ND = pd.DataFrame({"close": Dn.close.resample(rule).last(), "spread": Dn.spread.resample(rule).last() * 0.1}).dropna()
    rg = ALL[k].pnl.groupby(ALL[k].index.normalize()).sum()
    Rn = run(ND, position(np.log(ND.close), k[1], k[2]))
    rnas = Rn.pnl.groupby(Rn.index.normalize()).sum()
    d = pd.concat({"g": rg, "n": rnas}, axis=1, sort=True).dropna()
    d = d[d.index >= NAS_START]
    ins = d[d.index < IS_END]
    w = ins.g.std() / ins.n.std()
    d["mix"] = (d.g + w * d.n) / 2
    for per, g in [("樣本內", d[d.index < IS_END]), ("樣本外", d[d.index >= IS_END])]:
        sh = {c: g[c].mean() / g[c].std() * np.sqrt(252) for c in ["g", "n", "mix"]}
        P(f"  {tf:3s} L={k[1]} {k[2]:7s} {per}：只黃金 {sh['g']:+.2f}  只 NAS {sh['n']:+.2f}  各半 {sh['mix']:+.2f}  "
          f"兩策略日損益相關 {g.g.corr(g.n):+.2f}")

P("\n## 7. 是擇時還是只是做多黃金？日損益對黃金日報酬回歸：α（年化 bp, t）與 β；以及只做空單的 Sharpe")
gd = GOLD["H1"].close.resample("1D").last().dropna().pct_change() * 1e4
AL = []
for k, R in ALL.items():
    d = pd.concat({"p": R.pnl.groupby(R.index.normalize()).sum(), "g": gd}, axis=1, sort=True).dropna()
    for per, g in [("IS", d[d.index < IS_END]), ("OOS", d[d.index >= IS_END])]:
        X = np.column_stack([np.ones(len(g)), g.g])
        coef, res, *_ = np.linalg.lstsq(X, g.p, rcond=None)
        e = g.p - X @ coef
        se = np.sqrt(e.var() / len(g) * np.linalg.inv(X.T @ X / len(g))[0, 0])
        sh = R[(R.index < IS_END) if per == "IS" else (R.index >= IS_END)]
        short = sh.pnl.where(sh.pos.shift() < 0, 0.0)
        ds = short.groupby(short.index.normalize()).sum()
        AL.append(dict(tf=k[0], L=k[1], rule=k[2], per=per, alpha=coef[0] * 252, t=coef[0] / se, beta=coef[1],
                       short_sh=ds.mean() / ds.std() * np.sqrt(252) if ds.std() > 0 else np.nan))
AL = pd.DataFrame(AL)
for tf in GOLD:
    g = AL[AL.tf == tf].set_index("per")
    P(f"  {tf:3s}: α 中位數 IS {g.loc['IS'].alpha.median():+6.0f}bp (t>2 的 {(g.loc['IS'].t > 2).sum()}/18)  "
      f"OOS {g.loc['OOS'].alpha.median():+6.0f}bp (t>2 的 {(g.loc['OOS'].t > 2).sum()}/18)；β 中位數 OOS {g.loc['OOS'].beta.median():+.2f}；"
      f"只做空單 Sharpe 中位數 IS {g.loc['IS'].short_sh.median():+.2f} / OOS {g.loc['OOS'].short_sh.median():+.2f}")
for tf, k in BEST.items():
    g = AL[(AL.tf == tf) & (AL.L == k[1]) & (AL.rule == k[2])].set_index("per")
    P(f"  {tf:3s} 第一名 L={k[1]} {k[2]:7s}: IS α {g.loc['IS'].alpha:+6.0f}bp t={g.loc['IS'].t:+.2f} β {g.loc['IS'].beta:+.2f} | "
      f"OOS α {g.loc['OOS'].alpha:+6.0f}bp t={g.loc['OOS'].t:+.2f} β {g.loc['OOS'].beta:+.2f}")

fig, ax = plt.subplots(2, 1, figsize=(11, 7))
for tf, k in BEST.items():
    R = ALL[k]
    ax[0].plot(R.pnl.groupby(R.index.normalize()).sum().cumsum(), lw=1, label=f"{tf} L={k[1]} {k[2]}")
ax[0].axvline(IS_END, color="k", ls="--"); ax[0].legend(fontsize=7); ax[0].set_title("Gold trend: in-sample best per timeframe, cumulative bp")
for tf in GOLD:
    w = W.loc[tf]
    ax[1].scatter(w[("base", "IS")], w[("base", "OOS")], s=12, label=tf)
ax[1].axhline(0, color="k", lw=.5); ax[1].axvline(0, color="k", lw=.5); ax[1].legend(fontsize=7)
ax[1].set_xlabel("in-sample Sharpe"); ax[1].set_ylabel("out-of-sample Sharpe"); ax[1].set_title("All gold trend combos")
plt.tight_layout(); plt.savefig("part32_equity.png", dpi=110)
