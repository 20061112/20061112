"""第二十九部分：NAS100 × XAUUSD 價差回歸均值（配對交易）。
資料同第二十八部分：NAS100 M1（合成 M15 / H1）× XAUUSD M15，共同期間 2023/10/2 ~ 2026/10/9。

價差 s_t = log(NAS) − β_t · log(XAU)，z_t = (s_t − 滾動平均) / 滾動標準差（回看 L 根，只用 t 以前與 t 的資料）。
  z > E：做空價差（空 NAS、多 β 份 XAU）；z < −E：做多價差。
  出場：z 回到 0（回歸均值）；停損：|z| > E + 1.5（可關）；時間停損：持有超過 L 根。
  β 兩種：ols = 回看 L 根 log 價格回歸；vol = 回看 L 根報酬波動比（波動中性，永遠是正的）。
  intraday=True：伺服器時間 21 點後不開新倉、23 點前全部平倉（不留過夜 / 週末）。
部位：NAS 名目 1、XAU 名目 β，β 在進場時固定；損益以總名目 (1+β) 的報酬計（bp）。
成本：兩隻腳各付一次進場與出場當根的點差（NAS 0.1 點、XAU 0.01 美元單位）。
訊號在 t 收盤算出、t 收盤成交（同一根收盤價），沒有用到 t 之後的資料。

驗證：樣本內 2023/10 ~ 2024/12 選參數，2025/1 ~ 2026/10 完全不動。並拆兩隻腳的損益，
確認賺的是「價差回歸」而不是其中一隻自己的均值回歸。
輸出：results_part29.txt、part29_grid.csv、part29_equity.png
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
OUT = open("results_part29.txt", "w")


def P(*a):
    print(*a); print(*a, file=OUT)


def nas_m1():
    df = pd.concat([load(f) for f in sorted(glob.glob("data/NAS100_M1_*.csv"))])
    return df[~df.index.duplicated(keep="last")].sort_index()


n1 = nas_m1()
n15 = pd.DataFrame({"close": n1.close.resample("15min").last(), "spread": n1.spread.resample("15min").last() * 0.1}).dropna()
x15 = load("data/XAUUSD_M15_2023_2026.csv")[["close", "spread"]].assign(spread=lambda d: d.spread * 0.01)
M15 = pd.concat({"nas": n15, "xau": x15}, axis=1, sort=True).dropna()
M15.columns = ["n", "n_sp", "x", "x_sp"]
H1 = M15.resample("1h").last().dropna()
DATA = {"M15": M15, "H1": H1}
LOOKBACK = {"M15": [32, 96, 384], "H1": [24, 72, 240]}


def signals(D, L, beta_kind):
    ln, lx = np.log(D.n), np.log(D.x)
    mn, mx = ln.rolling(L).mean(), lx.rolling(L).mean()
    vn, vx, cv = ln.rolling(L).var(), lx.rolling(L).var(), ln.rolling(L).cov(lx)
    if beta_kind == "ols":
        b = cv / vx
    else:
        rn, rx = ln.diff(), lx.diff()
        b = rn.rolling(L).std() / rx.rolling(L).std()
    s = ln - b * lx
    mu = mn - b * mx
    sd = np.sqrt((vn + b * b * vx - 2 * b * cv).clip(lower=1e-12))
    return b, (s - mu) / sd


def backtest(D, L, E, stop, beta_kind, intraday):
    b, z = signals(D, L, beta_kind)
    b, z = b.to_numpy(), z.to_numpy()
    n, x, nsp, xsp = D.n.to_numpy(), D.x.to_numpy(), D.n_sp.to_numpy(), D.x_sp.to_numpy()
    hr = D.index.hour.to_numpy()
    t = D.index
    S = E + 1.5 if stop else np.inf
    trades, pos, i0 = [], 0, 0
    for i in range(len(D)):
        if pos:
            zi = z[i]
            why = None
            if pos * zi >= 0: why = "mean"
            elif abs(zi) > S: why = "stop"
            elif i - i0 >= L: why = "time"
            elif intraday and (hr[i] >= 23 or t[i].normalize() != t[i0].normalize()): why = "eod"
            if why:
                bb = abs(be)
                legn = pos * (n[i] / n[i0] - 1)
                legx = -pos * be * (x[i] / x[i0] - 1)
                cost = (nsp[i0] / n[i0] + nsp[i] / n[i]) / 2 + bb * (xsp[i0] / x[i0] + xsp[i] / x[i]) / 2
                g = 1 + bb
                trades.append((t[i0], t[i], pos, be, legn / g * 1e4, legx / g * 1e4, cost / g * 1e4, i - i0, why))
                pos = 0
        if not pos and np.isfinite(z[i]) and np.isfinite(b[i]) and abs(z[i]) > E and abs(z[i]) <= S:
            if intraday and hr[i] >= 21:
                continue
            pos, i0, be = (-1 if z[i] > 0 else 1), i, b[i]
    T = pd.DataFrame(trades, columns=["entry", "exit", "side", "beta", "nas_bp", "xau_bp", "cost_bp", "bars", "why"])
    T["pnl_bp"] = T.nas_bp + T.xau_bp - T.cost_bp
    return T


def stats(T):
    if len(T) < 5:
        return dict(n=len(T), bp=np.nan, t=np.nan, win=np.nan, pf=np.nan, sharpe=np.nan, tot=np.nan)
    r = T.pnl_bp
    d = r.groupby(T.exit.dt.normalize()).sum()
    gl = -r[r < 0].sum()
    return dict(n=len(r), bp=r.mean(), t=r.mean() / r.std() * np.sqrt(len(r)), win=(r > 0).mean(),
                pf=r[r > 0].sum() / gl if gl > 0 else np.inf, sharpe=d.mean() / d.std() * np.sqrt(252), tot=r.sum())


def fmt(s):
    return (f"n={s['n']:5d} 每筆 {s['bp']:+6.2f}bp t={s['t']:+5.2f} 勝率 {s['win']:.0%} PF {s['pf']:.2f} "
            f"Sharpe {s['sharpe']:+5.2f} 總 {s['tot']:+7.0f}bp")


def K(k):
    tf, L, E, stop, bk, intra = k
    return f"{tf} L={int(L)} E={float(E)} {'停損' if stop else '無停損'} β={bk} {'當沖' if intra else '可過夜'}"


rows, ALL = [], {}
for tf, L, E, stop, bk, intra in itertools.product(["M15", "H1"], [0, 1, 2], [1.5, 2.0, 2.5], [True, False], ["ols", "vol"], [False, True]):
    L = LOOKBACK[tf][L]
    T = backtest(DATA[tf], L, E, stop, bk, intra)
    key = (tf, L, E, stop, bk, intra)
    ALL[key] = T
    for per, m in [("IS", T.entry < IS_END), ("OOS", T.entry >= IS_END)]:
        rows.append(dict(tf=tf, L=L, E=E, stop=stop, beta=bk, intraday=intra, per=per, **stats(T[m])))
G = pd.DataFrame(rows)
G.to_csv("part29_grid.csv", index=False)
W = G.pivot_table(index=["tf", "L", "E", "stop", "beta", "intraday"], columns="per", values=["bp", "sharpe", "n", "pf"])

P(f"資料：共同 M15 {len(M15)} 根，{M15.index[0]} ~ {M15.index[-1]}；H1 {len(H1)} 根")
P(f"成本中位數：NAS {(M15.n_sp / M15.n).median() * 1e4:.2f}bp、XAU {(M15.x_sp / M15.x).median() * 1e4:.2f}bp（全點差）")
P(f"樣本內 < {IS_END.date()} ≤ 樣本外；參數組合 {len(ALL)} 個")

P("\n## 1. 全部組合：樣本內 vs 樣本外")
P(f"  樣本內每筆 bp > 0 的組合：{(W['bp']['IS'] > 0).sum()}/{len(W)}；樣本外 > 0：{(W['bp']['OOS'] > 0).sum()}/{len(W)}")
P(f"  樣本內 Sharpe 中位數 {W['sharpe']['IS'].median():+.2f}；樣本外 {W['sharpe']['OOS'].median():+.2f}")
gross = pd.Series({k: (T.pnl_bp + T.cost_bp).mean() for k, T in ALL.items()})
P(f"  扣成本前每筆 bp：中位數 {gross.median():+.2f}，最高 {gross.max():+.2f}（{K(gross.idxmax())}）；平均成本 {np.mean([T.cost_bp.mean() for T in ALL.values()]):.2f}bp")
P(f"  樣本內 vs 樣本外 Sharpe 的排名相關（Spearman）：{W['sharpe']['IS'].corr(W['sharpe']['OOS'], method='spearman'):+.2f}")
for col, vals in [("tf", ["M15", "H1"]), ("beta", ["ols", "vol"]), ("intraday", [False, True]), ("stop", [True, False]), ("E", [1.5, 2.0, 2.5])]:
    g = G.groupby([col, "per"]).bp.median().unstack()
    P(f"  依 {col:8s}（每筆 bp 中位數 IS / OOS）：" + "  ".join(f"{v}: {g.loc[v, 'IS']:+.2f} / {g.loc[v, 'OOS']:+.2f}" for v in vals))

P("\n## 2. 樣本內最好的 10 組（依 Sharpe，至少 50 筆）→ 樣本外")
ok = W[W["n"]["IS"] >= 50].sort_values(("sharpe", "IS"), ascending=False).head(10)
for k in ok.index:
    T = ALL[k]
    P(f"  {K(k):38s} IS {fmt(stats(T[T.entry < IS_END]))}")
    P(f"  {'':38s} OOS {fmt(stats(T[T.entry >= IS_END]))}")
best = ok.index[0]

P(f"\n## 3. 樣本內第一名 {K(best)} 細節")
T = ALL[best]
for y, g in T.groupby(T.entry.dt.year):
    P(f"  {y}: {fmt(stats(g))}")
P(f"  出場原因：" + "  ".join(f"{k} {v} 筆 ({T[T.why == k].pnl_bp.mean():+.1f}bp)" for k, v in T.why.value_counts().items()))
P(f"  拆腳（每筆平均 bp）：NAS 腳 {T.nas_bp.mean():+.2f}、XAU 腳 {T.xau_bp.mean():+.2f}、成本 −{T.cost_bp.mean():.2f}")
P(f"  做多價差 {fmt(stats(T[T.side == 1]))}")
P(f"  做空價差 {fmt(stats(T[T.side == -1]))}")

P("\n## 4. 拆腳：所有組合樣本外，賺的是價差還是單腳？")
legs = pd.DataFrame([(k, T[T.entry >= IS_END].nas_bp.mean(), T[T.entry >= IS_END].xau_bp.mean()) for k, T in ALL.items()],
                    columns=["k", "nas", "xau"])
P(f"  NAS 腳每筆中位數 {legs.nas.median():+.2f}bp（>0 的組合 {(legs.nas > 0).mean():.0%}）；"
  f"XAU 腳 {legs.xau.median():+.2f}bp（>0 的組合 {(legs.xau > 0).mean():.0%}）")

P("\n## 5. 對照：只做 NAS 單腳的 z 值回歸（同一套進出場，β=0）")
for tf in ["M15", "H1"]:
    for L in LOOKBACK[tf]:
        D = DATA[tf]
        ln = np.log(D.n)
        z = (ln - ln.rolling(L).mean()) / ln.rolling(L).std()
        P(f"  {tf} L={L:3d}：z>2 之後 {L // 4} 根 NAS 報酬平均 {((ln.shift(-(L // 4)) - ln)[z > 2]).mean() * 1e4:+.1f}bp，"
          f"z<−2 之後 {((ln.shift(-(L // 4)) - ln)[z < -2]).mean() * 1e4:+.1f}bp")

fig, ax = plt.subplots(2, 1, figsize=(11, 7))
for k in ok.index[:5]:
    T = ALL[k]
    ax[0].plot(T.exit, T.pnl_bp.cumsum(), lw=1, label=K(k).replace("停損", "stop").replace("無", "no ").replace("當沖", "intraday").replace("可過夜", "overnight"))
ax[0].axvline(IS_END, color="k", ls="--"); ax[0].legend(fontsize=7); ax[0].set_title("Top-5 in-sample combos, cumulative bp (dashed = OOS start)")
ax[1].scatter(W["sharpe"]["IS"], W["sharpe"]["OOS"], s=10)
ax[1].axhline(0, color="k", lw=.5); ax[1].axvline(0, color="k", lw=.5)
ax[1].set_xlabel("in-sample Sharpe"); ax[1].set_ylabel("out-of-sample Sharpe"); ax[1].set_title("All 144 combos")
plt.tight_layout(); plt.savefig("part29_equity.png", dpi=110)
