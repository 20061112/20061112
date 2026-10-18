"""第三十部分：NAS100 / XAUUSD 比值的 z 值回歸，加上「相關性 z 值」。
資料同第二十八、二十九部分（NAS100 M1 合成 × XAUUSD M15，2023/10/2 ~ 2026/10/9），另外合成 H4、D1。

1. 比值 z：ratio = log(NAS / XAU)（β = 1，兩邊同名目），z = (ratio − 滾動平均) / 滾動標準差。
   z > E 空 NAS 多 XAU；z < −E 反過來；z 回到 0 出場，停損 |z| > E+1.5，時間停損 L 根。
2. 相關性 z：短窗相關 c_s（報酬，WS 根）相對自己過去 WL 根的平均 / 標準差標準化：
   corr_z = (c_s − mean(c_s)) / std(c_s)。corr_z 很低 = 兩者突然脫鉤；很高 = 突然一起動。
3. 事件研究：|比值 z| > 2 之後 h 根，比值往均值走了幾個標準差（正 = 有回歸），依 corr_z 分組。
   → 先看「脫鉤時是不是比較會回歸」，再決定要不要拿來當濾網。
4. 回測：比值 z 進場 × corr_z 濾網（不篩 / 只在脫鉤 corr_z<−1 / 只在相關正常偏高 corr_z>0）。
   樣本內 2023/10 ~ 2024/12、樣本外 2025/1 ~ 2026/10。成本同第二十九部分。
輸出：results_part30.txt、part30_grid.csv
"""
import glob
import itertools
import numpy as np
import pandas as pd
from load import load

IS_END = pd.Timestamp("2025-01-01")
OUT = open("results_part30.txt", "w")


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
LOOKBACK = {"M15": [96, 384], "H1": [24, 120, 480], "H4": [30, 120], "D1": [20, 60, 120]}
CORR_WIN = {"M15": (96, 960), "H1": (24, 480), "H4": (30, 360), "D1": (20, 250)}   # (短窗, 標準化用的長窗)


def features(D, L):
    lr = np.log(D.n / D.x)
    z = (lr - lr.rolling(L).mean()) / lr.rolling(L).std()
    rn, rx = np.log(D.n).diff(), np.log(D.x).diff()
    ws, wl = CORR_WIN[tf_of(D)]
    cs = rn.rolling(ws).corr(rx)
    cz = (cs - cs.rolling(wl).mean()) / cs.rolling(wl).std()
    return lr, z, cs, cz


def tf_of(D):
    return next(k for k, v in DATA.items() if v is D)


def backtest(D, L, E, z, allow):
    n, x, nsp, xsp = D.n.to_numpy(), D.x.to_numpy(), D.n_sp.to_numpy(), D.x_sp.to_numpy()
    z, allow, t = z.to_numpy(), allow.to_numpy(), D.index
    S = E + 1.5
    trades, pos, i0 = [], 0, 0
    for i in range(len(D)):
        if pos:
            why = "mean" if pos * z[i] >= 0 else "stop" if abs(z[i]) > S else "time" if i - i0 >= L else None
            if why:
                legn, legx = pos * (n[i] / n[i0] - 1), -pos * (x[i] / x[i0] - 1)
                cost = (nsp[i0] / n[i0] + nsp[i] / n[i] + xsp[i0] / x[i0] + xsp[i] / x[i]) / 2
                trades.append((t[i0], t[i], pos, legn / 2 * 1e4, legx / 2 * 1e4, cost / 2 * 1e4, i - i0, why))
                pos = 0
        if not pos and np.isfinite(z[i]) and E < abs(z[i]) <= S and allow[i]:
            pos, i0 = (-1 if z[i] > 0 else 1), i
    T = pd.DataFrame(trades, columns=["entry", "exit", "side", "nas_bp", "xau_bp", "cost_bp", "bars", "why"])
    T["pnl_bp"] = T.nas_bp + T.xau_bp - T.cost_bp
    return T


def stats(T):
    if len(T) < 5:
        return dict(n=len(T), bp=np.nan, t=np.nan, win=np.nan, pf=np.nan, sharpe=np.nan)
    r = T.pnl_bp
    d = r.groupby(T.exit.dt.normalize()).sum()
    gl = -r[r < 0].sum()
    return dict(n=len(r), bp=r.mean(), t=r.mean() / r.std() * np.sqrt(len(r)), win=(r > 0).mean(),
                pf=r[r > 0].sum() / gl if gl > 0 else np.inf, sharpe=d.mean() / d.std() * np.sqrt(252))


def fmt(s):
    return f"n={s['n']:5d} 每筆 {s['bp']:+7.2f}bp t={s['t']:+5.2f} 勝率 {s['win']:.0%} PF {s['pf']:.2f} Sharpe {s['sharpe']:+5.2f}"


P(f"資料：共同 M15 {len(M15)} 根，{M15.index[0]} ~ {M15.index[-1]}；" + "、".join(f"{k} {len(v)} 根" for k, v in DATA.items()))
P(f"樣本內 < {IS_END.date()} ≤ 樣本外")

P("\n## 0. 相關性本身：短窗相關與 corr_z 的分佈")
for tf, D in DATA.items():
    _, _, cs, cz = features(D, LOOKBACK[tf][0])
    P(f"  {tf:3s} 短窗 {CORR_WIN[tf][0]} 根相關：中位數 {cs.median():+.2f} 10%~90% [{cs.quantile(.1):+.2f}, {cs.quantile(.9):+.2f}]；"
      f"corr_z<−1 佔 {(cz < -1).mean():.0%}、>+1 佔 {(cz > 1).mean():.0%}")

P("\n## 1. 事件研究：|比值 z|>2 之後 h=L/2 根，往均值回歸了幾個標準差（+ = 有回歸），依 corr_z 分組")
P("   （隨機漫步下期望 ≈ 0；括號內為 t 值與事件數，事件每 h 根只取一個）")
for tf, D in DATA.items():
    for L in LOOKBACK[tf]:
        lr, z, cs, cz = features(D, L)
        h = max(L // 2, 1)
        rev = -np.sign(z) * (lr.shift(-h) - lr) / lr.rolling(L).std()
        ev = pd.DataFrame({"rev": rev, "cz": cz})[z.abs() > 2].dropna()
        # 重疊事件很多：每 h 根只取第一個，t 值才不會灌水
        ev = ev[~ev.index.to_series().diff().lt(pd.Timedelta(D.index.to_series().diff().median() * h)).values]
        parts = []
        for name, m in [("全部", slice(None)), ("脫鉤 cz<−1", ev.cz < -1), ("正常", ev.cz.abs() <= 1), ("同動 cz>+1", ev.cz > 1)]:
            g = ev[m] if not isinstance(m, slice) else ev
            gi, go = g[g.index < IS_END].rev, g[g.index >= IS_END].rev
            tv = lambda r: r.mean() / r.std() * np.sqrt(len(r)) if len(r) > 2 else np.nan
            parts.append(f"{name} IS {gi.mean():+.2f}(t{tv(gi):+.1f},{len(gi)}) OOS {go.mean():+.2f}(t{tv(go):+.1f},{len(go)})")
        P(f"  {tf:3s} L={L:3d} h={h:3d}：" + " | ".join(parts))

rows, ALL = [], {}
for tf, D in DATA.items():
    for L, E, filt in itertools.product(LOOKBACK[tf], [1.5, 2.0, 2.5], ["none", "decouple", "coupled"]):
        lr, z, cs, cz = features(D, L)
        allow = {"none": pd.Series(True, D.index), "decouple": cz < -1, "coupled": cz > 0}[filt]
        T = backtest(D, L, E, z, allow)
        ALL[(tf, L, E, filt)] = T
        for per, m in [("IS", T.entry < IS_END), ("OOS", T.entry >= IS_END)]:
            rows.append(dict(tf=tf, L=L, E=E, filt=filt, per=per, **stats(T[m])))
G = pd.DataFrame(rows)
G.to_csv("part30_grid.csv", index=False)
W = G.pivot_table(index=["tf", "L", "E", "filt"], columns="per", values=["bp", "sharpe", "n"])

P(f"\n## 2. 比值 z 回測（{len(ALL)} 組）")
P(f"  樣本內每筆 > 0：{(W['bp']['IS'] > 0).sum()}/{len(W)}；樣本外 > 0：{(W['bp']['OOS'] > 0).sum()}/{len(W)}；"
  f"兩段都 > 0：{((W['bp']['IS'] > 0) & (W['bp']['OOS'] > 0)).sum()}")
P(f"  樣本內 vs 樣本外 Sharpe 排名相關：{W['sharpe']['IS'].corr(W['sharpe']['OOS'], method='spearman'):+.2f}")
for col in ["tf", "filt", "E"]:
    g = G.groupby([col, "per"]).bp.median().unstack()
    P(f"  依 {col:4s}（每筆 bp 中位數 IS / OOS）：" + "  ".join(f"{v}: {g.loc[v, 'IS']:+.2f} / {g.loc[v, 'OOS']:+.2f}" for v in g.index))

P("\n## 3. 各週期：corr_z 濾網的效果（同一組 L、E 下三種濾網並排，樣本內 → 樣本外每筆 bp）")
for (tf, L, E), g in G.groupby(["tf", "L", "E"], sort=False):
    s = g.set_index(["filt", "per"])
    P(f"  {tf:3s} L={L:3d} E={E}: " + "  ".join(
        f"{f:8s} {s.loc[(f, 'IS'), 'bp']:+6.1f}→{s.loc[(f, 'OOS'), 'bp']:+6.1f} ({int(s.loc[(f, 'IS'), 'n'])}/{int(s.loc[(f, 'OOS'), 'n'])})"
        for f in ["none", "decouple", "coupled"]))

P("\n## 4. 樣本內最好的 8 組（至少 30 筆）→ 樣本外")
ok = W[W["n"]["IS"] >= 30].sort_values(("sharpe", "IS"), ascending=False).head(8)
for k in ok.index:
    T = ALL[k]
    P(f"  {k[0]} L={k[1]} E={k[2]} {k[3]:8s} IS  {fmt(stats(T[T.entry < IS_END]))}")
    P(f"  {'':27s} OOS {fmt(stats(T[T.entry >= IS_END]))}  NAS 腳 {T[T.entry >= IS_END].nas_bp.mean():+.1f} XAU 腳 {T[T.entry >= IS_END].xau_bp.mean():+.1f}")
