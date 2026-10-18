"""第三十六部分：「紅線」當沖版反過來做 —— 缺口順勢（第三十五部分的反面）。
與第三十五部分相同的錨點 / 缺口 / z 定義，但 z > E → 多黃金、空 NAS（跟著強的那邊），z < −E 反過來。每天最多一筆。
出場兩種：eod0 = 抱到 23:00，途中缺口回到 0（強弱反轉）就出；half = z 退回 E/2 以內就出。
β：1 / vol，另加 gold = 只做黃金（β=0、缺口就是黃金自錨點的漲跌），檢查 NAS 那一腳有沒有用。
輸出：results_part36.txt、part36_grid.csv、part36_equity.png

以下為第三十五部分原說明：
「紅線」當沖版 —— 每天在錨點時間把 NAS100 與 XAUUSD 都設為 1，紅線 = 兩者平均；
盤中黃金偏離紅線太遠 → 空黃金、多 NAS（或反過來），回到紅線出場，23:00（伺服器時間）前強制平倉。
資料：XAUUSD M1（M5）/ M15、NAS100 M1 合成，2023/10/2 ~ 2026/10/9（XAU M1 缺 2026/4/3~4/14，那幾天 M5 沒有）。

缺口 gap_t = (ln XAU_t − ln XAU_a) − β (ln NAS_t − ln NAS_a)，a = 當天錨點那根收盤。紅線上方 / 下方各 gap/2（β=1 時）。
標準化：z_t = gap_t / (σ · √k)，σ = 過去 20 天每根 (r_xau − β r_nas) 的標準差（只用 t 以前），k = 錨點後第幾根。
  → z 就是「以隨機漫步來說，這個缺口有多不尋常」。
進場：z > E → 空黃金、多 β 份 NAS；z < −E 反過來。E = 1 / 1.5 / 2。出場：gap 回到 0；停損 |z| > E+1.5（有 / 無）；23:00 平倉。
β：1（照圖，同名目）或 vol（σ_xau/σ_nas，過去 20 天，兩腳同風險）。
錨點：01:00（一天開始）、10:00（倫敦）、16:30（紐約現貨開盤）。
成本：兩腳進出各付當根點差的一半。樣本內 < 2025/1/1 ≤ 樣本外。
（第三十五部分的輸出檔名不適用於本檔。）
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
OUT = open("results_part36.txt", "w")


def P(*a):
    print(*a); print(*a, file=OUT)


n1 = pd.concat([load(f) for f in sorted(glob.glob("data/NAS100_M1_*.csv"))])
n1 = n1[~n1.index.duplicated(keep="last")].sort_index()
xm1 = load("data/XAUUSD_M1_2023_2026.csv")
xm15 = load("data/XAUUSD_M15_2023_2026.csv")


def frame(nd, xd, rule):
    def rs(d, k):
        if rule is None:
            return d.close, d.spread * k
        return d.close.resample(rule).last(), d.spread.resample(rule).last() * k
    nc, ns = rs(nd, 0.1)
    xc, xs = (xd.close, xd.spread * 0.01) if rule == "15min" and xd is xm15 else rs(xd, 0.01)
    D = pd.concat({"n": nc, "n_sp": ns, "x": xc, "x_sp": xs}, axis=1, sort=True).dropna()
    return D


DATA = {"M5": frame(n1, xm1, "5min"), "M15": frame(n1, xm15, "15min")}
BPD = {"M5": 264, "M15": 88}            # 每天約幾根（22 小時）
ANCHORS = {"01:00": (1, 0), "10:00": (10, 0), "16:30": (16, 30)}


def prep(D, tf, beta_kind):
    rn, rx = np.log(D.n).diff(), np.log(D.x).diff()
    W = 20 * BPD[tf]
    if beta_kind == "1":
        b = pd.Series(1.0, D.index)
    elif beta_kind == "gold":
        b = pd.Series(0.0, D.index)
    else:
        b = (rx.rolling(W).std() / rn.rolling(W).std()).shift()
    sig = (rx - b * rn).rolling(W).std().shift()
    return b, sig


def backtest(D, tf, anchor, E, exit_kind, beta_kind):
    b, sig = prep(D, tf, beta_kind)
    n, x, nsp, xsp = D.n.to_numpy(), D.x.to_numpy(), D.n_sp.to_numpy(), D.x_sp.to_numpy()
    b, sig = b.to_numpy(), sig.to_numpy()
    t = D.index
    hh, mm = t.hour.to_numpy(), t.minute.to_numpy()
    day = t.normalize()
    ah, am = ANCHORS[anchor]
    trades, ia, pos, done = [], -1, 0, False
    for i in range(len(D)):
        if i > 0 and day[i] != day[i - 1]:
            ia, done = -1, False
        if hh[i] == ah and mm[i] == am:
            ia = i
            continue
        if ia < 0 or not np.isfinite(sig[ia]) or not np.isfinite(b[ia]):
            continue
        be = b[ia]
        gap = np.log(x[i] / x[ia]) - be * np.log(n[i] / n[ia])
        z = gap / (sig[ia] * np.sqrt(i - ia))
        eod = hh[i] >= 23 or (i + 1 < len(D) and day[i + 1] != day[i])
        if pos:
            if exit_kind == "eod0":
                why = "flip" if pos * gap <= 0 else "eod" if eod else None
            else:
                why = "fade" if pos * z < E / 2 else "eod" if eod else None
            if why:
                legx = pos * (x[i] / x[i0] - 1)
                legn = -pos * bb * (n[i] / n[i0] - 1)
                cost = (xsp[i0] / x[i0] + xsp[i] / x[i]) / 2 + bb * (nsp[i0] / n[i0] + nsp[i] / n[i]) / 2
                g = 1 + bb
                trades.append((t[i0], t[i], pos, legx / g * 1e4, legn / g * 1e4, cost / g * 1e4, i - i0, why))
                pos, done = 0, True
                continue
        if not pos and not done and not eod and hh[i] < 22 and abs(z) > E:
            pos, i0, bb = (1 if z > 0 else -1), i, be       # pos=+1：多黃金、空 NAS（黃金比較強）
    T = pd.DataFrame(trades, columns=["entry", "exit", "side", "xau_bp", "nas_bp", "cost_bp", "bars", "why"])
    T["pnl_bp"] = T.xau_bp + T.nas_bp - T.cost_bp
    return T


def stats(T):
    if len(T) < 5:
        return dict(n=len(T), bp=np.nan, t=np.nan, win=np.nan, pf=np.nan, sharpe=np.nan, gross=np.nan)
    r = T.pnl_bp
    d = r.groupby(T.exit.dt.normalize()).sum()
    gl = -r[r < 0].sum()
    return dict(n=len(r), bp=r.mean(), t=r.mean() / r.std() * np.sqrt(len(r)), win=(r > 0).mean(),
                pf=r[r > 0].sum() / gl if gl > 0 else np.inf, sharpe=d.mean() / d.std() * np.sqrt(252),
                gross=(r + T.cost_bp).mean())


def fmt(s):
    return (f"n={s['n']:5d} 每筆 {s['bp']:+6.2f}bp（扣成本前 {s['gross']:+6.2f}）t={s['t']:+5.2f} 勝率 {s['win']:.0%} "
            f"PF {s['pf']:.2f} Sharpe {s['sharpe']:+5.2f}")


P(f"資料：M5 {len(DATA['M5'])} 根、M15 {len(DATA['M15'])} 根，{DATA['M15'].index[0].date()} ~ {DATA['M15'].index[-1].date()}")
P(f"樣本內 < {IS_END.date()} ≤ 樣本外")

rows, ALL = [], {}
for tf, anchor, E, ex, bk in itertools.product(["M5", "M15"], ANCHORS, [1.0, 1.5, 2.0], ["eod0", "half"], ["1", "vol", "gold"]):
    T = backtest(DATA[tf], tf, anchor, E, ex, bk)
    ALL[(tf, anchor, E, ex, bk)] = T
    for per, m in [("IS", T.entry < IS_END), ("OOS", T.entry >= IS_END)]:
        rows.append(dict(tf=tf, anchor=anchor, E=E, exit=ex, beta=bk, per=per, **stats(T[m])))
G = pd.DataFrame(rows)
G.to_csv("part36_grid.csv", index=False)
GP = G[G.beta != "gold"]
W = GP.pivot_table(index=["tf", "anchor", "E", "exit", "beta"], columns="per", values=["bp", "gross", "sharpe", "n"])
WG = G[G.beta == "gold"].pivot_table(index=["tf", "anchor", "E", "exit"], columns="per", values=["bp", "sharpe", "n"])


def K(k):
    return f"{k[0]} 錨點{k[1]} E={k[2]} 出場{k[3]} β={k[4]}"


P(f"\n## 1. 配對（β=1 / vol）全部 {len(W)} 組")
P(f"  每筆 > 0：樣本內 {(W['bp']['IS'] > 0).sum()}/{len(W)}、樣本外 {(W['bp']['OOS'] > 0).sum()}/{len(W)}；"
  f"扣成本前 > 0：樣本內 {(W['gross']['IS'] > 0).sum()}、樣本外 {(W['gross']['OOS'] > 0).sum()}")
P(f"  Sharpe 中位數：樣本內 {W['sharpe']['IS'].median():+.2f}、樣本外 {W['sharpe']['OOS'].median():+.2f}；"
  f"排名相關 {W['sharpe']['IS'].corr(W['sharpe']['OOS'], method='spearman'):+.2f}")
for col in ["tf", "anchor", "E", "exit", "beta"]:
    g = GP.groupby([col, "per"]).bp.median().unstack()
    P(f"  依 {col:6s}（每筆 bp 中位數 IS / OOS）：" + "  ".join(f"{v}: {g.loc[v, 'IS']:+.2f} / {g.loc[v, 'OOS']:+.2f}" for v in g.index))

P(f"\n## 1b. 對照：只做黃金（同錨點同 z 規則）{len(WG)} 組")
P(f"  每筆 > 0：樣本內 {(WG['bp']['IS'] > 0).sum()}/{len(WG)}、樣本外 {(WG['bp']['OOS'] > 0).sum()}/{len(WG)}；"
  f"Sharpe 中位數 樣本內 {WG['sharpe']['IS'].median():+.2f}、樣本外 {WG['sharpe']['OOS'].median():+.2f}")
for bk in ["1", "vol"]:
    pr = W.xs(bk, level="beta")["sharpe"]
    P(f"  配對 β={bk:3s} 比只做黃金 Sharpe 高的組合：樣本內 {(pr['IS'] > WG['sharpe']['IS']).sum()}/{len(WG)}、"
      f"樣本外 {(pr['OOS'] > WG['sharpe']['OOS']).sum()}/{len(WG)}")

P("\n## 2. 每一組（每筆 bp：樣本內 → 樣本外，扣成本前樣本外；筆數）")
for k in W.index:
    kg = k[:4]
    P(f"  {K(k):34s} {W.loc[k, ('bp', 'IS')]:+6.2f} → {W.loc[k, ('bp', 'OOS')]:+6.2f}（扣成本前 {W.loc[k, ('gross', 'OOS')]:+6.2f}）"
      f"  n {int(W.loc[k, ('n', 'IS')])}/{int(W.loc[k, ('n', 'OOS')])}  Sharpe {W.loc[k, ('sharpe', 'IS')]:+.2f} → {W.loc[k, ('sharpe', 'OOS')]:+.2f}"
      f"｜只黃金 {WG.loc[kg, ('sharpe', 'IS')]:+.2f} → {WG.loc[kg, ('sharpe', 'OOS')]:+.2f}")

P("\n## 3. 樣本內前 5 名 → 樣本外細節")
top = W[W["n"]["IS"] >= 50].sort_values(("sharpe", "IS"), ascending=False).head(5)
for k in top.index:
    T = ALL[k]
    i, o = T[T.entry < IS_END], T[T.entry >= IS_END]
    P(f"  {K(k)}")
    P(f"    樣本內 {fmt(stats(i))}")
    P(f"    樣本外 {fmt(stats(o))}")
    P(f"    出場：" + "  ".join(f"{w} {len(g)} 筆 {g.pnl_bp.mean():+.1f}bp" for w, g in T.groupby("why")) +
      f"｜拆腳（全期每筆）XAU {T.xau_bp.mean():+.2f} NAS {T.nas_bp.mean():+.2f} 成本 −{T.cost_bp.mean():.2f}")
    TG = ALL[(*k[:4], "gold")]
    P(f"    只做黃金同規則：樣本內 {fmt(stats(TG[TG.entry < IS_END]))}")
    P(f"                    樣本外 {fmt(stats(TG[TG.entry >= IS_END]))}")
    P(f"    多黃金空NAS {fmt(stats(o[o.side == 1]))}")
    P(f"    空黃金多NAS {fmt(stats(o[o.side == -1]))}")
    P(f"    逐年 Sharpe：" + "  ".join(f"{y}: {stats(g)['sharpe']:+.2f}" for y, g in T.groupby(T.entry.dt.year)))

fig, ax = plt.subplots(1, 1, figsize=(11, 4.5))
for k in top.index[:3]:
    T = ALL[k]
    ax.plot(T.exit, T.pnl_bp.cumsum(), lw=1, label=K(k).replace("錨點", "anchor ").replace("出場", "exit "))
    TG = ALL[(*k[:4], "gold")]
    ax.plot(TG.exit, TG.pnl_bp.cumsum(), lw=1, ls=":", label=K(k[:4] + ("gold only",)).replace("錨點", "anchor ").replace("出場", "exit "))
ax.axvline(IS_END, color="k", ls="--"); ax.legend(fontsize=7); ax.set_title("Intraday gap momentum: top-3 in-sample pair combos (solid) vs gold-only same rule (dotted), cumulative bp")
plt.tight_layout(); plt.savefig("part36_equity.png", dpi=110)
