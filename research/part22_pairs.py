"""第二十二部分 G：兩兩配對的契合程度（β 框架）。

均值 = M5 EMA20 與 EMA50（分別跑）。全部 K 棒。
每一對量測 (a, b)，用 IS 三分位切成 3×3 = 9 格，每格估一個回復係數 β（前瞻 12 根報酬對 −D 的斜率）：
  ρ          兩個量測的 Spearman 相關（|ρ| 高 = 重複）
  IS 跨度     IS 中 9 格 β 的最大 − 最小
  OOS 邊際    IS 最高 β 格 與 IS 最低 β 格，在 OOS 的 β 差（> 0 = 配對的分辨力在樣本外還在）
  單獨最佳    a、b 各自三分位（高−低，方向取 IS）在 OOS 的 Δβ，取較大者
  綜效        OOS 邊際 − 單獨最佳（> 0 = 合在一起比任一個單獨強）
  格子一致    9 格 β 的 IS 與 OOS 排序相關
  契合分數    max(OOS 邊際,0) × max(格子一致,0) × (1 − |ρ|)
"""
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from part22_lib import load_tf, means, atr
from part22_beta import frame, beta
from part22_restore import NAMES

out = open("results_part22.txt", "a")
def P(s=""): print(s, flush=True); out.write(s + "\n"); out.flush()

PAIR = ["dist", "sig_ratio", "bbw_chg", "bbw_pct", "er10", "er20", "er_ratio", "er_leg", "atr_ratio", "atr_slope",
        "vel", "acc", "osc_c", "wick_out", "vol_rel", "htf", "day_used", "day_der", "vwap_side"]
EN = {"dist": "dist|D|", "sig_ratio": "sig10/60", "bbw_chg": "BBw chg5", "bbw_pct": "BBw level", "er10": "ER10",
      "er20": "ER20", "er_ratio": "ER5/ER20", "er_leg": "ER leg", "atr_ratio": "ATR5/60", "atr_slope": "ATR chg10",
      "vel": "velocity", "acc": "accel", "osc_c": "damping c", "wick_out": "outer wick", "vol_rel": "rel vol",
      "htf": "HTF align", "day_used": "day range used", "day_der": "day ER(dir)", "vwap_side": "VWAP same-side"}


def terc(X, f, IS):
    e = np.nanquantile(X.loc[IS, f], [1 / 3, 2 / 3])
    return pd.Series(np.digitize(X[f], e), X.index).where(X[f].notna())


def cell_betas(X, g, sel):
    return pd.Series({k: beta(X.D[sel & (g == k)], X.f[sel & (g == k)]) for k in range(int(g.max()) + 1 if g.notna().any() else 0)})


def run(X, tag):
    IS, OOS = X.per != "OOS", X.per == "OOS"
    T = {f: terc(X, f, IS) for f in PAIR}
    single = {}
    for f in PAIR:
        bi = cell_betas(X, T[f], IS); bo = cell_betas(X, T[f], OOS)
        sgn = np.sign(bi[2] - bi[0])
        single[f] = (bi[2] - bi[0], bo[2] - bo[0], sgn * (bo[2] - bo[0]))
    sub = X.iloc[::6]
    rows = []
    for a, b in itertools.combinations(PAIR, 2):
        g = T[a] * 3 + T[b]
        bi, bo = cell_betas(X, g, IS), cell_betas(X, g, OOS)
        best, worst = bi.idxmax(), bi.idxmin()
        edge = bo[best] - bo[worst]
        sb = max(single[a][2], single[b][2])
        rho = sub[a].rank().corr(sub[b].rank())
        cons = bi.rank().corr(bo.rank())
        rows.append(dict(a=a, b=b, rho=rho, span_is=bi.max() - bi.min(), edge=edge, single_best=sb, synergy=edge - sb,
                         grid_cons=cons, best=int(best), worst=int(worst), b_best_oos=bo[best], b_worst_oos=bo[worst],
                         score=max(edge, 0) * max(cons, 0) * (1 - abs(rho))))
    R = pd.DataFrame(rows)
    R.to_csv(f"part22_pairs_{tag}.csv", index=False)
    P(f"\n--- {tag}：單一量測三分位 Δβ（高 − 低）IS → OOS ---")
    for f in PAIR:
        P(f"  {NAMES[f]:12s} {single[f][0]:+.3f} → {single[f][1]:+.3f}  {'同向' if np.sign(single[f][0]) == np.sign(single[f][1]) else '反向'}")
    lab = lambda k: f"({k // 3},{k % 3})"
    P(f"\n--- {tag}：契合分數前 25 名（格子 (a三分位, b三分位)，0 = 低、2 = 高） ---")
    P(f"  {'a':12s} {'b':12s}    ρ   IS跨度  OOS邊際  單獨最佳  綜效   格子一致  IS最佳格→OOSβ   IS最差格→OOSβ")
    for r in R.sort_values("score", ascending=False).head(25).itertuples():
        P(f"  {NAMES[r.a]:12s} {NAMES[r.b]:12s} {r.rho:+.2f}  {r.span_is:.2f}   {r.edge:+.3f}   {r.single_best:+.3f}  {r.synergy:+.3f}  {r.grid_cons:+.2f}"
          f"     {lab(r.best)} {r.b_best_oos:+.3f}     {lab(r.worst)} {r.b_worst_oos:+.3f}")
    P(f"\n--- {tag}：使用者指定的核心（距離 / σ / ER / ATR）彼此配對 ---")
    core = ["dist", "sig_ratio", "bbw_chg", "bbw_pct", "er10", "er20", "er_ratio", "er_leg", "atr_ratio", "atr_slope"]
    C = R[R.a.isin(core) & R.b.isin(core)].sort_values("score", ascending=False)
    for r in C.itertuples():
        P(f"  {NAMES[r.a]:12s} {NAMES[r.b]:12s} ρ {r.rho:+.2f}  OOS邊際 {r.edge:+.3f}  綜效 {r.synergy:+.3f}  格子一致 {r.grid_cons:+.2f}  契合 {r.score:.3f}")
    P(f"\n--- {tag}：每個量測當配角的平均（OOS 邊際 / 格子一致 / 綜效 / 契合分數），以及最佳搭檔 ---")
    for f in PAIR:
        s = R[(R.a == f) | (R.b == f)]
        top = s.sort_values("score", ascending=False).iloc[0]
        mate = top.b if top.a == f else top.a
        P(f"  {NAMES[f]:12s} 邊際 {s.edge.mean():+.3f}  一致 {s.grid_cons.mean():+.2f}  綜效 {s.synergy.mean():+.3f}  契合 {s.score.mean():.3f}   最佳搭檔 {NAMES[mate]}（{top.score:.3f}）")

    n = len(PAIR)
    fig, ax = plt.subplots(2, 2, figsize=(17, 15))
    lab2 = [EN[f] for f in PAIR]
    for axx, (k, title, cm, lim) in zip(ax.flat, [("rho", "rho (redundancy)", "RdBu_r", 1), ("edge", "OOS edge: beta(best IS cell) - beta(worst IS cell)", "RdYlGn", .3),
                                                ("grid_cons", "3x3 grid rank consistency IS vs OOS", "RdYlGn", 1), ("synergy", "synergy = pair edge - best single (OOS)", "RdYlGn", .2)]):
        Mx = pd.DataFrame(np.nan, index=PAIR, columns=PAIR)
        for r in R.itertuples():
            Mx.loc[r.a, r.b] = Mx.loc[r.b, r.a] = getattr(r, k)
        im = axx.imshow(Mx.to_numpy(float), cmap=cm, vmin=-lim, vmax=lim)
        axx.set_xticks(range(n)); axx.set_yticks(range(n))
        axx.set_xticklabels(lab2, rotation=70, fontsize=8); axx.set_yticklabels(lab2, fontsize=8)
        for i in range(n):
            for j in range(n):
                v = Mx.iat[i, j]
                if not np.isnan(v):
                    axx.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.5)
        axx.set_title(f"{tag}: {title}", fontsize=10); fig.colorbar(im, ax=axx, shrink=.7)
    fig.tight_layout(); fig.savefig(f"part22_pairs_{tag}.png", dpi=100)
    return R


if __name__ == "__main__":
    P("\n" + "=" * 120)
    P("第二十二部分 G：兩兩配對契合度（β 框架，M5 全部 K 棒，前瞻 12 根）")
    d = load_tf("M5"); A = atr(d, 14); M = means(d)
    for mn in ("ema20", "ema50"):
        run(frame(d, A, M, mn), f"M5_{mn}")
