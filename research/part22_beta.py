"""第二十二部分 E：狀態相依的回復係數 β（核心框架）。

想法：把「回復力」直接定義成 β ——
    前瞻 12 根報酬（ATR） ≈ −β × D        （D = 距離均值 / ATR）
β > 0 = 被拉回（均值回歸），β < 0 = 越偏越遠（動能爆發）。
一個量測如果真的反映回復力，β 應該隨它單調變化：
    量測高 → β 大（回歸） / 量測低 → β 小甚至為負（動能）。
做法：用全部 M5 K 棒（不只事件），按量測的 IS 五分位分組，每組、每段（IS1/IS2/OOS）各估一個 β。
  Δβ = β(Q5) − β(Q1)；三段同向才算數。
另外算 |D| ≥ 1 的 K 棒「反向持有 12 根」平均報酬在 Q1 / Q5 的差，作為可交易的對照。
"""
import numpy as np
import pandas as pd
from part22_lib import load_tf, means, atr, fwd_returns, restoring_features, context_features, ema, CUT
from part22_restore import NAMES

out = open("results_part22.txt", "a")
def P(s=""): print(s); out.write(s + "\n"); out.flush()
CUT1 = pd.Timestamp("2026-04-01")

FE = ["dist", "sig_ratio", "bbw_chg", "bbw_pct", "er10", "er20", "er_ratio", "er_leg", "atr_ratio", "atr_slope",
      "vel", "acc", "pullback", "z_over_d", "ou_theta", "osc_k", "osc_c", "ac1", "vr6", "wick_out", "vol_rel",
      "leg_bars", "htf", "day_used", "day_der", "vwap_side", "beyond_pd"]


def beta(D, f):
    m = D.notna() & f.notna() & (D.abs() < 8)
    x, y = -D[m], f[m]
    if len(x) < 200:
        return np.nan
    return ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()


def frame(d, A, M, mean_name, H=12):
    F = restoring_features(d, M[mean_name], A)
    C = context_features(d, A, M[mean_name])
    X = pd.concat([F, C.drop(columns="sess")], axis=1)
    X["f"] = fwd_returns(d, A, H)
    X["cost"] = d.spread / 100 / A
    X["per"] = np.where(X.index < CUT1, "IS1", np.where(X.index < CUT, "IS2", "OOS"))
    return X.iloc[600:]


def state_beta(X, feat, nq=5):
    IS = X.per != "OOS"
    e = np.unique(np.nanquantile(X.loc[IS, feat], np.linspace(0, 1, nq + 1)))
    e[0], e[-1] = -np.inf, np.inf
    q = pd.cut(X[feat], e, labels=False)
    tab = {}
    for p in ("IS1", "IS2", "OOS"):
        sel = X.per == p
        tab[p] = [beta(X.D[sel & (q == k)], X.f[sel & (q == k)]) for k in range(len(e) - 1)]
        # 可交易對照：|D| ≥ 1 的反向 12 根報酬（扣成本），Q1 與 Qn
        far = sel & (X.D.abs() >= 1)
        g = (-np.sign(X.D) * X.f - X.cost)[far]
        tab[p + "_fade"] = [g[q[far] == k].mean() for k in range(len(e) - 1)]
    return tab


if __name__ == "__main__":
    P("\n" + "=" * 120)
    P("第二十二部分 E：狀態相依回復係數 β（M5 全部 K 棒，前瞻 12 根；β>0 回歸、β<0 延續）")
    d = load_tf("M5"); A = atr(d, 14); M = means(d)
    summary = []
    for mn in ("ema20", "ema50", "ema100", "kama", "linreg20", "rvwap20", "vwap_sess"):
        X = frame(d, A, M, mn)
        b0 = {p: beta(X.D[X.per == p], X.f[X.per == p]) for p in ("IS1", "IS2", "OOS")}
        P(f"\n--- 均值 {mn}：無條件 β  IS1 {b0['IS1']:+.3f}  IS2 {b0['IS2']:+.3f}  OOS {b0['OOS']:+.3f} ---")
        P(f"  {'量測':12s} {'β 五分位 Q1→Q5（IS1 | IS2 | OOS）':72s} Δβ IS1/IS2/OOS      反向12根 Q1/Q5（IS→OOS）")
        for f in FE:
            t = state_beta(X, f)
            db = [t[p][-1] - t[p][0] for p in ("IS1", "IS2", "OOS")]
            cons = np.sign(db[0]) == np.sign(db[1]) == np.sign(db[2])
            strength = np.sign(db[0]) * min(map(abs, db)) if cons else 0.0
            fade = [t["IS1_fade"][0] / 2 + t["IS2_fade"][0] / 2, t["IS1_fade"][-1] / 2 + t["IS2_fade"][-1] / 2, t["OOS_fade"][0], t["OOS_fade"][-1]]
            mark = "✓" if cons and min(map(abs, db)) >= 0.02 else " "
            P(f"  {NAMES[f]:12s} " + " | ".join(" ".join(f"{v:+.2f}" for v in t[p]) for p in ("IS1", "IS2", "OOS"))
              + f"   {db[0]:+.2f}/{db[1]:+.2f}/{db[2]:+.2f} {mark}   {fade[0]:+.2f}/{fade[1]:+.2f} → {fade[2]:+.2f}/{fade[3]:+.2f}")
            summary.append(dict(mean=mn, feat=f, db_IS1=db[0], db_IS2=db[1], db_OOS=db[2], strength=strength,
                                b_q1_OOS=t["OOS"][0], b_q5_OOS=t["OOS"][-1], fade_q1_OOS=fade[2], fade_q5_OOS=fade[3]))
    S = pd.DataFrame(summary)
    S.to_csv("part22_beta.csv", index=False)
    P("\n均值 × 量測：Δβ 三段同向時取最弱段（帶符號），否則 0；正 = 量測越高越會被拉回")
    piv = S.pivot(index="feat", columns="mean", values="strength").reindex(FE)
    cols = list(piv.columns)
    P(f"{'':12s}" + "".join(f"{c:>10s}" for c in cols))
    for f in FE:
        P(f"{NAMES[f]:12s}" + "".join(f"{piv.loc[f, c]:+10.3f}" for c in cols))
    P("\n各均值：|最弱段 Δβ| ≥ 0.02 的量測數、總和")
    for c in cols:
        g = piv[c].abs()
        P(f"  {c:10s} {int((g >= .02).sum()):2d} 個  總和 {g[g >= .02].sum():.3f}")
