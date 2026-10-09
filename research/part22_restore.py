"""第二十二部分 B：回復力量測 × 均值定義 的配對矩陣。

事件：|D| 第一次超過 IS 80% 分位（回到 1/3 門檻內才重新武裝）。
結果（回歸方向為正，扣點差，ATR 單位）：
  y6 / y12 / y24 = 反向持有 6 / 12 / 24 根；race = +1 先回到均值、−1 先再往外 1 ATR。
每個量測：Spearman IC（量測 vs y12），IS 前後兩半（1~3 月、4~5 月）與 OOS（6~10 月）。
好的回復力量測：IC 方向在三段一致。
  IC > 0：量測越大越容易回歸（= 回復力指標）
  IC < 0：量測越大越容易延續（= 動能爆發指標）
"""
import numpy as np
import pandas as pd
from part22_lib import (load_tf, means, atr, fwd_returns, events, race, restoring_features, context_features,
                        spearman, tstat_ic, CUT)

out = open("results_part22.txt", "a")
def P(s=""): print(s); out.write(s + "\n"); out.flush()

CUT1 = pd.Timestamp("2026-04-01")
MEANS = ["ema20", "ema50", "sma20", "kama", "linreg20", "rvwap20", "vwap_sess"]


def build(tf, mean_name, q=0.80, d=None, M=None, A=None):
    if d is None:
        d = load_tf(tf); A = atr(d, 14); M = means(d)
    mean = M[mean_name]
    F = restoring_features(d, mean, A)
    C = context_features(d, A, mean)
    D = F.D
    thr = D[d.index < CUT].abs().quantile(q)
    ev = events(D, d.seg, d0=thr, rearm=thr / 3)
    ev = ev[(ev > 600) & (ev + 25 < len(d))]
    s = -np.sign(D.to_numpy()[ev])                     # 回歸方向
    cost = (d.spread / 100 / A).to_numpy()[ev]
    E = pd.concat([F.iloc[ev], C.iloc[ev]], axis=1)
    # 快均值偏離 / 慢均值偏離（同方向為正）：< 1 代表短期已經往回拉（回復開始）
    fast = (d.close - (M.ema20 if mean_name != "ema20" else M.kama)) / A
    E["fast_slow"] = (fast.to_numpy()[ev] * np.sign(D.to_numpy()[ev])) / D.abs().to_numpy()[ev]
    E["side"] = s
    # 下一根開盤進場
    o1 = d.open.to_numpy()[ev + 1]; a = A.to_numpy()[ev]
    for H in (6, 12, 24):
        cH = d.close.to_numpy()[ev + H]
        same = d.seg.to_numpy()[ev + H] == d.seg.to_numpy()[ev]
        E[f"y{H}"] = np.where(same, s * (cH - o1) / a - cost, np.nan)
    E["race"] = race(d, ev, mean.to_numpy()[ev], a, -s)
    E["atr"] = a; E["ev"] = ev; E["thr"] = thr
    E["per"] = np.where(E.index < CUT1, "IS1", np.where(E.index < CUT, "IS2", "OOS"))
    return E, d, M, A


FEATS = ["dist", "sig_ratio", "bbw_chg", "bbw_pct", "er10", "er20", "er_ratio", "atr_ratio", "atr_slope",
         "er_leg", "pullback", "z_over_d", "fast_slow", "vel", "acc", "ou_theta", "osc_k", "osc_c", "ac1", "vr6", "wick_out", "vol_rel", "leg_bars",
         "htf", "day_used", "day_der", "vwap_side", "beyond_pd"]
NAMES = {"dist": "距離 |D|", "sig_ratio": "σ短/σ長", "bbw_chg": "帶寬5根變化", "bbw_pct": "帶寬水位", "er10": "ER10",
         "er20": "ER20", "er_ratio": "ER5/ER20", "atr_ratio": "ATR5/ATR60", "atr_slope": "ATR10根變化",
         "er_leg": "段ER", "pullback": "已拉回", "z_over_d": "z/D", "fast_slow": "快/慢偏離", "vel": "離開速度", "acc": "離開加速度", "ou_theta": "OU θ", "osc_k": "振盪 k", "osc_c": "阻尼 c",
         "ac1": "報酬自相關", "vr6": "變異數比", "wick_out": "外側影線", "vol_rel": "相對量", "leg_bars": "離開根數",
         "htf": "順高週期", "day_used": "日波幅已用", "day_der": "當天ER(順向)", "vwap_side": "VWAP同側距",
         "beyond_pd": "超出前日高低"}


def ic_table(E, y="y12"):
    rows = []
    for f in FEATS:
        r = dict(feat=f)
        for p in ("IS1", "IS2", "OOS"):
            sub = E[E.per == p]
            r[p] = spearman(sub[f], sub[y])
        r["IS"] = spearman(E[E.per != "OOS"][f], E[E.per != "OOS"][y])
        r["n_IS"] = (E.per != "OOS").sum(); r["n_OOS"] = (E.per == "OOS").sum()
        r["consistent"] = np.sign(r["IS1"]) == np.sign(r["IS2"]) == np.sign(r["OOS"])
        rows.append(r)
    return pd.DataFrame(rows).set_index("feat")


if __name__ == "__main__":
    P("\n" + "=" * 120)
    P("第二十二部分 B：回復力量測的 IC（y12 = 反向持有 12 根，扣點差；IS1=1~3月 IS2=4~5月 OOS=6~10月）")
    tf = "M5"
    d = load_tf(tf); A = atr(d, 14); M = means(d)
    allE, mat = {}, {}
    for mn in MEANS:
        E, *_ = build(tf, mn, d=d, M=M, A=A)
        allE[mn] = E
        T = ic_table(E)
        mat[mn] = T
        P(f"\n--- {tf} 均值 = {mn}  事件 IS {T.n_IS.iat[0]} / OOS {T.n_OOS.iat[0]}  門檻 {E.thr.iat[0]:.2f} ATR  "
          f"基準 y12 IS {E[E.per!='OOS'].y12.mean():+.3f} OOS {E[E.per=='OOS'].y12.mean():+.3f} ---")
        for f, r in T.iterrows():
            mark = "✓" if r.consistent and min(abs(r.IS1), abs(r.IS2), abs(r.OOS)) >= 0.03 else " "
            P(f"  {NAMES[f]:12s} IS1 {r.IS1:+.3f}  IS2 {r.IS2:+.3f}  OOS {r.OOS:+.3f}  {mark}")
    # 配對矩陣：均值 × 量測，值 = 三段 IC 中絕對值最小的那段（帶方向），三段不同向 = 0
    P("\n配對矩陣（均值 × 量測）：三段 IC 同向時取最弱一段（帶符號），不同向記 0；正 = 回復力、負 = 動能")
    P(f"{'':12s}" + "".join(f"{m:>10s}" for m in MEANS))
    grid = pd.DataFrame(index=FEATS, columns=MEANS, dtype=float)
    for f in FEATS:
        for mn in MEANS:
            r = mat[mn].loc[f]
            v = [r.IS1, r.IS2, r.OOS]
            grid.loc[f, mn] = np.sign(v[0]) * min(map(abs, v)) if r.consistent else 0.0
        P(f"{NAMES[f]:12s}" + "".join(f"{grid.loc[f, m]:+10.3f}" for m in MEANS))
    grid.to_csv("part22_pair_mean_feat.csv")
    P("\n每個均值：三段一致且最弱段 |IC| ≥ 0.03 的量測個數 / 這些量測的最弱段 |IC| 總和")
    for mn in MEANS:
        g = grid[mn].abs()
        P(f"  {mn:10s} {int((g >= 0.03).sum()):2d} 個   總和 {g[g >= 0.03].sum():.3f}")
