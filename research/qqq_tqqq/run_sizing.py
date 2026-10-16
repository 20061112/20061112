"""第二類用法：不再加濾網，而是改「部位大小」。
  ENS    參數集成（OU 策略 16 組、H1 六種背離各 1/N 的做法）：27 組 (MA × z 回看 × z 門檻) 投票，TQQQ 權重 = 開槓桿的比例
  ENS_T  趨勢集成：MA150/200/250 + TSMOM 126/189/252 投票 × ATR z < 0.5
  VT     波動目標（part44 vol sizing / OU 的 vr<1 → ×2）：MA200 之上時，曝險 = clip(σ* / σ_HAR, 1, 3)
  ZMIX   ATR z 與 VXN z 平均當波動訊號
  TSMOM  取代 MA200 的回看長度敏感度
輸出 results_sizing.csv。
"""
import itertools
import numpy as np
import pandas as pd
from engine import atr_z
import indicators as ind
from run_tests import q, OK0, Z, ABOVE, ZOK, c, r, w_rule, w_cont, evaluate

w0 = w_rule(OK0)
b, broll, bcal = evaluate(w0)
b, broll, bcal = evaluate(w0, broll, bcal)
rows = [dict(name="BASE", param="", band=0.1) | b]


def add(name, p, w, band=0.1):
    o, _, _ = evaluate(w, broll, bcal, band=band)
    rows.append(dict(name=name, param=p, band=band) | o)
    print(f"{name:7s} {p:28s} band {band:.2f} irr99 {o['irr99']:.3f} irr10 {o['irr10']:.3f} lev {o['lev']:.2f} "
          f"sw {o['sw']:.1f} win10 {o['win10']}/{o['n10']} dIRR10y {o['d_irr10y_med']:+.3f} (min {o['d_irr10y_min']:+.3f}) "
          f"worst {o['worst']:.2f} g10min {o['g10_min']:.2f} worstyr {o['worst_year']:.2f} sh {o['sharpe']:.2f}", flush=True)


# 參數集成
zs = {L: atr_z(q, 20, L) for L in [126, 252, 504]}
mas = {m: c > c.rolling(m).mean() for m in [150, 200, 250]}
votes = []
for m, L, th in itertools.product([150, 200, 250], [126, 252, 504], [0.0, 0.5, 1.0]):
    votes.append((mas[m] & (zs[L] < th)).astype(float))
ens = sum(votes) / len(votes)
for band in [0.1, 0.34]:
    add("ENS", "27 組比例", w_cont(ens), band)
add("ENS", "27 組多數決(>=0.5)", w_rule(ens >= 0.5))

# 趨勢集成
tr_votes = [mas[m].astype(float) for m in mas] + [(ind.tsmom(c, n) > 0).astype(float) for n in [126, 189, 252]]
tr = sum(tr_votes) / len(tr_votes)
for band in [0.1, 0.34]:
    add("ENS_T", "6 種趨勢比例 × z<0.5", w_cont(tr * ZOK), band)
add("ENS_T", "6 種趨勢多數決 × z<0.5", w_rule((tr >= 0.5) & ZOK))

# 波動目標
hv = ind.har_vol(q)
for tgt in [0.20, 0.25, 0.30, 0.35]:
    L = (tgt / hv).clip(1, 3)
    wt = ((L - 1) / 2).where(ABOVE, 0.0).fillna(0.0)
    add("VT", f"σ*={tgt:.0%}", w_cont(wt), 0.1)
    add("VT", f"σ*={tgt:.0%}, 只在 z<0.5", w_cont(wt.where(ZOK, 0.0)), 0.1)

# ATR z + VXN z
vz = ind.rz(np.log(q.vxn))
zmix = ((Z + vz) / 2).fillna(Z)
for th in [0.25, 0.5, 0.75]:
    add("ZMIX", f"(zATR+zVXN)/2<{th}", w_rule(ABOVE & (zmix < th)))

# TSMOM 回看長度
for n in range(120, 301, 20):
    add("TSMOM", f"{n}d>0 取代 MA200", w_rule((ind.tsmom(c, n) > 0) & ZOK))
for m in range(120, 301, 20):
    add("MA", f"MA{m}", w_rule((c > c.rolling(m).mean()) & ZOK))

pd.DataFrame(rows).to_csv("results_sizing.csv", index=False, float_format="%.4f")
