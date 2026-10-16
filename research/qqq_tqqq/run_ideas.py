"""新想法：用分支訊號「增加」有利的槓桿時間，而不是再加濾網。

前面的結論：超額 = 上升且平靜時整個部位 3 倍的時間；濾網一律砍掉這段時間所以變差。
σ̂ 預測反應快 → 當出場訊號會被洗出去；但「當進場訊號」也許能更早上車。

  A  GRADE    分級槓桿：z < 0.5 → 3 倍；0.5 ≤ z < z2 → 2 倍（QLD 等級）；其他 → 1 倍（只在 MA200 之上）
  B  VOL_OR   波動「任一」平靜就開：ATR z < 0.5 或 σ̂ z < th（σ̂ = har_rel 日線版）或 VXN z < th
  C  ASYM     快進慢出：進場看 σ̂ z < 0.5（且 ATR z < 1），出場只在 ATR z ≥ 1 或跌破 MA200
  D  TREND_OR 趨勢「任一」成立：MA200 之上，或 CUSUM 剛翻多，或 ER 很高（V 型反彈不用等 200 日線）
  E  BELOW2X  MA200 之下但非常平靜（ATR z < th）→ 2 倍
  F  PANIC    恐慌補倉時點：原本 z > 2；改成波動「見頂回落」（z > 1.5 且 σ̂ < 20 日實現波動）
輸出 results_ideas.csv、results_ideas.txt。
"""
import itertools
import numpy as np
import pandas as pd
import indicators as ind
import engine as E
from run_tests import q, OK0, Z, ABOVE, c, r, w_rule, w_cont, evaluate
from run_sigma import har_rel_daily

out = []
P = lambda s: (print(s, flush=True), out.append(s))
w0 = w_rule(OK0)
b, broll, bcal = evaluate(w0); b, broll, bcal = evaluate(w0, broll, bcal)
rows = [dict(idea="BASE", param="") | b]


def add(idea, p, w, band=0.1):
    o, _, _ = evaluate(w, broll, bcal, band=band)
    rows.append(dict(idea=idea, param=p) | o)
    P(f"{idea:8s} {p:34s} irr99 {o['irr99']:.3f} irr10 {o['irr10']:.3f} lev {o['lev']:.2f} sw {o['sw']:4.1f} "
      f"win10 {o['win10']:2d}/17 dIRR10y {o['d_irr10y_med']:+.3f}(min {o['d_irr10y_min']:+.3f}) worst {o['worst']:.2f} "
      f"g10min {o['g10_min']:.2f} worstyr {o['worst_year']:.2f} sh {o['sharpe']:.2f} mdd {o['mdd_lump']:.2f}")


P(f"BASE {'':37s} irr99 {b['irr99']:.3f} irr10 {b['irr10']:.3f} lev {b['lev']:.2f} sw {b['sw']:4.1f} "
  f"worst {b['worst']:.2f} g10min {b['g10_min']:.2f} worstyr {b['worst_year']:.2f} sh {b['sharpe']:.2f} mdd {b['mdd_lump']:.2f}")

sig = har_rel_daily(q, 5, "PK")
zs = ind.rz(np.log(sig))
vz = ind.rz(np.log(q.vxn))

# A 分級槓桿
for z2 in [0.75, 1.0, 1.5]:
    t = pd.Series(0.0, index=q.index)
    t[ABOVE & (Z < z2)] = 0.5
    t[ABOVE & (Z < 0.5)] = 1.0
    add("GRADE", f"0.5≤z<{z2} → 2x", w_cont(t))

# B 波動任一平靜
for th in [0.0, 0.25, 0.5]:
    add("VOL_OR", f"ATRz<0.5 或 σ̂z<{th}", w_rule(ABOVE & ((Z < 0.5) | (zs < th).fillna(False))))
for th in [0.0, 0.25, 0.5]:
    add("VOL_OR", f"ATRz<0.5 或 VXNz<{th}", w_rule(ABOVE & ((Z < 0.5) | (vz < th).fillna(False))))

# C 快進慢出
for zx in [0.75, 1.0]:
    ent = ABOVE & (zs.fillna(Z) < 0.5) & (Z < zx)
    stay = ABOVE & (Z < zx)
    add("ASYM", f"進 σ̂z<0.5&ATRz<{zx}，出 ATRz≥{zx}", w_rule(ent, stay))
    ent2 = ABOVE & (Z < 0.5)
    add("ASYM", f"進 ATRz<0.5，出 ATRz≥{zx}（對照）", w_rule(ent2, stay))

# D 趨勢任一成立
for h in [4.0, 6.0, 8.0]:
    cs = ind.cusum_state(r, 0.25, h)
    add("TREND_OR", f"MA200 或 CUSUM 翻多 h={h}", w_rule((ABOVE | (cs > 0)) & (Z < 0.5)))
for n, a in itertools.product([20, 60], [0.3, 0.5]):
    e = ind.er(c, n)
    add("TREND_OR", f"MA200 或 ER{n}>{a}", w_rule((ABOVE | (e > a).fillna(False)) & (Z < 0.5)))
for n in [42, 63]:
    m = ind.tsmom(c, n)
    add("TREND_OR", f"MA200 或 TSMOM{n}>0", w_rule((ABOVE | (m > 0).fillna(False)) & (Z < 0.5)))

# E MA200 之下但非常平靜 → 2 倍
for th in [-1.0, -0.5, 0.0]:
    t = pd.Series(0.0, index=q.index)
    t[~ABOVE & (Z < th)] = 0.5
    t[OK0] = 1.0
    add("BELOW2X", f"MA200 下且 ATRz<{th} → 2x", w_cont(t))

# F 恐慌補倉時點（只影響 20% 現金）
rv20 = r.rolling(20).std()
for zp in [1.5, 2.0]:
    trig = ((Z > zp) & (sig < rv20)).astype(float) * 3.0          # 偽造 z = 3 讓引擎觸發補倉
    zfake = pd.Series(np.where(trig > 0, 3.0, np.minimum(Z.fillna(0), 1.9)), index=q.index)
    o = E.stats(E.run(q, w0, z=Z)) if False else None
    full = E.stats(E.run(q, w0, z=Z.where(Z <= 2, 1.9).where(trig == 0, 3.0)))
    s10 = E.stats(E.run(q, w0, start="2010-02-12", z=Z.where(Z <= 2, 1.9).where(trig == 0, 3.0)))
    roll = E.rolling_windows(q, w0, z=Z.where(Z <= 2, 1.9).where(trig == 0, 3.0))
    d = roll.irr.values - broll.irr.values
    rows.append(dict(idea="PANIC", param=f"z>{zp} 且 σ̂<RV20", irr99=full["irr"], irr10=s10["irr"], worst=full["worst"],
                     win10=int((d > 0).sum()), d_irr10y_med=np.median(d), g10_min=roll.gain.min()))
    P(f"PANIC    z>{zp} 且 σ̂<RV20 補倉{'':17s} irr99 {full['irr']:.3f} irr10 {s10['irr']:.3f} win10 {(d > 0).sum():2d}/17 "
      f"dIRR10y {np.median(d):+.4f} worst {full['worst']:.2f} g10min {roll.gain.min():.2f}")

R = pd.DataFrame(rows)
R.to_csv("results_ideas.csv", index=False, float_format="%.4f")
open("results_ideas.txt", "w").write("\n".join(out) + "\n")
