"""第五十九部分延伸診斷：σ̂ 預測較準，為什麼策略變差？
把 ln σ̂ 先做 EMA 平滑（讓換倉頻率接近原版）再算 z；並看出場時點：各版本「換回 QQQ 後 20 天」QQQ 的報酬
（若 σ̂ 是被雜訊洗出去，下車後 QQQ 應該照樣上漲）。輸出附加到 results_sigma.txt。"""
import numpy as np
import pandas as pd
import indicators as ind
from run_tests import q, OK0, Z, ABOVE, w_rule, evaluate
from run_sigma import har_rel_daily

out = []
P = lambda s: (print(s, flush=True), out.append(s))
w0 = w_rule(OK0)
b, broll, bcal = evaluate(w0); b, broll, bcal = evaluate(w0, broll, bcal)
fwd = q.close.shift(-20) / q.close - 1


def exits(w):
    w = pd.Series(w, index=q.index)
    e = (w.shift(1) > 0) & (w == 0)
    f = fwd[e]
    return len(f), f.mean()


n0, f0 = exits(w0)
P(f"\n診斷：σ̂ 平滑後再替換（PK / TR，h=5，門檻 0.5）")
P(f"  BASE ATR z           irr99 {b['irr99']:.3f} sw {b['sw']:.1f} 出場 {n0} 次，出場後 20 天 QQQ 平均 {f0:+.2%}")
for kind in ["TR", "PK"]:
    s = har_rel_daily(q, 5, kind)
    for span in [1, 5, 10, 20]:
        ls = np.log(s).ewm(span=span, min_periods=1).mean() if span > 1 else np.log(s)
        zs = ind.rz(ls).fillna(Z)
        w = w_rule(ABOVE & (zs < 0.5))
        o, _, _ = evaluate(w, broll, bcal)
        n, f = exits(w)
        P(f"  {kind} EMA{span:<3d}          irr99 {o['irr99']:.3f} irr10 {o['irr10']:.3f} win10 {o['win10']:2d}/17 "
          f"dIRR10y {o['d_irr10y_med']:+.3f} sw {o['sw']:.1f} 出場 {n} 次，出場後 20 天 QQQ 平均 {f:+.2%}")
open("results_sigma.txt", "a").write("\n".join(out) + "\n")
