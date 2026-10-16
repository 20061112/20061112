"""雙動能的代價是 V 型反彈時還在 T-bill（2003、2009、2020、2023）。
組合：CUSUM 翻多（h = 4 / 6 / 8）時提早離開 T-bill；再看同時用 CUSUM 早期上車（第 D 類）的版本。只做歷史（拼貼法驗不了長空頭）。
輸出附加到 results_dualmom.txt。"""
import numpy as np
import pandas as pd
import engine as E
import indicators as ind
from run_tests import q, OK0, Z, ABOVE, c, r, w_rule
from run_drawdown import lump, mdd
from run_dualmom import dm_off

out = []
P = lambda s: (print(s, flush=True), out.append(s))
base_w = w_rule(OK0)
broll = E.rolling_windows(q, base_w)


def show(lab, w, off):
    f = E.stats(E.run(q, w, off=off)); roll = E.rolling_windows(q, w, off=off)
    d = roll.irr.values - broll.irr.values
    ls = lump(w, off); yr = (1 + ls).groupby(ls.index.year).prod() - 1
    P(f"  {lab:40s} irr99 {f['irr']:.3f} dIRR10y {np.median(d):+.3f} 最慘 {f['worst']:.2f} 10年段最慘 最差 {roll.worst.min():.2f} 中位 {roll.worst.median():.2f} "
      f"MDD {mdd(ls):.2f} 2003 {yr[2003]:+.0%} 2009 {yr[2009]:+.0%} 2023 {yr[2023]:+.0%} 換倉 {f['sw_per_year']:.1f}")


P("\n4. 雙動能 + CUSUM 翻多提早離開 T-bill（只做歷史）")
show("原策略", base_w, None)
dm = dm_off(c, q.rf, 200, 252)
show("雙動能 12 月", base_w, dm.values)
for h in [4.0, 6.0, 8.0]:
    up = ind.cusum_state(r, 0.25, h) > 0
    off = (dm & ~up.shift(1, fill_value=False)).values
    show(f"雙動能 + CUSUM h{h:.0f} 離開 T-bill", base_w, off)
    wD = w_rule((ABOVE | up) & (Z < 0.5))
    show(f"雙動能 + CUSUM h{h:.0f} 離開 T-bill 並早期上車", wD, off)
open("results_dualmom.txt", "a").write("\n".join(out) + "\n")
