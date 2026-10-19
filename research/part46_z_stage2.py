"""第四十六部分 第二階段：訊號源 —— 固定第一階段高原中心 Z1（純 z 回到帶內、M15、09:00~20:00），一次改一個面向。研究期 only。→ part46_z_stage2.csv
基準其餘：z20 門檻 2、突破訊號 K 高（低）點 3 根內進場、停損近 3 根極值外 0.1 ATR14、2R、最多 16 根。"""
import numpy as np, pandas as pd
from z_core import run, summ

BASE = dict(k=15, src="zcross", hours=(9, 20))
VARS = [("基準", {})]
VARS += [(f"z 長度 {n}", dict(zlen=n)) for n in (10, 30, 50, 100)]
VARS += [(f"z 門檻 {t}", dict(zthr=t)) for t in (1.5, 1.75, 2.25, 2.5, 3.0)]
VARS += [("訊號源：型態 + z（XAU D）", dict(src="pattern")), ("訊號源：只要型態（對照）", dict(src="pattern_noz"))]
VARS += [("收盤直接進場", dict(entry_close=True))] + [(f"等待 {w} 根", dict(wait=w)) for w in (1, 2, 5)]
VARS += [(f"停損緩衝 {b} ATR", dict(buf=b)) for b in (0.0, 0.05, 0.2, 0.5)]
VARS += [(f"RR {r}", dict(rr=r)) for r in (1.0, 1.5, 3.0, 4.0)] + [("中軌出場", dict(mid_exit=True))]
VARS += [(f"持有 {h} 根", dict(hold=h)) for h in (4, 8, 32)]
VARS += [(f"滑價 {s} 點", dict(slip=s)) for s in (1.0, 3.0)]
if __name__ == "__main__":
    rows = [dict(var=lab, **summ(run(**{**BASE, **kw}))) for lab, kw in VARS]
    R = pd.DataFrame(rows)
    R.to_csv("part46_z_stage2.csv", index=False)
    pd.set_option("display.width", 250)
    print(R.round(3).to_string(index=False))
