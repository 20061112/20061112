"""第四十二部分 第二階段：訊號源 —— 固定第一階段高原中心 P（K 線 10 分、01:00~04:00 進場、順日線方向），一次改一個面向。研究期 only。→ part42_piv_stage2.csv
基準其餘：前一根高低觸價、停損 2×ATR10m、RR 3、當天收盤平倉。每格也列出對照（每根開盤直接順方向進場，同停損停利）。"""
import numpy as np, pandas as pd
from piv_core import run, summ, direction

DIRS = direction()
BASE = dict(k=10, mode=1, hours=(1, 4), dirs=DIRS)
VARS = [("基準", {})]
VARS += [(f"點位 前 {n} 根高低", dict(nprev=n)) for n in (2, 3, 4, 6, 8)]
VARS += [(f"停損 {s}×ATR", dict(sm=s)) for s in (1.0, 1.5, 3.0, 4.0)]
VARS += [(f"RR {r}", dict(rr=r)) for r in (1.0, 2.0, 4.0, 6.0)] + [("不設停利", dict(rr=0.0))]
VARS += [(f"出場 {h:02d}:00", dict(exit_min=h * 60)) for h in (8, 12, 16, 20, 23)]
VARS += [(f"滑價 {s} 點", dict(slip=s)) for s in (1.0, 3.0)]
if __name__ == "__main__":
    rows = []
    for lab, kw in VARS:
        s = summ(run(**{**BASE, **kw}))
        c = summ(run(**{**BASE, **kw}, ctrl=True)) if "nprev" not in kw else dict(mR=np.nan)
        rows.append(dict(var=lab, **s, ctrl_mR=c["mR"], excess=s["mR"] - c["mR"]))
    R = pd.DataFrame(rows)
    R.to_csv("part42_piv_stage2.csv", index=False)
    pd.set_option("display.width", 250)
    print(R.drop(columns=["nL", "nS"]).round(3).to_string(index=False))
