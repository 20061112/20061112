"""第三十四部分補充：通道順勢網格的滑價敏感度（每完成一筆來回多付 k × 日 ATR10）。"""
import sys, os
import numpy as np, pandas as pd
from multiprocessing import Pool
import part33_band_grid as B
from part28_ib_grid import t
SP0 = B.SP.copy()
atr_bar = B.ATRD.reindex(B.IDX.normalize()).values
def job(a):
    c, k = a
    B.SP = SP0 + np.nan_to_num(atr_bar) * k
    mid, u = B.bands(B.OHLC0, c[0], c[1])
    return a, B.sim(B.OHLC0, mid, u, c[2], c[3], -1)
cfg = [("1h","KC",4,"inward"),("1h","KC",4,"mean"),("1h","KC",10,"inward"),("1h","KC",10,"mean"),("1h","BB",10,"inward"),("5min","KC",4,"inward"),("15min","KC",4,"inward")]
jobs = [(c,k) for c in cfg for k in (0, 0.005, 0.01, 0.02)]
if __name__ == "__main__":
  with Pool(4) as p: res = dict(p.map(job, jobs, chunksize=1))
  for c in cfg:
    r0 = res[(c,0)]; tw = r0.trades.mean()
    a0 = r0.pnl/r0.atr
    line = f"{c}: 每週筆 {tw:.1f}, 每筆 {a0.mean()/tw:+.4f} ATR |"
    for k in (0, 0.005, 0.01, 0.02):
        r = res[(c,k)]; a = r.pnl/r.atr; IS = r.year < 2025
        line += f" 滑價{k}: {a[IS].mean():+.2f}/{a[~IS].mean():+.2f}"
    print(line)
