"""隨機進場基準：相同數量、相同停損距離分布、相同 2R / 4 小時出場規則，但進場時點與方向隨機。"""
import numpy as np
import pandas as pd
from load import load
from backtest import resample, backtest_m1, _arrays
from strategy import CFG, TF, TF_MIN, run

m1 = load("data/XAUUSD_M1_full.csv")
d = resample(m1, TF)
_, h, l, c, atr, _ = _arrays(d)
real = run(m1)
idx = d.index.get_indexer(real.setup)
r_atr = (real.R_usd.to_numpy() / atr[idx])
cfg = CFG.__class__(**{**CFG.__dict__, "entry": "close", "buf": 0.0})
valid = np.where(np.isfinite(atr) & (np.arange(len(d)) > 60))[0][:-20]
res, ns = [], []
for k in range(300):
    rng = np.random.default_rng(k)
    ts = np.sort(rng.choice(valid, int(len(real) * 1.15), replace=False))
    st = [(t, s, "rand", c[t] - s * ra * atr[t], atr[t], 5.0)
          for t, s, ra in zip(ts, rng.choice([1, -1], len(ts)), rng.choice(r_atr, len(ts)))]
    tr = backtest_m1(d, m1, cfg, TF_MIN, st)
    res.append(tr.pnl_R.mean()); ns.append(len(tr))
res = np.array(res)
print(f"隨機組平均筆數 {np.mean(ns):.0f}")
print(f"策略 exp = {real.pnl_R.mean():+.3f}R (n={len(real)})")
print(f"隨機進場 300 次：平均 {res.mean():+.3f}R，95% 區間 [{np.percentile(res,2.5):+.3f}, {np.percentile(res,97.5):+.3f}]，"
      f"隨機 >= 策略 的比例 {np.mean(res >= real.pnl_R.mean()):.3f}")
