"""第二十八部分：類馬丁 —— 連虧達 N 筆之後才開始放大部位（每再虧一筆 × m，上限 cap 倍；贏一筆就回到 1 倍）。
用 CUSUM H1 定案版（不加停利）的逐筆 R。輸出 results_part28.txt
"""
import math, warnings, itertools
import numpy as np, pandas as pd
warnings.simplefilter("ignore", RuntimeWarning)
from load import load
from cusum_lab import bars, zscore, backtest, m1_backtest, M1_PATH
from changepoint import cusum_signals

out = open("results_part28.txt", "w")
def P(s=""):
    print(s); out.write(s + "\n"); out.flush()

hb = bars("H1"); sig = cusum_signals(zscore(hb), 0.25, 6)
T = backtest(hb, sig, 2.0)
TM = m1_backtest(hb, sig, load(M1_PATH), 2.0)
r = T.pnl_R.to_numpy()

P("0) 前面連虧幾筆之後，下一筆的表現（H1 模擬 173 筆）——馬丁要有用，連虧後勝率必須變高")
run = 0; rows = []
for x in r:
    rows.append((run, x)); run = run + 1 if x < 0 else 0
D = pd.DataFrame(rows, columns=["prev_losses", "r"])
for k in range(0, 9):
    g = D[D.prev_losses >= k]
    P(f"  前面已連虧 ≥ {k} 筆：{len(g):3d} 筆，勝率 {(g.r > 0).mean():.0%}，平均 {g.r.mean():+.2f}R")

def sizes(rr, N, m, cap):
    s = np.empty(len(rr)); run = 0
    for i, x in enumerate(rr):
        s[i] = min(cap, m ** max(0, run - N + 1)) if run >= N else 1.0
        run = run + 1 if x < 0 else 0
    return s

def stats(rr, s):
    p = rr * s; eq = np.cumsum(p); dd = (np.maximum.accumulate(np.r_[0, eq])[1:] - eq).max()
    return p.sum(), dd, p.sum() / s.sum(), s.max()

rng = np.random.default_rng(0)
BOOT = [rng.choice(r, 138) for _ in range(5000)]                     # 3 年（138 筆）
def boot(N, m, cap):
    dds, tots = [], []
    for x in BOOT:
        s = sizes(x, N, m, cap); p = x * s; e = np.cumsum(p)
        dds.append((np.maximum.accumulate(np.r_[0, e])[1:] - e).max()); tots.append(p.sum())
    dds, tots = np.array(dds), np.array(tots)
    return np.median(dds), np.percentile(dds, 95), np.percentile(dds, 99), (dds > 30).mean(), np.median(tots), (tots < 0).mean()

P("\n1) 規則比較（實際順序 H1 2023/1~2026/10 + 自助法 3 年 5,000 次）")
P(f"  {'規則':28s} {'累計R':>7s} {'回撤':>6s} {'總R/回撤':>7s} {'每單位風險':>8s} {'最大倍數':>6s} | {'自助:回撤中位':>10s} {'95%':>6s} {'99%':>6s} {'回撤>30R機率':>10s} {'3年R中位':>8s} {'3年虧損機率':>9s}")
cfgs = [("基準（固定 1R）", 99, 1, 1)]
for N, m, cap in itertools.product((2, 3, 4, 5), (1.5, 2.0), (2, 4, 8)):
    if m == 1.5 and cap == 8: continue
    cfgs.append((f"連虧{N}筆後 ×{m} 上限{cap}倍", N, m, cap))
cfgs += [("（對照）每筆固定 1.5R", 99, 1, 1)]
res = []
for lab, N, m, cap in cfgs:
    s = sizes(r, N, m, cap)
    if lab.startswith("（對照）"):
        s = s * 1.5
    tot, dd, per, mx = stats(r, s)
    if lab.startswith("（對照）"):
        b = boot(99, 1, 1); b = (b[0] * 1.5, b[1] * 1.5, b[2] * 1.5, np.nan, b[4] * 1.5, b[5])
    else:
        b = boot(N, m, cap)
    res.append((lab, tot, dd, tot / dd, b))
    P(f"  {lab:28s} {tot:+7.1f} {dd:6.1f} {tot / dd:7.1f} {per:+8.3f} {mx:6.1f} | {b[0]:10.1f} {b[1]:6.1f} {b[2]:6.1f} {b[3]:10.0%} {b[4]:+8.1f} {b[5]:9.0%}")

P("\n2) 美元（1 單位 = 1 盎司，M1 逐根 2023/6~2026/10）")
p = TM.pnl.to_numpy(); rr = TM.pnl_R.to_numpy()
for lab, N, m, cap in (("基準 1 盎司", 99, 1, 1), ("連虧3筆後 ×2 上限4倍", 3, 2.0, 4), ("連虧4筆後 ×2 上限8倍", 4, 2.0, 8), ("連虧2筆後 ×2 上限8倍", 2, 2.0, 8)):
    s = sizes(rr, N, m, cap); q = p * s; e = np.cumsum(q); dd = (np.maximum.accumulate(np.r_[0, e])[1:] - e).max()
    P(f"  {lab:22s} 合計 {q.sum():+7,.0f} 美元，最大回撤 {dd:6,.0f}，最大單筆虧損 {q.min():+6,.0f}，最大部位 {s.max():.0f} 盎司")
