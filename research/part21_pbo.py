"""第二十一部分 A：回測過擬合機率（PBO，Bailey et al. 的 CSCV 方法）。

用第十七部分掃描的 144 組設定（滾動前推門檻），取每組的「日損益」矩陣（3/3~10/7）。
把交易日依時間切成 S = 10 段，所有「挑 5 段當樣本內、另 5 段當樣本外」的組合（252 種）：
  在樣本內挑 Sharpe 最高的設定 → 看它在樣本外的排名（相對位置 ω）；
  PBO = 樣本外排名落在中位數以下（logit ≤ 0）的比例。
另外報告：樣本內最佳在樣本外的平均 Sharpe、樣本外 Sharpe 為負的比例、版本 A 本身在各組合中的排名分佈。
"""
import itertools
import numpy as np
import pandas as pd
from part14_deep import DAYS
from part17_balance import gen, wf

configs = []
for (confirm, buf), hours, step, (qt, qe), monday in itertools.product(
        [("close5", 0.05), ("close5", 0.02), ("touch", 0.0)], [(2, 10), (1, 12), (2, 14)], [60, 30],
        [(0.8, 0.8), (0.9, 0.9), (0.8, 1.0), (1.0, 1.0)], [False, True]):
    configs.append((confirm, buf, hours, step, qt, qe, monday))

cols = {}
for c in configs:
    confirm, buf, hours, step, qt, qe, monday = c
    T = wf(gen(confirm, buf, hours, step, monday), qt, qe)
    cols[c] = T.groupby("day").pnl.sum()
days = [d for d in DAYS if pd.Timestamp("2026-03-01") <= d <= pd.Timestamp("2026-10-07")]
M = pd.DataFrame(cols).reindex(days).fillna(0.0)
M.columns = [f"{c[0]}+{c[1]}|{c[2][0]:02d}-{c[2][1]:02d}|{c[3]}|t{c[4]}/e{c[5]}|mon{int(c[6])}" for c in configs]
M.to_csv("part21_daily_matrix.csv")
A = "close5+0.05|02-10|30|t0.9/e0.9|mon0"

S = 10
blocks = np.array_split(np.arange(len(M)), S)
sh = lambda X: X.mean() / X.std().replace(0, np.nan) * np.sqrt(250)
logits, oos_best, rankA = [], [], []
for ins in itertools.combinations(range(S), S // 2):
    i_idx = np.concatenate([blocks[k] for k in ins]); o_idx = np.concatenate([blocks[k] for k in range(S) if k not in ins])
    s_in, s_out = sh(M.iloc[i_idx]), sh(M.iloc[o_idx])
    best = s_in.idxmax()
    w = (s_out.rank(pct=True)[best])
    w = min(max(w, 1e-6), 1 - 1e-6)
    logits.append(np.log(w / (1 - w))); oos_best.append(s_out[best]); rankA.append(s_out.rank(pct=True)[A])
logits, oos_best, rankA = map(np.array, (logits, oos_best, rankA))
with open("results_part21.txt", "w") as f:
    def P(s=""): print(s); f.write(s + "\n")
    P("第二十一部分 A：回測過擬合機率（CSCV，144 組設定 × 日損益，10 段、252 種切法）")
    P(f"  PBO（樣本內最佳在樣本外落到後半段的機率）= {(logits <= 0).mean() * 100:.1f}%")
    P(f"  樣本內最佳 → 樣本外 Sharpe：平均 {oos_best.mean():.2f}，中位數 {np.median(oos_best):.2f}，為負的比例 {(oos_best < 0).mean() * 100:.1f}%")
    P(f"  全部 144 組全期 Sharpe：中位數 {sh(M).median():.2f}，最低 {sh(M).min():.2f}，為負的組數 {(sh(M) < 0).sum()}")
    P(f"  版本 A 在各切法樣本外的百分位排名：平均 {rankA.mean() * 100:.0f}%，落在前半的比例 {(rankA > .5).mean() * 100:.0f}%")
