"""只用 2/2~5/31 選特徵與門檻，6/1~10/8 檢驗。"""
import numpy as np
import pandas as pd
from feature_stability import T, FEATS
from backtest import stats

TRAIN_END = pd.Timestamp("2026-06-01")
tr_, te_ = T[T.entry_time < TRAIN_END], T[T.entry_time >= TRAIN_END]
halves = [tr_[tr_.entry_time < "2026-04-01"], tr_[tr_.entry_time >= "2026-04-01"]]

sel = []
for f in FEATS:
    if f == "hour":
        continue
    q1, q2 = tr_[f].quantile([1 / 3, 2 / 3])
    eff = []
    for h in halves:
        lo, hi = h[h[f] <= q1].pnl_R.mean(), h[h[f] > q2].pnl_R.mean()
        eff.append(hi - lo)
    if np.sign(eff[0]) == np.sign(eff[1]) and min(abs(eff[0]), abs(eff[1])) >= 0.15:
        sel.append((f, q1, q2, int(np.sign(eff[0])), [round(e, 2) for e in eff]))

print("樣本內選出的特徵（特徵, 下三分位, 上三分位, 方向, 兩半效果）")
for s in sel:
    print("  ", s[0], round(s[1], 3), round(s[2], 3), s[3], s[4])


def score(D):
    sc = np.zeros(len(D))
    for f, q1, q2, sg, _ in sel:
        good = D[f] > q2 if sg > 0 else D[f] <= q1
        bad = D[f] <= q1 if sg > 0 else D[f] > q2
        sc += good.to_numpy().astype(int) - bad.to_numpy().astype(int)
    return sc


for name, D in (("樣本內 2~5 月", tr_), ("樣本外 6~10 月", te_)):
    sc = score(D)
    print(f"\n{name}：依分數分組的平均 R（候選可重疊）")
    print(D.groupby(sc).pnl_R.agg(n="size", 平均R="mean").round(3).T.to_string())

# 不重疊的實際交易：分數 >= k
def nonoverlap(D, k):
    D = D[score(D) >= k].sort_values("entry_time")
    keep, busy = [], pd.Timestamp.min
    for r in D.itertuples():
        if r.entry_time > busy:
            keep.append(r.Index); busy = r.exit_time
    return D.loc[keep]

for k in (1, 2, 3):
    for name, D in (("IS", tr_), ("OOS", te_)):
        s = stats(nonoverlap(D, k))
        print(f"score>={k} {name}: n={s['n']} 勝率={s['win']:.2f} 期望={s['expR']:+.3f}R t={s['t']:+.2f} PF={s['pf']:.2f} 累計={s['totR']:+.1f}R DD={s['maxddR']:.1f}R")
