"""把多個時段的 IB 突破合成一個組合，看交易次數與日報酬（ATR）。"""
import numpy as np, pandas as pd
from part10_freq import trades, SPLIT
import part10_freq  # noqa  (掃描結果會一起印出)

def leg(h, win, rev, zmin=None, sign=1):
    E = trades(h, win, rev)
    m = E.z.notna() & ((E.z > zmin) if zmin is not None else True)
    E = E[m]
    return (sign * E.ret).rename(f"{h}h"), E.n.fillna(0)

CFG = {
    "A 03h（全部日子, 反手）": [(3, 3, True, None, 1)],
    "B A + 10h z>0（反手）": [(3, 3, True, None, 1), (10, 3, True, 0, 1)],
    "C B + 05h（全部日子）": [(3, 3, True, None, 1), (10, 3, True, 0, 1), (5, 3, False, None, 1)],
    "D 01~10h 每個整點都做（無反手）": [(h, 3, False, None, 1) for h in range(1, 11)],
}
print("\n組合：每天加總的報酬（ATR），t 用日報酬算")
for name, legs in CFG.items():
    rets, ns = zip(*[leg(*l) for l in legs])
    R = pd.concat(rets, axis=1, sort=True).fillna(0); N = pd.concat(ns, axis=1, sort=True).fillna(0)
    day = R.sum(axis=1)
    t = day.mean() / (day.std(ddof=1) / np.sqrt(len(day)))
    h1, h2 = day[day.index < SPLIT], day[day.index >= SPLIT]
    cum = day.cumsum(); dd = (cum - cum.cummax()).min()
    per = R.values[R.values != 0]
    print(f"  {name:28s} 交易 {int(N.values.sum()):4d} 筆 / {len(day)} 天（{N.values.sum() / len(day):.1f} 筆/天）"
          f" 每筆 {per.mean():+.3f} ATR  日均 {day.mean():+.3f}  t={t:+.2f}  H1 {h1.mean():+.3f} H2 {h2.mean():+.3f}"
          f"  累積 {cum.iloc[-1]:+.1f} ATR  最大回撤 {dd:.1f} ATR  正報酬天 {(day > 0).mean() * 100:.0f}%")
