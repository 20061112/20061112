"""第二十四部分 A：為什麼第二十三部分的空單 2023~2025 虧、2026 賺？

檢查：
  1. 空單結果 vs 進場當時的大週期趨勢（日線收盤 vs 日線 SMA50、20 日報酬）
  2. 空單 vs 同年度隨機空單基準（扣掉 drift 後是否還有差）
  3. 下軌突破後的「延續幅度」與「反抽」是否多空不對稱（逢低買盤）
"""
import numpy as np
import pandas as pd

from load import load
from part23_bb_breakout import Data, simulate, random_baseline, HOLD

m1 = load("data/XAUUSD_M1_2023_2026.csv")
d = Data(m1, "H1", n=20)
tr = simulate(d, 2, 2, "trail_mid", HOLD["H1"])
base = random_baseline(d, tr, 2, "trail_mid", HOLD["H1"], reps=20)

daily = m1.close.resample("1D").last().dropna()
sma50 = daily.rolling(50).mean().shift(1)          # 用前一天以前的資料
ret20 = (daily / daily.shift(20) - 1).shift(1)
day = tr.time.dt.normalize()
tr["above50"] = (daily.shift(1).reindex(day).to_numpy() > sma50.reindex(day).to_numpy())
tr["ret20"] = ret20.reindex(day).to_numpy()
tr["year"] = tr.time.dt.year
base["year"] = base.time.dt.year
bday = base.time.dt.normalize()
base["above50"] = (daily.shift(1).reindex(bday).to_numpy() > sma50.reindex(bday).to_numpy())

pd.set_option("display.width", 200)
print("1) 各年度：金價年報酬、日線在 SMA50 上方的比例")
y = daily.groupby(daily.index.year)
print(pd.DataFrame({"年報酬": y.last() / y.first() - 1,
                    "SMA50上方比例": (daily > sma50).groupby(daily.index.year).mean()}).round(3))

print("\n2) 空單 / 多單：訊號 vs 隨機基準（平均 σ）")
rows = []
for yr in sorted(tr.year.unique()):
    for s, name in ((-1, "空"), (1, "多")):
        a = tr[(tr.year == yr) & (tr.side == s)].pnl_sd
        b = base[(base.year == yr) & (base.side == s)].pnl_sd
        rows.append(dict(year=yr, side=name, n=len(a), signal=a.mean(), random=b.mean(), edge=a.mean() - b.mean(),
                         edge_t=(a.mean() - b.mean()) / np.sqrt(a.var() / len(a) + b.var() / len(b))))
print(pd.DataFrame(rows).round(3).to_string(index=False))

print("\n3) 依大週期趨勢分組（日線收盤在 SMA50 上 / 下）")
rows = []
for s, name in ((-1, "空"), (1, "多")):
    for flag, lab in ((True, "SMA50上方"), (False, "SMA50下方")):
        a = tr[(tr.side == s) & (tr.above50 == flag)].pnl_sd
        b = base[(base.side == s) & (base.above50 == flag)].pnl_sd
        rows.append(dict(side=name, regime=lab, n=len(a), signal=a.mean(), random=b.mean(), edge=a.mean() - b.mean(),
                         win=(a > 0).mean()))
print(pd.DataFrame(rows).round(3).to_string(index=False))

print("\n4) 依進場前 20 日報酬分五組（空單）")
sh = tr[tr.side == -1].dropna(subset=["ret20"]).copy()
sh["q"] = pd.qcut(sh.ret20, 5, labels=False)
print(sh.groupby("q").agg(ret20=("ret20", "mean"), n=("pnl_sd", "size"), avg=("pnl_sd", "mean"),
                          win=("pnl_sd", lambda x: (x > 0).mean())).round(3))

print("\n5) 碰帶後 24 根 H1 的最大順向 / 最大逆向（以進場 σ 為單位，全部碰帶事件，不考慮持倉）")
o, h, l, c = d.o, d.h, d.l, d.c
mid, sd = d.mid, d.sd
ev = []
for t in range(21, len(c) - 25):
    if not sd[t] > 0:
        continue
    for side, lvl in ((1, mid[t] + 2 * sd[t]), (-1, mid[t] - 2 * sd[t])):
        if (h[t] >= lvl) if side == 1 else (l[t] <= lvl):
            fh, fl = h[t + 1:t + 25].max(), l[t + 1:t + 25].min()
            mfe = (fh - lvl) if side == 1 else (lvl - fl)
            mae = (lvl - fl) if side == 1 else (fh - lvl)
            ev.append((d.time[t].year, side, mfe / sd[t], mae / sd[t], side * (c[t + 24] - lvl) / sd[t]))
ev = pd.DataFrame(ev, columns=["year", "side", "mfe", "mae", "ret24"])
print(ev.groupby(["year", "side"]).agg(n=("mfe", "size"), mfe=("mfe", "median"), mae=("mae", "median"),
                                       ret24=("ret24", "mean")).round(2))
