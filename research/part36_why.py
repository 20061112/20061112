"""第三十六部分：為什麼候選策略的利潤集中在 2025~2026 —— 行情特徵、依當天型態分組、方向預測力、IB 寬度 / 成本。輸出 results_part36.txt"""
import numpy as np, pandas as pd
import part28_ib_grid as P
_, A = P.job((60, 0.1, True, None))
A["a"] = A.BASE / A.atr
# 每日特徵（事後）：當天 01:00~23:50 的 |收−開| / ATR、|收−開| / 高低區間；點差 / ATR
f = []
for k, atr, mins, o, sp in P.DAYS:
    m = mins < P.FLAT
    oo, hh, ll, cc = o[m, 0][0], o[m, 1].max(), o[m, 2].min(), o[m, 3][-1]
    f.append(dict(day=k, move=abs(cc - oo) / atr, eff=abs(cc - oo) / (hh - ll), cost=np.median(sp[m]) / atr, atr=atr, close=cc))
F = pd.DataFrame(f); F["year"] = F.day.dt.year
X = A.merge(F[["day", "move", "eff", "cost"]], on="day")
X["year"] = X.day.dt.year
print("== 各年行情特徵（每天）")
g = F.groupby("year")
print(pd.DataFrame(dict(ATR中位=g.atr.median(), 收開除ATR=g.move.mean(), 趨勢日比例_move_gt1=g.move.apply(lambda s: (s > 1).mean()),
      效率=g.eff.mean(), 點差除ATR_千分=g.cost.median() * 1000, 年漲跌pct=g.close.apply(lambda s: s.iloc[-1] / s.iloc[0] - 1) * 100)).round(3))
print("\n== 策略 A 每天 ATR 損益：依當天『收−開 / ATR』分組 × 年")
X["bin"] = pd.cut(X.move, [0, .3, .6, 1, 99], labels=["<0.3", "0.3~0.6", "0.6~1", ">1"])
print(X.pivot_table(index="bin", columns="year", values="a", aggfunc="mean", observed=False).round(3))
print(X.pivot_table(index="bin", columns="year", values="a", aggfunc="count", observed=False))
print("\n== 用 2023~24 的分組平均 × 各年的分組比例 → 預期 vs 實際")
base = X[X.year < 2025].groupby("bin", observed=False).a.mean()
for y, x in X.groupby("year"):
    w = x.bin.value_counts(normalize=True).reindex(base.index)
    print(y, "預期", round((w * base).sum(), 4), "實際", round(x.a.mean(), 4))
print("\n== 突破方向對當天剩餘走勢的預測力（不設停損：進場 → 23:50）")
r = []
for k, atr, mins, o, sp in P.DAYS:
    s0 = 60; ib = (mins >= s0) & (mins < s0 + 60)
    if ib.sum() < 48: continue
    ibh, ibl = o[ib, 1].max(), o[ib, 2].min()
    after = np.where((mins >= s0 + 60) & (mins < P.FLAT))[0]; c = o[after, 3]
    up, dn = np.where(c > ibh + .05 * atr)[0], np.where(c < ibl - .05 * atr)[0]
    if not len(up) and not len(dn): continue
    side, i = (1, after[up[0]]) if len(up) and (not len(dn) or up[0] < dn[0]) else (-1, after[dn[0]])
    end = after[-1]
    r.append(dict(day=k, year=k.year, hit=np.sign(o[end, 3] - o[i + 1, 0]) == side, mv=side * (o[end, 3] - o[i + 1, 0]) / atr,
                  hour=mins[i] / 60))
R = pd.DataFrame(r)
print(R.groupby("year").agg(順向收盤比例=("hit", "mean"), 平均順向移動ATR=("mv", "mean"), 進場時間中位=("hour", "median")).round(3))
# 自相關：前 1 小時 vs 之後的日內報酬，逐年（全部日子）
print("\n== 日內延續性：02:00~10:00 報酬 vs 10:00~23:50 報酬 的相關（逐年）")
q = []
for k, atr, mins, o, sp in P.DAYS:
    def px(t):
        j = np.searchsorted(mins, t); return o[min(j, len(o) - 1), 0]
    q.append(dict(year=k.year, a=(px(600) - px(120)) / atr, b=(px(1430) - px(600)) / atr, c=(px(120) - px(60)) / atr, d=(px(1430) - px(120)) / atr))
Q = pd.DataFrame(q)
print(Q.groupby("year").apply(lambda x: pd.Series(dict(亞洲首小時_vs_其後=x.c.corr(x.d), 早盤_vs_午後=x.a.corr(x.b)))).round(3))
print("\n== 亞洲 IB 寬度、停損距離、勝率（逐年）")
r = []
for k, atr, mins, o, sp in P.DAYS:
    ib = (mins >= 60) & (mins < 120); m = mins < P.FLAT
    if ib.sum() < 48: continue
    r.append(dict(year=k.year, IB寬除ATR=(o[ib, 1].max() - o[ib, 2].min()) / atr, 日區間除ATR=(o[m, 1].max() - o[m, 2].min()) / atr))
print(pd.DataFrame(r).groupby("year").median().round(3))
A["year"] = A.day.dt.year
print(A.groupby("year").agg(停損距離除ATR中位=("R", lambda s: np.median(s / A.loc[s.index, "atr"])),
                            每筆R=("BASE", lambda s: (s / A.loc[s.index, "R"]).mean()), 勝率=("BASE", lambda s: (s > 0).mean())).round(3))
