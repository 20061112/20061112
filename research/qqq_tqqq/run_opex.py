"""網路說法檢驗：「OPEX（每月第三個星期五）當天跌超過 1%，之後 1 天、5 天會反彈。」

事件：月選擇權結算日（第三個星期五；遇休市改前一個交易日）收盤報酬 ≤ −1%。
比較：
  1. 所有交易日之後 1 / 5 天的平均報酬（無條件）
  2. 非 OPEX 的 −1% 大跌日之後 1 / 5 天（一般大跌後的反彈）
  3. 所有 OPEX 日之後 1 / 5 天
報 t 值（事件互不重疊：每月最多一次）與勝率。資料：SPY（2005 起）、QQQ（1999 起）。
輸出 results_opex.txt。
"""
import numpy as np
import pandas as pd
import engine as E

out = []
P = lambda s: (print(s, flush=True), out.append(s))


def opex_days(idx):
    s = pd.Series(idx, index=idx)
    res = []
    for (y, m), g in s.groupby([idx.year, idx.month]):
        fridays = pd.date_range(f"{y}-{m:02d}-01", periods=31, freq="D")
        fridays = fridays[(fridays.month == m) & (fridays.dayofweek == 4)]
        third = fridays[2]
        cand = g[g <= third]
        if len(cand):
            res.append(cand.iloc[-1])
    return pd.DatetimeIndex(res)


def stats(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3:
        return f"n={len(x)}"
    t = x.mean() / x.std(ddof=1) * np.sqrt(len(x))
    return f"n={len(x):4d} 平均 {x.mean():+.2%} 中位 {np.median(x):+.2%} 上漲比例 {(x > 0).mean():.0%} t={t:+.2f}"


for t in ["spy", "qqq"]:
    d = E._read(f"{t}_us_d.csv")
    c = d.close; r = c.pct_change()
    f1 = c.shift(-1) / c - 1; f5 = c.shift(-5) / c - 1
    ox = opex_days(d.index)
    is_ox = d.index.isin(ox)
    big = r <= -0.01
    P(f"\n=== {t.upper()}（{d.index[0].date()} ~ {d.index[-1].date()}）")
    for lab, m in [("所有交易日", np.ones(len(d), bool)), ("所有 OPEX 日", is_ox),
                   ("非 OPEX 的 −1% 大跌日", big & ~is_ox), ("OPEX 且 −1% 大跌（網路說法）", big & is_ox)]:
        P(f"  {lab:24s} 之後 1 天：{stats(f1[m])}")
        P(f"  {'':24s} 之後 5 天：{stats(f5[m])}")
    ev = d.index[big & is_ox]
    diff5 = f5[big & is_ox].mean() - f5[big & ~is_ox].mean()
    P(f"  OPEX 大跌 vs 一般大跌，5 天後平均差 {diff5:+.2%}")
    # 分期：2000~2012 / 2013 以後
    for a, b in [("1999", "2012"), ("2013", "2026")]:
        m = big & is_ox & (d.index >= a) & (d.index <= f"{b}-12-31")
        P(f"  {a}~{b} OPEX 大跌 之後 5 天：{stats(f5[m])}")
    P("  事件清單（日期、當天、之後 1 天、之後 5 天）：")
    P("   " + "；".join(f"{x.date()} {r[x]:+.1%}/{f1[x]:+.1%}/{f5[x]:+.1%}" for x in ev[-12:]) + "（最近 12 次）")
open("results_opex.txt", "w").write("\n".join(out) + "\n")
