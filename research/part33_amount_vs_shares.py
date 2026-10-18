"""第三十三部分：定額 vs 定股，哪個年化（IRR）比較高？

定額：每週投入固定金額。定股：每週買固定股數 → 投入金額 ∝ 當週價格（以起點價格標準化，IRR 不受規模影響）。
不考慮成本與資金負擔，只比 IRR（考慮現金流時點的年化報酬）。
對象：純定投（QQQ / SPY / DIA / SOXX）與策略（全期貨 L=3，策略內的布林加碼照常）。
期間：全期（QQQ 1999 起，其餘 2006/3 起）、10 年滾動、5 年滾動（每年起一段）。
"""
import numpy as np
import pandas as pd
from part22_core import load, signals, run, metrics, year_starts

out = open("results_part33.txt", "w")


def P(s=""):
    print(s); out.write(s + "\n"); out.flush()


BOOKS = [("QQQ", "MNQ", 41.0, None), ("SPY", "MES", 10.0, "2006-03-01"), ("DIA", "MYM", 100.0, "2006-03-01"),
         ("SOXX", "MNQ", 41.0, "2006-03-01")]
P("第三十三部分：定額 vs 定股（年化 IRR）\n")
P("  差 = 定額 IRR − 定股 IRR（正 = 定額較好）；平均成本 / 平均價格：定額的平均買入成本相對期間平均價格\n")
for name, fut, ratio, start in BOOKS:
    S = signals(load(f"data/{name.lower()}_us_d.csv"))
    i0 = 0 if start is None else S.index.searchsorted(pd.Timestamp(start))
    Pc = S.close.values
    y0 = S.index[i0].year + 1
    YS = year_starts(S, y0, 2021)
    P(f"{name}（{S.index[i0].date()} 起）")
    for lab, kw in (("純定投", dict(eng="dca")),
                    ("策略（期貨 3 倍）", dict(eng="allfut", L=3.0, fut_name=fut, ratio=ratio))):
        def irr(a, b, shares):
            mult = Pc / Pc[a] if shares else None
            return metrics(run(S, i0=a, i1=b, amt_mult=mult, **kw))["irr"]
        fa, fs = irr(i0, len(S), False), irr(i0, len(S), True)
        line = f"  {lab:14s} 全期：定額 {fa * 100:5.1f}%｜定股 {fs * 100:5.1f}%｜差 {(fa - fs) * 100:+.1f}%"
        for n in (10, 5):
            ys = [y for y in range(y0, 2027 - n) if y + n in YS]
            d = np.array([irr(YS[y], YS[y + n], False) - irr(YS[y], YS[y + n], True) for y in ys])
            line += f"｜{n} 年滾動 {len(d)} 段：定額勝 {(d > 0).sum()}/{len(d)}、差中位 {np.median(d) * 100:+.1f}%"
        P(line)
    # 平均成本 vs 平均價格（純定投，週投入日）
    wk = S.ws.values.copy(); wk[:i0] = False
    px = Pc[wk]
    avg_cost = len(px) / (1 / px).sum()
    P(f"  定額平均成本 {avg_cost:.2f} vs 平均價格（= 定股平均成本）{px.mean():.2f} → 低 {(1 - avg_cost / px.mean()) * 100:.0f}%\n")
out.close()
