"""第二十一部分 D：精簡 + ER。ER 門檻也用滾動前推（每月用過去的訊號算 80% 分位，刪掉 ER 最高的 20%）。"""
import numpy as np, pandas as pd
from part17_balance import gen, wf, score
from part21_er import er_feats, show

out = open("results_part21.txt", "a")
def P(s=""): print(s); out.write(s + "\n"); out.flush()

X = gen("close5", 0.05, (2, 10), 30, False)
E = pd.DataFrame([er_feats(r) for r in X.itertuples()], index=X.index)
X = pd.concat([X, E[["er_day"]]], axis=1)


def wf_er(X, qt, qe, q_er=0.8, q_ib=0.4):
    X = X.copy(); X["ym"] = X.day.dt.to_period("M"); parts = []
    for m in sorted(X.ym.unique())[2:]:
        past, cur = X[X.ym < m], X[X.ym == m]
        k = ~(cur.aligned & (cur.ib_atr < past.ib_atr.quantile(q_ib))) & (cur["tail"] <= past["tail"].quantile(qt)) \
            & (cur.ext_letter.fillna(0) <= past.ext_letter.quantile(qe)) & (cur.er_day.fillna(0) <= past.er_day.quantile(q_er))
        parts.append(cur[k])
    return pd.concat(parts).sort_values(["t_in", "h"]).drop_duplicates(["t_in", "side"])


P("\n" + "=" * 120 + "\n第二十一部分 D：精簡版與 ER（全部門檻滾動前推）")
for lab, Y in [("版本 A（tail + ext）", wf(X, 0.9, 0.9)), ("A + ER", wf_er(X, 0.9, 0.9)),
               ("精簡：拿掉 ext（tail）", wf(X, 0.9, 1.0)), ("精簡 + ER（tail + ER）", wf_er(X, 0.9, 1.0)),
               ("只用 ER（不用 tail、ext）", wf_er(X, 1.0, 1.0))]:
    s = score(Y)
    P("  " + show(Y, lab) + f" Sharpe {s['sharpe']:.2f} 總/DD {s['ret_dd']:.1f} DD {s['dd']:.0f} 去前10天 {s['ex10']:+.0f} {s['per_day']:.1f}筆/交易日")
out.close()
