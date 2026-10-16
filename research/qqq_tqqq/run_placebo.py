"""雜訊基準：星期幾、訊號延遲這類「不該有差」的改動，結果會差多少？
用來判斷其他參數的差異是不是超出雜訊。輸出附加到 results_params.txt。"""
import numpy as np
import pandas as pd
import engine as E
from run_tests import q, OK0
from run_params import ev, weekday_mask, broll, b_full

out = []
P = lambda s: (print(s, flush=True), out.append(s))
P("\n雜訊基準（安慰劑）：每週檢查日 × 訊號延遲，共 5 × 4 = 20 組")
rows = []
for d, nm in enumerate(["一", "二", "三", "四", "五"]):
    m = weekday_mask(d)
    for lag in [0, 1, 2, 3]:
        ok = OK0.shift(lag - 1, fill_value=False)
        w = E.weights_from_rules(q, ok, ~ok, entry_mask=m)
        f, s10, roll = ev(w, dca_mask=m)
        dd = roll.irr.values - broll.irr.values
        rows.append(dict(day=nm, lag=lag, irr99=f["irr"], d10=np.median(dd), win10=int((dd > 0).sum()), worst=f["worst"]))
R = pd.DataFrame(rows)
P("1999 起 IRR（列 = 星期，欄 = 延遲天數；原版 = 星期一、延遲 1）")
P(R.pivot(index="day", columns="lag", values="irr99").reindex(list("一二三四五")).to_string(float_format=lambda x: f"{x:.3f}"))
P("10 年滾動 IRR 差中位（相對原版）")
P(R.pivot(index="day", columns="lag", values="d10").reindex(list("一二三四五")).to_string(float_format=lambda x: f"{x:+.3f}"))
P(f"20 組：1999 起 IRR {R.irr99.min():.3f} ~ {R.irr99.max():.3f}（標準差 {R.irr99.std():.3f}）；10 年 IRR 差 {R.d10.min():+.3f} ~ {R.d10.max():+.3f}；"
  f"10 年勝出段數 {R.win10.min()} ~ {R.win10.max()} / 17；最慘 {R.worst.min():.2f} ~ {R.worst.max():.2f}")
open("results_params.txt", "a").write("\n".join(out) + "\n")
