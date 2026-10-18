"""第二十七部分：版本 A 與複雜度階梯在黃金 2023/6 ~ 2025/12 的樣本外（版本 A 研究只用過 2026/1 以後的資料）。

回答「是不是參數沒有等比例放大」：同一個商品、同一個時段，只換年份。
  XAUUSD_2023    原始價格（TPO 格子 1 美元，相對 ATR 比 2026 粗 2~5 倍）
  XAUUSD_2023X3  價格 ×3（格子 / ATR 接近 2026 年）
L0~L3 只用 ATR 倍數，本來就不受價格比例影響；只有 L4、L5（tail、ext_letter）會受格子大小影響。
用法：python3 part27_gold_2023.py XAUUSD_2023 ; python3 part27_gold_2023.py XAUUSD_2023X3 → results_part27.txt（附加）
"""
import os
import sys
SYM = sys.argv[1] if len(sys.argv) > 1 else "XAUUSD_2023"
os.environ["SYMBOL"] = SYM
import numpy as np
import pandas as pd
from tpo import INSTR
from part14_deep import DAYS
from part17_balance import gen, wf, score

K = INSTR[SYM][1]
out = open("results_part27.txt", "a")
def P(s=""): print(s); out.write(s + "\n"); out.flush()


def pf(p):
    return p[p > 0].sum() / max(-p[p <= 0].sum(), 1e-12)


def line(Y, lab):
    r = Y.R; y = Y.day.dt.year; s = score(Y)
    yrs = "  ".join(f"{k}: {v.mean():+.3f}R PF {pf(Y.pnl[y == k]):.2f} ({len(v)})" for k, v in Y.R.groupby(y))
    return (f"  {lab:34s} {len(Y):5d}筆 {s['per_day']:.1f}/日 每筆 {r.mean():+.3f}R t={r.mean() / r.std() * np.sqrt(len(r)):+.2f} "
            f"PF {pf(Y.pnl):.2f} Sharpe {s['sharpe']:+.2f} | {yrs}")


P(f"\n## {SYM}（價格 × {1 / K:g}）{DAYS[0].date()} ~ {DAYS[-1].date()}，{len(DAYS)} 個交易日；黃金 2026 研究期：L0 PF 1.37~1.39 → A PF 1.92")
steps = [("L0 觸價、停損另一側、收盤出", ("touch", 0.0, True), (1.0, 1.0, 0.0)),
         ("L1 + M5 收盤 + 0.05 ATR", ("close5", 0.05, True), (1.0, 1.0, 0.0)),
         ("L2 + 不做星期一", ("close5", 0.05, False), (1.0, 1.0, 0.0)),
         ("L3 + 順前日方向的小 IB 不做", ("close5", 0.05, False), (1.0, 1.0, 0.4)),
         ("L4 + 反向尾巴", ("close5", 0.05, False), (0.9, 1.0, 0.4)),
         ("L5 + 反向極值字母（= 版本 A）", ("close5", 0.05, False), (0.9, 0.9, 0.4))]
for lab, (cf, bf, mon), (qt, qe, qi) in steps:
    P(line(wf(gen(cf, bf, (2, 10), 30, mon), qt, qe, qi), lab))

# 核心 L1 的穩健性：靠少數大單？多空？星期一？
Y = wf(gen("close5", 0.05, (2, 10), 30, True), 1.0, 1.0, 0.0)
p = Y.pnl.sort_values(ascending=False); r = Y.R.sort_values(ascending=False)
m = Y.groupby(Y.day.dt.to_period("M")).pnl.sum()
P(f"  [L1 穩健性] PF {pf(Y.pnl):.2f} → 拿掉最好 1% 的單 PF {pf(p.iloc[len(p) // 100:]):.2f}；總 R {r.sum():+.0f} → 去前 20 筆 {r.iloc[20:].sum():+.0f}、去前 50 筆 {r.iloc[50:].sum():+.0f}；"
  f"正月 {(m > 0).sum()}/{len(m)}；多 PF {pf(Y.pnl[Y.side == 1]):.2f} / 空 PF {pf(Y.pnl[Y.side == -1]):.2f}")
mon = Y.day.dt.dayofweek == 0
P("  [星期一] " + "  ".join(f"{k}: 週一 PF {pf(Y.pnl[(Y.day.dt.year == k) & mon]):.2f} / 其他 PF {pf(Y.pnl[(Y.day.dt.year == k) & ~mon]):.2f}"
                          for k in sorted(Y.day.dt.year.unique())))
out.close()
