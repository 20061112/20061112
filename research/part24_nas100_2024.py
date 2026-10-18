"""第二十四部分：第二十三部分的候選在 NAS100 更早年份（2024/10/1 ~ 2025/9/30）的真正樣本外驗證。

這段資料在第二十二、二十三部分都沒有用過；與新檔重疊的 2025/10/1~10/2 已排除（tpo.END）。
事先登記（看結果之前就決定，規則一個都不改）：
  主要：第十二部分 B（單一部位、反手、IB30、停損另一側、1×IB 保本、5 小時窗、02~10 點）
  次要：第十二部分 A、M15 H6 量能高潮、紐約 OR15 假突破反向抱到 21:00
  參考：版本 A（第二十二部分已判定無效）
輸出：results_part24.txt、part24_equity.png
第二十五部分（2023/10 ~ 2024/9）用同一支程式：python3 part24_nas100_2024.py NAS100_2023 → results_part25.txt、part25_equity.png
"""
import os
import sys
SYM = sys.argv[1] if len(sys.argv) > 1 else "NAS100_OLD"
os.environ["SYMBOL"] = SYM
import numpy as np
import pandas as pd
from tpo import load_m1, INSTR
from backtest import resample

K = INSTR[SYM][1]
PART, CUT, PERIOD = {"NAS100_OLD": (24, pd.Timestamp("2025-04-01"), "2024/10 ~ 2025/9"),
                     "NAS100_2023": (25, pd.Timestamp("2024-04-01"), "2023/10 ~ 2024/9")}[SYM]
import part12_strategy as p12
D = p12.DD
ATR = p12.d.groupby("day").apply(lambda g: g.high.max() - g.low.min())      # 只用來顯示
M1 = p12.d
M15 = resample(M1[["open", "high", "low", "close", "tickvol", "spread"]], "15min")

out = open(f"results_part{PART}.txt", "w")
def P(s=""): print(s); out.write(s + "\n"); out.flush()


def pf(x):
    x = pd.Series(x).dropna()
    return x[x > 0].sum() / max(-x[x <= 0].sum(), 1e-12)


def tst(x):
    x = pd.Series(x).dropna()
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))


def months(p, when):
    m = pd.Series(np.asarray(p)).groupby(pd.to_datetime(np.asarray(when)).to_period("M")).sum()
    return m


P(f"第{'二十四' if PART == 24 else '二十五'}部分：NAS100 {PERIOD} 樣本外（{M1.index.min()} ~ {M1.index.max()}，{len(p12.DAYS)} 個交易日）")
P(f"前半 < {CUT.date()} ≤ 後半；日波幅中位數 {ATR.median() * K:.0f} 點；點差中位數 {M1.spread.median() * 0.01 * K:.2f} 點")
P("候選在 2025/10 ~ 2026/10（第二十三部分）的結果：B PF 1.30、t 2.4；A PF 1.28；H6 +0.14R；紐約 OR15 抱到 21:00 +0.23R")

# ---- 第十二部分 A / B ----
from tpo import build_days
DB, _ = build_days(M1)
atr10 = DB.atr10
H210 = tuple(range(2, 11))
CFG = {"B（主要）反手、IB30、停損另一側、1×IB 保本、5h": dict(hours=H210, ib_min=30, win=5, stop="opp", be=1.0, reverse=True),
       "A（次要）不反手、IB60、停損中點、2h": dict(hours=H210, ib_min=60, win=2, stop="mid", be=None, reverse=False)}
curves = {}
for name, cfg in CFG.items():
    T = p12.backtest(**cfg)
    T["atr"] = atr10.reindex(T.day).values
    T = T[np.isfinite(T.atr)]
    a, p = T.pnl / T.atr, T.pnl
    h = T.day < CUT
    eq = (p * K).cumsum()
    m = months(p * K, T.t_in)
    P(f"\n## 第十二部分 {name}")
    P(f"  {len(T)} 筆 {len(T) / T.day.nunique():.1f}/有單日  勝率 {(p > 0).mean() * 100:.0f}%  每筆 {a.mean():+.4f} ATR（t {tst(a):+.2f}）"
      f"  PF {pf(p):.2f}  總 {p.sum() * K:+.0f} 點  最大回撤 {(eq - eq.cummax()).min():.0f} 點  正月 {(m > 0).sum()}/{len(m)}")
    P(f"  前半 PF {pf(p[h]):.2f}（{a[h].mean():+.4f} ATR） | 後半 PF {pf(p[~h]):.2f}（{a[~h].mean():+.4f} ATR）")
    P(f"  多 PF {pf(p[T.side == 1]):.2f} / 空 PF {pf(p[T.side == -1]):.2f}；出場 {T.why.value_counts().to_dict()}")
    mo = T.day.dt.to_period("M"); best = (p.groupby(mo).sum()).idxmax()
    P(f"  拿掉最好的月份（{best}）→ PF {pf(p[mo != best]):.2f}；同期指數 {D.o.loc[T.day.min()] * K:.0f} → {D.c.loc[T.day.max()] * K:.0f}"
      f"（{(D.c.loc[T.day.max()] / D.o.loc[T.day.min()] - 1) * 100:+.1f}%）")
    for extra in (0.5, 1.0):
        q = p - extra / K
        P(f"  成本壓力：每筆再扣 {extra} 點 → PF {pf(q):.2f}，總 {q.sum() * K:+.0f} 點")
    rng = np.random.default_rng(0)
    daily = p.groupby(T.day).sum()
    bs = [pf(daily.values[rng.integers(0, len(daily), len(daily))]) for _ in range(2000)]
    P(f"  以天重抽 PF 5% ~ 95%：{np.quantile(bs, .05):.2f} ~ {np.quantile(bs, .95):.2f}，PF < 1 的機率 {(np.array(bs) < 1).mean() * 100:.0f}%")
    P("  逐月（點）：" + "  ".join(f"{k.strftime('%y/%m')} {v:+.0f}" for k, v in m.items()))
    curves[name[0]] = (T.sort_values("t_out").t_out, (T.sort_values("t_out").pnl * K).cumsum())

# ---- H6 量能高潮（第三部分，M15，非重疊）----
from m15_features import candidate_table as m15_table
T3 = m15_table(M15)
X = T3[T3.tv_e > 1.5].sort_values("entry_time")
keep, busy = [], pd.Timestamp.min
for r in X.itertuples():
    if r.entry_time > busy:
        keep.append(r.Index); busy = r.exit_time
X = X.loc[keep]; r = X.pnl_R; h = X.entry_time < CUT
P(f"\n## H6 量能高潮（M15，次要）\n  {len(r)} 筆  每筆 {r.mean():+.3f}R  t {tst(r):+.2f}  勝率 {(r > 0).mean() * 100:.0f}%  PF {pf(r):.2f}"
  f" | 前半 {r[h].mean():+.3f}R 後半 {r[~h].mean():+.3f}R")
base = T3.sort_values("entry_time")
P(f"  對照：全部候選 {base.pnl_R.mean():+.3f}R（{len(base)} 筆，可重疊）")

# ---- 紐約 OR15 假突破反向、抱到 21:00（第九部分）----
from orb_fade import fade
m15s = M15.copy(); m15s["sp"] = m15s.spread * 0.01
F = fade(m15s, 15, (15, 30), 15, mode="touch", target=False)
r = F.pnl_R; h = F.day < CUT
P(f"\n## 紐約 OR15 假突破反向、抱到 21:00（次要）\n  {len(r)} 筆  每筆 {r.mean():+.3f}R  t {tst(r):+.2f}  勝率 {(r > 0).mean() * 100:.0f}%  PF {pf(r):.2f}"
  f" | 前半 {r[h].mean():+.3f}R 後半 {r[~h].mean():+.3f}R")

# ---- 參考：版本 A ----
from part17_balance import gen, wf
A = wf(gen("close5", 0.05, (2, 10), 30, False), 0.9, 0.9)
h = A.day < CUT
P(f"\n## 參考：版本 A（滾動前推）\n  {len(A)} 筆  每筆 {A.R.mean():+.3f}R  t {tst(A.R):+.2f}  PF(點) {pf(A.pnl):.2f}"
  f" | 前半 {A.R[h].mean():+.3f}R 後半 {A.R[~h].mean():+.3f}R")
out.close()

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(11, 4.5))
for k_, col, lab in (("B", "#3b6ea8", "Part 12 B: reverse, IB30, stop opposite, BE 1xIB"), ("A", "#c9824a", "Part 12 A: no reverse, IB60, stop IB mid")):
    t, c = curves[k_]
    ax.plot(t, c, color=col, lw=1.3, drawstyle="steps-post", label=lab)
ax.axhline(0, color="k", lw=.6); ax.set_ylabel("cumulative PnL (index points, 1 unit)"); ax.legend(loc="upper left"); ax.grid(alpha=.3)
ax.set_title(f"NAS100 {PERIOD} — unseen data, rules frozen")
plt.tight_layout(); plt.savefig(f"part{PART}_equity.png", dpi=110)
