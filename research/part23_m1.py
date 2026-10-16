"""第二十三部分 (1)：CUSUM H1（k=0.25, h=6, 停損 2×ATR14(H1), 反手）用 M1 逐根驗證。
訊號一律在 H1（由 M15 合成）收盤產生；M1 只負責進場、停損、反手的成交。M1 資料 2023/6/1 ~ 2026/10/8。
輸出：results_part23_m1.txt、part23_m1_trades.csv
"""
import numpy as np, pandas as pd
from load import load
from cusum_lab import bars, zscore, backtest, m1_backtest, summ, line, M1_PATH, detrend_R
from changepoint import cusum_signals

K, H, STOP = 0.25, 6, 2.0
out = open("results_part23_m1.txt", "w")
def P(s=""):
    print(s); out.write(s + "\n"); out.flush()

hb = bars("H1")
sig = cusum_signals(zscore(hb), K, H)
m1 = load(M1_PATH)
t0 = m1.index[0]

TH = backtest(hb, sig, STOP)
TH = TH[TH.entry_time >= t0].reset_index(drop=True)
TM = m1_backtest(hb, sig, m1, STOP)
TM.to_csv("part23_m1_trades.csv", index=False)

P(f"CUSUM H1 k={K} h={H} 停損 {STOP}×ATR14(H1)，比較期間 {t0.date()} ~ {m1.index[-1].date()}")
P(f"  H1 模擬 : {line(summ(TH, hb))}")
P(f"  M1 逐根 : {line(summ(TM, hb))}")

# 逐筆比對（以進場的小時對齊）
TM["hour"] = TM.entry_time.dt.floor("h")
J = TH.merge(TM, left_on="entry_time", right_on="hour", suffixes=("_h", "_m"))
P(f"\n  逐筆對齊：H1 {len(TH)} 筆、M1 {len(TM)} 筆、對上 {len(J)} 筆；方向一致 {(J.side_h == J.side_m).mean():.0%}；"
  f"出場原因一致 {(J.why_h == J.why_m).mean():.0%}")
P(f"  進場價差 中位 {(J.entry_m - J.entry_h).abs().median():.2f} 美元；R 差 中位 {(J.R_m - J.R_h).abs().median():.3f}")
d = J.pnl_R_m - J.pnl_R_h
P(f"  每筆 R 差（M1 − H1）：平均 {d.mean():+.3f}、中位 {d.median():+.3f}、|差|>0.1R 的比例 {(d.abs() > .1).mean():.0%}")
st = J[(J.why_h == "stop") & (J.why_m == "stop")]
P(f"  停損單的成交：M1 比 H1 平均差 {(st.pnl_R_m - st.pnl_R_h).mean():+.3f}R（跳空穿越停損時 M1 用實際那一分鐘開盤）")
dis = J[J.why_h != J.why_m]
if len(dis):
    P(f"  出場原因不同的 {len(dis)} 筆：" + "；".join(f"{r.entry_time.date()} H1={r.why_h}/M1={r.why_m}" for r in dis.head(8).itertuples()))

# 用 M1 自己合成的 H1 算訊號（資料來源檢查）
hb1 = bars("H1", src=m1)
sig1 = cusum_signals(zscore(hb1), K, H)
s0 = pd.Series(sig, index=hb.index); s1 = pd.Series(sig1, index=hb1.index)
c = pd.concat([s0, s1], axis=1, join="inner").loc[hb1.index[0] + pd.Timedelta(days=30):]
nz = c[(c.iloc[:, 0] != 0) | (c.iloc[:, 1] != 0)]
P(f"\n  資料來源檢查：M15 合成 vs M1 合成的 H1，訊號（非 0）{len(nz)} 個中，同一根同方向 {(nz.iloc[:, 0] == nz.iloc[:, 1]).mean():.0%}")
TM1 = m1_backtest(hb1, sig1, m1, STOP)
P(f"  M1 合成 H1 的訊號 + M1 執行：{line(summ(TM1[TM1.entry_time >= c.index[0]], hb))}")

P("\n  成本敏感度（M1 執行，每邊再加滑價）")
for slip in (0.0, 0.2, 0.5, 1.0, 2.0):
    T = m1_backtest(hb, sig, m1, STOP, slip=slip)
    P(f"    滑價 {slip:.1f}/邊：{line(summ(T, hb))}")
T = m1_backtest(hb, sig, m1, STOP, fin=0.0)
P(f"    不扣融資：{line(summ(T, hb))}")
T = m1_backtest(hb, sig, m1, STOP, fin=0.0004)
P(f"    融資加倍 0.04%/日：{line(summ(T, hb))}")

D = summ(TM, hb)
P("\n  M1 逐根 逐年（原始 / 去漂移 / 筆數 / 多 / 空）")
dt = detrend_R(hb, TM)
for y, g in TM.groupby(TM.entry_time.dt.year):
    P(f"    {y}: {g.pnl_R.mean():+.2f}R / {dt[g.index].mean():+.2f}R / {len(g)} / 多 {g[g.side > 0].pnl_R.mean():+.2f} 空 {g[g.side < 0].pnl_R.mean():+.2f}")
P(f"\n  出場原因：" + "  ".join(f"{k} {v}筆 平均 {TM[TM.why == k].pnl_R.mean():+.2f}R" for k, v in TM.why.value_counts().items()))
P(f"  R 中位 {TM.R.median():.1f} 美元；成本中位 {TM.cost.median():.2f} 美元（{(TM.cost / TM.R).median():.3f}R）；"
  f"最大單筆虧損 {TM.pnl_R.min():+.2f}R；最長連虧 {max((len(list(g)) for k, g in __import__('itertools').groupby(TM.pnl_R < 0) if k), default=0)} 筆")
