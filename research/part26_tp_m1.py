"""第二十六部分補充：固定停利（盈虧比）用 M1 逐根驗證（H1 核心訊號，2023/6 ~ 2026/10）。
停利後若之後又出現同方向訊號，會重新進場；同一分鐘同時碰到停損與停利算停損。
Sharpe 用逐日盯市（持倉中的浮動損益也算），比「出場日實現」更公平。輸出 results_part26_tp.txt
"""
import math, warnings
import numpy as np, pandas as pd
warnings.simplefilter("ignore", RuntimeWarning)
from load import load
from cusum_lab import bars, zscore, m1_backtest, summ, line, M1_PATH
from changepoint import cusum_signals, daily_pnl_R

out = open("results_part26_tp.txt", "w")
def P(s=""):
    print(s); out.write(s + "\n"); out.flush()
hb = bars("H1"); sig = cusum_signals(zscore(hb), 0.25, 6); m1 = load(M1_PATH)
yrs = (m1.index[-1] - m1.index[0]).days / 365.25
P("固定停利 × 停損（M1 逐根；H1 k0.25 h6）")
P(f"{'':18s} {'筆/年':>5s} {'每筆R':>7s} {'t':>5s} {'PF':>5s} {'勝率':>4s} {'R/年':>6s} {'盯市Sharpe':>9s} {'回撤':>5s} {'R/回撤':>6s} {'去漂移':>7s}  逐年")
res = {}
for st in (2.0, 2.5, 3.0):
    for tp in (2, 3, 4, 5, 6, 8, None):
        T = m1_backtest(hb, sig, m1, st, tp_R=tp)
        d = summ(T, hb)
        dp = daily_pnl_R(hb, T); dp = dp[dp.index >= m1.index[0].normalize()]
        eq = dp.cumsum(); mdd = (eq.cummax() - eq).max()
        yr = T.groupby(T.entry_time.dt.year).pnl_R.mean()
        res[(st, tp)] = T
        P(f"  停{st:.1f} 利{('無' if tp is None else str(tp) + 'R'):>4s}     {len(T) / yrs:5.0f} {d['mean']:+7.3f} {d['t']:+5.2f} {d['pf']:5.2f} {d['win']:4.0%} "
          f"{T.pnl_R.sum() / yrs:+6.1f} {dp.mean() / dp.std() * math.sqrt(252):9.2f} {mdd:5.1f} {T.pnl_R.sum() / mdd:6.1f} {d['adj']:+7.3f}  "
          + " ".join(f"{y % 100}:{v:+.2f}" for y, v in yr.items()))
T = res[(2.0, 4)]
P(f"\n停 2 利 4R：出場原因 " + "  ".join(f"{k} {v}筆 平均 {T[T.why == k].pnl_R.mean():+.2f}R" for k, v in T.why.value_counts().items()))
P(f"  多 {T[T.side > 0].pnl_R.mean():+.2f}R（{(T.side > 0).sum()}）空 {T[T.side < 0].pnl_R.mean():+.2f}R（{(T.side < 0).sum()}）；"
  f"最好 10 筆合計 {T.pnl_R.nlargest(10).sum():+.1f}R / 總 {T.pnl_R.sum():+.1f}R；拿掉後每筆 {T.pnl_R.sort_values().iloc[:-10].mean():+.3f}R")
