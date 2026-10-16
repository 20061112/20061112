"""第二十二部分補充：網格裡唯二「研究期、驗證期、去漂移後」都為正的區域，做鄰近參數與逐年檢查。
（這兩個區域是看過 480 組結果後才挑的 → 屬於探索性，需要之後的新資料前推驗證。）
  1) CUSUM H1，k 0.25~0.5、h 6（訊號稀少，持有約 5~7 天）
  2) BOCPD H4，λ=100、θ=0.5
輸出：results_part22_cand.txt
"""
import itertools, math
import numpy as np, pandas as pd
from changepoint import bars, zscore, bocpd_ng, bocpd_signals, cusum_signals, backtest, stats, fmt, detrend
from part22_bocpd_helpers import event_study, random_dir

out = open("results_part22_cand.txt", "w")
def P(s=""):
    print(s); out.write(s + "\n"); out.flush()

def line(b, T):
    dt = detrend(b, T)
    yy = T.groupby(T.entry_time.dt.year)
    ys = "  ".join(f"{y}:{v.pnl_R.mean():+.2f}/{detrend(b, v).mean():+.2f}({len(v)})" for y, v in yy)
    return (f"{fmt(stats(T))} | 去漂移 {dt.mean():+.3f}R t={dt.mean() / dt.std() * math.sqrt(len(dt)):+.2f} | "
            f"多 {T[T.side > 0].pnl_R.mean():+.2f}({(T.side > 0).sum()}) 空 {T[T.side < 0].pnl_R.mean():+.2f}({(T.side < 0).sum()}) | "
            f"持有 {T.days.mean():.1f} 天 | 逐年 原始/去漂移: {ys}")

P("0) 偵測延遲的數學限制（日線）")
bd = bars("D1"); zd = zscore(bd, "D1")
m20 = zd.rolling(20).mean().dropna()
drift_sd = math.sqrt(max(m20.std() ** 2 - 1 / 20, 0))
P(f"  20 日平均 z 的標準差 {m20.std():.3f}（純雜訊應為 {1 / math.sqrt(20):.3f}）→ 真實漂移的典型大小 ≈ {drift_sd:.2f} σ/日")
for d in (0.1, drift_sd, 0.2, 0.3, 0.5):
    P(f"  漂移 {d:.2f} σ/日：要有 80% 把握判斷方向，大約需要 (0.84/{d:.2f})² ≈ {(0.84 / d) ** 2:.0f} 根")
P("")
P("1) CUSUM H1 鄰近參數（停損 2 ATR，反手）")
b = bars("H1"); z = zscore(b, "H1").to_numpy()
for k, h in itertools.product((0.15, 0.25, 0.35, 0.5, 0.65), (5, 6, 7)):
    T = backtest(b, cusum_signals(z, k, h), 2.0, "flip")
    P(f"  k={k:.2f} h={h}: {line(b, T)}")
T = backtest(b, cusum_signals(z, 0.25, 6), 2.0, "flip")
real, rnd, pct, lo = random_dir(T, b)
P(f"\n  k=0.25 h=6：方向勝過 {pct:.0%} 的隨機方向；同時間全做多 {lo:+.3f}R")
P("  事件研究（訊號後 h 根 H1，報酬×方向÷ATR，扣漂移）")
sig = cusum_signals(z, 0.25, 6)
for r in event_study(b, sig, [4, 12, 24, 60, 120]).itertuples():
    P(f"    h={r.h:3d}: n={r.n} adj {r.adj:+.3f}（t {r.t_adj:+.2f}）命中 {r.hit:.2f}｜不重疊 n={r.n_nov} adj {r.adj_nov:+.3f}（t {r.t_nov:+.2f}）")
T.to_csv("part22_trades_cusum_H1.csv", index=False)

P("\n2) BOCPD H4 λ=100 / θ=0.5 附近（w、c、停損、出場）")
b = bars("H4"); z = zscore(b, "H4").to_numpy()
for lam in (75, 100, 150):
    for w in (5, 10):
        cp, conf, _ = bocpd_ng(z, lam, w=w)
        for theta, c in itertools.product((0.4, 0.5, 0.6), (0.8,)):
            sig, ep = bocpd_signals(cp, conf, theta, c)
            T = backtest(b, sig, 2.0, "flip", ep_start=ep)
            if len(T) > 3:
                P(f"  λ={lam} w={w} θ={theta}: {line(b, T)}")

P("\n3) CUSUM H1 (k=0.25, h=6) 是不是只是「比較慢的動能 / 均線」？")
b = bars("H1"); z = zscore(b, "H1").to_numpy()
from changepoint import bar_pnl, daily_pnl_R
from part22_bocpd_helpers import ols
sigC = cusum_signals(z, 0.25, 6)
TC = backtest(b, sigC, 2.0, "flip")
P(f"  CUSUM            : {line(b, TC)}")
def flips(s):
    e = np.zeros(len(s), int); ch = (s != np.r_[0, s[:-1]]) & (s != 0); e[ch] = s[ch]; return e
base = {}
for L in (20, 60, 120, 240):
    s = np.sign(b.close - b.close.shift(L)).fillna(0).to_numpy().copy(); s[:L + 1] = 0
    T = backtest(b, flips(s), 2.0, "flip"); base[f"TSMOM{L}"] = (s, T)
    P(f"  TSMOM{L:<3d} 翻轉+停損: {line(b, T)}")
for f_, s_ in ((10, 40), (24, 96), (50, 200)):
    s = np.sign(b.close.ewm(span=f_).mean() - b.close.ewm(span=s_).mean()).fillna(0).to_numpy().copy(); s[:s_] = 0
    T = backtest(b, flips(s), 2.0, "flip"); base[f"EMA{f_}/{s_}"] = (s, T)
    P(f"  EMA{f_}/{s_} 交叉+停損: {line(b, T)}")
yC, pC = bar_pnl(b, TC)
o = b.open.to_numpy(float); rr = np.nan_to_num(np.r_[o[1:] - o[:-1], 0.0] / b.atr.shift().to_numpy())
P("  部位相關：" + "  ".join(f"{k} {np.corrcoef(pC, np.r_[0, v[0][:-1]])[0, 1]:+.2f}" for k, v in base.items()))
X = [np.r_[0, v[0][:-1]] * rr for v in base.values()] + [rr]
bb, tt, r2 = ols(yC, X)
P(f"  CUSUM 逐根報酬對 {list(base)} + 買進持有 迴歸：截距 {bb[0]:+.4f}（t {tt[0]:+.2f}），R² {r2:.2f}；"
  + "  ".join(f"{k} t {t_:+.1f}" for k, t_ in zip(list(base) + ['BH'], tt[1:])))
dp = daily_pnl_R(b, TC)
L = pd.read_csv("oos_lonbrk_trades.csv", parse_dates=["entry_time"]); lon = L.groupby(L.entry_time.dt.normalize()).pnl_R.sum()
A = pd.read_csv("final_A_trades.csv", parse_dates=["交易日"]); fa = A.groupby("交易日")["損益_R"].sum()
for nm, other in (("LONBRK 2023~26", lon), ("版本A 2026", fa)):
    j = pd.concat([dp, other], axis=1, join="inner").fillna(0); j = j[j.iloc[:, 1] != 0]
    P(f"  與 {nm} 每日損益相關 {np.corrcoef(j.iloc[:, 0], j.iloc[:, 1])[0, 1]:+.2f}（{len(j)} 天）")
