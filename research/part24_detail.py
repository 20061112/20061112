"""第二十四部分：CUSUM H1 核心的逐筆詳細數據（M1 執行）+ 增加交易數的各種做法。
輸出：results_part24.txt、part24_trades_detail.csv（逐筆，含 MAE/MFE）、part24_monthly.csv
"""
import itertools, math, warnings
import numpy as np, pandas as pd
warnings.simplefilter("ignore", RuntimeWarning)
from load import load
from cusum_lab import bars, zscore, backtest, backtest_pyr, m1_backtest, summ, line, detrend_R, M1_PATH
from changepoint import cusum_signals, daily_pnl_R

out = open("results_part24.txt", "w")
def P(s=""):
    print(s); out.write(s + "\n"); out.flush()

hb = bars("H1"); z = zscore(hb)
sig = cusum_signals(z, 0.25, 6)
m1 = load(M1_PATH)
T = m1_backtest(hb, sig, m1, 2.0)

# ---------------------------------------------------------------- MAE / MFE（M1）
mt = m1.index; mh, ml = m1.high.to_numpy(), m1.low.to_numpy()
mae, mfe, t_mfe = [], [], []
for r in T.itertuples():
    a, b_ = mt.searchsorted(r.entry_time), mt.searchsorted(r.exit_time)
    hi, lo = mh[a:b_ + 1], ml[a:b_ + 1]
    if r.side > 0:
        mae.append((r.entry - lo.min()) / r.R); f = (hi - r.entry) / r.R
    else:
        mae.append((hi.max() - r.entry) / r.R); f = (r.entry - lo) / r.R
    mfe.append(f.max()); t_mfe.append((mt[a + f.argmax()] - r.entry_time).total_seconds() / 3600)
T["MAE_R"], T["MFE_R"], T["hours_to_MFE"] = mae, mfe, t_mfe
T["hours"] = (T.exit_time - T.entry_time).dt.total_seconds() / 3600
T["adj_R"] = detrend_R(hb, T)
T["giveback_R"] = T.MFE_R - T.pnl_R
T.to_csv("part24_trades_detail.csv", index=False)

P("第二十四部分：CUSUM H1 核心（k0.25 h6 停損 2ATR 反手）— M1 執行，2023/6 ~ 2026/10")
P(f"  {line(summ(T, hb))}")
yrs = (T.exit_time.max() - T.entry_time.min()).days / 365.25
r = T.pnl_R
P("\nA) 損益分布")
P(f"  每年 {len(T) / yrs:.0f} 筆；勝率 {(r > 0).mean():.0%}；平均獲利 {r[r > 0].mean():+.2f}R、平均虧損 {r[r <= 0].mean():+.2f}R、"
  f"賺賠比 {r[r > 0].mean() / -r[r <= 0].mean():.2f}")
P("  R 分位數：" + "  ".join(f"{q:.0%}:{r.quantile(q):+.2f}" for q in (.05, .1, .25, .5, .75, .9, .95)) + f"  最大 {r.max():+.2f}  最小 {r.min():+.2f}")
srt = r.sort_values(ascending=False)
P(f"  最好 5 筆合計 {srt.head(5).sum():+.1f}R、最好 10 筆 {srt.head(10).sum():+.1f}R（總 {r.sum():+.1f}R）→ 拿掉最好 10 筆後每筆 {srt.iloc[10:].mean():+.3f}R")
bins = [-9, -1.2, -0.8, 0, 1, 2, 4, 8, 99]
hist = pd.cut(r, bins).value_counts().sort_index()
P("  區間筆數：" + "  ".join(f"{iv}:{c}" for iv, c in hist.items()))
P(f"  美元（每盎司）：每筆平均 {T.pnl.mean():+.2f}、R 中位 {T.R.median():.1f}、成本中位 {T.cost.median():.2f}")

P("\nB) 持有時間與出場")
P("  持有小時分位：" + "  ".join(f"{q:.0%}:{T.hours.quantile(q):.0f}h" for q in (.1, .25, .5, .75, .9)) + f"；最長 {T.hours.max() / 24:.1f} 天")
for why, g in T.groupby("why"):
    P(f"  {why:5s}: {len(g):3d} 筆 平均 {g.pnl_R.mean():+.2f}R 持有中位 {g.hours.median():.0f}h")
for lab, g in (("1 天內", T[T.hours <= 24]), ("1~3 天", T[(T.hours > 24) & (T.hours <= 72)]),
               ("3~7 天", T[(T.hours > 72) & (T.hours <= 168)]), ("7 天以上", T[T.hours > 168])):
    P(f"  持有 {lab:6s}: {len(g):3d} 筆 平均 {g.pnl_R.mean():+.2f}R 勝率 {(g.pnl_R > 0).mean():.0%}")
P(f"  市場曝險時間：約 {T.hours.sum() / ((T.exit_time.max() - T.entry_time.min()).total_seconds() / 3600):.0%} 的時間有持倉")

P("\nC) MAE / MFE（以 R 計，M1 最高/最低）")
w, l_ = T[T.pnl_R > 0], T[T.why == "stop"]
P("  獲利單的最大浮虧 MAE 分位：" + "  ".join(f"{q:.0%}:{w.MAE_R.quantile(q):.2f}" for q in (.5, .75, .9, 1.0)))
P("  停損單在被停損前的最大浮盈 MFE 分位：" + "  ".join(f"{q:.0%}:{l_.MFE_R.quantile(q):.2f}" for q in (.5, .75, .9)) +
  f"；曾經浮盈 ≥1R 後才停損 {(l_.MFE_R >= 1).mean():.0%}、≥2R {(l_.MFE_R >= 2).mean():.0%}")
P("  獲利單 MFE 分位：" + "  ".join(f"{q:.0%}:{w.MFE_R.quantile(q):.2f}" for q in (.5, .75, .9)) +
  f"；平均回吐 {w.giveback_R.mean():.2f}R（MFE − 實際），達到 MFE 的時間中位 {w.hours_to_MFE.median():.0f}h")

P("\nD) 時間分布（進場）")
for lab, key in (("星期", T.entry_time.dt.dayofweek), ("broker 小時（4 段）", T.entry_time.dt.hour // 6 * 6)):
    g = T.groupby(key).pnl_R
    P(f"  {lab}: " + "  ".join(f"{k}:{v.mean():+.2f}({len(v)})" for k, v in g))
P("  多 / 空：" + "  ".join(f"{'多' if k > 0 else '空'} {len(g)} 筆 {g.pnl_R.mean():+.2f}R 勝率 {(g.pnl_R > 0).mean():.0%} 去漂移 {g.adj_R.mean():+.2f}R"
                            for k, g in T.groupby("side")))

P("\nE) 連續性與回撤")
neg = (r < 0).to_numpy()
runs = [len(list(g)) for k, g in itertools.groupby(neg) if k]
P(f"  連虧次數分布：" + "  ".join(f"{n}筆×{runs.count(n)}" for n in sorted(set(runs))))
eq = r.cumsum().to_numpy(); peak = np.maximum.accumulate(np.r_[0, eq])[1:]
dd = peak - eq
P(f"  最大回撤 {dd.max():.1f}R；回撤 > 5R 的筆數比例 {(dd > 5).mean():.0%}")
# 回撤時間（日曆天）
ts = T.exit_time.to_numpy(); top = 0; start = ts[0]; longest = (0, None)
pk = -1e9
for i, e in enumerate(eq):
    if e >= pk:
        dur = (ts[i] - start) / np.timedelta64(1, "D")
        if dur > longest[0]:
            longest = (dur, pd.Timestamp(start))
        pk, start = e, ts[i]
P(f"  最長「創新高之間」的時間 {longest[0]:.0f} 天（從 {longest[1].date()} 起）")
dp = daily_pnl_R(hb, T.assign(atr=T.R / 2))
dp = dp[dp.index >= T.entry_time.min().normalize()]
P(f"  逐日盯市：日 Sharpe（年化）{dp.mean() / dp.std() * math.sqrt(252):.2f}；最差單日 {dp.min():+.2f}R；最好單日 {dp.max():+.2f}R")
mon = T.groupby(T.exit_time.dt.to_period("M")).pnl_R.agg(["sum", "count"])
mon.to_csv("part24_monthly.csv")
P(f"  月份：{len(mon)} 個月，正 {(mon['sum'] > 0).sum()}、負 {(mon['sum'] < 0).sum()}；最差月 {mon['sum'].min():+.1f}R、最好月 {mon['sum'].max():+.1f}R")
P("  每月 R（出場月份）：")
for y in sorted(set(mon.index.year)):
    m = mon[mon.index.year == y]
    P(f"    {y}: " + " ".join(f"{p.month:02d}:{v:+5.1f}" for p, v in m["sum"].items()) + f"  | 年 {m['sum'].sum():+.1f}R")

# ================================================================ 增加交易數
P("\n" + "=" * 100)
P("F) 增加交易數的做法（H1 模擬，2023/1 ~ 2026/10；每筆風險都算 1R）")
P("   比較用：每年筆數、每筆 R、每年總 R、日 Sharpe（不受部位大小影響）、總 R / 最大回撤")
yrs_all = (hb.index[-1] - hb.index[0]).days / 365.25
def portfolio(Ts, weights=None):
    """多個子系統的逐日損益相加（各自 1R 或依權重）。"""
    ds = [daily_pnl_R(hb if b_ is None else b_, t) * (1 if weights is None else w_) for (t, b_), w_ in
          zip(Ts, weights or [1] * len(Ts))]
    d = pd.concat(ds, axis=1).fillna(0).sum(axis=1).sort_index()
    eq = d.cumsum(); mdd = (eq.cummax() - eq).max()
    return d, mdd
def row(lab, Ts, weights=None):
    allT = pd.concat([t for t, _ in Ts])
    d, mdd = portfolio(Ts, weights)
    tot = d.sum()
    dt = pd.concat([detrend_R(hb if b_ is None else b_, t) for t, b_ in Ts])
    P(f"  {lab:34s} 每年 {len(allT) / yrs_all:5.0f} 筆 | 每筆 {allT.pnl_R.mean():+.3f}R（去漂移 {dt.mean():+.3f}）| 每年 {tot / yrs_all:+6.1f}R"
      f" | 日Sharpe {d.mean() / d.std() * math.sqrt(252):.2f} | 總R/回撤 {tot / mdd:5.1f}（回撤 {mdd:.1f}R）")
    return d

core = backtest(hb, sig, 2.0)
base_d = row("核心 k0.25 h6", [(core, None)])

P("\n  1) 降低門檻（沿著斜帶往訊號多的方向走）")
for k, h in ((0.25, 5.5), (0.25, 5.0), (0.25, 4.5), (0.25, 4.0), (0.15, 5.0), (0.1, 5.0), (0.0, 5.0), (0.5, 4.0), (0.6, 4.0)):
    row(f"k{k} h{h}", [(backtest(hb, cusum_signals(z, k, h), 2.0), None)])

P("\n  2) 同方向訊號加碼（每個新訊號再開一單，各自停損；最多 N 單）")
for mu in (2, 3, 5):
    row(f"加碼 最多 {mu} 單（k0.25 h6）", [(backtest_pyr(hb, sig, 2.0, mu), None)])
for k, h in ((0.25, 5.0), (0.25, 4.0)):
    row(f"加碼 最多 3 單（k{k} h{h}）", [(backtest_pyr(hb, cusum_signals(z, k, h), 2.0, 3), None)])

P("\n  3) 多組參數一起跑（斜帶上 5 組，各自獨立）")
SETS = [(0.1, 7.5), (0.2, 6.5), (0.25, 6.0), (0.4, 5.0), (0.6, 4.5)]
TS = [(backtest(hb, cusum_signals(z, k, h), 2.0), None) for k, h in SETS]
row("5 組 各 1R", TS)
row("5 組 各 0.2R（總風險同核心）", TS, [0.2] * 5)
ds = [daily_pnl_R(hb, t) for t, _ in TS]
C = pd.concat(ds, axis=1).fillna(0).corr().to_numpy()
P(f"     5 組之間的日損益相關：平均 {C[np.triu_indices(5, 1)].mean():.2f}（{C[np.triu_indices(5, 1)].min():.2f} ~ {C[np.triu_indices(5, 1)].max():.2f}）")

P("\n  4) 多週期一起跑（同參數 k0.25 h6，各自 ATR 停損）")
TF = {}
for tf in ("M30", "H1", "H2", "H3"):
    b_ = hb if tf == "H1" else bars(tf)
    TF[tf] = (backtest(b_, cusum_signals(zscore(b_), 0.25, 6), 2.0), b_)
    row(f"{tf} 單獨", [TF[tf]])
row("H1 + H2", [TF["H1"], TF["H2"]])
row("H1 + H2 + H3", [TF["H1"], TF["H2"], TF["H3"]])
row("M30 + H1 + H2", [TF["M30"], TF["H1"], TF["H2"]])
d1, d2 = daily_pnl_R(hb, TF["H1"][0]), daily_pnl_R(TF["H2"][1], TF["H2"][0])
P(f"     H1 與 H2 日損益相關 {pd.concat([d1, d2], axis=1).fillna(0).corr().iloc[0, 1]:.2f}")

P("\n  5) 組合：斜帶 5 組 + H2 核心（各 1R）")
row("斜帶 5 組 + H2", TS + [TF["H2"]])
