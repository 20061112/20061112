"""第二十五部分：把 H1 的 CUSUM 參數「換算」到 M30 / M15，而不是直接沿用。
換算（隨機漫步下讓兩個週期的 CUSUM 等價）：一根 H1 的 z ≈ (兩根 M30 的 z 相加)/√2，所以
  k_M30 = k_H1 / √2、h_M30 = h_H1 × √2、停損 ATR 倍數 × √2（ATR_M30 ≈ ATR_H1/√2）
M15 再除/乘一次 √2。核心 H1：k0.25 h6 停損 2 → M30：k0.177 h8.49 停損 2.83 → M15：k0.125 h12 停損 4。
輸出：results_part25.txt、part25_grid_m30.csv
"""
import itertools, math, warnings
import numpy as np, pandas as pd
warnings.simplefilter("ignore", RuntimeWarning)
from load import load
from cusum_lab import bars, zscore, backtest, m1_backtest, summ, line, detrend_R, M1_PATH
from changepoint import cusum_signals, daily_pnl_R

out = open("results_part25.txt", "w")
def P(s=""):
    print(s); out.write(s + "\n"); out.flush()
yrs = lambda b: (b.index[-1] - b.index[0]).days / 365.25
r2 = math.sqrt(2)

B = {tf: bars(tf) for tf in ("M15", "M30", "H1")}
Z = {tf: zscore(b) for tf, b in B.items()}
MAP = {"H1": (0.25, 6.0, 2.0), "M30": (0.25 / r2, 6 * r2, 2 * r2), "M15": (0.125, 12.0, 4.0)}

P("A) 換算後的參數 vs 直接沿用 H1 參數（2023/1 ~ 2026/10，週期本身 K 棒模擬）")
T_core = {}
for tf in ("H1", "M30", "M15"):
    k, h, st = MAP[tf]
    T = backtest(B[tf], cusum_signals(Z[tf], k, h), st); T_core[tf] = T
    P(f"  {tf:3s} 換算 k{k:.3f} h{h:.2f} 停損{st:.2f}: 每年 {len(T) / yrs(B[tf]):4.0f} 筆 | {line(summ(T, B[tf]))}")
    if tf != "H1":
        T2 = backtest(B[tf], cusum_signals(Z[tf], 0.25, 6), 2.0)
        P(f"  {tf:3s} 直接沿用 k0.25 h6 停損2   : 每年 {len(T2) / yrs(B[tf]):4.0f} 筆 | {line(summ(T2, B[tf]))}")

# 訊號對照：換算後的 M30 和 H1 是否抓到同一批趨勢？早多少？
P("\nB) 換算後 M30 / M15 訊號與 H1 訊號的關係")
sH = pd.Series(cusum_signals(Z["H1"], 0.25, 6), index=B["H1"].index)
evH = sH[sH != 0]
for tf in ("M30", "M15"):
    k, h, st = MAP[tf]
    s = pd.Series(cusum_signals(Z[tf], k, h), index=B[tf].index); ev = s[s != 0]
    # 每個 H1 訊號：前後 24 小時內最近的同方向低週期訊號
    lead = []
    for t, d in evH.items():
        tclose = t + pd.Timedelta(hours=1)
        w = ev[(ev.index >= tclose - pd.Timedelta(hours=24)) & (ev.index <= tclose + pd.Timedelta(hours=24)) & (ev == d)]
        if len(w):
            dt = ((w.index + pd.Timedelta(minutes=30 if tf == "M30" else 15)) - tclose).total_seconds() / 3600
            lead.append(dt[np.argmin(np.abs(dt))])
        else:
            lead.append(np.nan)
    lead = pd.Series(lead)
    P(f"  {tf}: 訊號 {len(ev)} 個（H1 {len(evH)} 個）；H1 訊號 24h 內有同向 {tf} 訊號 {lead.notna().mean():.0%}；"
      f"其中 {tf} 比較早 {(lead < 0).mean() / max(lead.notna().mean(), 1e-9):.0%}，時間差中位 {lead.median():+.1f} 小時（負 = {tf} 先）")
    dH = daily_pnl_R(B["H1"], T_core["H1"]); dL = daily_pnl_R(B[tf], T_core[tf])
    P(f"     每日損益相關（與 H1 核心）{pd.concat([dH, dL], axis=1).fillna(0).corr().iloc[0, 1]:.2f}")

P("\nC) M30 換算點附近的網格（停損 2.83 ATR；k × h）")
rows = []
for k, h in itertools.product((0.05, 0.1, 0.15, 0.177, 0.2, 0.25, 0.3, 0.4), (6, 7, 8, 8.49, 9, 10, 11, 12)):
    T = backtest(B["M30"], cusum_signals(Z["M30"], k, h), 2 * r2)
    d = summ(T, B["M30"])
    rows.append(dict(k=k, h=h, n=d["n"], mean=d.get("mean"), adj=d.get("adj"), t=d.get("t"), minyr=d.get("minyr"), hold=d.get("hold")))
G = pd.DataFrame(rows); G.to_csv("part25_grid_m30.csv", index=False)
for col, f in (("mean", "{:+.2f}"), ("adj", "{:+.2f}"), ("minyr", "{:+.2f}"), ("n", "{:5.0f}")):
    M = G.pivot(index="k", columns="h", values=col)
    P(f"  [{ {'mean': '每筆 R 原始', 'adj': '每筆 R 去漂移', 'minyr': '最差年', 'n': '筆數'}[col] }]")
    P("        h→ " + " ".join(f"{x:>5.2f}" for x in M.columns))
    for k, r in M.iterrows():
        P(f"  k={k:.3f} " + " ".join(f.format(v) for v in r.values))
ok = G[G.n >= 30]
P(f"  筆數≥30 的 {len(ok)} 組：原始>0 {(ok['mean'] > 0).mean():.0%}、去漂移>0 {(ok.adj > 0).mean():.0%}、每年都>0 {(ok.minyr > 0).mean():.0%}，"
  f"平均 原始 {ok['mean'].mean():+.3f} / 去漂移 {ok.adj.mean():+.3f}")
P("  （對照 H1 第二十三部分：筆數≥30 的組 原始>0 97%、去漂移>0 87%、每年都>0 32%）")

P("\nD) M30 換算版的停損倍數")
k, h, _ = MAP["M30"]
sig30 = cusum_signals(Z["M30"], k, h)
for st in (2.0, 2.5, 2.83, 3.5, 4.0):
    P(f"  停損 {st:.2f} ATR(M30): {line(summ(backtest(B['M30'], sig30, st), B['M30']))}")

P("\nE) M1 逐根驗證（2023/6 起）：H1 核心 vs M30 換算版")
m1 = load(M1_PATH)
TH = m1_backtest(B["H1"], cusum_signals(Z["H1"], 0.25, 6), m1, 2.0)
TM = m1_backtest(B["M30"], sig30, m1, 2 * r2)
P(f"  H1 核心   : {line(summ(TH, B['H1']))}")
P(f"  M30 換算版: {line(summ(TM, B['M30']))}")
for lab, T in (("H1", TH), ("M30", TM)):
    P(f"    {lab} 逐年：" + "  ".join(f"{y}:{v:+.2f}({c})" for y, v, c in
                                 zip(*[T.groupby(T.entry_time.dt.year).pnl_R.agg(x).index for x in ['mean']],
                                     T.groupby(T.entry_time.dt.year).pnl_R.mean(), T.groupby(T.entry_time.dt.year).size())))

P("\nF) H1 + M30 換算版一起跑（各 0.5R，總風險同 H1 核心）")
def port(Ts):
    d = pd.concat([daily_pnl_R(b, t) * w for t, b, w in Ts], axis=1).fillna(0).sum(axis=1)
    eq = d.cumsum(); return d, (eq.cummax() - eq).max()
for lab, Ts in (("H1 核心 1R", [(T_core["H1"], B["H1"], 1)]), ("M30 換算 1R", [(T_core["M30"], B["M30"], 1)]),
                ("H1 0.5R + M30 0.5R", [(T_core["H1"], B["H1"], .5), (T_core["M30"], B["M30"], .5)]),
                ("H1 + M30 + M15 各 1/3R", [(T_core[x], B[x], 1 / 3) for x in ("H1", "M30", "M15")])):
    d, mdd = port(Ts)
    n = sum(len(t) for t, _, _ in Ts)
    P(f"  {lab:24s} 每年 {n / yrs(B['H1']):4.0f} 筆 | 每年 {d.sum() / yrs(B['H1']):+6.1f}R | 日Sharpe {d.mean() / d.std() * math.sqrt(252):.2f} | 回撤 {mdd:.1f}R | 總R/回撤 {d.sum() / mdd:.1f}")
