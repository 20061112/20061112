"""第三十二部分：用「最近的盤面回測」當開單基準 —— 哪一邊最近太負就先不開。
兩種做法（都只用前一天以前的資料）：
  一、策略自己的近期績效（主規則，事先登記）：
      每天開盤前，分別算「最近 N 個交易日」多單、空單的累計 R；某一邊 < 門檻 → 今天那一邊不開。
      N = 10 / 20 / 40 / 60 / 120；門檻 0R / −5R / −10R。主規則 = N 40、門檻 0R。
  二、價格漂移（近期黃金每天平均漲跌，單位 ATR10）：
      最近 N 日漲跌 ÷ N ÷ ATR10；> +k → 不做空；< −k → 不做多。N = 10 / 20 / 60；k = 0 / 0.05 / 0.1 / 0.2。
套用在：L1 核心 + 反向尾巴（主要）、版本 A（凍結門檻）。
"""
import numpy as np
import pandas as pd
from load import load
from part29_dir_filter import BASES, OOS_END, pf

out = open("results_part32.txt", "w")


def P(s=""):
    print(s)
    out.write(s + "\n")
    out.flush()


m1 = load("data/XAUUSD_M1_2023_2026.csv")
dk = (m1.index - pd.Timedelta(hours=1)).normalize()
Dd = m1.groupby(dk).agg(high=("high", "max"), low=("low", "min"), close=("close", "last"), n=("close", "size"))
Dd = Dd[Dd.n >= 600]
tr = np.maximum(Dd.high - Dd.low, np.maximum((Dd.high - Dd.close.shift()).abs(), (Dd.low - Dd.close.shift()).abs()))
atr10 = tr.rolling(10).mean()
DAYS = Dd.index


def stat(T):
    if len(T) == 0:
        return "    0 筆"
    return f"{len(T):5d} 筆 PF {pf(T.R):.2f} 每筆 {T.R.mean():+.3f}R 累計 {T.R.sum():+6.0f}R"


def perf_gate(T, N, thr):
    """策略自己最近 N 個交易日、每一邊的累計 R（只用前一天以前）。"""
    keep = np.ones(len(T), bool)
    for s in (1, -1):
        daily = T[T.side == s].groupby("day").R.sum().reindex(DAYS, fill_value=0)
        trail = daily.rolling(N, min_periods=N).sum().shift(1)        # 到昨天為止
        v = trail.reindex(T.day).to_numpy()
        keep &= ~((T.side.to_numpy() == s) & (v < thr))
    return T[keep]


def drift_gate(T, N, k):
    dr = ((Dd.close - Dd.close.shift(N)) / N / atr10).shift(1)         # 到昨天為止，每天平均漲跌（ATR10）
    v = dr.reindex(T.day).to_numpy()
    s = T.side.to_numpy()
    keep = ~((s == -1) & (v > k)) & ~((s == 1) & (v < -k))
    return T[keep]


def report(T, lab):
    o, i = T[T.day < OOS_END], T[T.day >= OOS_END]
    yr = T.groupby(T.day.dt.year)
    yrs = " ".join(f"{y}:{pf(g.R):.2f}" for y, g in yr)
    lo, sh = T[T.side == 1], T[T.side == -1]
    P(f"  {lab:24s} 全期 {stat(T)} | 樣本外 {stat(o)} | 2026 {stat(i)} | 逐年PF {yrs} | 多 {len(lo)}筆 PF {pf(lo.R):.2f} 空 {len(sh)}筆 PF {pf(sh.R):.2f}")
    return dict(lab=lab, n=len(T), pf=pf(T.R), pf_o=pf(o.R), pf_i=pf(i.R), worst_year=min(pf(g.R) for _, g in yr))


P("第三十二部分：用近期盤面回測當開單基準（哪一邊最近太負就不開）")
P("PF 一律以 R 計（不受金價高低影響）。樣本外 = 2023/6~2025/12。\n")
P("# 為什麼漂移不是 0：每年黃金的漲跌")
for y in (2023, 2024, 2025, 2026):
    s = Dd.close[Dd.index.year == y]
    a = atr10[atr10.index.year == y].mean()
    P(f"  {y}: {s.iloc[0]:.0f} → {s.iloc[-1]:.0f}（{100 * (s.iloc[-1] / s.iloc[0] - 1):+.0f}%），{len(s)} 個交易日，平均每天 {(s.iloc[-1] - s.iloc[0]) / len(s):+.2f} 美元 = {(s.iloc[-1] - s.iloc[0]) / len(s) / a:+.3f} ATR10")

rows = []
for name in ("L1 核心 + 反向尾巴（前推）", "版本 A（凍結門檻）"):
    T = BASES[name]
    P(f"\n\n# {name}")
    rows.append(dict(base=name, **report(T, "不加過濾")))
    P("\n  一、策略自己的近期績效（某一邊最近 N 天累計 R < 門檻 → 那一邊今天不開）")
    for N in (10, 20, 40, 60, 120):
        for thr in (0, -5, -10):
            lab = f"N={N:3d} 門檻 {thr:+d}R" + ("  ← 主規則" if (N, thr) == (40, 0) else "")
            rows.append(dict(base=name, **report(perf_gate(T, N, thr), lab)))
    P("\n  二、價格漂移（最近 N 天平均每天漲跌 > +k ATR10 不做空；< −k 不做多）")
    for N in (10, 20, 60):
        for k in (0, 0.05, 0.1, 0.2):
            rows.append(dict(base=name, **report(drift_gate(T, N, k), f"漂移 N={N:2d} k={k:.2f}")))
    P("\n  對照")
    rows.append(dict(base=name, **report(T[T.side == 1], "只做多")))
    rows.append(dict(base=name, **report(T[T.side == -1], "只做空")))

R = pd.DataFrame(rows)
R.to_csv("part32_summary.csv", index=False, float_format="%.3f")
P("\n\n# 摘要：樣本外與 2026 都比「不加過濾」好的設定")
for name, g in R.groupby("base", sort=False):
    b = g[g.lab == "不加過濾"].iloc[0]
    better = g[(g.pf_o > b.pf_o) & (g.pf_i > b.pf_i) & (g.lab != "不加過濾")]
    P(f"  {name}（不加過濾：樣本外 {b.pf_o:.2f} / 2026 {b.pf_i:.2f}）→ {len(better)} / {len(g) - 1} 組兩段都更好")
    for _, r in better.sort_values("pf_o", ascending=False).iterrows():
        P(f"     {r.lab.strip():24s} 筆數 {r.n:5d} 全期 PF {r.pf:.2f} | 樣本外 {r.pf_o:.2f} | 2026 {r.pf_i:.2f} | 最差一年 {r.worst_year:.2f}")
out.close()


out = open("results_part32.txt", "a")
P("\n\n# 補充（看過上面結果後才加的，只能當描述）：長視窗 N 的鄰近檢查，門檻 0R")
for name in ("L1 核心 + 反向尾巴（前推）", "版本 A（凍結門檻）"):
    T = BASES[name]
    P(f"\n## {name}")
    for N in (60, 80, 100, 120, 150, 180, 250):
        Y = perf_gate(T, N, 0)
        eq = Y.sort_values("exit_time").R.cumsum()
        dd = (eq - eq.cummax()).min()
        o, i = Y[Y.day < OOS_END], Y[Y.day >= OOS_END]
        sh = Y[Y.side == -1]
        P(f"  N={N:3d}: {len(Y):5d} 筆 全期 PF {pf(Y.R):.2f} 累計 {Y.R.sum():+5.0f}R 平倉回撤 {dd:5.0f}R | 樣本外 {pf(o.R):.2f} | 2026 {pf(i.R):.2f} | 逐年 "
          + " ".join(f"{y}:{pf(g.R):.2f}" for y, g in Y.groupby(Y.day.dt.year)) + f" | 空單 {len(sh)} 筆 PF {pf(sh.R):.2f}")
    eq = T.sort_values("exit_time").R.cumsum()
    P(f"  不加過濾: {len(T):5d} 筆 全期 PF {pf(T.R):.2f} 累計 {T.R.sum():+5.0f}R 平倉回撤 {(eq - eq.cummax()).min():5.0f}R")
out.close()
