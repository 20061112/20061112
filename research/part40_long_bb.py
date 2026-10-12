"""第四十部分：長週期布林帶（日線 20 / 50 / 100 日）的 sigma 與帶內位置 —— 用來控制最終版的回撤。

交易：final_m15_trades.csv（最終版，只扣點差），0.02 口美元。
因子（進場當下已知；均線與標準差用到前一日為止）：
  z_N   = (進場價 − SMA_N) / STD_N × 方向           → sigma；+2 = 順方向在上軌（2σ）
  bw_N  = 4·STD_N / SMA_N 的 250 日百分位            → 帶寬（大 = 波動擴張）
  %b    = (z_N + 2) / 4                              → 帶內位置（1 = 上軌、0 = 下軌，已乘方向）
評估：(1) 每個因子五分位的平均 R（訓練切點）；(2) 8 段大回撤開始時的 z；(3) 「z > k 時減半」與第三十九部分「離 50MA > 3 ATR 減半」比較。
訓練 2023~2024 / 驗證 2025~2026；比值 = 總損益 / 日結 MDD。
"""
import numpy as np
import pandas as pd
import part24_prev_bar_breakout as B

T = pd.read_csv("final_m15_trades.csv", parse_dates=["進場時間", "出場時間"])
T["usd"] = T.淨利_每盎司 * 2
T["date"] = T.進場時間.dt.normalize()
T["dir"] = np.where(T.方向 == "多", 1, -1)
T["train"] = T.進場時間 < "2025-01-01"
px = T["進場價(=點位)"].values
dl = B.daily.copy()
dl.index = pd.DatetimeIndex(dl.index)
dl["atr"] = (dl.H - dl.L).rolling(10).mean()
feat = {}
for n in (20, 50, 100):
    sma, sd = dl.C.rolling(n).mean(), dl.C.rolling(n).std()
    bw = (4 * sd / sma).rolling(250, min_periods=60).rank(pct=True)
    prev = pd.DataFrame({"sma": sma, "sd": sd, "bw": bw}).shift(1).reindex(T.date)
    T[f"z{n}"] = (px - prev.sma.values) / prev.sd.values * T.dir.values
    T[f"bw{n}"] = prev.bw.values
prev50 = pd.DataFrame({"sma": dl.C.rolling(50).mean(), "atr": dl.atr}).shift(1).reindex(T.date)
T["dist50_atr"] = (px - prev50.sma.values) / prev50.atr.values * T.dir.values
lines = []


def p(s=""):
    print(s)
    lines.append(s)


tr, te = T[T.train], T[~T.train]
p("第四十部分：長週期布林帶 sigma / 帶寬（最終版，0.02 口，只扣點差）\n")
p("=== 1. 五分位平均 R（Q1 小 → Q5 大；訓練切點）===")
for f in ["z20", "z50", "z100", "bw20", "bw50", "bw100", "dist50_atr"]:
    qs = np.unique(np.nanquantile(tr[f], [0.2, 0.4, 0.6, 0.8]))
    bins = [-np.inf, *qs, np.inf]
    a = tr.groupby(pd.cut(tr[f], bins), observed=True).R.mean()
    b = te.groupby(pd.cut(te[f], bins), observed=True).R.mean()
    p(f"  {f:10s} 切點 {np.round(qs, 2)}")
    p(f"     訓練 " + " ".join(f"{v:+.3f}" for v in a) + f"   驗證 " + " ".join(f"{v:+.3f}" for v in b))

# 2. 大回撤開始時的 z（part39 的 8 段）
p("\n=== 2. 大回撤開始時（高點日收盤）的 sigma ===")
starts = ["2023-10-02", "2024-04-15", "2024-07-16", "2025-04-21", "2025-05-06", "2025-10-08", "2025-10-16", "2026-01-28"]
z_all = {}
for n in (20, 50, 100):
    sma, sd = dl.C.rolling(n).mean(), dl.C.rolling(n).std()
    z_all[n] = (dl.C - sma) / sd
for s in starts:
    t = pd.Timestamp(s)
    p(f"  {s}: z20 {z_all[20][t]:+.2f}  z50 {z_all[50][t]:+.2f}  z100 {z_all[100][t]:+.2f}")
rng_ = dl.index[(dl.index >= T.date.min()) & (dl.index <= T.date.max())]
for n in (20, 50, 100):
    z = z_all[n][rng_]
    p(f"  對照：全部交易日 z{n} 中位數 {z.median():+.2f}；|z| > 2 的比例 {(z.abs() > 2).mean():.0%}")


# 3. 減半規則
def rat(x, w):
    d = (x.usd * w).groupby(x.date).sum(); e = d.cumsum(); m = (e.cummax() - e).max()
    return (x.usd * w).sum(), m


yrs = (T.進場時間.max() - T.進場時間.min()).days / 365
p("\n=== 3. 減倉規則（新單 0.02 → 0.01）===")
p(f"{'規則':28s} {'總損益':>8s} {'MDD':>6s} {'年化/MDD':>8s} | {'訓練比值':>6s} {'驗證比值':>6s} | 被減半的比例")
rules = [("最終版", np.zeros(len(T), bool)), ("離 50MA > 3 ATR（第三十九部分）", T.dist50_atr > 3)]
for n in (20, 50, 100):
    for k in (1.0, 1.5, 2.0, 2.5, 3.0):
        rules.append((f"z{n} > {k}", T[f"z{n}"] > k))
for n in (20, 50, 100):
    for k in (0.7, 0.8, 0.9):
        rules.append((f"帶寬{n} 百分位 > {k:.0%}", T[f"bw{n}"] > k))
rules.append(("z50 > 2 且 帶寬50 > 70%", (T.z50 > 2) & (T.bw50 > 0.7)))
rules.append(("z100 > 2 或 離50MA > 3 ATR", (T.z100 > 2) | (T.dist50_atr > 3)))
for lab, m in rules:
    w = np.where(np.asarray(m), 0.5, 1.0)
    a = rat(tr, w[T.train.values]); b = rat(te, w[~T.train.values]); c = rat(T, w)
    p(f"  {lab:28s} {c[0]:+8,.0f} {c[1]:6,.0f} {c[0] / yrs / c[1]:8.2f} | {a[0] / a[1]:6.1f} {b[0] / b[1]:6.1f} | {np.mean(m):.0%}")
open("results_part40.txt", "w").write("\n".join(lines) + "\n")
