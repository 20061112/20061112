"""第三十五部分：最終版 M15 突破的倉位放大 —— (1) 依狀態調整大小 (2) 每筆風險可以放多大。

樣本：最終版實際下單（含權益濾網）7,492 筆，每筆 R。因子來自 part34（需要先跑 part34_factors.py）。
(1) 依狀態加權：權重 w（好狀態 > 1、差狀態 < 1），損益 = w × R；比較 年化 R / MDD（與總規模無關）、訓練/驗證比值。
    因子切點全部來自訓練期（2023~2024）。
(2) 複利模擬：每筆風險 = 帳戶 f%，同一天的單用當天開盤的權益計算；
    歷史路徑 + 區塊自助抽樣（每塊 20 個交易日，保留連虧的群聚）2,000 次 → 年化報酬、MDD 分布、P(MDD > 20% / 30% / 50%)。
    Kelly：以「每日 R 總和」最大化 E[log(1 + f·日R)]。
"""
import numpy as np
import pandas as pd

X = pd.read_pickle("part34_X.pkl")
S = pd.read_csv("final_m15_signals.csv", parse_dates=["進場時間", "出場時間"])
X["進場時間"] = S.進場時間.values
T = X[S.實際下單.values].copy().reset_index(drop=True)
T["date"] = T.進場時間.dt.date
tr = X[X.train]
lines = []


def p(s=""):
    print(s)
    lines.append(s)


# ---------- (1) 依狀態加權 ----------
c1 = np.nanquantile(tr["ATR15/ATR日"], 0.8)
c2 = np.nanquantile(tr["BB寬度百分位(M15)"], 0.8)
c3 = np.nanquantile(tr["日ER10"], 0.2)
F1 = (T["ATR15/ATR日"] <= c1).values
F2 = (T["BB寬度百分位(M15)"] <= c2).values
F3 = (T["日ER10"] >= c3).values
score = F1.astype(int) + F2 + F3
asia = T.hour.between(1, 7).values
nopen = T["進場時已持倉數"].values
eqv = T["權益濾網(累積R−均線)"].values
eq_hi = eqv > np.nanquantile(eqv[T.train.values], 0.5)


def ratio(w, mask=None):
    x = T if mask is None else T[mask]
    ww = w if mask is None else w[mask]
    d = (x.R * ww).groupby(x.date).sum()
    e = d.cumsum()
    m = (e.cummax() - e).max()
    return (x.R * ww).sum(), m


def show(w, lab):
    tot, m = ratio(w)
    ttr, mtr = ratio(w, T.train.values)
    tte, mte = ratio(w, ~T.train.values)
    yrs = (T.進場時間.max() - T.進場時間.min()).days / 365
    p(f"  {lab:40s} 平均權重 {w.mean():.2f}  每單位風險 {tot / w.sum():+.3f}R  訓練比值 {ttr / mtr:4.1f}  驗證比值 {tte / mte:4.1f}  年化/MDD {tot / yrs / m:4.2f}")


p("第三十五部分：倉位放大")
p("\n=== (1) 依狀態調整倉位（不刪單，只調大小）===")
one = np.ones(len(T))
show(one, "固定 1R（最終版）")
show(np.select([score == 3, score == 2, score == 1], [1.5, 1.0, 0.75], 0.5), "三因子分數 3/2/1/0 → 1.5/1/0.75/0.5")
show(np.where(F1 & F3, 1.5, 1.0), "F1+F3 成立 → 1.5")
show(np.where(F1 & F3, 1.0, 0.5), "F1+F3 不成立 → 0.5")
show(np.where(F3, 1.0, 0.5), "日線盤整（F3 不成立）→ 0.5")
show(np.where(F1, 1.0, 0.5), "M15 波動大（F1 不成立）→ 0.5")
show(np.where(asia, 1.5, 1.0), "亞洲盤 01~08 點 → 1.5")
show(np.where(eq_hi, 1.5, 1.0), "權益遠高於均線 → 1.5")
show(np.where(nopen >= 4, 0.5, 1.0), "已持倉 ≥ 4 筆 → 0.5")
show(1.0 / np.sqrt(1 + nopen), "權重 = 1/√(1+已持倉數)")
p("  （訓練與驗證都提高的：權益遠高於均線 → 1.5、已持倉 ≥ 4 → 0.5；組合如下）")
show(np.where(eq_hi, 1.5, 1.0) * np.where(nopen >= 4, 0.5, 1.0), "權益高 ×1.5 且 已持倉 ≥ 4 ×0.5")

# ---------- (2) 每筆風險可以多大 ----------
SLIP = 0.05  # 每筆滑價（R）
d = (T.R - SLIP).groupby(T.date).sum()
dates = list(d.index)
dR = d.values
yrs = (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days / 365
ntd = len(dR) / yrs  # 每年有交易的天數


def path(f, seq):
    eq = np.cumprod(1 + f * seq)
    eq = np.maximum(eq, 1e-9)
    peak = np.maximum.accumulate(eq)
    return eq[-1], (1 - eq / peak).max()


p("\n=== (2) 每筆風險（帳戶 %）—— 複利，同一天的單用當天開盤權益；每筆已再扣 0.05R 滑價 ===")
fs = np.linspace(0.0005, 0.03, 600)
g = [np.mean(np.log(np.maximum(1 + f * dR, 1e-9))) for f in fs]
fk = fs[int(np.argmax(g))]
p(f"  每日 R 總和：平均 {dR.mean():+.2f}R、標準差 {dR.std():.2f}R、最差日 {dR.min():+.1f}R、最好日 {dR.max():+.1f}R（{len(dR)} 個交易日）")
p(f"  Kelly（每筆風險）≈ {fk * 100:.2f}%；建議用 1/4 ~ 1/2 Kelly ≈ {fk * 25:.2f}% ~ {fk * 50:.2f}%")

rng = np.random.default_rng(7)
BL = 20
nb = int(np.ceil(len(dR) / BL))
sims = []
for _ in range(2000):
    st = rng.integers(0, len(dR) - BL, nb)
    sims.append(np.concatenate([dR[s:s + BL] for s in st])[:len(dR)])
sims = np.array(sims)
p(f"\n  {'每筆風險':>8s} | {'歷史 年化':>9s} {'歷史 MDD':>8s} | {'抽樣 年化中位':>11s} {'MDD 中位':>8s} {'MDD 95%':>8s} {'P(MDD>20%)':>10s} {'P(>30%)':>8s} {'P(>50%)':>8s}")
for f in (0.0005, 0.001, 0.0015, 0.002, 0.003, 0.005, 0.0075, 0.01):
    fe, fm = path(f, dR)
    res = np.array([path(f, s) for s in sims])
    cagr = res[:, 0] ** (1 / yrs) - 1
    md = res[:, 1]
    p(f"  {f * 100:7.2f}% | {fe ** (1 / yrs) - 1:+9.0%} {fm:8.1%} | {np.median(cagr):+11.0%} {np.median(md):8.1%} {np.quantile(md, .95):8.1%} "
      f"{(md > .2).mean():10.0%} {(md > .3).mean():8.0%} {(md > .5).mean():8.0%}")
p(f"\n  注意：同時最多 8 筆，但一天內會陸續進出很多筆，歷史最差一天是 {dR.min():+.0f}R（已扣滑價）。")
open("results_part35.txt", "w").write("\n".join(lines) + "\n")
