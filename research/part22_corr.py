"""第二十二部分：NAS100 與 XAUUSD 配對交易 — 第一步：相關係數。
用法：python3 part22_corr.py data/NAS100_M15_2023_2026.csv [data/XAUUSD_M15_2023_2026.csv]
兩個檔案都是 MT5 匯出格式（同一家券商、伺服器時間），只用兩邊都有報價的 K 棒。

檢查項目：
  1. 價格水準 vs 報酬率的相關（價格相關是假象，報酬才是真的）
  2. 多週期報酬相關：M15 / H1 / H4 / D1 / W1，Pearson 與 Spearman
  3. 滾動相關（日報酬 20/60/120 日）：穩不穩、正負切換多頻繁
  4. 逐年、逐月、分時段（亞洲/倫敦/紐約）
  5. 領先落後（XAU_t vs NAS_{t+k}）
  6. 極端日（NAS 大跌 / 大漲）時的條件相關
  7. 共整合（Engle-Granger）、避險比率、價差半衰期 — 配對交易真正需要的是這個
"""
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint, adfuller
from load import load

warnings.filterwarnings("ignore", category=FutureWarning)

NAS_PATH = sys.argv[1] if len(sys.argv) > 1 else "data/NAS100_M15_2023_2026.csv"
XAU_PATH = sys.argv[2] if len(sys.argv) > 2 else "data/XAUUSD_M15_2023_2026.csv"

nas, xau = load(NAS_PATH)["close"], load(XAU_PATH)["close"]
px = pd.concat({"nas": nas, "xau": xau}, axis=1).dropna()
lp = np.log(px)
print(f"資料：NAS {nas.index[0]} ~ {nas.index[-1]} ({len(nas)} 根)，XAU {xau.index[0]} ~ {xau.index[-1]} ({len(xau)} 根)")
print(f"共同 K 棒：{len(px)} 根，{px.index[0]} ~ {px.index[-1]}")


def rets(rule):
    p = lp if rule is None else lp.resample(rule).last().dropna()
    return p.diff().dropna()


def corr_line(r):
    return f"Pearson {r['nas'].corr(r['xau']):+.3f}  Spearman {r['nas'].corr(r['xau'], method='spearman'):+.3f}  n={len(r)}"


print("\n## 1. 價格水準 vs 報酬")
d = lp.resample("1D").last().dropna()
print(f"  日收盤 log 價格相關：{d['nas'].corr(d['xau']):+.3f}   ← 兩個都在漲就會很高，不代表會一起動")
print(f"  日報酬相關        ：{d.diff()['nas'].corr(d.diff()['xau']):+.3f}")

print("\n## 2. 多週期報酬相關")
for name, rule in [("M15", None), ("H1", "1h"), ("H4", "4h"), ("D1", "1D"), ("W1", "W-FRI")]:
    print(f"  {name:4s} {corr_line(rets(rule))}")

print("\n## 3. 滾動相關（日報酬）")
rd = rets("1D")
roll = {}
for w in (20, 60, 120):
    c = rd["nas"].rolling(w).corr(rd["xau"]).dropna()
    roll[w] = c
    flips = (np.sign(c).diff().iloc[1:] != 0).sum()
    print(f"  {w:3d} 日：中位數 {c.median():+.2f}  10%~90% [{c.quantile(.1):+.2f}, {c.quantile(.9):+.2f}]  "
          f"最低 {c.min():+.2f} 最高 {c.max():+.2f}  >0 佔 {(c > 0).mean():.0%}  正負翻轉 {flips} 次")

print("\n## 4a. 逐年（H1 報酬 / D1 報酬）")
rh = rets("1h")
for y in sorted(set(rh.index.year)):
    a, b = rh[rh.index.year == y], rd[rd.index.year == y]
    print(f"  {y}: H1 {a['nas'].corr(a['xau']):+.3f} (n={len(a)})   D1 {b['nas'].corr(b['xau']):+.3f} (n={len(b)})")

print("\n## 4b. 逐月（H1 報酬）")
m = rh.groupby(rh.index.to_period("M")).apply(lambda g: g["nas"].corr(g["xau"]))
print(f"  月相關 中位數 {m.median():+.2f}，>0 的月份 {(m > 0).sum()}/{len(m)}，最低 {m.min():+.2f} ({m.idxmin()})，最高 {m.max():+.2f} ({m.idxmax()})")
print("  " + "  ".join(f"{k}:{v:+.2f}" for k, v in m.items()))

print("\n## 4c. 分時段（M15 報酬，伺服器時間小時）")
r15 = rets(None)
r15 = r15[r15.index.to_series().diff().le(pd.Timedelta("15min")).values]  # 去掉跨休市的報酬
for name, hrs in [("亞洲 01-09", range(1, 9)), ("倫敦 09-15", range(9, 15)), ("紐約 15-23", range(15, 23)), ("尾盤 23-01", [23, 0])]:
    g = r15[r15.index.hour.isin(list(hrs))]
    print(f"  {name}: {corr_line(g)}")

print("\n## 5. 領先落後 corr(XAU_t, NAS_t+k)")
for name, r in [("M15", r15), ("H1", rh)]:
    print(f"  {name}: " + "  ".join(f"k={k:+d}:{r['xau'].corr(r['nas'].shift(-k)):+.3f}" for k in range(-3, 4)))

print("\n## 6. 極端日（D1）")
s = rd["nas"].std()
for name, mask in [("NAS 跌 >1.5σ", rd["nas"] < -1.5 * s), ("NAS 漲 >1.5σ", rd["nas"] > 1.5 * s), ("其餘", rd["nas"].abs() <= 1.5 * s)]:
    g = rd[mask]
    print(f"  {name}: n={len(g):3d}  相關 {g['nas'].corr(g['xau']):+.3f}  當日 XAU 平均 {g['xau'].mean() * 100:+.2f}%  XAU 同向比例 {(np.sign(g['xau']) == np.sign(g['nas'])).mean():.0%}")

print("\n## 7. 共整合（log 日收盤）")
def eg(p, label):
    beta = np.polyfit(p["xau"], p["nas"], 1)[0]
    spread = p["nas"] - beta * p["xau"]
    t, pv, _ = coint(p["nas"], p["xau"])
    ds = spread.diff().dropna()
    lag = spread.shift().loc[ds.index]
    phi = np.polyfit(lag - lag.mean(), ds, 1)[0]
    hl = -np.log(2) / np.log(1 + phi) if phi < 0 else np.inf
    print(f"  {label}: 避險比 β={beta:+.2f}  EG t={t:+.2f} p={pv:.3f}  價差 ADF p={adfuller(spread)[1]:.3f}  半衰期 {hl:.0f} 日  n={len(p)}")
    return spread
spread = eg(d, "全期")
for y in sorted(set(d.index.year)):
    g = d[d.index.year == y]
    if len(g) > 60:
        eg(g, str(y))
print("  （p<0.05 才算共整合；半衰期太長代表價差不會在可交易的時間內回歸）")

fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
(px / px.iloc[0]).resample("1D").last().plot(ax=ax[0], title="Normalized price (start = 1)")
for w, c in roll.items():
    c.plot(ax=ax[1], label=f"{w}d", lw=1)
ax[1].axhline(0, color="k", lw=.5); ax[1].legend(); ax[1].set_title("Rolling correlation of daily returns")
((spread - spread.mean()) / spread.std()).plot(ax=ax[2], title="Log-price spread z-score (full-sample beta)")
ax[2].axhline(0, color="k", lw=.5)
plt.tight_layout(); plt.savefig("part22_corr.png", dpi=110)
pd.concat({f"roll{w}": c for w, c in roll.items()}, axis=1).to_csv("part22_rolling_corr.csv")
