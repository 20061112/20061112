"""第三十九部分：最終版的大回撤期間都發生了什麼？

交易：final_m15_trades.csv（最終版實際下單，只扣點差），0.02 口 = 淨利_每盎司 × 2。
回撤段：日結權益從高點到谷底（取前 8 大、深度 ≥ 600 美元），列出：
  期間、深度、持續、回到新高天數；
  市場：金價變化、期間高低、50 日均線斜率、價格是否穿越 50 日均線、日 ER10、日 ATR 相對前 60 日、
        日內效率（|收−開| / (高−低)，越低 = 越來回）、隔日反轉比例（今天方向和昨天相反）；
  交易：筆數/天、做多比例、勝率、停損比例、平均 R、最差一天。
最後和「非回撤期間」的同樣指標比較，找共同點。
"""
import numpy as np
import pandas as pd
import part24_prev_bar_breakout as B

T = pd.read_csv("final_m15_trades.csv", parse_dates=["進場時間", "出場時間"])
T["usd"] = T.淨利_每盎司 * 2
T["date"] = T.進場時間.dt.normalize()
dl = B.daily.copy()
dl.index = pd.DatetimeIndex(dl.index)
dl["sma50"] = dl.C.rolling(50).mean()
dl["slope"] = dl.sma50 - dl.sma50.shift(10)
dl["atr"] = (dl.H - dl.L).rolling(10).mean()
dl["atr_rel"] = dl.atr / dl.atr.shift(10).rolling(60).mean()
dl["er10"] = (dl.C - dl.C.shift(10)).abs() / dl.C.diff().abs().rolling(10).sum()
dl["eff"] = (dl.C - dl.O).abs() / (dl.H - dl.L)
dl["rev"] = (np.sign(dl.C - dl.O) != np.sign(dl.C.shift(1) - dl.O.shift(1))).astype(float)
dl["above"] = dl.C > dl.sma50

d = T.groupby("date").usd.sum()
d = d.reindex(pd.DatetimeIndex(dl.index[(dl.index >= d.index.min()) & (dl.index <= d.index.max())]), fill_value=0.0)
eq = d.cumsum()
peak = eq.cummax()
dd = peak - eq

# 找回撤段
eps = []
used = pd.Series(False, eq.index)
for _ in range(12):
    x = dd[~used]
    if x.empty or x.max() < 600:
        break
    tr = x.idxmax()
    pk = eq[:tr].idxmax()
    after = eq[tr:]
    rec = after[after >= eq[pk]]
    end = rec.index[0] if len(rec) else None
    eps.append((pk, tr, end, dd[tr]))
    used[pk:(end if end is not None else eq.index[-1])] = True
eps = sorted(eps, key=lambda e: e[0])[:8] if len(eps) > 8 else sorted(eps, key=lambda e: e[0])

lines = []


def p(s=""):
    print(s)
    lines.append(s)


def ctx(a, b):
    """a < 日期 ≤ b 的市場與交易狀態。"""
    m = (dl.index > a) & (dl.index <= b)
    g = dl[m]
    t = T[(T.date > a) & (T.date <= b)]
    pre = dl[dl.index <= a].iloc[-1]
    cross = int((g.above != g.above.shift(1)).iloc[1:].sum()) + int(g.above.iloc[0] != pre.above) if len(g) else 0
    out = dict(
        天數=len(g), 金價變化=(g.C.iloc[-1] / pre.C - 1) * 100 if len(g) else np.nan,
        區間_ATR=(g.H.max() - g.L.min()) / pre.atr if len(g) else np.nan,
        淨移動_ATR=abs(g.C.iloc[-1] - pre.C) / pre.atr if len(g) else np.nan,
        斜率同號=(np.sign(g.slope) == np.sign(pre.slope)).mean() if len(g) else np.nan,
        穿越50MA=cross, ER10=g.er10.mean(), ATR相對=g.atr_rel.mean(), 日內效率=g.eff.mean(), 隔日反轉=g.rev.mean(),
        筆每天=len(t) / max(t.date.nunique(), 1), 做多=(t.方向 == "多").mean(), 勝率=(t.R > 0).mean(),
        停損比例=(t.出場原因 == "停損").mean(), 平均R=t.R.mean(), 損益=t.usd.sum(),
        最差日=t.groupby("date").usd.sum().min() if len(t) else np.nan,
        最差日期=t.groupby("date").usd.sum().idxmin().date() if len(t) else None,
    )
    return out


p("第三十九部分：最終版（0.02 口，只扣點差）的大回撤期間")
p(f"日結 MDD {dd.max():,.0f}；回撤段（深度 ≥ 600 美元）共 {len(eps)} 段\n")
rows = []
for pk, tr, end, amt in eps:
    c = ctx(pk, tr)
    rec_days = (end - tr).days if end is not None else None
    rows.append(c)
    p(f"■ {pk.date()} → {tr.date()}　深度 {amt:,.0f}　下跌 {(tr - pk).days} 天　{'回到新高 ' + str(end.date()) + f'（再 {rec_days} 天）' if end is not None else '尚未回到新高'}")
    p(f"   市場：金價 {c['金價變化']:+.1f}%，期間高低 {c['區間_ATR']:.1f} 個日 ATR、淨移動只有 {c['淨移動_ATR']:.1f} 個；"
      f"穿越 50 日均線 {c['穿越50MA']} 次；ER10 {c['ER10']:.2f}；日 ATR 相對前 60 日 {c['ATR相對']:.2f}；"
      f"日內效率 {c['日內效率']:.2f}；隔日反轉 {c['隔日反轉']:.0%}")
    p(f"   交易：{c['筆每天']:.1f} 筆/天，做多 {c['做多']:.0%}，勝率 {c['勝率']:.0%}，停損 {c['停損比例']:.0%}，平均 {c['平均R']:+.2f}R，"
      f"期間損益 {c['損益']:+,.0f}，最差一天 {c['最差日']:+,.0f}（{c['最差日期']}）")

# 非回撤期間
in_dd = pd.Series(False, eq.index)
for pk, tr, end, amt in eps:
    in_dd[(eq.index > pk) & (eq.index <= tr)] = True
norm_days = eq.index[~in_dd]
g = dl.loc[dl.index.isin(norm_days)]
t = T[T.date.isin(norm_days)]
gd = dl.loc[dl.index.isin(eq.index[in_dd])]
td = T[T.date.isin(eq.index[in_dd])]
p("\n=== 回撤期間 vs 其他期間 ===")
p(f"{'':12s} {'ER10':>6s} {'日ATR相對':>9s} {'日內效率':>8s} {'隔日反轉':>8s} {'勝率':>6s} {'停損比例':>8s} {'平均R':>7s} {'筆/天':>6s} {'做多':>6s}")
for lab, gg, tt in (("回撤期間", gd, td), ("其他期間", g, t)):
    p(f"{lab:12s} {gg.er10.mean():6.2f} {gg.atr_rel.mean():9.2f} {gg.eff.mean():8.2f} {gg.rev.mean():8.0%} {(tt.R > 0).mean():6.0%} "
      f"{(tt.出場原因 == '停損').mean():8.0%} {tt.R.mean():+7.2f} {len(tt) / max(tt.date.nunique(), 1):6.1f} {(tt.方向 == '多').mean():6.0%}")

# 圖
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True, gridspec_kw=dict(height_ratios=[2, 2, 1]))
    ax[0].plot(eq.index, eq.values, color="tab:blue"); ax[0].set_ylabel("cum USD (0.02 lot)"); ax[0].grid(alpha=.3)
    m = (dl.index >= eq.index.min())
    ax[1].plot(dl.index[m], dl.C[m], color="goldenrod", label="XAUUSD")
    ax[1].plot(dl.index[m], dl.sma50[m], color="gray", lw=1, label="SMA50"); ax[1].legend(); ax[1].grid(alpha=.3)
    ax[2].plot(dl.index[m], dl.er10[m].rolling(5).mean(), color="tab:green"); ax[2].set_ylabel("ER10 (5d avg)"); ax[2].grid(alpha=.3)
    for pk, tr, end, amt in eps:
        for a_ in ax:
            a_.axvspan(pk, tr, color="tab:red", alpha=.15)
    ax[0].set_title("Final M15 breakout: drawdown episodes (red)")
    plt.tight_layout(); plt.savefig("part39_dd_episodes.png", dpi=110)
except ImportError:
    pass
open("results_part39.txt", "w").write("\n".join(lines) + "\n")

# 回撤開始時的位置：是否剛創高、離 50 日均線多遠
dl["hi60"] = dl.H.rolling(60).max()
dl["dist50"] = (dl.C - dl.sma50) / dl.atr
dl["run20"] = (dl.C - dl.C.shift(20)) / dl.atr
out = ["", "=== 回撤開始那天（高點日）的位置 ==="]
for pk, tr, end, amt in eps:
    r = dl.loc[pk]
    out.append(f"  {pk.date()}：收盤離 60 日最高 {(r.C / r.hi60 - 1) * 100:+.1f}%，離 50 日均線 {r.dist50:+.1f} 個日 ATR，前 20 日漲 {r.run20:+.1f} 個日 ATR")
allq = dl.loc[eq.index]
out.append(f"  對照（全部交易日）：離 60 日最高中位數 {((allq.C / allq.hi60 - 1) * 100).median():+.1f}%，離 50 日均線中位數 {allq.dist50.median():+.1f} ATR，前 20 日漲中位數 {allq.run20.median():+.1f} ATR")
out.append(f"  全部交易日中，離 50 日均線 > 4 ATR 的比例 {(allq.dist50 > 4).mean():.0%}")
print("\n".join(out))
open("results_part39.txt", "a").write("\n".join(out) + "\n")

# 事前規則：前一日「離 50 日均線」或「前 20 日漲幅」過大時減半 / 不做（只用前一日資料）
prev = dl[["dist50", "run20"]].shift(1)
T = T.join(prev, on="date")
T["ext50"] = T.dist50 * np.where(T.方向 == "多", 1, -1)
T["ext20"] = T.run20 * np.where(T.方向 == "多", 1, -1)
T["train"] = T.進場時間 < "2025-01-01"


def rat(x, w):
    dd_ = (x.usd * w).groupby(x.date).sum(); e = dd_.cumsum(); m = (e.cummax() - e).max()
    return (x.usd * w).sum(), m


out = ["", "=== 事前規則：漲過頭時減倉（0.02 口美元；比值 = 總損益 / 日結 MDD）==="]
yrs = (T.進場時間.max() - T.進場時間.min()).days / 365
for lab, w in [("最終版", np.ones(len(T)))] + [
        (f"離 50MA > {k} ATR 時{nm}", np.where(T.ext50 > k, v, 1.0)) for k in (3, 4, 5, 6) for nm, v in (("減半", .5), ("不做", 0.0))] + [
        (f"前 20 日漲 > {k} ATR 時{nm}", np.where(T.ext20 > k, v, 1.0)) for k in (4, 5, 6) for nm, v in (("減半", .5), ("不做", 0.0))]:
    w = np.asarray(w, float)
    a = rat(T[T.train.values], w[T.train.values]); b = rat(T[~T.train.values], w[~T.train.values]); c = rat(T, w)
    out.append(f"  {lab:22s} 總 {c[0]:+7,.0f}  MDD {c[1]:6,.0f}  年化/MDD {c[0] / yrs / c[1]:4.2f} | 訓練比值 {a[0] / a[1]:4.1f}  驗證比值 {b[0] / b[1]:4.1f}")
print("\n".join(out))
open("results_part39.txt", "a").write("\n".join(out) + "\n")
