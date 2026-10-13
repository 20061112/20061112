"""第二十三部分：v 的 z-score 事件，換不同週期（M1 / M5 / M15 / H1）+ 停損 + 樣本外。

每個週期：v = ewm(收盤價變化, span 20) / ATR60（該週期的 ATR60），z = 滾動 W 根（約等於 M1 的 1~10 天，見 ZWIN）。
訊號：急跌買（z≤−thr 做多）、急漲追多（z≥+thr 做多）、急跌追空（z≤−thr 做空）、急漲做空（z≥+thr 做空）。
進場：z 首次越過門檻 → 下一根開盤；出場：持有 hold 根後收盤，或停損（stop × ATR，觸價即出，價格 = 停損價）。
同一訊號同時只有一筆（冷卻 = 持有期間）。成本：進場那分鐘的點差。
損益單位：bp（÷ 進場價 × 1e4），可以跨年、跨週期比較。去漂移：減「過去 20 個交易日每分鐘平均報酬 × 持有分鐘」。
樣本內 2023/6 ~ 2024/12 挑參數；樣本外 2025/1 ~ 2026/10 只看結果。
輸出：results_part23.txt、part23_grid.csv、part23_best_trades.csv、part23_equity.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from load import load

TFS = {"M1": 1, "M5": 5, "M15": 15, "H1": 60}
ZWIN = {"M1": 1380, "M5": 576, "M15": 480, "H1": 240}     # 約 1 天 / 2 天 / 5 天 / 10 天
THRS = (2.0, 2.5, 3.0)
HOLDS = (3, 12, 48)                                         # 根數（隨週期放大）
STOPS = (None, 3.0, 6.0)                                    # × 該週期 ATR60
SIGS = {"急跌買": (-1, 1), "急漲追多": (1, 1), "急跌追空": (-1, -1), "急漲做空": (1, -1)}   # (觸發方向, 交易方向)
SPLIT = pd.Timestamp("2025-01-01")
out = []


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    out.append(s)


m1 = load("data/XAUUSD_M1_2023_2026.csv")
gap1 = m1.index.to_series().diff() > pd.Timedelta(minutes=5)
r1 = (m1.close.diff() / m1.close.shift() * 1e4).where(~gap1)                 # bp / 分鐘
drift_bp = r1.rolling(27600, min_periods=5000).mean()                        # 過去 20 交易日每分鐘平均（bp）


def build(tf):
    k = TFS[tf]
    if k == 1:
        b = m1.copy()
    else:
        b = m1.resample(f"{k}min", label="left", closed="left").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "spread": "first"}).dropna()
    c = b.close
    pc = c.shift()
    tr = pd.concat([b.high - b.low, (b.high - pc).abs(), (b.low - pc).abs()], axis=1).max(axis=1)
    b["atr"] = tr.rolling(60).mean()
    gap = b.index.to_series().diff() > pd.Timedelta(minutes=k + 5 if k < 60 else 180)
    d = c.diff().where(~gap)
    v = d.ewm(span=20, adjust=False).mean() / b.atr
    W = ZWIN[tf]
    b["z"] = (v - v.rolling(W).mean()) / v.rolling(W).std()
    b["drift"] = drift_bp.reindex(b.index, method="ffill")
    return b


def run(b, tf, sig, thr, hold, stop, keep=False):
    k = TFS[tf]
    zdir, side = SIGS[sig]
    z = b.z.to_numpy()
    o, h, l, c, a, spr, dr = (b[x].to_numpy() for x in ("open", "high", "low", "close", "atr", "spread", "drift"))
    ts = b.index.to_numpy()
    trig = (z <= -thr) if zdir == -1 else (z >= thr)
    prev = np.r_[False, trig[:-1]]
    pos = np.flatnonzero(trig & ~prev)                                       # 首次越過
    rows, busy = [], -1
    maxspan = np.timedelta64(int(hold * k * 1.6) + 30, "m")
    for t in pos:
        if t <= busy or t + 1 + hold >= len(b) or not np.isfinite(a[t]) or not np.isfinite(dr[t]):
            continue
        e, x = t + 1, t + hold
        if ts[x] - ts[t] > maxspan:                                          # 跨週末
            continue
        ep = o[e]
        xp, xi = c[x], x
        if stop is not None:
            sp = ep - side * stop * a[t]
            hit = np.flatnonzero((l[e:x + 1] <= sp) if side == 1 else (h[e:x + 1] >= sp))
            if len(hit):
                xi = e + hit[0]
                xp = sp
        busy = xi
        mins = (ts[xi] - ts[e]) / np.timedelta64(1, "m") + k
        pnl = (side * (xp - ep) - spr[e] * 0.01) / ep * 1e4
        rows.append((ts[t], ts[e], ep, ts[xi], xp, pnl, pnl - side * dr[t] * mins, xp == (ep - side * stop * a[t]) if stop else False))
    T = pd.DataFrame(rows, columns=["觸發時間", "進場時間", "進場價", "出場時間", "出場價", "損益_bp", "去漂移_bp", "停損"])
    return T


def stats(T):
    if len(T) < 30:
        return dict(n=len(T), mean=np.nan, t=np.nan, dd=np.nan)
    r = T.損益_bp
    eq = r.cumsum()
    return dict(n=len(T), mean=r.mean(), t=r.mean() / r.std() * np.sqrt(len(r)), x=T.去漂移_bp.mean(),
                pf=r[r > 0].sum() / -r[r <= 0].sum(), win=(r > 0).mean(), dd=(eq - eq.cummax()).min(), tot=r.sum())


rows = []
bars = {}
for tf in TFS:
    b = build(tf)
    bars[tf] = b
    for sig in SIGS:
        for thr in THRS:
            for hold in HOLDS:
                for stop in STOPS:
                    T = run(b, tf, sig, thr, hold, stop)
                    I, O = T[T.觸發時間 < SPLIT], T[T.觸發時間 >= SPLIT]
                    si, so = stats(I), stats(O)
                    rows.append(dict(週期=tf, 訊號=sig, 門檻=thr, 持有根數=hold, 持有分鐘=hold * TFS[tf], 停損=stop or 0,
                                     IS_n=si["n"], IS_bp=si["mean"], IS_t=si["t"], IS_去漂移=si.get("x"),
                                     OOS_n=so["n"], OOS_bp=so["mean"], OOS_t=so["t"], OOS_去漂移=so.get("x"),
                                     OOS_PF=so.get("pf"), OOS_勝率=so.get("win"), OOS_DD_bp=so["dd"]))
    print("done", tf)
G = pd.DataFrame(rows)
G.round(3).to_csv("part23_grid.csv", index=False, encoding="utf-8-sig")

p("=" * 120)
p("第二十三部分：v 的 z-score 事件 × 週期（M1/M5/M15/H1）× 門檻 × 持有 × 停損；樣本內 2023/6~2024/12、樣本外 2025/1~2026/10")
p(f"  共 {len(G)} 組；損益單位 bp（扣點差）；去漂移 = 減過去 20 日平均漂移")
p("=" * 120)
ok = G.dropna(subset=["IS_t", "OOS_t"])
p(f"樣本內 t 與樣本外 t 的排名相關（全部組合）：{ok.IS_t.rank().corr(ok.OOS_t.rank()):+.2f}")
p(f"樣本內 t > 2 的組合 {(ok.IS_t > 2).sum()} 組，其中樣本外也 > 0 的比例 {(ok[ok.IS_t > 2].OOS_bp > 0).mean():.0%}")
p("")
p("--- 每個週期 × 訊號：各組合平均（樣本內 / 樣本外 bp，括號為去漂移）")
agg = ok.groupby(["週期", "訊號"]).agg(IS=("IS_bp", "mean"), IS_x=("IS_去漂移", "mean"), OOS=("OOS_bp", "mean"),
                                         OOS_x=("OOS_去漂移", "mean"), OOS為正=("OOS_bp", lambda s: (s > 0).mean()))
p(agg.round(2).to_string())
p("")
p("--- 每個週期：樣本內 t 最高的 3 組 → 樣本外")
cols = ["週期", "訊號", "門檻", "持有分鐘", "停損", "IS_n", "IS_bp", "IS_t", "OOS_n", "OOS_bp", "OOS_t", "OOS_去漂移", "OOS_PF", "OOS_DD_bp"]
for tf in TFS:
    p(ok[ok.週期 == tf].nlargest(3, "IS_t")[cols].round(2).to_string(index=False))
    p("")
p("--- 全部組合中，樣本內 t 最高的 10 組 → 樣本外")
best10 = ok.nlargest(10, "IS_t")
p(best10[cols].round(2).to_string(index=False))

# ---------------------------------------------------------------- 最佳（只依樣本內選）與基準的曲線
best = best10.iloc[0]
pick = [("樣本內最佳", best.週期, best.訊號, best.門檻, int(best.持有根數), best.停損 or None),
        ("第二十二部分基準（無停損）", "M1", "急跌買", 2.5, 48, None)]
for tf in ("M1", "M5", "M15"):
    r = ok[(ok.週期 == tf)].nlargest(1, "IS_t").iloc[0]
    pick.append((f"{tf} 樣本內最佳", tf, r.訊號, r.門檻, int(r.持有根數), r.停損 or None))
fig, ax = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
allT = []
p("")
p("--- 曲線用的設定（全期）")
for lbl, tf, sig, thr, hold, stop in pick:
    T = run(bars[tf], tf, sig, thr, hold, stop)
    T.insert(0, "設定", f"{lbl}: {tf} {sig} z{'≤−' if SIGS[sig][0] < 0 else '≥+'}{thr} 持有{hold}根 停損{stop or '無'}")
    allT.append(T)
    s = stats(T)
    yr = T.groupby(T.觸發時間.dt.year).損益_bp.mean()
    p(f"{T.設定.iloc[0]:60s} n={s['n']} 每筆 {s['mean']:+.2f}bp t={s['t']:+.2f} PF {s['pf']:.2f} 勝率 {s['win']:.0%} "
      f"總 {s['tot']:+.0f}bp 回撤 {s['dd']:.0f}bp | 逐年 " + " ".join(f"{y}:{v:+.2f}" for y, v in yr.items()))
    name = f"{tf} {'dip-buy' if sig == '急跌買' else 'spike-long' if sig == '急漲追多' else 'dip-short' if sig == '急跌追空' else 'spike-short'} thr{thr} hold{hold} stop{stop or '-'}"
    ax[0].plot(T.觸發時間, T.損益_bp.cumsum(), label=name)
    ax[1].plot(T.觸發時間, T.去漂移_bp.cumsum(), label=name)
for a_, t_ in zip(ax, ("cumulative P&L (bp, after spread)", "cumulative P&L minus trailing 20d drift (bp)")):
    a_.axhline(0, color="grey", lw=0.8)
    a_.axvline(SPLIT, color="k", ls="--", lw=0.8)
    a_.set_title(t_)
    a_.legend(fontsize=8)
fig.tight_layout()
fig.savefig("part23_equity.png", dpi=110)
pd.concat(allT, ignore_index=True).round({"進場價": 3, "出場價": 3, "損益_bp": 3, "去漂移_bp": 3}).to_csv("part23_best_trades.csv", index=False, encoding="utf-8-sig")
open("results_part23.txt", "w").write("\n".join(out) + "\n")
