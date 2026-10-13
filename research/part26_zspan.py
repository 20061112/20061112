"""第二十六部分：z 長度 × EWMA span（速度平滑）怎麼調。

固定：門檻 −2.5（首次跌破）、停損 3 ATR、不設停利、持有 120 根（第二十五部分基準）。
掃：span ∈ {5,10,15,20,30,40,60}（v 的 EWMA）× z 長度 ∈ {360,720,1380,2880,7200} × z 算法 {滾動, EWMA}。
另做：同頻率比較（每個 span 把門檻調到每日筆數與基準相同）、逐年前推（每年用之前的資料挑 span × z 長度）。
輸出：results_part26.txt、part26_grid.csv、part26_heatmap.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from load import load

SPANS = (5, 10, 15, 20, 30, 40, 60)
ZWS = (360, 720, 1380, 2880, 7200)
THR, SL, HOLD = 2.5, 3.0, 120
SPLIT = pd.Timestamp("2025-01-01")
out = []


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    out.append(s)


m = load("data/XAUUSD_M1_2023_2026.csv")
c = m.close
pc = c.shift()
tr = pd.concat([m.high - m.low, (m.high - pc).abs(), (m.low - pc).abs()], axis=1).max(axis=1)
atr = tr.rolling(60).mean()
gap = m.index.to_series().diff() > pd.Timedelta(minutes=6)
dlt = c.diff().where(~gap)
o, h, l, cl, A = m.open.to_numpy(), m.high.to_numpy(), m.low.to_numpy(), c.to_numpy(), atr.to_numpy()
spr = m.spread.to_numpy() * 0.01
ts = m.index
tsn = ts.to_numpy()
ndays = ((ts - pd.Timedelta(hours=1)).normalize()).nunique()


def zcalc(span, zw, kind):
    v = dlt.ewm(span=span, adjust=False).mean() / atr
    if kind == "滾動":
        z = (v - v.rolling(zw).mean()) / v.rolling(zw).std()
    else:
        z = (v - v.ewm(span=zw, adjust=False).mean()) / v.ewm(span=zw, adjust=False).std()
    return z.to_numpy()


def trades(Z, thr):
    trig = Z <= -thr
    pos = np.flatnonzero(trig & ~np.r_[False, trig[:-1]])
    res, busy = [], -1
    for t in pos:
        if t < 7500 or t <= busy or t + HOLD + 1 >= len(m) or not np.isfinite(A[t]):
            continue
        e, x = t + 1, t + HOLD
        if tsn[x] - tsn[t] > np.timedelta64(int(HOLD * 1.6 + 30), "m"):
            continue
        ep, sd = o[e], SL * A[t]
        sp = ep - sd
        hit = np.flatnonzero(l[e:x + 1] <= sp)
        if len(hit):
            xi = e + hit[0]
            xp = min(o[xi], sp) if xi > e else sp
        else:
            xi, xp = x, cl[x]
        busy = xi
        pnl = xp - ep - spr[e]
        res.append((tsn[t], pnl / sd, pnl / ep * 1e4))
    return pd.DataFrame(res, columns=["t", "R", "bp"])


def st(T):
    if len(T) < 30:
        return dict(n=len(T))
    R = T.R
    eq = R.cumsum()
    i, oo = T[T.t < SPLIT].R, T[T.t >= SPLIT].R
    yr = R.groupby(T.t.dt.year).mean()
    return dict(n=len(T), 每日=len(T) / ndays, R=R.mean(), bp=T.bp.mean(), t=R.mean() / R.std() * np.sqrt(len(R)),
                IS=i.mean(), OOS=oo.mean(), DD=(eq - eq.cummax()).min(), 最差年=yr.min(), **{f"y{k}": v for k, v in yr.items()})


rows, Zc = [], {}
for kind in ("滾動", "EWMA"):
    for span in SPANS:
        for zw in ZWS:
            Z = zcalc(span, zw, kind)
            Zc[(kind, span, zw)] = Z
            rows.append(dict(z算法=kind, span=span, z長度=zw, **st(trades(Z, THR))))
G = pd.DataFrame(rows)
G.round(4).to_csv("part26_grid.csv", index=False, encoding="utf-8-sig")

p("=" * 110)
p("第二十六部分：span（速度的 EWMA）× z 長度 × z 算法；門檻 −2.5、停損 3 ATR、持有 120")
p("  span 的半衰期 ≈ ln2 / ln(1/(1−α))，α = 2/(span+1)：span 5 ≈ 2.4 分、10 ≈ 3.5 分、20 ≈ 6.9 分、40 ≈ 13.5 分、60 ≈ 20.4 分")
p("  z 長度：360 = 6 小時、720 = 半天、1380 = 1 天、2880 = 2 天、7200 = 5 天（M1 根數）")
p("=" * 110)
for kind in ("滾動", "EWMA"):
    S = G[G.z算法 == kind]
    for val, lab in (("R", "全期每筆 R"), ("IS", "樣本內每筆 R"), ("OOS", "樣本外每筆 R"), ("最差年", "最差一年每筆 R"), ("每日", "每日筆數")):
        p(f"\n--- {kind} z：{lab}（列 = span，欄 = z 長度）")
        p(S.pivot(index="span", columns="z長度", values=val).round(3).to_string())

p("\n--- 單一參數平均（滾動 + EWMA 全部格子）")
for k in ("span", "z長度", "z算法"):
    p(G.groupby(k)[["R", "IS", "OOS", "最差年", "每日"]].mean().round(3).to_string())
    p("")

# ---------------------------------------------------------------- 同頻率比較：每個 span 調門檻讓每日筆數 ≈ 基準
p("--- 同頻率：每個 span 把門檻調到每日筆數 ≈ 基準（滾動 z 1380、span 20 的每日筆數），只比較訊號品質")
base_n = G[(G.z算法 == "滾動") & (G.span == 20) & (G.z長度 == 1380)].每日.iloc[0]
for span in SPANS:
    Z = Zc[("滾動", span, 1380)]
    best = None
    for thr in np.arange(1.75, 4.01, 0.05):
        T = trades(Z, thr)
        d = abs(len(T) / ndays - base_n)
        if best is None or d < best[0]:
            best = (d, thr, T)
    s = st(best[2])
    p(f"  span {span:2d}：門檻 {best[1]:.2f} 每日 {s['每日']:.2f} 每筆 {s['R']:+.3f}R (t {s['t']:+.2f}) IS {s['IS']:+.3f} OOS {s['OOS']:+.3f} 最差年 {s['最差年']:+.3f}")

# ---------------------------------------------------------------- 逐年前推
p("\n--- 逐年前推：每年年初用『之前全部』的交易挑 span × z 長度（滾動 z，依每筆 R），套用到當年")
cache = {(span, zw): trades(Zc[("滾動", span, zw)], THR) for span in SPANS for zw in ZWS}
wf, fixed = [], []
for y in (2024, 2025, 2026):
    start = pd.Timestamp(f"{y}-01-01")
    score = {k: T[T.t < start].R.mean() for k, T in cache.items() if (T.t < start).sum() >= 200}
    pick = max(score, key=score.get)
    Ty = cache[pick][(cache[pick].t >= start) & (cache[pick].t < pd.Timestamp(f"{y + 1}-01-01"))]
    Tb = cache[(20, 1380)][(cache[(20, 1380)].t >= start) & (cache[(20, 1380)].t < pd.Timestamp(f"{y + 1}-01-01"))]
    rk = pd.Series(score).rank(ascending=False)
    p(f"  {y}：挑到 span {pick[0]} / z {pick[1]}（過去每筆 {score[pick]:+.3f}R）→ 當年 {Ty.R.mean():+.3f}R（{len(Ty)} 筆）；"
      f"固定 span 20 / z 1380 當年 {Tb.R.mean():+.3f}R（過去排名第 {int(rk[(20, 1380)])} / {len(rk)}）")
    wf.append(Ty)
    fixed.append(Tb)
W, F = pd.concat(wf), pd.concat(fixed)
p(f"  2024~2026 合計：前推挑選 {W.R.mean():+.3f}R（{len(W)} 筆）vs 固定 {F.R.mean():+.3f}R（{len(F)} 筆）")

# ---------------------------------------------------------------- 圖
fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
for a_, (kind, val, title) in zip(ax, (("滾動", "IS", "rolling z: in-sample R/trade"), ("滾動", "OOS", "rolling z: out-of-sample R/trade"),
                                       ("EWMA", "OOS", "EWMA z: out-of-sample R/trade"))):
    P = G[G.z算法 == kind].pivot(index="span", columns="z長度", values=val)
    im = a_.imshow(P.values, cmap="RdYlGn", vmin=-0.2, vmax=0.3, aspect="auto")
    a_.set_xticks(range(len(P.columns)), P.columns)
    a_.set_yticks(range(len(P.index)), P.index)
    a_.set_xlabel("z length (M1 bars)")
    a_.set_ylabel("EWMA span")
    a_.set_title(title)
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            a_.text(j, i, f"{P.values[i, j]:+.2f}", ha="center", va="center", fontsize=8)
fig.colorbar(im, ax=ax, shrink=0.8)
fig.savefig("part26_heatmap.png", dpi=110, bbox_inches="tight")
open("results_part26.txt", "w").write("\n".join(out) + "\n")
