"""第三十部分：急跌買放到 M3 / M5 / M15 / M30 / H1，以及和 M1 版本的相關性。

每個週期 k 分鐘：
  v = EWMA(收盤變化, span 20 根) / ATR60（該週期）；z = 滾動 max(1380/k, 240) 根
  訊號：該週期 K 棒收盤時 z 首次 ≤ −門檻 → 下一分鐘開盤做多（= 下一根該週期 K 棒開盤）
  停損：進場價 − 停損倍數 × 該週期 ATR60；出場：持有 H 分鐘，用 M1 逐根判斷停損（較精確）
  同時只持有一筆；成本 = 進場那分鐘的點差
掃：門檻 {2.5, 3.0} × 停損 {3, 6} × 持有 {120, 480, 1440 分}；樣本內 < 2025、樣本外 ≥ 2025，依樣本內 t 挑每個週期的代表設定。
相關性：各週期代表設定 vs M1 定案的「日損益（R）」相關；以及等風險合併的效果。
輸出：results_part30.txt、part30_grid.csv、part30_equity.png
"""
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from load import load

TFS = (1, 3, 5, 15, 30, 60)
THRS, SLS, HOLDS = (2.5, 3.0), (3.0, 6.0), (120, 480, 1440)
SPLIT = pd.Timestamp("2025-01-01")
M1_FINAL = (1, 2.5, 3.0, 120)
out = []


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    out.append(s)


m = load("data/XAUUSD_M1_2023_2026.csv")
o1, h1, l1, c1 = (m[k].to_numpy() for k in ("open", "high", "low", "close"))
spr = m.spread.to_numpy() * 0.01
ts = m.index
tsn = ts.to_numpy()
tday = ((ts - pd.Timedelta(hours=1)).normalize()).to_numpy()
WARM = pd.Timestamp("2023-07-01")


def tf_signals(k):
    if k == 1:
        b = m
    else:
        b = m.resample(f"{k}min", label="left", closed="left").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    c = b.close
    pc = c.shift()
    tr = pd.concat([b.high - b.low, (b.high - pc).abs(), (b.low - pc).abs()], axis=1).max(axis=1)
    atr = tr.rolling(60).mean()
    gap = b.index.to_series().diff() > pd.Timedelta(minutes=k + 5 if k < 60 else 180)
    v = c.diff().where(~gap).ewm(span=20, adjust=False).mean() / atr
    zw = max(1380 // k, 240)
    z = (v - v.rolling(zw).mean()) / v.rolling(zw).std()
    close_time = b.index + pd.Timedelta(minutes=k)                 # 該 K 棒收盤（= 下一根開盤）時間
    return pd.DataFrame({"z": z.to_numpy(), "atr": atr.to_numpy()}, index=close_time)


def run(S, thr, sl, hold):
    z = S.z.to_numpy()
    trig = z <= -thr
    first = trig & ~np.r_[False, trig[:-1]]
    sig = S[first & np.isfinite(S.atr.to_numpy()) & (S.index >= WARM)]
    e_idx = ts.searchsorted(sig.index)                              # 下一分鐘開盤進場
    rows, busy = [], -1
    for (t_sig, r), e in zip(sig.iterrows(), e_idx):
        if e >= len(m) - hold - 1 or e <= busy:
            continue
        x = e + hold - 1
        if tsn[x] - tsn[e] > np.timedelta64(int(hold * 1.6 + 60), "m"):
            continue
        ep, sd = o1[e], sl * r.atr
        sp = ep - sd
        hit = np.flatnonzero(l1[e:x + 1] <= sp)
        if len(hit):
            xi = e + hit[0]
            xp = min(o1[xi], sp) if xi > e else sp
        else:
            xi, xp = x, c1[x]
        busy = xi
        pnl = xp - ep - spr[e]
        rows.append((tsn[e], tday[e], pnl / sd, pnl / ep * 1e4))
    return pd.DataFrame(rows, columns=["t", "day", "R", "bp"])


def st(T):
    if len(T) < 30:
        return dict(n=len(T))
    i, o = T[T.t < SPLIT], T[T.t >= SPLIT]
    d = T.groupby("day").R.sum()
    eq = T.R.cumsum()
    return dict(n=len(T), R=T.R.mean(), bp=T.bp.mean(), t=T.R.mean() / T.R.std() * np.sqrt(len(T)), 勝率=(T.R > 0).mean(),
                IS_R=i.R.mean(), IS_t=i.R.mean() / i.R.std() * np.sqrt(len(i)), OOS_R=o.R.mean(), OOS_t=o.R.mean() / o.R.std() * np.sqrt(len(o)),
                日Sharpe=d.mean() / d.std() * np.sqrt(252), DD=(eq - eq.cummax()).min(),
                **{f"y{y}": v for y, v in T.R.groupby(pd.DatetimeIndex(T.t).year).mean().items()})


rows, trades = [], {}
for k in TFS:
    S = tf_signals(k)
    for thr, sl, hold in itertools.product(THRS, SLS, HOLDS):
        if k == 1 and hold == 1440:
            pass
        T = run(S, thr, sl, hold)
        trades[(k, thr, sl, hold)] = T
        rows.append(dict(週期=f"M{k}", k=k, 門檻=thr, 停損=sl, 持有分=hold, **st(T)))
    print("done", k)
G = pd.DataFrame(rows)
G.round(4).to_csv("part30_grid.csv", index=False, encoding="utf-8-sig")

p("=" * 120)
p("第三十部分：急跌買 × 週期（訊號在各週期收盤、出場用 M1 逐根）；門檻 {2.5,3} × 停損 {3,6}×該週期 ATR60 × 持有 {2,8,24 小時}")
p("=" * 120)
cols = ["週期", "門檻", "停損", "持有分", "n", "R", "bp", "t", "勝率", "IS_R", "IS_t", "OOS_R", "OOS_t", "日Sharpe", "DD", "y2023", "y2024", "y2025", "y2026"]
p("\n--- 全部組合")
p(G[cols].round(3).to_string(index=False))
p("\n--- 每個週期：組合平均（樣本內 / 樣本外每筆 R、為正比例）")
p(G.groupby("週期", sort=False).agg(IS=("IS_R", "mean"), OOS=("OOS_R", "mean"), OOS為正=("OOS_R", lambda s: (s > 0).mean()),
                                    每日筆數=("n", lambda s: s.mean() / 855)).round(3).to_string())

# 每個週期的代表設定：依樣本內 t 挑（M1 直接用定案）
pick = {1: M1_FINAL}
for k in TFS[1:]:
    r = G[G.k == k].sort_values("IS_t", ascending=False).iloc[0]
    pick[k] = (k, r.門檻, r.停損, int(r.持有分))
p("\n--- 每個週期的代表設定（樣本內 t 最高；M1 = 定案）")
for k, key in pick.items():
    r = G[(G.k == key[0]) & (G.門檻 == key[1]) & (G.停損 == key[2]) & (G.持有分 == key[3])].iloc[0]
    p(f"  M{k:<3d} 門檻 {key[1]} 停損 {key[2]} 持有 {key[3]} 分：n={int(r.n)} 每筆 {r.R:+.3f}R {r.bp:+.2f}bp (t {r.t:+.2f}) "
      f"IS {r.IS_R:+.3f} → OOS {r.OOS_R:+.3f}（t {r.OOS_t:+.2f}） 日Sharpe {r.日Sharpe:.2f} 回撤 {r.DD:.0f}R")

# ---------------------------------------------------------------- 相關性
days = pd.Index(np.unique(tday[ts.searchsorted(WARM):]))
D = pd.DataFrame({f"M{k}": trades[key].groupby("day").R.sum().reindex(days).fillna(0) for k, key in pick.items()})
p("\n--- 日損益（R）相關係數（代表設定）")
p(D.corr().round(2).to_string())
W = D.resample("W").sum()
p("\n--- 週損益相關係數")
p(W.corr().round(2).to_string())
p("\n--- 交易重疊：其他週期的進場時間，落在 M1 持倉期間內的比例")
m1T = trades[M1_FINAL]
m1_hold = np.zeros(len(m), bool)
for t in m1T.t:
    e = ts.searchsorted(t)
    m1_hold[e:e + 120] = True
for k, key in pick.items():
    if k == 1:
        continue
    T = trades[key]
    p(f"  M{k}：{m1_hold[ts.searchsorted(T.t)].mean():.0%}")

p("\n--- 合併（每個週期每筆 1R 風險、日損益相加）")


def comb(cs):
    d = D[cs].sum(axis=1)
    eq = d.cumsum()
    return f"日Sharpe {d.mean() / d.std() * np.sqrt(252):.2f} 總 {d.sum():+.0f}R 回撤 {(eq - eq.cummax()).min():.0f}R 總/回撤 {d.sum() / -(eq - eq.cummax()).min():.1f}"


p(f"  M1 單獨               {comb(['M1'])}")
for k in TFS[1:]:
    p(f"  M1 + M{k:<3d}           {comb(['M1', f'M{k}'])}  | M{k} 單獨 {comb([f'M{k}'])}")
p(f"  全部                  {comb(list(D.columns))}")
p(f"  M1 + M15 + M60        {comb(['M1', 'M15', 'M60'])}")

fig, ax = plt.subplots(1, 2, figsize=(15, 5))
for c_ in D.columns:
    ax[0].plot(D.index, D[c_].cumsum(), label=c_)
ax[0].axvline(SPLIT, color="k", ls="--", lw=0.8)
ax[0].axhline(0, color="grey", lw=0.8)
ax[0].set_title("dip-buy by timeframe: cumulative R (representative configs)")
ax[0].legend(fontsize=8)
im = ax[1].imshow(D.corr().values, cmap="RdBu_r", vmin=-1, vmax=1)
ax[1].set_xticks(range(len(D.columns)), D.columns)
ax[1].set_yticks(range(len(D.columns)), D.columns)
for i in range(len(D.columns)):
    for j in range(len(D.columns)):
        ax[1].text(j, i, f"{D.corr().values[i, j]:.2f}", ha="center", va="center", fontsize=9)
ax[1].set_title("daily P&L correlation")
fig.colorbar(im, ax=ax[1], shrink=0.8)
fig.tight_layout()
fig.savefig("part30_equity.png", dpi=110)
open("results_part30.txt", "w").write("\n".join(out) + "\n")
