"""第二十五部分：M1 急跌買（停損 3 ATR、不設停利、持有 120 根）的進場樣貌、穩健性與失效情境。

A. 開單當下「速度」長什麼樣（前 5/15/30 分跌了多少、v、z、波動）
B. 穩健性：參數單一維度微調、單日虧損上限 / 連虧停手、日區塊 bootstrap 回撤分布
C. 什麼時候沒效：依趨勢、波動、新聞時段、當日第幾次訊號、是否破前日低、點差、近期績效分組（樣本內 / 樣本外分開看）
樣本內 < 2025-01-01、樣本外 ≥ 2025-01-01；單位 R（停損距離）與 bp。
輸出：results_part25.txt、part25_trades.csv、part25_robust.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from load import load

SPLIT = pd.Timestamp("2025-01-01")
BASE = dict(thr=2.5, sl=3.0, hold=120, zw=1380, span=20)
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
# 交易日 = broker 01:00 起算（與定案 A 相同）
tday = (ts - pd.Timedelta(hours=1)).normalize()
day_low = pd.Series(l, index=ts).groupby(tday).cummin().to_numpy()
day_open = pd.Series(o, index=ts).groupby(tday).transform("first").to_numpy()
dl = pd.Series(l, index=ts).groupby(tday).min()
prev_low = pd.Series(tday, index=ts).map(dl.shift()).to_numpy()
dc = pd.Series(cl, index=ts).groupby(tday).last()
do = pd.Series(o, index=ts).groupby(tday).first()
prev_ret = pd.Series(tday, index=ts).map(((dc - do) / do * 1e4).shift()).to_numpy()
atr_rel = (atr / atr.rolling(1380).mean()).to_numpy()
r1 = (c.diff() / c.shift() * 1e4).where(~gap)
mu = r1.rolling(27600, min_periods=5000).mean().to_numpy()
# 日線 ATR（前 10 個交易日）相對於過去 60 個交易日 → 大週期波動
dh = pd.Series(h, index=ts).groupby(tday).max()
drng = (dh - dl) / dc * 1e4
datr_rel = pd.Series(tday, index=ts).map((drng.rolling(10).mean() / drng.rolling(60).mean()).shift()).to_numpy()
zcache = {}


def zser(zw, span):
    k = (zw, span)
    if k not in zcache:
        v = dlt.ewm(span=span, adjust=False).mean() / atr
        zcache[k] = ((v - v.rolling(zw).mean()) / v.rolling(zw).std()).to_numpy(), v.to_numpy()
    return zcache[k]


def gen(thr, sl, hold, zw, span, feats=False):
    Z, V = zser(zw, span)
    trig = Z <= -thr
    pos = np.flatnonzero(trig & ~np.r_[False, trig[:-1]])
    rows, busy, nday, lastday = [], -1, 0, None
    for t in pos:
        if t + hold + 1 >= len(m) or not np.isfinite(A[t]) or t < 300:
            continue
        if tday[t] != lastday:
            nday, lastday = 0, tday[t]
        nday += 1                                                   # 當日第幾次訊號（含持倉中被忽略的）
        if t <= busy:
            continue
        e, x = t + 1, t + hold
        if ts[x] - ts[t] > pd.Timedelta(minutes=hold * 1.6 + 30):
            continue
        ep, sd = o[e], sl * A[t]
        sp = ep - sd
        hit = np.flatnonzero(l[e:x + 1] <= sp)
        if len(hit):
            xi = e + hit[0]
            xp, why = (min(o[xi], sp) if xi > e else sp), "停損"
        else:
            xi, xp, why = x, cl[x], "時間"
        busy = xi
        pnl = xp - ep - spr[e]
        r = dict(訊號時間=ts[t], 交易日=tday[t], 進場價=ep, 出場時間=ts[xi], 出場原因=why, 停損距離=sd,
                 損益_R=pnl / sd, 損益_bp=pnl / ep * 1e4, 去漂移_bp=pnl / ep * 1e4 - mu[t] * (xi - e + 1))
        if feats:
            r.update(z=Z[t], v=V[t], ATR60=A[t],
                     跌5分_ATR=(cl[t] - cl[t - 5]) / A[t], 跌15分_ATR=(cl[t] - cl[t - 15]) / A[t], 跌30分_ATR=(cl[t] - cl[t - 30]) / A[t],
                     跌5分_美元=cl[t] - cl[t - 5], 跌15分_美元=cl[t] - cl[t - 15], 跌15分_bp=(cl[t] / cl[t - 15] - 1) * 1e4,
                     前4小時_bp=(cl[t] / cl[t - 240] - 1) * 1e4, 今日_bp=(cl[t] / day_open[t] - 1) * 1e4, 前日_bp=prev_ret[t],
                     ATR相對1天=atr_rel[t], 日波動相對=datr_rel[t], 在今日低點=cl[t] - day_low[t] <= 0.5 * A[t],
                     破前日低=cl[t] < prev_low[t], 當日第幾次=nday, 點差_R=spr[e] / sd, 時=ts[e].hour, 分=ts[e].minute)
        rows.append(r)
    T = pd.DataFrame(rows)
    T["樣本"] = np.where(T.訊號時間 < SPLIT, "IS", "OOS")
    return T


def line(T, label):
    R = T.損益_R
    eq = R.cumsum()
    t = R.mean() / R.std() * np.sqrt(len(R)) if len(R) > 2 else np.nan
    i, oo = T[T.樣本 == "IS"].損益_R, T[T.樣本 == "OOS"].損益_R
    return (f"{label:34s} n={len(T):5d} 每筆 {R.mean():+.3f}R {T.損益_bp.mean():+.2f}bp (t {t:+.2f}) 勝率 {(R > 0).mean():.0%} "
            f"| IS {i.mean():+.3f} OOS {oo.mean():+.3f} | 總 {R.sum():+.0f}R 回撤 {(eq - eq.cummax()).min():.0f}R")


T = gen(**BASE, feats=True)
T.round(4).to_csv("part25_trades.csv", index=False, encoding="utf-8-sig")
p("=" * 120)
p("第二十五部分：M1 急跌買（z_v20 首次 ≤ −2.5、停損 3 ATR、不設停利、持有 120 根）")
p("=" * 120)
p(line(T, "基準"))

# ================================================================ A
p("\n--- A. 開單當下的樣子（中位數 [10% ~ 90%]）")
for k in ("z", "v", "跌5分_ATR", "跌15分_ATR", "跌30分_ATR", "跌5分_美元", "跌15分_美元", "跌15分_bp", "ATR60", "ATR相對1天", "前4小時_bp", "今日_bp"):
    s = T[k]
    p(f"  {k:10s} {s.median():+9.2f}  [{s.quantile(.1):+9.2f} ~ {s.quantile(.9):+9.2f}]")
p(f"  訊號時剛好在今日低點附近（0.5 ATR 內）：{T.在今日低點.mean():.0%}；跌破前一日低點：{T.破前日低.mean():.0%}")
p("  逐年 15 分鐘跌幅中位（美元 / bp）：" + " ".join(
    f"{y}: {g.跌15分_美元.median():+.2f} / {g.跌15分_bp.median():+.1f}" for y, g in T.groupby(T.訊號時間.dt.year)))

# ================================================================ B1 參數微調
p("\n--- B1. 參數單一維度微調（其他維持基準）")
grid = dict(thr=(2.0, 2.25, 2.5, 2.75, 3.0), sl=(2.0, 2.5, 3.0, 3.5, 4.0), hold=(60, 90, 120, 180, 240),
            zw=(720, 1380, 2880, 7200), span=(10, 15, 20, 30, 40))
nb = []
for k, vals in grid.items():
    for val in vals:
        cfg = dict(BASE, **{k: val})
        S = gen(**cfg)
        p(line(S, f"{k}={val}"))
        nb.append((k, val, S.groupby("樣本").損益_bp.mean().to_dict()))

# ================================================================ B2 單日停手
p("\n--- B2. 單日虧損上限 / 連虧停手（觸發後當天不再開新單；以交易日 01:00 切）")


def daily_rule(T, max_loss=None, max_consec=None, max_trades=None):
    keep = []
    for _, g in T.groupby("交易日", sort=True):
        cum, consec, n = 0.0, 0, 0
        for i, r in g.iterrows():
            if (max_loss is not None and cum <= -max_loss) or (max_consec is not None and consec >= max_consec) or \
               (max_trades is not None and n >= max_trades):
                break
            keep.append(i)
            cum += r.損益_R
            n += 1
            consec = consec + 1 if r.損益_R <= 0 else 0
    return T.loc[keep]


for lbl, kw in [("無限制", {}), ("單日虧 ≥ 2R 停手", dict(max_loss=2)), ("單日虧 ≥ 3R 停手", dict(max_loss=3)),
                ("單日虧 ≥ 4R 停手", dict(max_loss=4)), ("單日虧 ≥ 5R 停手", dict(max_loss=5)),
                ("連虧 2 筆停手", dict(max_consec=2)), ("連虧 3 筆停手", dict(max_consec=3)), ("連虧 4 筆停手", dict(max_consec=4)),
                ("每日最多 3 筆", dict(max_trades=3)), ("每日最多 5 筆", dict(max_trades=5))]:
    S = daily_rule(T, **kw)
    dd = S.groupby("交易日").損益_R.sum()
    p(line(S, lbl) + f" | 最差日 {dd.min():+.1f}R")
p("  單日損益分布（R）：" + " ".join(f"{q:.0%}:{T.groupby('交易日').損益_R.sum().quantile(q):+.1f}" for q in (.01, .05, .1, .5, .9, .95, .99)))
cl_ = (T.損益_R <= 0).astype(int)
runs = cl_.groupby((cl_ != cl_.shift()).cumsum()).sum()
p("  連虧長度分布：" + " ".join(f"≥{k}:{(runs >= k).sum()}次" for k in (5, 8, 10, 12, 15, 20)) + f"，最長 {runs.max()}")
p("  連虧之後下一筆的表現：" + " ".join(
    f"連虧≥{k}後 {T.損益_R[(cl_.groupby((cl_ != cl_.shift()).cumsum()).cumsum().shift() >= k)].mean():+.3f}R" for k in (3, 5, 8)))

# ================================================================ B3 bootstrap
p("\n--- B3. 回撤分布（以交易日為單位重抽 2000 次，長度同全期）")
rng = np.random.default_rng(0)
dR = T.groupby("交易日").損益_R.sum().to_numpy()
mdd, tot = [], []
for _ in range(2000):
    s = rng.choice(dR, len(dR), replace=True)
    eq = np.cumsum(s)
    mdd.append((eq - np.maximum.accumulate(eq)).min())
    tot.append(eq[-1])
mdd, tot = np.array(mdd), np.array(tot)
p(f"  最大回撤 R：中位 {np.median(mdd):.0f}，95% 分位 {np.quantile(mdd, .05):.0f}，99% 分位 {np.quantile(mdd, .01):.0f}；實際 {((T.損益_R.cumsum()) - T.損益_R.cumsum().cummax()).min():.0f}")
p(f"  總 R ≤ 0 的機率 {(tot <= 0).mean():.1%}；總 R 5% 分位 {np.quantile(tot, .05):+.0f}")
p(f"  → 每筆風險 r%：回撤 ≈ r × R；例 0.2% × 99% 分位 {np.quantile(mdd, .01):.0f}R = {0.2 * np.quantile(mdd, .01):.0f}%")

# ================================================================ C 失效情境
p("\n--- C. 什麼時候沒效（每格：筆數、每筆 R、IS / OOS 每筆 R、去漂移 bp）")


def split(col, bins=None, labels=None, q=None):
    s = T[col]
    if q:
        g = pd.qcut(s, q, duplicates="drop")
    elif bins is not None:
        g = pd.cut(s, bins, labels=labels)
    else:
        g = s
    rows = []
    for key, G in T.groupby(g, observed=True):
        rows.append(dict(分組=str(key), n=len(G), R=G.損益_R.mean(), IS=G[G.樣本 == "IS"].損益_R.mean(),
                         OOS=G[G.樣本 == "OOS"].損益_R.mean(), 去漂移bp=G.去漂移_bp.mean(), 勝率=(G.損益_R > 0).mean()))
    p(f"[{col}]")
    p(pd.DataFrame(rows).round(3).to_string(index=False))


split("前4小時_bp", q=5)
split("今日_bp", q=5)
split("前日_bp", q=5)
split("ATR相對1天", q=5)
split("日波動相對", q=4)
split("跌15分_ATR", q=5)
split("z", bins=[-99, -4, -3.5, -3, -2.5])
split("當日第幾次", bins=[0, 1, 2, 3, 5, 99], labels=["1", "2", "3", "4-5", "6+"])
split("在今日低點")
split("破前日低")
split("點差_R", q=4)
T["新聞窗"] = np.select([(T.時 == 15) & (T.分 >= 25) | (T.時 == 16) & (T.分 < 5), (T.時 == 21) & (T.分 < 40), (T.時 == 17) & (T.分 < 20)],
                      ["15:25-16:05（美數據 8:30 NY）", "21:00-21:40（FOMC 14:00 NY）", "17:00-17:20（10:00 NY 數據）"], "其他")
split("新聞窗")
T["時段"] = pd.cut(T.時, [0, 4, 8, 11, 14, 17, 20, 24], right=False, labels=["00-03", "04-07", "08-10", "11-13", "14-16", "17-19", "20-23"])
split("時段")
# 近期績效（只用已出場的單）→ 是否該暫停
T["前30筆"] = T.損益_R.rolling(30).sum().shift()
split("前30筆", q=5)
T["前20日"] = T.訊號時間.map(T.set_index("訊號時間").損益_R.rolling("20D").sum().shift())
split("前20日", q=5)

# ---------------------------------------------------------------- 圖
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
ax[0].hist(mdd, bins=50, color="tab:red", alpha=0.7)
ax[0].axvline(((T.損益_R.cumsum()) - T.損益_R.cumsum().cummax()).min(), color="k", ls="--")
ax[0].set_title("bootstrap max drawdown (R); dashed = actual")
roll = T.set_index("訊號時間").損益_R.rolling("90D").mean()
ax[1].plot(roll.index, roll.values)
ax[1].axhline(0, color="grey", lw=0.8)
ax[1].set_title("rolling 90-day avg R per trade")
g = T.groupby(pd.qcut(T["前4小時_bp"], 10)).損益_R.mean()
ax[2].bar(range(len(g)), g.values, color=np.where(g.values > 0, "tab:green", "tab:red"))
ax[2].set_title("avg R by prior-4h return decile (left = fell most)")
fig.tight_layout()
fig.savefig("part25_robust.png", dpi=110)
open("results_part25.txt", "w").write("\n".join(out) + "\n")
