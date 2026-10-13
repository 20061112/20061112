"""第二十四部分：
A. 第二十三部分「走最短光程 + 超光速」那格（W=60，t 2.5）拆開看：改成獨立事件（不重疊）、以天為單位算 t、逐年 / 多空 / 新價位 / 時段、
   門檻與視窗的鄰近參數、和簡單對照組比較、事件後平均路徑。
B. 對更多東西取 d（一階、二階導數），系統性掃描和之後行情的關係。
輸出：results_part24.txt、part24_cell_path.png
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from load import load
import physics_features as pf
from fermat_core import _fermat

out = open("results_part24.txt", "w")


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    out.write(s + "\n")
    out.flush()


def day_t(x, when):
    """以天聚合後的 t（同一天多筆事件不當成獨立樣本）。"""
    s = pd.Series(np.asarray(x), index=pd.DatetimeIndex(when)).dropna()
    g = s.groupby(s.index.date).mean()
    return g.mean() / g.std() * np.sqrt(len(g)) if len(g) > 2 else np.nan


def summ(E, col="c60"):
    x = E[col].dropna()
    return pd.Series({"n": len(x), "平均": x.mean(), "中位": x.median(), "勝率": (x > 0).mean(),
                      "t(逐筆)": x.mean() / x.std() * np.sqrt(len(x)), "t(以天)": day_t(x, E.loc[x.index, "time"])})


d = pf.base(load("data/XAUUSD_M1_2023_2026.csv"))
hi, lo, cl, atr = d.high.values, d.low.values, d.close.values, d.atr.values
seg = d.seg.values
N = len(d)
HMAX = 120
fwd = {h: (np.r_[cl[h:], np.full(h, np.nan)] - cl) / atr for h in (15, 30, 60, 120)}
same = {h: np.r_[seg[h:], np.full(h, -1)] == seg for h in (15, 30, 60, 120)}
for h in fwd:
    fwd[h][~same[h]] = np.nan
lmin = {h: pd.Series(lo).rolling(h).min().shift(-h).values for h in (60,)}
hmax = {h: pd.Series(hi).rolling(h).max().shift(-h).values for h in (60,)}

STEP = 5
cache = {}


def window(W):
    if W in cache:
        return cache[W]
    ts = np.arange(1440 + W + 1, N - HMAX - 1, STEP)
    ts = ts[seg[ts - W] == seg[ts]]
    onet, opath, n_now, n_start = _fermat(hi, lo, cl, atr, ts.astype(np.int64), W, 1440, 0.1, 1.0)
    net = cl[ts] - cl[ts - W]
    path = pd.Series(np.abs(np.diff(cl, prepend=cl[0]))).rolling(W).sum().values[ts]
    X = pd.DataFrame({"t": ts, "time": d.index[ts], "PhiP": onet / opath, "Phit": onet / W, "ER": np.abs(net) / path,
                      "speed": np.abs(net) / (atr[ts] * W), "net": net / atr[ts], "n_now": n_now, "n_start": n_start})
    X["side"] = np.sign(X.net)
    for h in (15, 30, 60, 120):
        X[f"c{h}"] = X.side * fwd[h][ts]
    X["mae60"] = np.where(X.side > 0, cl[ts] - lmin[60][ts], hmax[60][ts] - cl[ts]) / atr[ts]
    cache[W] = X
    return X


def events(X, cond, W, cool=None):
    """條件第一次成立（上一個取樣點不成立）才算事件，之後 cool 根內不再算。"""
    cool = cool or W
    c = cond.values & (X.net.abs().values >= 1.0)
    on = c & ~np.r_[False, c[:-1]]
    keep, last = [], -10 ** 9
    for i in np.where(on)[0]:
        t = X.t.values[i]
        if t - last >= cool:
            keep.append(i)
            last = t
    return X.iloc[keep].copy()


# ================================================================== A
P("=" * 110 + "\nA. 「走最短光程（Φ_P ≥ 0.5）+ 超光速（Φ_t ≥ 1）」W=60 那格的細節\n" + "=" * 110)
X = window(60)
samp = X[(X.t % 15 == X.t.iloc[0] % 15)]  # 重現第二十三部分（每 15 根取樣、會重疊）
c23 = samp[(samp.PhiP >= 0.5) & (samp.Phit >= 1) & (samp.net.abs() >= 1)]
P("A-0 第二十三部分的寫法（每 15 根取樣，同一段行情會被重複計算）：")
P(summ(c23.assign(time=c23.time)).round(3).to_string())
E = events(X, (X.PhiP >= 0.5) & (X.Phit >= 1), 60)
E["yr"] = E.time.dt.year
E["新價位"] = E.n_now <= 1.5
E["hour"] = E.time.dt.hour
P(f"\nA-1 改成獨立事件（條件第一次成立、60 根冷卻）：{len(E)} 筆，分布在 {E.time.dt.date.nunique()} 天")
tab = pd.DataFrame({h: summ(E, f"c{h}") for h in (15, 30, 60, 120)}).T
P("  順著視窗方向的後續報酬（ATR）：")
P(tab.round(3).to_string())
P(f"  60 根內最大逆向（MAE）中位 {E.mae60.median():.2f} ATR，平均 {E.mae60.mean():.2f}")
P("\nA-2 逐年 / 多空 / 新價位 / 時段（c60）")
for k, lab in (("yr", "年"), ("side", "方向(+1 多 / −1 空)"), ("新價位", "現價在過去 1 天剖面外")):
    P(f"  依{lab}")
    P(E.groupby(k).apply(lambda g: summ(g)).round(3).to_string())
E["時段"] = pd.cut(E.hour, [0, 9, 15, 19, 24], right=False, labels=["亞洲 0-9", "歐洲 9-15", "美盤 15-19", "美盤後 19-24"])
P("  依時段（伺服器時間）")
P(E.groupby("時段", observed=True).apply(lambda g: summ(g)).round(3).to_string())
E["月"] = E.time.dt.to_period("M")
m = E.groupby("月").c60.sum()
P(f"  按月累計：正月份 {(m > 0).sum()}/{len(m)}；最好的 3 個月貢獻總和的 {m.nlargest(3).sum() / m.sum():.0%}")

P("\nA-3 鄰近參數（獨立事件；c60 平均 / 以天 t / 筆數）—— 只有一格好看就是巧合")
rows = []
for W in (30, 45, 60, 90):
    XW = window(W)
    for pp in (0.4, 0.5, 0.6):
        for pt in (1, 2, 3):
            e = events(XW, (XW.PhiP >= pp) & (XW.Phit >= pt), W)
            s = summ(e)
            rows.append({"W": W, "Φ_P≥": pp, "Φ_t≥": pt, "n": int(s.n), "c60": s["平均"], "t天": s["t(以天)"]})
G = pd.DataFrame(rows)
P(G.pivot_table(index=["W", "Φ_P≥"], columns="Φ_t≥", values="c60").round(3).to_string())
P("  以天 t：")
P(G.pivot_table(index=["W", "Φ_P≥"], columns="Φ_t≥", values="t天").round(2).to_string())
P("  筆數：")
P(G.pivot_table(index=["W", "Φ_P≥"], columns="Φ_t≥", values="n").to_string())
P(f"  36 組中 c60 > 0 的有 {(G.c60 > 0).sum()} 組，以天 t > 2 的有 {(G.t天 > 2).sum()} 組")

P("\nA-4 對照組（同樣的獨立事件寫法）")
base = {
    "普通 ER ≥ 0.3 且速度前 20%": (X.ER >= 0.3) & (X.speed >= X.speed.quantile(0.8)),
    "只要 Φ_t ≥ 1（不管路徑）": X.Phit >= 1,
    "只要 Φ_P ≥ 0.5（不管時間）": X.PhiP >= 0.5,
    "現價在剖面外（新價位）": X.n_now <= 1.5,
    "現價在剖面外 且 ER ≥ 0.3": (X.n_now <= 1.5) & (X.ER >= 0.3),
    "全部 |淨移動| ≥ 1 ATR": X.net.abs() >= 0,
}
P(pd.DataFrame({k: summ(events(X, v, 60)) for k, v in base.items()}).T.round(3).to_string())
Eall = events(X, X.net.abs() >= 0, 60)
P("  對照組「全部 |淨移動| ≥ 1 ATR」依方向（檢查多空不對稱是不是黃金大漲的趨勢造成）：")
P(Eall.groupby("side").apply(lambda g: summ(g)).round(3).to_string())
P(f"  費馬那格 vs 「現價在剖面外 且 ER≥0.3」重疊：{E.t.isin(events(X, base['現價在剖面外 且 ER ≥ 0.3'], 60).t).mean():.0%} 的事件相同")

# 事件後平均路徑圖
path = np.array([[(cl[t + k] - cl[t]) / atr[t] * s if seg[t + k] == seg[t] else np.nan for k in range(HMAX + 1)]
                 for t, s in zip(E.t.values, E.side.values)])
Eb = events(X, X.net.abs() >= 0, 60)
pathb = np.array([[(cl[t + k] - cl[t]) / atr[t] * s if seg[t + k] == seg[t] else np.nan for k in range(HMAX + 1)]
                  for t, s in zip(Eb.t.values, Eb.side.values)])
plt.rcParams["font.family"] = "WenQuanYi Zen Hei"
plt.rcParams["axes.unicode_minus"] = False
fig, ax = plt.subplots(figsize=(8.5, 4.8), facecolor="#fcfcfb")
ax.set_facecolor("#fcfcfb")
for arr, col, lab in ((path, "#2a78d6", f"走最短光程 + 超光速（{len(E)} 筆）"), (pathb, "#eb6834", f"對照：所有 |淨移動|≥1 ATR（{len(Eb)} 筆）")):
    mu = np.nanmean(arr, 0)
    se = np.nanstd(arr, 0) / np.sqrt(np.sum(~np.isnan(arr), 0))
    ax.fill_between(range(HMAX + 1), mu - 2 * se, mu + 2 * se, color=col, alpha=0.15, lw=0)
    ax.plot(mu, color=col, lw=2, label=lab)
ax.axhline(0, color="#52514e", lw=0.8)
ax.axhline(0.09, color="#52514e", lw=0.8, ls=":")
ax.text(HMAX, 0.1, "點差 ≈ 0.09 ATR", ha="right", va="bottom", fontsize=9, color="#52514e")
ax.set_xlabel("事件後第幾根 M1", color="#52514e")
ax.set_ylabel("順視窗方向累計報酬（ATR）", color="#52514e")
ax.set_title("事件後平均路徑（陰影 = ±2 標準誤）", color="#0b0b0b", loc="left")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.tick_params(colors="#52514e")
ax.grid(axis="y", color="#e6e5e0", lw=0.6)
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig("part24_cell_path.png", dpi=130)

# ================================================================== B
P("\n" + "=" * 110 + "\nB. 對更多東西取 d：一階 / 二階導數掃描（每 15 根取樣，W=30 視窗）\n" + "=" * 110)
P("目標：r60 = 未來 60 根報酬（有方向）；c60 = 順最近 30 根方向的延續；vol = 未來 15 根路徑長（波動）")
P("每個 d 量算每日 Spearman IC；前半（2023/6~2024/12）與後半（2025~2026/10）分開，兩半同號且 |t|>2 才算穩定。")
X30 = window(30)
X30 = X30[(X30.t % 15) == (X30.t.iloc[0] % 15)].set_index("t")
X30p = window(30).set_index("t")  # 用來取 30 根前的值
t = X30.index.values
s = pd.Series
c = d.close
A = d.atr
lnv = np.log(d.rvol.clip(lower=1e-3))
sp = d.spread.astype(float)
PY = pf.poynting(d)
v5 = (c - c.shift(5)) / A / 5
v30 = (c - c.shift(30)) / A / 30
acc = v5 - v5.shift(5)
rv = d.ret.abs() / A
OFI = PY.H
side30 = np.sign(c - c.shift(30))
hh = d.high.rolling(240).max()
ll = d.low.rolling(240).min()
pos = (c - ll) / (hh - ll)  # 在 240 根區間的位置 0~1
lag = lambda f: pd.Series(f.reindex(X30p.index).values, index=X30p.index).reindex(t - 30).values
feat = {
    # 價格對時間
    "dP/dt (5根速度)": v5,
    "dP/dt (30根速度)": v30,
    "d²P/dt² (加速度)": acc,
    "d³P/dt³ (急動度)": acc - acc.shift(5),
    "d|P|/dt 路徑速度 (5根)": rv.rolling(5).mean(),
    "d(路徑速度)/dt": rv.rolling(5).mean() - rv.rolling(5).mean().shift(5),
    # 量
    "d lnV/dt (量的變化)": lnv.rolling(5).mean() - lnv.rolling(5).mean().shift(5),
    "d²lnV/dt²": (lnv.rolling(5).mean() - lnv.rolling(5).mean().shift(5)).diff(5),
    "dP/dV (每單位量推多少價)": (c - c.shift(30)) / A / d.rvol.rolling(30).sum(),
    "dV/d|P| (每 ATR 需要多少量)": d.rvol.rolling(30).sum() / ((c - c.shift(30)).abs() / A + 0.1),
    # 成本 / 波動
    "d spread/dt": (sp.rolling(5).mean() - sp.rolling(30).mean()) / sp.rolling(30).mean(),
    "d lnATR/dt": np.log(A) - np.log(A.shift(30)),
    "d²lnATR/dt²": (np.log(A) - np.log(A.shift(30))).diff(30),
    # 委託流 / 偏離
    "d OFI/dt (委託流變化)": OFI - OFI.shift(15),
    "dE/dt (偏離變化)": PY.E - PY.E.shift(5),
    "d pos/dt (區間位置變化)": pos - pos.shift(30),
    # 時間對價格
    "dt/dP (每 ATR 花幾根)": 30 / ((c - c.shift(30)).abs() / A + 0.1),
    "距 240 根極值多久 (dt)": None,
}
# 距離最近一次 240 根新高 / 新低的根數（有方向：最近是新高 → +，新低 → −）
isH = (d.high >= hh).values
isL = (d.low <= ll).values
lastH = pd.Series(np.where(isH, np.arange(N), np.nan)).ffill().values
lastL = pd.Series(np.where(isL, np.arange(N), np.nan)).ffill().values
ar = np.arange(N)
feat["距 240 根極值多久 (dt)"] = pd.Series(np.where(lastH > lastL, 1, -1) * (ar - np.fmax(lastH, lastL)), index=d.index)
F = pd.DataFrame({k: (v.values if hasattr(v, "values") else v)[t] for k, v in feat.items()}, index=d.index[t])
# 費馬量與它們的 d
F["Φ_P"] = X30.PhiP.values
F["Φ_t"] = X30.Phit.values
F["dΦ_P/dt"] = X30.PhiP.values - X30p.PhiP.reindex(t - 30).values
F["dΦ_t/dt"] = np.log(X30.Phit.values) - np.log(X30p.Phit.reindex(t - 30).values)
F["dn/dt (介質變稠/變稀)"] = np.log(X30.n_now.values) - np.log(X30.n_start.values)
F["dn/dP (順方向的介質梯度)"] = (np.log(X30.n_now.values) - np.log(X30.n_start.values)) / (np.abs(X30.net.values) + 0.1)
signed = {"dP/dt (5根速度)", "dP/dt (30根速度)", "d²P/dt² (加速度)", "d³P/dt³ (急動度)", "dP/dV (每單位量推多少價)",
          "d OFI/dt (委託流變化)", "dE/dt (偏離變化)", "d pos/dt (區間位置變化)", "距 240 根極值多久 (dt)"}
Y = pd.DataFrame({"r60": fwd[60][t], "c60": np.sign(X30.net.values) * fwd[60][t], "vol": pf.forward(d)["path15"].values[t]},
                 index=d.index[t])


def ic_split(x, y):
    df = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    ic = df.groupby(df.index.date).apply(lambda g: g.x.rank().corr(g.y.rank()) if len(g) > 10 else np.nan).dropna()
    ic.index = pd.to_datetime(ic.index)
    res = []
    for part in (ic, ic[ic.index < "2025-01-01"], ic[ic.index >= "2025-01-01"]):
        res += [part.mean(), part.mean() / part.std() * np.sqrt(len(part))]
    return res


rows = []
for k in F.columns:
    for tg in ("r60", "c60", "vol"):
        x = F[k]
        if tg == "c60" and k in signed:
            x = x * np.sign(X30.net.values)  # 有方向的量先轉成「順最近方向」
        m, tt, m1, t1, m2, t2 = ic_split(x, Y[tg])
        rows.append({"d 量": k, "目標": tg, "IC": m, "t": tt, "前半IC": m1, "前半t": t1, "後半IC": m2, "後半t": t2,
                     "穩定": (np.sign(t1) == np.sign(t2)) and abs(t1) > 2 and abs(t2) > 2})
R = pd.DataFrame(rows)
for tg, lab in (("r60", "未來 60 根報酬（方向）"), ("c60", "順最近 30 根方向的延續"), ("vol", "未來 15 根波動")):
    P(f"\n目標：{lab}（依 |t| 排序）")
    P(R[R.目標 == tg].drop(columns="目標").sort_values("t", key=abs, ascending=False).round(4).to_string(index=False))

P("\nB-2 方向類的穩定訊號扣掉「最簡單的對照」後還剩多少（r60 扣掉 30 根速度、c60 扣掉 ER、vol 扣掉最近 5 根路徑速度）")


def resid_on(x, cc):
    df = pd.DataFrame({"x": x, "c": cc}).replace([np.inf, -np.inf], np.nan).dropna()
    rx, rc = df.x.rank(), df.c.rank()
    return (rx - np.cov(rx, rc)[0, 1] / rc.var() * rc).reindex(x.index)


ctrl = {"r60": F["dP/dt (30根速度)"], "c60": pd.Series(X30.ER.values, index=F.index), "vol": F["d|P|/dt 路徑速度 (5根)"]}
rows = []
for _, r in R[R.穩定].iterrows():
    if r["d 量"] in ("dP/dt (30根速度)",) and r.目標 == "r60" or r["d 量"] == "d|P|/dt 路徑速度 (5根)" and r.目標 == "vol":
        continue
    x = F[r["d 量"]]
    if r.目標 == "c60" and r["d 量"] in signed:
        x = x * np.sign(X30.net.values)
    m, tt, m1, t1, m2, t2 = ic_split(resid_on(x, ctrl[r.目標]), Y[r.目標])
    rows.append({"d 量": r["d 量"], "目標": r.目標, "原IC": r.IC, "扣掉對照後IC": m, "t": tt, "前半t": t1, "後半t": t2})
if rows:
    P(pd.DataFrame(rows).round(4).to_string(index=False))
P("\nB-3 方向類穩定訊號的五分位（未來 60 根報酬，ATR；點差 ≈ 0.09）")
for k in ("dP/dt (30根速度)", "距 240 根極值多久 (dt)", "dP/dV (每單位量推多少價)", "d pos/dt (區間位置變化)"):
    df = pd.DataFrame({"x": F[k], "y": Y.r60}).dropna()
    b = pd.qcut(df.x.rank(method="first"), 5, labels=False) + 1
    g = df.groupby(b)
    P(f"  {k}")
    P(pd.DataFrame({"x中位": g.x.median(), "r60平均": g.y.mean(), "t天": [day_t(gg.y, gg.index) for _, gg in g], "n": g.size()}).round(3).to_string())
P(f"\n共測 {len(R)} 組（{F.shape[1]} 個 d 量 × 3 個目標），隨機情況下 |t|>2 約會出現 {0.046 * len(R):.0f} 組；"
  f"實際「兩半都 |t|>2 且同號」有 {R.穩定.sum()} 組。")
out.close()
