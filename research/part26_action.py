"""第二十六部分：路徑作用量策略原型 — 向上 / 向下 / 區間三條最優路徑的作用量差，是否能預測可交易的淨報酬。

設計（細節見 path_action.py）：
  每 15 根取樣；H = 30 根；終點 ±D（D = √H，約 1 個 H 根的標準差）或 0（區間）。
  狀態：因果三次多項式（最後 15 根）估端點 v0、a0、j0；μ = 0 或 240 根回歸斜率。
  介質 R(x) = log(過去 1 天停留密度)：非二次的部分，是唯一可能超出「線性動能濾波」的地方。
  訊號 C = (A↓ − A↑)/(τ + |A↓| + |A↑|)。
樣本切分：2023/6 ~ 2024/12 選參數（樣本內），2025/1 ~ 2026/10 凍結驗證（樣本外）。
評估：日 IC、|C| 前 20% 才交易的扣點差淨報酬（以天計 t）、逐年；並與「只看速度」等對照比較。
輸出：results_part26.txt
"""
import warnings
warnings.filterwarnings("ignore")
import itertools
import time
import numpy as np
import pandas as pd
from load import load
import physics_features as pf
from path_action import Action, initial_state, endpoint_derivs, profiles, interp_rows, NB, GRID_STEP

out = open("results_part26.txt", "w")


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    out.write(s + "\n")
    out.flush()


H, STEP = 30, 15
SPLIT = pd.Timestamp("2025-01-01")
d = pf.base(load("data/XAUUSD_M1_2023_2026.csv"))
cl, hi, lo = d.close.values, d.high.values, d.low.values
seg = d.seg.values
N = len(d)
s_all = d.close.diff().rolling(60).std().to_numpy(copy=True)
s_all[s_all <= 0] = np.nan
ts = np.arange(1500, N - 61, STEP)
ts = ts[(seg[ts - 60] == seg[ts]) & np.isfinite(s_all[ts]) & np.isfinite(d.atr.values[ts])]
idx = d.index[ts]
IS = idx < SPLIT
P(f"樣本：{len(ts):,} 個起點（樣本內 {IS.sum():,} / 樣本外 {(~IS).sum():,}），H={H}，每 {STEP} 根")

der = endpoint_derivs(cl, s_all, K=15, deg=3)[ts]
v0, a0, j0 = der[:, 0], der[:, 1], der[:, 2]
L = 240
wt = np.arange(L) - (L - 1) / 2
wt = wt / (wt ** 2).sum()
slope = np.full(N, np.nan)
slope[L - 1:] = np.convolve(cl, wt[::-1], mode="valid")
mu_slope = slope[ts] / s_all[ts]
zK = initial_state(v0, a0, j0)
Fw = pf.forward(d, hs=(15, 30, 60))
fwd = {h: Fw[h].values[ts] for h in (15, 30, 60)}
vol30 = Fw["path15"].values[ts]
cost = d.spread.values[ts] * 0.01 / d.atr.values[ts]
P(f"每筆成本（點差/ATR）中位 {np.median(cost):.3f}；s/ATR 中位 {np.nanmedian(s_all[ts] / d.atr.values[ts]):.2f}")

t0 = time.time()
R = profiles(hi, lo, cl, s_all, ts.astype(np.int64), 1440, 4)
_, _, d2 = interp_rows(R, np.tile(np.linspace(-15, 15, 61), (len(ts), 1)))
cbound = float(np.nanpercentile(np.abs(d2), 99))
P(f"剖面完成 {time.time() - t0:.0f}s；|R''| 99% 分位 = {cbound:.3f}（MM 的 c）")


def day_ic(x, y, mask):
    df = pd.DataFrame({"x": x[mask], "y": y[mask]}, index=idx[mask]).replace([np.inf, -np.inf], np.nan).dropna()
    ic = df.groupby(df.index.date).apply(lambda g: g.x.rank().corr(g.y.rank()) if len(g) > 10 else np.nan).dropna()
    return ic.mean(), ic.mean() / ic.std() * np.sqrt(len(ic))


def day_t(r, when):
    s = pd.Series(r, index=when).dropna()
    g = s.groupby(s.index.date).sum()
    return g.mean() / g.std() * np.sqrt(len(g))


def run(lam_a, lam_j, lam_r, mu_mode, Dmul=1.0, want=("C",)):
    mu = np.zeros(len(ts)) if mu_mode == "0" else np.nan_to_num(mu_slope)
    act = Action(H, lam_a, lam_j, lam_r)
    Rr = R if lam_r > 0 else None
    D = np.full(len(ts), np.sqrt(H) * Dmul)
    zu, Au = act.solve(zK, mu, D, Rr, c=cbound)
    zd, Ad = act.solve(zK, mu, -D, Rr, c=cbound)
    o = {"Au": Au, "Ad": Ad}
    if "rng" in want:
        zr, Ar = act.solve(zK, mu, np.zeros(len(ts)), Rr, c=cbound)
        o["Ar"] = Ar
    if "free" in want:
        zf, Af = act.solve(zK, mu, None, Rr, c=cbound)
        o["zf"], o["Af"] = zf, Af
    if "ld" in want:
        o["ldu"], o["ldd"] = act.logdet(zu, Rr), act.logdet(zd, Rr)
    tau = np.nanmedian((np.abs(Au) + np.abs(Ad))[IS])
    o["C"] = (Ad - Au) / (tau + np.abs(Ad) + np.abs(Au))
    o["mu"] = mu
    return o


# ================================================================== 1. 參數掃描（只看樣本內選）
P("\n" + "=" * 110 + "\n1. 參數掃描：λa × λj × λr × μ（24 組）。選擇標準 = 樣本內 C 對未來 30 根報酬的日 IC\n" + "=" * 110)
rows = []
grid = list(itertools.product((0.3, 3.0), (0.3, 3.0), (0.0, 1.0, 3.0), ("0", "slope")))
for la, lj, lr, mm in grid:
    o = run(la, lj, lr, mm)
    ic_is, t_is = day_ic(o["C"], fwd[30], IS)
    ic_os, t_os = day_ic(o["C"], fwd[30], ~IS)
    rows.append(dict(λa=la, λj=lj, λr=lr, μ=mm, IC樣本內=ic_is, t樣本內=t_is, IC樣本外=ic_os, t樣本外=t_os,
                     corr_v0=np.corrcoef(o["C"], v0)[0, 1]))
G = pd.DataFrame(rows)
P(G.round(4).to_string(index=False))
best = G.loc[G.IC樣本內.abs().idxmax()]
P(f"\n樣本內 |IC| 最大：λa={best.λa} λj={best.λj} λr={best.λr} μ={best.μ}（IC {best.IC樣本內:+.4f}）→ 凍結")
P(f"樣本內 IC 與樣本外 IC 的跨參數相關：{G.IC樣本內.corr(G.IC樣本外):+.2f}；樣本外 IC 同號比例 "
  f"{(np.sign(G.IC樣本外) == np.sign(best.IC樣本內)).mean():.0%}")

# ================================================================== 2. 凍結模型的細節
la, lj, lr, mm = best.λa, best.λj, best.λr, best.μ
lr_use = lr if lr > 0 else 1.0  # 若選到 λr=0，另外也看有介質的版本
M = run(la, lj, lr_use, mm, want=("C", "rng", "free", "ld"))
Q = run(la, lj, 0.0, mm, want=("C", "free"))
P("\n" + "=" * 110 + f"\n2. 凍結模型 λa={la} λj={lj} μ={mm}；有介質 λr={lr_use} vs 純二次 λr=0\n" + "=" * 110)
mfree = Q["zf"][:, -1]
P(f"2-0 數學檢查：純二次的 C 與「自由終點 m」相關 {np.corrcoef(Q['C'], mfree)[0, 1]:.4f}（理論上 = 1：二次作用量差 = 線性動能濾波）")
X = np.c_[v0, a0, j0, Q["mu"], np.ones(len(ts))]
b = np.linalg.lstsq(X, mfree, rcond=None)[0]
P(f"    m = {b[0]:.3f}·v0 + {b[1]:.3f}·a0 + {b[2]:.3f}·j0 + {b[3]:.2f}·μ")
P(f"    有介質的 C 與純二次 C 的相關 {np.corrcoef(M['C'], Q['C'])[0, 1]:.4f}；與 v0 的相關 {np.corrcoef(M['C'], v0)[0, 1]:.4f}")


def resid(x, c):
    ok = np.isfinite(x) & np.isfinite(c)
    rx = pd.Series(x[ok]).rank().values
    rc = pd.Series(c[ok]).rank().values
    r = np.full(len(x), np.nan)
    r[ok] = rx - np.cov(rx, rc)[0, 1] / rc.var() * rc
    return r


sig = {
    "對照：v0（只看速度）": v0,
    "對照：μ（240 根斜率）": mu_slope,
    "純二次 C（= 線性動能）": Q["C"],
    "有介質 C": M["C"],
    "有介質 C 扣掉純二次 C（介質的增量）": resid(M["C"], Q["C"]),
    "介質單獨：R↓ − R↑ 路徑上累積密度差": None,
}
# 介質單獨：沿著兩條最優路徑的 Σ R 差（正 = 往下比較貴 → 偏多）
act = Action(H, la, lj, lr_use)
D = np.full(len(ts), np.sqrt(H))
zu, _ = act.solve(zK, M["mu"], D, R, c=cbound)
zd, _ = act.solve(zK, M["mu"], -D, R, c=cbound)
sig["介質單獨：R↓ − R↑ 路徑上累積密度差"] = interp_rows(R, zd[:, 4:])[0].sum(1) - interp_rows(R, zu[:, 4:])[0].sum(1)
P("\n2-1 日 IC（未來 15 / 30 / 60 根報酬），樣本內 → 樣本外")
for k, x in sig.items():
    line = []
    for h in (15, 30, 60):
        a1, t1 = day_ic(x, fwd[h], IS)
        a2, t2 = day_ic(x, fwd[h], ~IS)
        line.append(f"{h}: {a1:+.4f}({t1:+.1f}) → {a2:+.4f}({t2:+.1f})")
    P(f"  {k:<34s} " + " | ".join(line))

P("\n2-2 可交易性：|訊號| 在樣本內前 20% 才進場（門檻凍結），方向 = 訊號正負，持有 30 根，扣點差")


def trade(x, name, q=0.8, h=30):
    thr = np.nanquantile(np.abs(x[IS]), q)
    on = np.abs(x) >= thr
    r = np.sign(x) * fwd[h] - cost
    res = []
    for lab, m in (("樣本內", IS), ("樣本外", ~IS)):
        mm_ = on & m & np.isfinite(r)
        res.append(f"{lab} n={mm_.sum():5d} 淨 {np.nanmean(r[mm_]):+.3f} 毛 {np.nanmean((r + cost)[mm_]):+.3f} t天 {day_t(r[mm_], idx[mm_]):+.2f}")
    yrs = pd.Series(r[on], index=idx[on]).groupby(idx[on].year).mean()
    P(f"  {name:<26s} " + " | ".join(res) + " | 逐年淨 " + " ".join(f"{y}:{v:+.3f}" for y, v in yrs.items()))


for k in ("對照：v0（只看速度）", "對照：μ（240 根斜率）", "純二次 C（= 線性動能）", "有介質 C"):
    trade(sig[k], k)
trade(-sig["有介質 C"], "反向：−有介質 C")
trade(sig["介質單獨：R↓ − R↑ 路徑上累積密度差"], "介質單獨")

# ================================================================== 3. 區間路徑
P("\n" + "=" * 110 + "\n3. 區間路徑：A_區間 − min(A↑, A↓)（小 = 留在原區間是最便宜的路徑）\n" + "=" * 110)
Srng = M["Ar"] - np.minimum(M["Au"], M["Ad"])
for lab, y in (("未來 15 根路徑長（波動）", vol30), ("|未來 30 根報酬|", np.abs(fwd[30]))):
    a1, t1 = day_ic(Srng, y, IS)
    a2, t2 = day_ic(Srng, y, ~IS)
    b1, u1 = day_ic(np.abs(v0), y, IS)
    b2, u2 = day_ic(np.abs(v0), y, ~IS)
    r1, w1 = day_ic(resid(Srng, np.abs(v0)), y, ~IS)
    P(f"  vs {lab:<16s} 區間分數 IC {a1:+.4f}({t1:+.1f}) → {a2:+.4f}({t2:+.1f}) | 對照 |v0| {b1:+.4f} → {b2:+.4f} | 扣掉 |v0| 後樣本外 {r1:+.4f}({w1:+.1f})")

# ================================================================== 4. 接近最優的路徑數（Laplace）與穩定性
P("\n" + "=" * 110 + "\n4. 路徑穩定性\n" + "=" * 110)
Fu, Fd = M["Au"] + 0.5 * M["ldu"], M["Ad"] + 0.5 * M["ldd"]
tauF = np.nanmedian((np.abs(Fu) + np.abs(Fd))[IS])
CF = (Fd - Fu) / (tauF + np.abs(Fd) + np.abs(Fu))
P(f"4-1 Laplace 自由能 F = A* + ½ log det（把接近最優的路徑數量也算進去）：C_F 與 C 相關 {np.corrcoef(np.nan_to_num(CF), M['C'])[0, 1]:.4f}")
for h in (30, 60):
    a2, t2 = day_ic(resid(CF, M["C"]), fwd[h], ~IS)
    P(f"    C_F 扣掉 C → 未來 {h} 根（樣本外）IC {a2:+.4f} t {t2:+.1f}")
P("4-2 擾動穩定性：D ×0.7 / ×1.3、λa ×0.5 / ×2，四個版本與原版同號的比例")
pert = [run(la, lj, lr_use, mm, Dmul=0.7)["C"], run(la, lj, lr_use, mm, Dmul=1.3)["C"],
        run(la * 0.5, lj, lr_use, mm)["C"], run(la * 2, lj, lr_use, mm)["C"]]
agree = np.mean([np.sign(p) == np.sign(M["C"]) for p in pert], 0)
P(f"    全部同號的比例 {(agree == 1).mean():.1%}")
thr = np.nanquantile(np.abs(M["C"][IS]), 0.8)
on = np.abs(M["C"]) >= thr
r = np.sign(M["C"]) * fwd[30] - cost
for lab, m in (("穩定（4/4 同號）", agree == 1), ("不穩定", agree < 1)):
    for sl, sm in (("樣本內", IS), ("樣本外", ~IS)):
        k = on & m & sm & np.isfinite(r)
        P(f"    {lab:<12s} {sl} n={k.sum():5d} 淨 {np.nanmean(r[k]):+.3f}")

# ================================================================== 5. 實際價格偏離最優路徑
P("\n" + "=" * 110 + "\n5. 實際價格偏離 15 根前算出的最優（自由終點）路徑\n" + "=" * 110)
pos = {t: i for i, t in enumerate(ts)}
prev = np.array([pos.get(t - STEP, -1) for t in ts])
ok = (prev >= 0)
ok[ok] &= seg[ts[ok] - STEP] == seg[ts[ok]]
dev = np.full(len(ts), np.nan)
pi = prev[ok]
pred = M["zf"][pi, 4 + STEP]          # 15 根前預測的 x(15)
actual = (cl[ts[ok]] - cl[ts[pi]]) / s_all[ts[pi]]
dev[ok] = actual - pred
P(f"  偏離（以 s 為單位）中位 {np.nanmedian(np.abs(dev)):.2f}；實際 15 根移動的中位 |x| {np.nanmedian(np.abs(actual)):.2f}"
  f"（模型預測的 15 根路徑解釋了實際移動變異的 {1 - np.nanvar(dev) / np.nanvar(actual):+.1%}）")
rec = (cl[ts] - cl[np.maximum(ts - STEP, 0)]) / s_all[ts]
for h in (15, 30, 60):
    a1, t1 = day_ic(dev, fwd[h], IS)
    a2, t2 = day_ic(dev, fwd[h], ~IS)
    r2, w2 = day_ic(resid(dev, rec), fwd[h], ~IS)
    P(f"  偏離 → 未來 {h:2d} 根  IC {a1:+.4f}({t1:+.1f}) → {a2:+.4f}({t2:+.1f})；扣掉「最近 15 根移動」後樣本外 {r2:+.4f}({w2:+.1f})")
trade(-dev, "偏離反向（回到路徑）")
trade(dev, "偏離順向")

# ================================================================== 6. IC 與交易結果方向相反：分十組看
P("\n" + "=" * 110 + "\n6. 分十組（依樣本內分位切）：未來 30 根報酬（ATR，未扣成本）— 平均 / 中位 / 上漲機率 / 以天 t\n" + "=" * 110)
# 簡單對照：不走變分法，直接比較現價上方 D 與下方 D 的停留密度
grid_x = (np.arange(NB) - (NB - 1) / 2) * GRID_STEP
Dv = np.sqrt(H)
up_band = (grid_x > 0) & (grid_x <= Dv)
dn_band = (grid_x < 0) & (grid_x >= -Dv)
simple = R[:, dn_band].mean(1) - R[:, up_band].mean(1)   # 正 = 下方比較稠密（往下比較貴）
sig["對照：上下方密度差（不用變分法）"] = simple
P(f"  有介質 C 與「上下方密度差」的相關 {np.corrcoef(M['C'], simple)[0, 1]:.3f}；與「介質單獨」的相關 "
  f"{np.corrcoef(sig['介質單獨：R↓ − R↑ 路徑上累積密度差'], simple)[0, 1]:.3f}")
for k in ("對照：μ（240 根斜率）", "純二次 C（= 線性動能）", "有介質 C", "介質單獨：R↓ − R↑ 路徑上累積密度差", "對照：上下方密度差（不用變分法）"):
    x = sig[k]
    edges = np.nanquantile(x[IS], np.linspace(0, 1, 11)[1:-1])
    b = np.digitize(x, edges)
    rows = []
    for g in range(10):
        r_ = []
        for m in (IS, ~IS):
            sel = (b == g) & m & np.isfinite(fwd[30])
            y = fwd[30][sel]
            r_ += [y.mean(), np.median(y), (y > 0).mean(), day_t(y, idx[sel])]
        rows.append(r_)
    T_ = pd.DataFrame(rows, columns=pd.MultiIndex.from_product([["樣本內", "樣本外"], ["平均", "中位", "上漲率", "t天"]]),
                      index=[f"D{g + 1}" for g in range(10)])
    P(f"\n  {k}")
    P(T_.round(3).to_string())
trade(simple, "對照：上下方密度差")
P("\n  成本（點差/ATR）逐年中位：" + " ".join(f"{y}:{v:.3f}" for y, v in pd.Series(cost, index=idx).groupby(idx.year).median().items()))
x = sig["有介質 C"]
thr = np.nanquantile(np.abs(x[IS]), 0.8)
on = np.abs(x) >= thr
g = pd.DataFrame({"毛": np.sign(x) * fwd[30], "成本": cost, "多空": np.sign(x)}, index=idx)[on]
P("  有介質 C 前 20%：逐年毛報酬 / 成本 / 多空")
P(g.groupby([g.index.year, "多空"]).agg(n=("毛", "size"), 毛=("毛", "mean"), 成本=("成本", "mean")).round(3).to_string())
out.close()
