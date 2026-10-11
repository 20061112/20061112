"""第二十二部分：ATR 的時間尺度比值（ATR_Mx / (ATR_M1·√x)）與它的 z-score。

理論：價格若是隨機漫步，x 分鐘 K 棒的波幅 ≈ 1 分鐘的 √x 倍，所以
    VR_x = ATR(Mx) / (ATR(M1) · √x) ≈ 1
VR > 1：x 分鐘內的走勢比 1 分鐘「走得更遠」→ 同方向延續（趨勢、正自相關）
VR < 1：1 分鐘來回很多、x 分鐘卻沒走多遠 → 來回 / 均值回歸（負自相關）
這其實就是 Lo-MacKinlay variance ratio 的「波幅版」。

做法（全部只用當下以前的資料）：
- 兩個 ATR 用「同一段時間」：Mx 最近 N 根 = M1 最近 N·x 根，避免把波動率高低混進比值。
- √x 改用每根 Mx 實際包含的 M1 根數（缺分鐘時較準）。
- z-score：log(VR) 先扣掉「同一時段過去 60 天」的平均（盤中季節性），再除以過去 20 天的標準差。
- 結果（未來 H 根 Mx）：波動變化、未來 VR、未來 ER、順勢延續（過去方向 × 未來報酬）、未來波幅。
- 迴歸斜率用「以交易日為群」的穩健標準誤；也控制 ATR 本身的高低，看 z 有沒有額外資訊。
資料：data/XAUUSD_M1_2023_2026.csv（2023/6 ~ 2026/10）。輸出 results_part22.txt。
"""
import numpy as np
import pandas as pd
from load import load

N = 14             # ATR 長度（Mx 根數）
TOD_DAYS = 60      # 季節性基準：同時段過去 60 天
Z_DAYS = 20        # z-score 標準差：過去 20 天

m1 = load("data/XAUUSD_M1_2023_2026.csv")
m1 = m1[~m1.index.duplicated()]
pc = m1.close.shift()
# 跨過週末 / 休市的第一根不要算跳空
gap = m1.index.to_series().diff() > pd.Timedelta("10min")
tr1 = np.maximum(m1.high, pc) - np.minimum(m1.low, pc)
tr1[gap] = (m1.high - m1.low)[gap]
m1["tr"] = tr1


def build(x):
    g = m1.resample(f"{x}min", label="left", closed="left")
    b = pd.DataFrame({"open": g.open.first(), "high": g.high.max(), "low": g.low.min(),
                      "close": g.close.last(), "tr1_sum": g.tr.sum(), "cnt": g.close.count()})
    b = b[b.cnt >= max(1, int(0.6 * x))].copy()
    bpc = b.close.shift()
    bgap = b.index.to_series().diff() > pd.Timedelta(f"{x + 10}min")
    tr = np.maximum(b.high, bpc) - np.minimum(b.low, bpc)
    tr[bgap] = (b.high - b.low)[bgap]
    b["tr"] = tr
    b["atrx"] = b.tr.rolling(N).mean()
    b["atr1"] = b.tr1_sum.rolling(N).sum() / b.cnt.rolling(N).sum()          # 同一段時間的 M1 ATR
    b["sqrtx"] = np.sqrt(b.cnt.rolling(N).mean())
    b["vr"] = b.atrx / (b.atr1 * b.sqrtx)
    b["lvr"] = np.log(b.vr)
    # 收盤價版本的 variance ratio（對照）：x 分鐘報酬的變異 / (x · 1 分鐘報酬的變異)
    r1 = np.log(m1.close).diff()
    r1[gap] = np.nan
    v1 = (r1 ** 2).resample(f"{x}min", label="left", closed="left").sum().reindex(b.index)
    rx = np.log(b.close).diff()
    rx[bgap] = np.nan
    b["cvr"] = (rx ** 2).rolling(N).sum() / v1.rolling(N).sum()

    # 盤中季節性：同一時段過去 TOD_DAYS 個交易日的平均（不含今天）
    b["date"] = b.index.normalize()
    b["slot"] = b.index.hour * 60 + b.index.minute
    piv = b.pivot_table(index="date", columns="slot", values="lvr")
    base = piv.shift(1).rolling(TOD_DAYS, min_periods=20).mean()
    b["tod"] = base.stack().reindex(pd.MultiIndex.from_arrays([b.date, b.slot])).values
    b["dev"] = b.lvr - b.tod
    per_day = int(round(len(b) / b.date.nunique()))
    w = Z_DAYS * per_day
    b["z"] = b.dev / b.dev.rolling(w, min_periods=w // 2).std().shift(1)
    # 不扣季節性的簡單 z（滾動平均 / 標準差，過去 20 天）
    b["z_raw"] = (b.lvr - b.lvr.rolling(w, min_periods=w // 2).mean().shift(1)) / \
        b.lvr.rolling(w, min_periods=w // 2).std().shift(1)
    # 波動率本身的高低（控制變數）：現在的 M1 ATR / 過去 20 天平均
    b["lvol"] = np.log(b.atr1 / b.atr1.rolling(w, min_periods=w // 2).mean().shift(1))
    return b, per_day


def er(close, h):
    ch = (close.shift(-h) - close).abs()
    path = close.diff().abs().rolling(h).sum().shift(-h)
    return ch / path


def outcomes(b, H):
    o = pd.DataFrame(index=b.index)
    fut_atr1 = (b.tr1_sum.rolling(H).sum() / b.cnt.rolling(H).sum()).shift(-H)
    o["vol_chg"] = np.log(fut_atr1 / b.atr1)                              # 未來 M1 波動 / 現在
    fut_x = b.tr.rolling(H).mean().shift(-H)
    o["fut_lvr"] = np.log(fut_x / (fut_atr1 * b.sqrtx)) - b.tod             # 未來 VR（扣季節性）
    o["fut_er"] = er(b.close, H)                                           # 未來 H 根效率比
    past = np.sign(b.close - b.close.shift(N))
    unit = b.atr1 * b.sqrtx                                                # 「理論」Mx ATR：用 M1 推算，不受 z 影響
    o["cont"] = past * (b.close.shift(-H) - b.close) / unit                # 順過去 N 根方向的未來報酬（理論 ATR）
    hi = b.high[::-1].rolling(H).max()[::-1].shift(-1)
    lo = b.low[::-1].rolling(H).min()[::-1].shift(-1)
    o["fut_rng"] = np.log((hi - lo) / (unit * np.sqrt(H)))                  # 未來波幅（相對理論 ATR）
    o["fut_rng_x"] = np.log((hi - lo) / (b.atrx * np.sqrt(H)))           # 同上，但除以實際 ATRx（會有機械相關）
    o["abs_ret"] = (b.close.shift(-H) - b.close).abs() / (unit * np.sqrt(H))
    # 未來跨過休市的樣本不要用
    span = b.index.to_series().shift(-H) - b.index.to_series()
    o[span > pd.Timedelta(minutes=int(H * b.attrs["x"] * 1.5) + 30)] = np.nan
    return o


def cluster_ols(y, X, groups):
    """OLS + 以交易日為群的穩健標準誤。回傳係數與 t。"""
    X = np.column_stack([np.ones(len(y)), X])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    e = y - X @ beta
    S = pd.DataFrame(X * e[:, None]).groupby(groups).sum().values
    G = S.shape[0]
    V = XtX_inv @ (S.T @ S) @ XtX_inv * G / (G - 1)
    return beta[1:], beta[1:] / np.sqrt(np.diag(V)[1:])


def qtable(d, zcol, ycol):
    q = pd.cut(d[zcol], [-np.inf, -2, -1, -0.5, 0.5, 1, 2, np.inf])
    g = d.groupby(q, observed=True)[ycol]
    return "  ".join(f"{str(k):>12s}:{v.mean():+.3f}" for k, v in g)


out = []
P = out.append
P("第二十二部分：ATR 時間尺度比值 VR_x = ATR(Mx) / (ATR(M1)·√x) 與它的 z-score")
P(f"資料 {m1.index[0]} ~ {m1.index[-1]}，M1 {len(m1):,} 根；ATR 長度 {N} 根 Mx，兩者用同一段時間")
P("=" * 118)

for x in (5, 15, 60):
    b, per_day = build(x)
    b.attrs["x"] = x
    P(f"\n######## x = {x}（M{x}，每天約 {per_day} 根） ########")
    v = b.vr.dropna()
    P(f"VR 分布：平均 {v.mean():.3f}  中位 {v.median():.3f}  10%~90% [{v.quantile(.1):.3f}, {v.quantile(.9):.3f}]"
      f"  （收盤價 variance ratio 中位 {b.cvr.median():.3f}）")
    yr = b.groupby(b.index.year).vr.median()
    P("  逐年中位：" + "  ".join(f"{k}: {val:.3f}" for k, val in yr.items()))
    hr = b.groupby(b.index.hour).vr.median()
    P("  各小時中位（伺服器時間）：" + " ".join(f"{h}:{val:.2f}" for h, val in hr.items()))
    P(f"  VR 與 收盤 VR 的相關（log）：{np.corrcoef(*np.log(b[['vr', 'cvr']].dropna().values.T))[0, 1]:.2f}；"
      f"z 與 z_raw 相關：{b[['z', 'z_raw']].corr().iloc[0, 1]:.2f}；z 與波動高低(lvol) 相關：{b[['z', 'lvol']].corr().iloc[0, 1]:+.2f}")
    zz = b.z.dropna()
    P(f"  z 分布：標準差 {zz.std():.2f}  |z|>2 比例 {(zz.abs() > 2).mean():.3f}（常態 0.046）  "
      f"z>2 {(zz > 2).mean():.3f}  z<-2 {(zz < -2).mean():.3f}")

    for H in (N, 4):
        o = outcomes(b, H)
        d = pd.concat([b[["z", "z_raw", "lvol", "date"]], o], axis=1).dropna()
        P(f"\n  -- 未來 H = {H} 根 M{x}（{H * x} 分鐘），n = {len(d):,}，交易日 {d.date.nunique()} --")
        for ycol, name in [("vol_chg", "未來波動變化 log(ATR1未來/現在)"),
                           ("fut_lvr", "未來 VR（log、扣季節性）"),
                           ("fut_er", "未來效率比 ER"),
                           ("fut_rng", "未來波幅 log(高−低 / ATR1·√x·√H)"),
                           ("fut_rng_x", "未來波幅 log(高−低 / ATRx·√H)（分母含 z，機械相關）"),
                           ("abs_ret", "未來 |報酬| / ATR1·√x·√H"),
                           ("cont", "順過去方向的未來報酬（ATR1·√x）")]:
            g = d.date.values
            (b1,), (t1,) = cluster_ols(d[ycol].values, d[["z"]].values, g)
            (b2, b3), (t2, t3) = cluster_ols(d[ycol].values, d[["z", "lvol"]].values, g)
            yrs = []
            for yv, dd in d.groupby(d.index.year):
                if len(dd) > 500:
                    (by,), (ty,) = cluster_ols(dd[ycol].values, dd[["z"]].values, dd.date.values)
                    yrs.append(f"{yv}:{ty:+.1f}")
            P(f"  {name}")
            P(f"     z 斜率 {b1:+.4f} (t {t1:+.1f}) | 控制波動高低後 z {b2:+.4f} (t {t2:+.1f})、lvol {b3:+.4f} (t {t3:+.1f})"
              f" | 逐年 t：{' '.join(yrs)}")
            P(f"     z 分組平均：{qtable(d, 'z', ycol)}")

    # 極端事件：z 由 <2 跨到 >2、或跨到 <−2 之後的盤面
    H = N
    o = outcomes(b, H)
    d = pd.concat([b[["z", "date", "vr"]], o], axis=1)
    up = (d.z > 2) & (d.z.shift() <= 2)
    dn = (d.z < -2) & (d.z.shift() >= -2)
    base = d.dropna()
    P(f"\n  極端事件（未來 {H} 根）：            n     波動變化   未來ER   未來波幅   順勢延續")
    for nm, m in [("全部樣本", slice(None)), ("z 剛升破 +2", up), ("z 剛跌破 −2", dn)]:
        s = d[m].dropna() if not isinstance(m, slice) else base
        P(f"    {nm:12s} {len(s):6d}   {s.vol_chg.mean():+.3f}    {s.fut_er.mean():.3f}    {s.fut_rng.mean():+.3f}    {s.cont.mean():+.3f}")

open("results_part22.txt", "w").write("\n".join(out) + "\n")
print("\n".join(out))
