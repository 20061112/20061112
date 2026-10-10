"""第五十五部分：波動預測（Volatility Engine）到底有沒有用 —— 嚴格驗證。

問題：第五十四部分的 log-HAR × 時段因子在 2026 M5（M1 算 RV）上看起來很好。這裡確認：
  1. 更長、真正新的資料：M15 2023/1 ~ 2026/10（3.75 年）、M5 2025/5 ~ 2026/10；每根 RV = r²（沒有更細的資料）。
  2. 更強的對照組：交易員常用的 ATR、GARCH(1,1)、EWMA、同長度歷史 RV、只有時段因子、沒有時段因子的 HAR。
  3. 統計檢定：QLIKE、Mincer–Zarnowitz（log RV 對 log 預測：斜率 ≈ 1、截距 ≈ 0）、
     Diebold–Mariano（Newey–West，lag = h），逐月勝場，凍結係數（不重估）跨年套用。
  4. 跟交易有關的檢查：未來 h 根「高低範圍」/ 預測、碰到 ±1σ̂ 的機率，在預測十分位間是否平坦（= 停損距離是否被正確縮放）；
     與 ATR 縮放比較。另依時段（亞洲 / 倫敦 / 紐約）分開。
所有模型逐月 walk-forward：測試月只用之前的資料估計（含時段因子、ATR 縮放常數、HAR / GARCH 係數）。
"""
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter
from load import load

OUT = "results_part55.txt"
GAP_MIN = 180          # 超過 3 小時的空檔（週末 / 假日 / 缺資料）不讓目標跨過去
pd.set_option("display.width", 220)


# ================================================================ 資料與特徵
def prep(path, start=None):
    b = load(path)
    if start:
        b = b[b.index >= start]
    b = b[~b.index.duplicated()]
    b["r"] = np.log(b.close).diff().fillna(0.0)
    step = int(pd.Series(b.index).diff().dt.total_seconds().median() // 60)
    gap = np.r_[False, np.diff(b.index.values).astype("timedelta64[m]").astype(int) > GAP_MIN]
    b.loc[gap, "r"] = 0.0            # 跨週末的跳空不算進 RV（也不在目標內）
    b["gapid"] = np.cumsum(gap)
    return b, step


def tod_factor(b, step, halflife_days=40):
    """時段因子：同一時段 r² 的 EWM（只用之前的天數）/ 全部 K 棒 r² 的 EWM（只用之前的 K 棒）。"""
    key = (b.index.hour * 60 + b.index.minute) // step
    r2 = b.r.to_numpy() ** 2
    df = pd.DataFrame({"r2": r2, "key": key})
    per = df.groupby("key").r2.transform(lambda v: v.shift(1).ewm(halflife=halflife_days, min_periods=10).mean())
    glob = pd.Series(r2).shift(1).ewm(halflife=halflife_days * 1440 / step, min_periods=500).mean()
    f = (per / glob).to_numpy().copy()
    f[~np.isfinite(f)] = 1.0
    return np.clip(f, 0.05, 20), key.to_numpy()


def fut_sum(x_by_key_latest, key, h, nkeys):
    """未來 h 根的時段因子總和，用到 t 為止每個時段的最新估計值。"""
    n = len(key)
    out = np.empty(n)
    latest = np.ones(nkeys)
    f = x_by_key_latest
    offs = np.arange(1, h + 1)
    for i in range(n):
        latest[key[i]] = f[i]
        out[i] = latest[(key[i] + offs) % nkeys].sum()
    return out


def features(b, step, h):
    bph = 60 // step                         # 每小時 K 棒數
    n = len(b)
    r = b.r.to_numpy()
    r2 = r ** 2
    f, key = tod_factor(b, step)
    nkeys = 1440 // step
    futf = fut_sum(f, key, h, nkeys)
    d = r2 / f                               # 去季節化 r²
    s = pd.Series(d)
    W = dict(S=bph, M=4 * bph, D=24 * bph, W5=120 * bph)
    X = {k: np.log(s.rolling(w, min_periods=max(2, w // 2)).mean().to_numpy() + 1e-14) for k, w in W.items()}
    absr = np.abs(r)
    bp = (np.pi / 2) * pd.Series(absr * np.r_[0, absr[:-1]]).rolling(4 * bph).sum()
    rv4 = pd.Series(r2).rolling(4 * bph).sum()
    X["JUMP"] = np.clip(1 - bp / rv4, 0, 1).to_numpy()
    # ATR14（含時段的原始值）
    h_, l_, c_ = b.high.to_numpy(), b.low.to_numpy(), b.close.to_numpy()
    pc = np.r_[c_[0], c_[:-1]]
    tr = np.maximum(h_ - l_, np.maximum(abs(h_ - pc), abs(l_ - pc)))
    atr = pd.Series(tr).rolling(14).mean().to_numpy()
    # 目標
    cs = np.r_[0, np.cumsum(r2)]
    rvf = np.full(n, np.nan)
    rvf[:n - h] = cs[h + 1:n + 1] - cs[1:n + 1 - h]
    g = b.gapid.to_numpy()
    ok = np.zeros(n, bool)
    ok[:n - h] = g[h:] == g[:n - h]
    rvf[~ok] = np.nan
    lc = np.log(c_)
    y = np.full(n, np.nan)
    y[:n - h] = lc[h:] - lc[:n - h]
    y[~ok] = np.nan
    # 未來 h 根的高低（從 t+1 起），相對 close_t
    hh = pd.Series(h_[::-1]).rolling(h, min_periods=h).max().to_numpy()[::-1]
    ll = pd.Series(l_[::-1]).rolling(h, min_periods=h).min().to_numpy()[::-1]
    up = np.full(n, np.nan)
    dn = np.full(n, np.nan)
    up[:n - 1] = np.log(hh[1:]) - lc[:n - 1]
    dn[:n - 1] = lc[:n - 1] - np.log(ll[1:])
    up[~ok] = np.nan
    dn[~ok] = np.nan
    rvpast = pd.Series(r2).rolling(h).sum().to_numpy()
    df = pd.DataFrame(dict(rvf=rvf, y=y, up=up, dn=dn, f=f, futf=futf, atr=atr, close=c_, rvpast=rvpast,
                           ewma=pd.Series(r2).ewm(alpha=0.06, adjust=False).mean().to_numpy(), u=r / np.sqrt(f),
                           **{f"L{k}": v for k, v in X.items() if k != "JUMP"}, JUMP=X["JUMP"]), index=b.index)
    df["month"] = b.index.to_period("M")
    df["hour"] = b.index.hour
    return df


# ================================================================ 模型
def ols(X, y):
    Xc = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    return beta


def garch_fit(u):
    u = u[np.isfinite(u)]
    u2 = u ** 2
    v0 = u2.mean()

    def nll(p):
        w, a, bb = np.exp(p[0]) * v0, 1 / (1 + np.exp(-p[1])), 1 / (1 + np.exp(-p[2]))
        if a + bb >= 0.9995:
            return 1e12
        s2 = w / (1 - bb) + lfilter([a], [1, -bb], np.r_[v0, u2[:-1]])
        s2 = np.maximum(s2, 1e-18)[500:]
        return 0.5 * np.sum(np.log(s2) + u2[500:] / s2)

    best = None
    for a0, b0 in ((0.05, 0.90), (0.10, 0.85), (0.03, 0.96)):
        p0 = [np.log((1 - a0 - b0) + 1e-6), np.log(a0 / (1 - a0)), np.log(b0 / (1 - b0))]
        res = minimize(nll, p0, method="Nelder-Mead", options=dict(maxiter=400, xatol=1e-4, fatol=1e-3))
        if best is None or res.fun < best.fun:
            best = res
    p = best.x
    return np.exp(p[0]) * v0, 1 / (1 + np.exp(-p[1])), 1 / (1 + np.exp(-p[2]))


def garch_path(u, w, a, bb):
    u2 = np.nan_to_num(u ** 2)
    v = w / (1 - a - bb)
    # s2[t] = 第 t 根收盤後對 t+1 的條件變異數
    s2 = lfilter([a], [1, -bb], u2) + w / (1 - bb)
    return s2, v


def rel_mats(XT, ly, futf):
    """XT 欄位 = log RV 1h、4h、1d、5d（去季節化）、跳躍。回傳相對 5 天的 X 與 y。"""
    W5 = XT[:, 3]
    Xr = np.column_stack([XT[:, 0] - W5, XT[:, 1] - W5, XT[:, 2] - W5, XT[:, 4]])
    return Xr, ly - np.log(futf) - W5


def forecasts(df, months, h, step):
    n = len(df)
    mon = df.month.to_numpy()
    rvf = df.rvf.to_numpy()
    futf = df.futf.to_numpy()
    P = {k: np.full(n, np.nan) for k in ("hist", "atr", "ewma", "garch", "tod_only", "har", "har_tod", "har_rel")}
    coefs = []
    XH = np.column_stack([df.LS, df.LM, df.LD, df.LW5]).astype(float)
    XT = np.column_stack([XH, df.JUMP.to_numpy()])
    # 「沒有時段」的 HAR 用原始（含時段）的 log RV
    raw = df.f.to_numpy()
    XR = np.column_stack([XH[:, 0] + np.log(pd.Series(raw).rolling(60 // step).mean().to_numpy()),
                          XH[:, 1] + np.log(pd.Series(raw).rolling(240 // step).mean().to_numpy()), XH[:, 2], XH[:, 3]])
    atr2 = (df.atr.to_numpy() / df.close.to_numpy()) ** 2 * h
    ly = np.log(rvf)
    for m in months:
        tr = (mon < m) & np.isfinite(ly) & np.isfinite(XT).all(1) & np.isfinite(XR).all(1) & (rvf > 0)
        te = mon == m
        # 歷史 RV（同長度）與 EWMA：不需估計
        P["hist"][te] = df.rvpast.to_numpy()[te]
        P["ewma"][te] = df.ewma.to_numpy()[te] * h
        # ATR：常數縮放（訓練段 mean(RVfut) / mean(ATR² h)）
        c = np.nanmean(rvf[tr]) / np.nanmean(atr2[tr])
        P["atr"][te] = c * atr2[te]
        # 只有時段：最近 1 天去季節化水準 × 未來時段因子
        c2 = np.nanmean(rvf[tr]) / np.nanmean(np.exp(XH[tr, 2]) * futf[tr])
        P["tod_only"][te] = c2 * np.exp(XH[te, 2]) * futf[te]
        # HAR（無時段）
        bt = ols(XR[tr], ly[tr])
        s2 = np.var(ly[tr] - np.column_stack([np.ones(tr.sum()), XR[tr]]) @ bt)
        P["har"][te] = np.exp(np.column_stack([np.ones(te.sum()), XR[te]]) @ bt + s2 / 2)
        # HAR × 時段
        yt = ly - np.log(futf)
        bt2 = ols(XT[tr], yt[tr])
        s2b = np.var(yt[tr] - np.column_stack([np.ones(tr.sum()), XT[tr]]) @ bt2)
        P["har_tod"][te] = np.exp(np.column_stack([np.ones(te.sum()), XT[te]]) @ bt2 + s2b / 2) * futf[te]
        coefs.append([str(m)] + list(np.round(bt2, 3)))
        # 尺度無關版：全部相對 5 天 RV（係數和自動 = 1，不受長期水準漂移影響）+ Duan smearing 修正平均偏誤
        Xr, yr = rel_mats(XT, ly, futf)
        bt3 = ols(Xr[tr], yr[tr])
        sm = np.mean(np.exp(yr[tr] - np.column_stack([np.ones(tr.sum()), Xr[tr]]) @ bt3))
        P["har_rel"][te] = np.exp(np.column_stack([np.ones(te.sum()), Xr[te]]) @ bt3 + XT[te, 3]) * sm * futf[te]
        # GARCH(1,1) 去季節化報酬（每 3 個月重估一次以節省時間）
        if len(coefs) == 1 or m.month % 3 == 0:
            gpar = garch_fit(df.u.to_numpy()[mon < m][-100000:])
        w, a, bb = gpar
        s2p, v = garch_path(df.u.to_numpy(), w, a, bb)
        ab = a + bb
        k = np.arange(h)
        avg = v * h + (s2p[te] - v) * (ab ** k).sum()     # 未來 h 根去季節化變異數總和
        P["garch"][te] = avg * futf[te] / h
    return P, coefs, gpar


# ================================================================ 評估
def nw_t(d, lag):
    d = d[np.isfinite(d)]
    n = len(d)
    e = d - d.mean()
    g0 = e @ e / n
    s = g0
    for k in range(1, lag + 1):
        s += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k] / n)
    return d.mean() / np.sqrt(s / n)


def qlike(rv, p):
    x = rv / p
    return x - np.log(x) - 1


def evaluate(df, P, months, h, pr, tag):
    mon = df.month.to_numpy()
    rvf = df.rvf.to_numpy()
    test = np.isin(mon, months) & np.isfinite(rvf) & (rvf > 0)
    for p in P.values():
        test &= np.isfinite(p) & (p > 0)
    pr(f"\n--- {tag}：h = {h} 根；測試 {months[0]} ~ {months[-1]}（{len(months)} 個月），樣本 {test.sum()} ---")
    pr(f"{'model':9s} {'QLIKE':>7s} {'R2log':>6s} {'MZ a':>6s} {'MZ b':>5s} {'level':>6s} {'DM t':>6s} {'勝月':>6s}")
    L = {k: qlike(rvf[test], p[test]) for k, p in P.items()}
    ref = L["har_tod"]
    mt = mon[test]
    lr = np.log(rvf[test])
    rows = {}
    for k, p in P.items():
        lp = np.log(p[test])
        b = np.polyfit(lp, lr, 1)
        r2 = 1 - np.var(lr - lp - np.mean(lr - lp)) / np.var(lr)
        lvl = rvf[test].mean() / p[test].mean()
        dm = nw_t(L[k] - ref, max(h, 1)) if k != "har_tod" else np.nan
        wins = sum(ref[mt == m].mean() < L[k][mt == m].mean() for m in months) if k != "har_tod" else np.nan
        pr(f"{k:9s} {L[k].mean():7.4f} {r2:6.3f} {b[1]:+6.2f} {b[0]:5.2f} {lvl:6.2f} {dm:+6.1f} {'' if k == 'har_tod' else f'{wins}/{len(months)}':>6s}")
        rows[k] = L[k].mean()
    pr("  R2log = 1 − var(log RV − log 預測) / var(log RV)；MZ：log RV = a + b·log 預測（b ≈ 1 為校準）；level = 平均 RV / 平均預測")
    pr("  DM t > 0：har_tod 的 QLIKE 顯著較小（Newey–West lag = h）；勝月 = har_tod 在幾個月份勝過該模型")
    return test, rows


def practical(df, P, test, h, pr):
    """停損縮放：未來 h 根內碰到 ±k·σ̂ 的機率、範圍 / σ̂，按預測十分位。對照 ATR。"""
    up = df.up.to_numpy()
    dn = df.dn.to_numpy()
    y = df.y.to_numpy()
    m = test & np.isfinite(up) & np.isfinite(dn) & np.isfinite(y)
    pr(f"\n  停損縮放檢查（未來 {h} 根，從 close_t 起算；十分位依「預測值」切）")
    pr(f"  {'十分位':6s} " + " ".join(f"{'D' + str(i + 1):>6s}" for i in range(10)) + "   max/min  CV")
    for k in ("har_rel", "har_tod", "atr", "hist", "tod_only"):
        s = np.sqrt(P[k][m])
        rng = (up[m] + dn[m]) / s
        touch = ((up[m] >= s) | (dn[m] >= s)).astype(float)
        absz = np.abs(y[m]) / s
        dec = pd.qcut(s, 10, labels=False, duplicates="drop")
        for nm, v in (("範圍/σ̂", rng), ("碰±1σ̂", touch), ("|y|/σ̂", absz)):
            g = pd.Series(v).groupby(dec).mean().to_numpy()
            pr(f"  {k:8s} {nm:7s} " + " ".join(f"{x:6.2f}" for x in g) + f"   {g.max() / g.min():5.2f}  {g.std() / g.mean():.3f}")
    pr("  理想：各十分位相同（max/min ≈ 1、CV ≈ 0）→ 用 k·σ̂ 當停損時，每筆被掃到的機率與報酬尺度不隨行情改變")
    # 對 ATR 的整體比較：|y|/σ̂ 的離散程度（越小 = 風險單位越穩定）
    pr("  單筆風險穩定度：std(log(範圍/σ̂))（越小越好，代表用 σ̂ 定部位時每筆實際波動幅度更一致）")
    for k in ("har_rel", "har_tod", "har", "garch", "ewma", "atr", "hist", "tod_only"):
        s = np.sqrt(P[k][m])
        pr(f"    {k:9s} {np.std(np.log((up[m] + dn[m]) / s)):.3f}")


def by_session(df, P, test, pr):
    hr = df.hour.to_numpy()
    rvf = df.rvf.to_numpy()
    pr("\n  依時段的 QLIKE（伺服器時間；亞洲 1–8、倫敦 9–14、紐約 15–23）")
    pr(f"  {'model':9s} {'亞洲':>7s} {'倫敦':>7s} {'紐約':>7s}")
    for k in ("har_rel", "har_tod", "har", "garch", "atr", "hist"):
        vals = []
        for lo, hi in ((1, 8), (9, 14), (15, 23)):
            mm = test & (hr >= lo) & (hr <= hi)
            vals.append(qlike(rvf[mm], P[k][mm]).mean())
        pr(f"  {k:9s} " + " ".join(f"{v:7.4f}" for v in vals))


def mats_raw(df):
    XT = np.column_stack([df.LS, df.LM, df.LD, df.LW5, df.JUMP]).astype(float)
    return XT, np.log(df.rvf.to_numpy())


def frozen(df_tr, df_te, h, pr, tag):
    """在 df_tr 全部資料上估計 har_tod 係數一次，原封不動套到 df_te（不重估），對照 df_te 上 walk-forward 的結果。"""
    def mats(df):
        X = np.column_stack([df.LS, df.LM, df.LD, df.LW5, df.JUMP]).astype(float)
        return X, np.log(df.rvf.to_numpy()) - np.log(df.futf.to_numpy())
    X, y = mats(df_tr)
    ok = np.isfinite(X).all(1) & np.isfinite(y)
    b = ols(X[ok], y[ok])
    s2 = np.var(y[ok] - np.column_stack([np.ones(ok.sum()), X[ok]]) @ b)
    X2, _ = mats(df_te)
    p = np.exp(np.column_stack([np.ones(len(X2)), X2]) @ b + s2 / 2) * df_te.futf.to_numpy()
    rv = df_te.rvf.to_numpy()
    m = np.isfinite(p) & np.isfinite(rv) & (rv > 0)
    q = qlike(rv[m], p[m]).mean()
    hist = df_te.rvpast.to_numpy()
    mh = m & (hist > 0)
    XT, ly = mats_raw(df_tr)
    Xr, yr = rel_mats(XT, ly, df_tr.futf.to_numpy())
    ok = np.isfinite(Xr).all(1) & np.isfinite(yr)
    b3 = ols(Xr[ok], yr[ok])
    sm = np.mean(np.exp(yr[ok] - np.column_stack([np.ones(ok.sum()), Xr[ok]]) @ b3))
    XT2, _ = mats_raw(df_te)
    Xr2, _ = rel_mats(XT2, np.zeros(len(XT2)), df_te.futf.to_numpy())
    p3 = np.exp(np.column_stack([np.ones(len(Xr2)), Xr2]) @ b3 + XT2[:, 3]) * sm * df_te.futf.to_numpy()
    m3 = m & np.isfinite(p3)
    pr(f"  {tag}: hist {qlike(rv[mh], hist[mh]).mean():.4f} | har_tod QLIKE {q:.4f} level {rv[m].mean() / p[m].mean():.2f} | "
       f"har_rel QLIKE {qlike(rv[m3], p3[m3]).mean():.4f} level {rv[m3].mean() / p3[m3].mean():.2f}  係數 {np.round(b3, 3)}")


# ================================================================ main
def run_dataset(path, start, warm_months, hs, pr, label):
    b, step = prep(path, start)
    pr(f"\n================ {label}：{b.index[0]} ~ {b.index[-1]}，{len(b)} 根，週期 {step} 分鐘 ================")
    allm = sorted(b.index.to_period("M").unique())
    months = allm[warm_months:]
    res = {}
    for h in hs:
        df = features(b, step, h)
        P, coefs, g = forecasts(df, months, h, step)
        test, rows = evaluate(df, P, months, h, pr, f"{label} {h * step} 分鐘")
        res[h] = rows
        practical(df, P, test, h, pr)
        by_session(df, P, test, pr)
        pr("  HAR×時段 係數（常數、log RV 1h、4h、1d、5d、跳躍）逐月：")
        for c in coefs[:: max(1, len(coefs) // 8)]:
            pr("    " + " ".join(str(x) for x in c))
        pr(f"  GARCH(1,1)（去季節化）最後一次估計：ω={g[0]:.3g} α={g[1]:.3f} β={g[2]:.3f} α+β={g[1] + g[2]:.4f}")
    return b, step


def main():
    out = open(OUT, "w")

    def pr(*a):
        print(*a, file=out, flush=True)
    pr("第五十五部分：波動預測驗證（Volatility Engine）")
    pr("模型：hist = 過去同長度 RV；atr = ATR14² × h × 常數；ewma = RiskMetrics λ=0.94；garch = GARCH(1,1)（去季節化報酬）× 時段；")
    pr("      tod_only = 最近 1 天水準 × 未來時段因子；har = log-HAR（1h/4h/1d/5d，無時段）；har_tod = 去季節化 log-HAR + 跳躍 × 未來時段因子（第五十四部分）；")
    pr("      har_rel = har_tod 改成相對 5 天 RV 的尺度無關形式 + Duan smearing（修正 har_tod 的平均低估與跨年水準漂移）")

    bm15, s15 = run_dataset("data/XAUUSD_M15_2023_2026.csv", None, 3, [4, 16, 96], pr, "M15 2023–2026")
    bm5, s5 = run_dataset("data/XAUUSD_M5_2025_2026.csv", None, 3, [6, 12, 48], pr, "M5 2025–2026")

    pr("\n================ 凍結係數跨年套用（不重估）================")
    for h in (4, 16, 96):
        df = features(bm15, s15, h)
        yr = df.index.year
        frozen(df[yr <= 2024], df[yr == 2025], h, pr, f"M15 h={h * 15}分 2023–24 → 2025")
        frozen(df[yr <= 2024], df[yr == 2026], h, pr, f"M15 h={h * 15}分 2023–24 → 2026")
    for h in (12, 48):
        df = features(bm5, s5, h)
        frozen(df[df.index < "2026-01-01"], df[df.index >= "2026-01-01"], h, pr, f"M5  h={h * 5}分 2025 → 2026")
    out.close()
    print(open(OUT).read())


if __name__ == "__main__":
    main()
