"""第五十四部分：從零設計的 Entropy × Volatility 研究架構（XAUUSD，M5）。

核心原則：Entropy / Volatility 只描述市場，不直接決定買賣；買賣要建立在
「對未來報酬有增量預測能力，且扣成本後仍為正」之上。驗證分三階段：
  階段一  特徵是否有資訊（互資訊、條件互資訊、波動預測、狀態空間、轉換事件）
  階段二  預測能力（逐月 walk-forward：基準 vs +波動 vs +熵；IC、方向 AUC、校準；Model A/B/C）
  階段三  交易價值（門檻 μ̂ > C + B、預測波動定停損與部位、成本壓力、隨機方向、消融）

資料：data/XAUUSD_M1_2026.csv（2026/1/2 ~ 10/8，4/3~4/14 缺資料），M1 → M5。
目標：h 根後的 log 報酬 y，標準化 z = y / σ̂_h（σ̂_h = 波動引擎的事前預測）；跨缺口的樣本剔除。
walk-forward：每個測試月只用之前的月份訓練（3 月起），所有門檻 / 分位 / 標準化皆在訓練段估計。
"""
import sys
import numpy as np
import pandas as pd
from load import load
from ev_features import (to_bars, entropy_engine, volatility_engine, ewma_sigma,
                         future_tod_sum, base_directional, LN3)

H = int(sys.argv[1]) if len(sys.argv) > 1 else 12   # 預測 / 持有期（M5 根數）
SPLIT = pd.Timestamp("2026-06-01")
RNG = np.random.default_rng(23)
STOP_K = 2.0       # 停損 = STOP_K × σ̂_h
pd.set_option("display.width", 200)


# ================================================================ 資料與特徵
def build():
    m1 = load("data/XAUUSD_M1_2026.csv")
    b = to_bars(m1, "5min")
    idx = b.index
    n = len(b)
    vol, key = volatility_engine(b)
    sig_prev = np.sqrt(pd.Series(b.rv.to_numpy()).ewm(alpha=0.06, adjust=False).mean().shift(1).to_numpy())
    ent = entropy_engine(b.r.to_numpy(), sig_prev, close=b.close.to_numpy())
    ent144p3 = entropy_engine(b.r.to_numpy(), sig_prev, W=48, close=b.close.to_numpy())
    ent["H_T48"] = ent144p3.H_T.to_numpy()
    ent["H_P48"] = ent144p3.H_P.to_numpy()
    sig_bar = np.sqrt(pd.Series(b.rv.to_numpy()).rolling(288).mean().to_numpy())
    dirf = base_directional(b, sig_bar)
    df = pd.concat([b.reset_index(), vol.reset_index(drop=True), ent, dirf], axis=1).set_index("time")

    # 目標：close_t → close_{t+H}；交易：open_{t+1} → close_{t+H}
    c = np.log(b.close.to_numpy())
    gap = np.r_[False, np.diff(idx.values).astype("timedelta64[m]").astype(int) > 15]
    gapcum = np.cumsum(gap)
    ok = np.zeros(n, bool)
    ok[:-H] = gapcum[H:] == gapcum[:-H]
    y = np.full(n, np.nan)
    y[:-H] = c[H:] - c[:-H]
    y[~ok] = np.nan
    rv = b.rv.to_numpy()
    rvf = np.full(n, np.nan)
    cs = np.r_[0, np.cumsum(rv)]
    rvf[:-H] = cs[H + 1:n + 1] - cs[1:n + 1 - H]
    rvf[~ok] = np.nan
    df["y"] = y
    df["rvf"] = rvf
    df["futtod"] = future_tod_sum(vol.tod.to_numpy(), key, H)
    df["ewma"] = ewma_sigma(b.r.to_numpy()) ** 2
    df["month"] = idx.to_period("M")
    df = df.iloc[288 * 2:]          # 暖機（最長窗口 288 + 時段因子）
    return df


ENT = ["H_D", "H_T", "I1", "I2", "TF", "PERS", "H_P", "H_T48", "H_P48"]
VOLF = ["V_SL", "V_ML", "V_des", "rho1", "JUMP", "KURT"]
DIRF = ["MOM12", "MOM48", "Z48"]


# ================================================================ 小工具
def ols(X, y, lam=0.0):
    Xc = np.column_stack([np.ones(len(X)), X])
    P = lam * np.eye(Xc.shape[1])
    P[0, 0] = 0
    return np.linalg.solve(Xc.T @ Xc + P, Xc.T @ y)


def pred(beta, X):
    return beta[0] + X @ beta[1:]


def logit_fit(X, y, lam=1.0, it=25):
    Xc = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(Xc.shape[1])
    P = lam * np.eye(len(w))
    P[0, 0] = 0
    for _ in range(it):
        p = 1 / (1 + np.exp(-Xc @ w))
        g = Xc.T @ (p - y) + P @ w
        Hs = (Xc * (p * (1 - p))[:, None]).T @ Xc + P
        w -= np.linalg.solve(Hs, g)
    return w


def auc(score, lab):
    o = np.argsort(score)
    rk = np.empty(len(score))
    rk[o] = np.arange(1, len(score) + 1)
    n1 = lab.sum()
    n0 = len(lab) - n1
    return (rk[lab == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def rank(v):
    return pd.Series(v).rank().to_numpy()


def tstat(v):
    v = v[np.isfinite(v)]
    return v.mean() / v.std(ddof=1) * np.sqrt(len(v)) if len(v) > 2 else np.nan


def disc_mi(a, b):
    """plug-in MI（nats）+ Miller–Madow 修正；a, b 為非負整數。"""
    ka, kb = a.max() + 1, b.max() + 1
    c = np.zeros((ka, kb))
    np.add.at(c, (a, b), 1)
    N = c.sum()
    p = c / N
    pa, pb = p.sum(1), p.sum(0)
    nz = p > 0
    mi = (p[nz] * np.log(p[nz] / (pa[:, None] * pb[None, :])[nz])).sum()
    return mi - (((pa > 0).sum() - 1) * ((pb > 0).sum() - 1)) / (2 * N)


def mi_test(x_bins, ybins, cond=None, nperm=200):
    """MI（或條件 MI）與「循環平移」虛無分配（保留兩邊各自的自相關）。"""
    def stat(yb):
        if cond is None:
            return disc_mi(x_bins, yb)
        tot = 0.0
        for g in np.unique(cond):
            m = cond == g
            tot += m.mean() * disc_mi(x_bins[m], yb[m])
        return tot
    s0 = stat(ybins)
    n = len(ybins)
    null = np.array([stat(np.roll(ybins, RNG.integers(n // 10, n - n // 10))) for _ in range(nperm)])
    return s0, (null >= s0).mean(), null.mean()


def qbins(x, edges):
    return np.searchsorted(edges, x)


# ================================================================ 波動預測（Volatility Engine 輸出）
def vol_forecast(df, months):
    """逐月 walk-forward：naive / naive×時段 / EWMA / HAR×時段（log 迴歸）。回傳 σ̂²_h 與評估表。"""
    rvS = (df.sigS.to_numpy() ** 2)
    tod = df.tod.to_numpy()
    fut = df.futtod.to_numpy()
    rvf = df.rvf.to_numpy()
    des = lambda s: np.log(s ** 2 / 1.0 + 1e-14)
    X = np.column_stack([des(df.sigS), des(df.sigM), des(df.sigL), df.JUMP.to_numpy()])
    # 用「去季節化」版本：log RV_fut − log futtod ~ log σ² 各尺度（已含時段的 σ 也減掉目前時段因子）
    X[:, 0] -= np.log(pd.Series(tod).rolling(12).mean().to_numpy())
    X[:, 1] -= np.log(pd.Series(tod).rolling(48).mean().to_numpy())
    ylog = np.log(rvf / fut)
    preds = {k: np.full(len(df), np.nan) for k in ("naive", "naive_tod", "ewma", "har_tod")}
    preds["naive"] = rvS * H
    preds["naive_tod"] = np.exp(X[:, 0]) * fut
    preds["ewma"] = df.ewma.to_numpy() * H
    mon = df.month.to_numpy()
    for m in months:
        tr = (mon < m) & np.isfinite(ylog) & np.isfinite(X).all(1)
        te = (mon == m) & np.isfinite(X).all(1)
        beta = ols(X[tr], ylog[tr])
        s2 = np.var(ylog[tr] - pred(beta, X[tr]))
        preds["har_tod"][te] = np.exp(pred(beta, X[te]) + 0.5 * s2) * fut[te]
    # 第一個測試月之前沒有可用的樣本外 HAR → 用不需估計的 naive×時段（同樣是事前可知）
    pre = mon < months[0]
    preds["har_tod"][pre] = preds["naive_tod"][pre]
    test = np.isin(mon, months) & np.isfinite(rvf) & (rvf > 0)
    rows = []
    for k, p in preds.items():
        m = test & np.isfinite(p) & (p > 0)
        ratio = rvf[m] / p[m]
        q = np.mean(ratio - np.log(ratio) - 1)
        lr = np.log(rvf[m]) - np.log(p[m])
        r2 = 1 - np.var(lr) / np.var(np.log(rvf[m]))
        rows.append(dict(model=k, QLIKE=q, log_bias=lr.mean(), R2_log=r2))
    return preds["har_tod"], pd.DataFrame(rows)


# ================================================================ 階段一：特徵資訊
def stage1(df, out):
    pr = lambda *a: print(*a, file=out)
    pr(f"\n=== 階段一：特徵是否有資訊（H={H} 根 M5，非重疊取樣 每 {H} 根一點）===")
    s = df.iloc[::H]
    s = s[np.isfinite(s.z)]
    z = s.z.to_numpy()
    yb = np.where(z > 0.5, 2, np.where(z < -0.5, 0, 1))      # 未來報酬三態（σ̂ 單位，θ = 0.5）
    absz = np.abs(z)
    vb = qbins(s.V_des.to_numpy(), np.nanquantile(s.V_des, [1 / 3, 2 / 3]))
    pr(f"樣本 {len(s)}；Y 分布 下/中/上 = {np.bincount(yb) / len(yb)}")
    pr("\n(a) I(X;Y) 與 I(X;Y|V)：X 為五分位，Y 為未來三態；p = 循環平移虛無（200 次）；單位 1e-3 nats（已 Miller–Madow 修正）")
    pr(f"{'feature':8s} {'I(X;Y)':>8s} {'p':>6s} {'null':>7s} | {'I(X;Y|V)':>9s} {'p':>6s} | {'I(X;|Y|大)':>10s} {'p':>6s} | corr(X,z) t(H1) t(H2)")
    big = (absz > np.quantile(absz, 2 / 3)).astype(int)
    h1 = s.index < SPLIT
    for f in ENT + VOLF + DIRF:
        x = s[f].to_numpy()
        ok = np.isfinite(x)
        xb = qbins(x[ok], np.quantile(x[ok], [.2, .4, .6, .8]))
        mi, p, nl = mi_test(xb, yb[ok])
        cmi, cp, _ = mi_test(xb, yb[ok], cond=vb[ok])
        bmi, bp, _ = mi_test(xb, big[ok])
        c1 = np.corrcoef(rank(x[ok & h1]), rank(z[ok & h1]))[0, 1] * np.sqrt((ok & h1).sum())
        c2 = np.corrcoef(rank(x[ok & ~h1]), rank(z[ok & ~h1]))[0, 1] * np.sqrt((ok & ~h1).sum())
        pr(f"{f:8s} {mi * 1e3:8.2f} {p:6.3f} {nl * 1e3:7.2f} | {cmi * 1e3:9.2f} {cp:6.3f} | {bmi * 1e3:10.2f} {bp:6.3f} | {c1:+6.2f} {c2:+6.2f}")
    pr("  （I(X;|Y|大)：特徵對「未來波動是否大」的資訊；corr t = Spearman × √n，前半 / 後半）")

    pr("\n(b) 熵是否改變方向訊號的意義：Spearman corr(訊號, z) 於熵 / 波動三分位內（t 值；前半 | 後半）")
    for sigf in ["MOM12", "MOM48", "Z48", "TF"]:
        for cf in ["H_T", "H_P", "I1", "V_des"]:
            x = s[cf].to_numpy()
            e = np.nanquantile(x, [1 / 3, 2 / 3])
            line = f"  {sigf:5s} | {cf:6s}"
            for g in range(3):
                parts = []
                for half in (h1, ~h1):
                    m = (qbins(x, e) == g) & half & np.isfinite(s[sigf].to_numpy()) & np.isfinite(x)
                    t = np.corrcoef(rank(s[sigf].to_numpy()[m]), rank(z[m]))[0, 1] * np.sqrt(m.sum())
                    parts.append(t)
                line += f"  q{g + 1}: {parts[0]:+5.2f}|{parts[1]:+5.2f}"
            pr(line)

    pr("\n(c) 波動聚集：corr(|r_t|, |r_{t-k}|)（M5，全樣本 / 去時段因子）")
    a = np.abs(df.r.to_numpy())
    ad = a / np.sqrt(df.tod.to_numpy())
    row = "  lag " + "  ".join(f"{k:>4d}" for k in (1, 2, 6, 12, 48, 144, 288))
    pr(row)
    pr("  raw " + "  ".join(f"{pd.Series(a).autocorr(k):4.2f}" for k in (1, 2, 6, 12, 48, 144, 288)))
    pr("  des " + "  ".join(f"{pd.Series(ad).autocorr(k):4.2f}" for k in (1, 2, 6, 12, 48, 144, 288)))


def stage1_states(df, months, out):
    pr = lambda *a: print(*a, file=out)
    pr("\n(d) 二維狀態空間：熵軸 = H_T（轉移熵），波動軸 = V_des（去時段的短 / 長波動比）")
    pr("    分界 = 前一段訓練資料的三分位（walk-forward），以下為 3~10 月測試段彙總，非重疊取樣")
    mon = df.month.to_numpy()
    st = np.full(len(df), -1)
    for m in months:
        tr = mon < m
        te = mon == m
        eH = np.nanquantile(df.H_T[tr], [1 / 3, 2 / 3])
        eV = np.nanquantile(df.V_des[tr], [1 / 3, 2 / 3])
        st[te] = qbins(df.H_T.to_numpy()[te], eH) * 3 + qbins(df.V_des.to_numpy()[te], eV)
    df["state"] = st
    s = df.iloc[::H]
    s = s[(s.state >= 0) & np.isfinite(s.z)]
    md = np.sign(s.MOM12.to_numpy())
    zc = s.z.to_numpy() * md                    # 順 12 根動能方向
    zr = -s.z.to_numpy() * np.sign(s.Z48.to_numpy())   # 往 EMA48 回歸方向
    pr(f"{'state':18s} {'n':>5s} {'|z|':>5s} {'P(|z|>2)':>8s} {'續 z':>7s} {'t':>5s} {'H1':>6s} {'H2':>6s} {'歸 z':>7s} {'t':>5s} {'H1':>6s} {'H2':>6s}")
    lab = ["低熵", "中熵", "高熵"]
    vl = ["低波", "中波", "高波"]
    h1 = s.index < SPLIT
    for k in range(9):
        m = (s.state == k).to_numpy()
        pr(f"{lab[k // 3] + '×' + vl[k % 3]:16s} {m.sum():5d} {np.abs(s.z[m]).mean():5.2f} {(np.abs(s.z[m]) > 2).mean():8.3f} "
           f"{zc[m].mean():+7.3f} {tstat(zc[m]):+5.1f} {zc[m & h1].mean():+6.3f} {zc[m & ~h1].mean():+6.3f} "
           f"{zr[m].mean():+7.3f} {tstat(zr[m]):+5.1f} {zr[m & h1].mean():+6.3f} {zr[m & ~h1].mean():+6.3f}")
    pr("  續 z = 往過去 12 根方向的未來報酬（σ̂ 單位）；歸 z = 往 EMA48 回歸方向；|z| ≈ 1 代表波動預測尺度正確")

    pr(f"\n(e) 狀態轉移矩陣 P(Z_(t+{H}) | Z_t)（測試段，非重疊）")
    a = s.state.to_numpy()
    T = np.zeros((9, 9))
    np.add.at(T, (a[:-1], a[1:]), 1)
    T = T / T.sum(1, keepdims=True)
    pr("        " + " ".join(f"{lab[j // 3][0]}{vl[j % 3][0]:>3s}" for j in range(9)))
    for i in range(9):
        pr(f"  {lab[i // 3][0]}{vl[i % 3][0]}  " + " ".join(f"{T[i, j]:5.2f}" for j in range(9)))
    pers_e = np.mean([T[i, j] for i in range(9) for j in range(9) if i // 3 == j // 3]) * 3
    pers_v = np.mean([T[i, j] for i in range(9) for j in range(9) if i % 3 == j % 3]) * 3
    pr(f"  熵軸留在同一三分位的機率 ≈ {pers_e:.2f}，波動軸 ≈ {pers_v:.2f}（獨立 = 0.33）")

    pr("\n(f) 狀態轉換事件：V_des 在 12 根內從訓練段下三分位升到上三分位（波動擴張）")
    vb = np.full(len(df), -1)
    for m in months:
        tr = mon < m
        te = mon == m
        vb[te] = qbins(df.V_des.to_numpy()[te], np.nanquantile(df.V_des[tr], [1 / 3, 2 / 3]))
    was_low = pd.Series(vb == 0).rolling(12).max().shift(1).fillna(0).to_numpy() > 0
    ev = (vb == 2) & was_low & (np.r_[-1, vb[:-1]] != 2)
    ev_i = np.flatnonzero(ev)
    keep = []
    last = -10 ** 9
    for i in ev_i:
        if i - last >= H:
            keep.append(i)
            last = i
    e = df.iloc[keep]
    e = e[np.isfinite(e.z)]
    d = np.sign(e.MOM12.to_numpy())
    zc = e.z.to_numpy() * d
    h1 = e.index < SPLIT
    pr(f"  事件 {len(e)}；未來 {H} 根往擴張方向 z = {zc.mean():+.3f}（t {tstat(zc):+.1f}；前半 {zc[h1].mean():+.3f} / 後半 {zc[~h1].mean():+.3f}）；|z| 平均 {np.abs(e.z).mean():.2f}")
    for g, nm in enumerate(lab):
        m = (e.state.to_numpy() // 3) == g
        pr(f"    事件當下 {nm}: n={m.sum():4d}  續 z {zc[m].mean():+.3f} (t {tstat(zc[m]):+.1f})")


# ================================================================ 階段二：預測模型
def design(df, kind):
    """回傳特徵矩陣（kind 決定哪幾組進模型）。交互作用 = 方向訊號 × 狀態描述。"""
    cols = {}
    for f in DIRF:
        cols[f] = df[f].to_numpy()
    if kind in ("vol", "full", "cont_full", "mr_full"):
        for f in ("V_des", "JUMP", "rho1"):
            cols[f] = df[f].to_numpy()
        for f in ("MOM12", "MOM48", "Z48"):
            cols[f + "×V"] = df[f].to_numpy() * df.V_des.to_numpy()
    if kind in ("ent", "full", "cont_full", "mr_full"):
        for f in ("H_T", "H_P", "I1", "TF", "PERS"):
            cols[f] = df[f].to_numpy()
        for f in ("MOM12", "MOM48", "Z48"):
            cols[f + "×H_T"] = df[f].to_numpy() * df.H_T.to_numpy()
            cols[f + "×H_P"] = df[f].to_numpy() * df.H_P.to_numpy()
        cols["TF×I1"] = df.TF.to_numpy() * df.I1.to_numpy()
    if kind == "full":
        cols["MOM12×H_T×V"] = df.MOM12.to_numpy() * df.H_T.to_numpy() * df.V_des.to_numpy()
    if kind.startswith("cont"):
        for f in list(cols):
            if "Z48" in f:
                del cols[f]
    if kind.startswith("mr"):
        for f in list(cols):
            if "MOM" in f or f in ("TF", "TF×I1", "PERS"):
                del cols[f]
    return pd.DataFrame(cols, index=df.index)


KINDS = ["base", "vol", "ent", "full"]


def walk_forward(df, months, kinds, lam_frac=0.01):
    mon = df.month.to_numpy()
    z = df.z.to_numpy()
    preds = {}
    for kind in kinds:
        X = design(df, kind).to_numpy()
        fin = np.isfinite(X).all(1)
        p = np.full(len(df), np.nan)
        pp = np.full(len(df), np.nan)
        for m in months:
            tr = (mon < m) & fin & np.isfinite(z)
            te = (mon == m) & fin
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12
            Xt = (X[tr] - mu) / sd
            zt = np.clip(z[tr], -5, 5)
            beta = ols(Xt, zt, lam=lam_frac * tr.sum())
            p[te] = pred(beta, (X[te] - mu) / sd)
            w = logit_fit(Xt[::3], (z[tr][::3] > 0).astype(float), lam=lam_frac * tr.sum() / 3)
            pp[te] = 1 / (1 + np.exp(-(w[0] + ((X[te] - mu) / sd) @ w[1:])))
        preds[kind] = p
        preds[kind + "_p"] = pp
    return preds


def model_b(df, months):
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
    except ImportError:
        return None
    X = design(df, "full")
    X = pd.concat([X, df[ENT + VOLF]], axis=1).loc[:, lambda d: ~d.columns.duplicated()].to_numpy()
    mon = df.month.to_numpy()
    z = df.z.to_numpy()
    p = np.full(len(df), np.nan)
    for m in months:
        tr = (mon < m) & np.isfinite(z)
        te = mon == m
        g = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.03, max_iter=200,
                                          min_samples_leaf=400, l2_regularization=1.0, random_state=0)
        g.fit(X[tr][::2], np.clip(z[tr][::2], -5, 5))
        p[te] = g.predict(X[te])
    return p


def model_c(df, months, k=200):
    """狀態條件：H_T 三分位 × V_des 三分位 × MOM12 正負 → 訓練段平均 z（收縮至 0）。"""
    mon = df.month.to_numpy()
    z = df.z.to_numpy()
    p = np.full(len(df), np.nan)
    for m in months:
        tr = (mon < m) & np.isfinite(z)
        te = mon == m
        eH = np.nanquantile(df.H_T[tr], [1 / 3, 2 / 3])
        eV = np.nanquantile(df.V_des[tr], [1 / 3, 2 / 3])
        cell = qbins(df.H_T.to_numpy(), eH) * 6 + qbins(df.V_des.to_numpy(), eV) * 2 + (df.MOM12.to_numpy() > 0)
        for c in range(18):
            mm = tr & (cell == c)
            nn = mm.sum()
            p[te & (cell == c)] = z[mm].sum() / (nn + k) if nn else 0.0
    return p


def stage2(df, months, preds, out):
    pr = lambda *a: print(*a, file=out)
    pr("\n=== 階段二：預測能力（逐月 walk-forward，測試 3~10 月）===")
    mon = df.month.to_numpy()
    test = np.isin(mon, months) & np.isfinite(df.z.to_numpy())
    z = df.z.to_numpy()
    pr("IC = Spearman(預測, z)；t 用非重疊（每 H 根）樣本；AUC = 預測 P(z>0) 的方向辨識；月 IC>0 = 正 IC 的月份數")
    pr(f"{'model':10s} {'IC':>7s} {'t':>6s} {'IC H1':>7s} {'IC H2':>7s} {'月IC>0':>7s} {'AUC':>6s} {'slope':>6s}")
    stride = np.zeros(len(df), bool)
    stride[::H] = True
    h1 = df.index < SPLIT
    for k, p in preds.items():
        if k.endswith("_p") or p is None:
            continue
        m = test & np.isfinite(p)
        ic = np.corrcoef(rank(p[m]), rank(z[m]))[0, 1]
        ms = m & stride
        t = np.corrcoef(rank(p[ms]), rank(z[ms]))[0, 1] * np.sqrt(ms.sum())
        ic1 = np.corrcoef(rank(p[m & h1]), rank(z[m & h1]))[0, 1]
        ic2 = np.corrcoef(rank(p[m & ~h1]), rank(z[m & ~h1]))[0, 1]
        mics = [np.corrcoef(rank(p[m & (mon == mm)]), rank(z[m & (mon == mm)]))[0, 1] for mm in months]
        pp = preds.get(k + "_p")
        a = auc(pp[m], (z[m] > 0).astype(int)) if pp is not None else auc(p[m], (z[m] > 0).astype(int))
        slope = np.polyfit(p[m], np.clip(z[m], -5, 5), 1)[0]
        pr(f"{k:10s} {ic:+7.4f} {t:+6.2f} {ic1:+7.4f} {ic2:+7.4f} {sum(x > 0 for x in mics):>4d}/{len(months)} {a:6.3f} {slope:6.2f}")
    pr("  slope = z 對預測的迴歸斜率（≈1 代表預測幅度有校準；<<1 代表過度自信）")
    pp = preds["full_p"]
    m = test & np.isfinite(pp)
    pr("\n機率校準（full logistic，測試段）：預測 P(上漲) 分組 vs 實際上漲比例")
    e = np.quantile(pp[m], np.linspace(0, 1, 6))
    for i in range(5):
        mm = m & (pp >= e[i]) & (pp <= e[i + 1])
        pr(f"  P̂ {e[i]:.3f}~{e[i + 1]:.3f}: 平均 {pp[mm].mean():.3f}  實際 {(z[mm] > 0).mean():.3f}  n={mm.sum()}")


# ================================================================ 階段三：交易
def simulate(df, signal, mask, b, cost_mult=1.0, stop_k=STOP_K, rand=False):
    """signal = 預測 z；交易條件 |μ̂| > C + B：μ̂ = signal × σ̂，C = 點差，B = b × σ̂。
    進場 open_{t+1}，停損 stop_k × σ̂，到期 close_{t+H}；一次只持一筆；R = 損益 / 停損距離。"""
    o = df.open.to_numpy()
    hi = df.high.to_numpy()
    lo = df.low.to_numpy()
    c = df.close.to_numpy()
    sh = df.sig_h.to_numpy()
    spr = df.spread.to_numpy() * 0.01 * cost_mult
    tm = df.index
    n = len(df)
    yok = np.isfinite(df.y.to_numpy())
    trades = []
    busy = -1
    for t in np.flatnonzero(mask):
        if t <= busy or t + H >= n or not yok[t] or not np.isfinite(signal[t]):
            continue
        mu = signal[t] * sh[t] * c[t]
        C = spr[t + 1]
        Bt = b * sh[t] * c[t]
        if abs(mu) <= C + Bt:
            continue
        d = np.sign(mu) if not rand else RNG.choice([-1, 1])
        e = o[t + 1]
        risk = stop_k * sh[t] * c[t]
        stop = e - d * risk
        x, out = c[t + H], t + H
        for j in range(t + 1, t + H + 1):
            if (lo[j] <= stop) if d > 0 else (hi[j] >= stop):
                x, out = stop, j
                break
        busy = out
        trades.append((tm[t], d, ((x - e) * d - C) / risk, C / risk, out == t + H))
    return pd.DataFrame(trades, columns=["time", "dir", "R", "costR", "timeexit"])


def tstats(tr, days):
    if len(tr) < 5:
        return dict(n=len(tr))
    R = tr.R.to_numpy()
    daily = tr.groupby(tr.time.dt.normalize()).R.sum().reindex(days, fill_value=0.0)
    eq = daily.cumsum()
    dd = (eq - eq.cummax()).min()
    neg = daily[daily < 0]
    gp, gl = R[R > 0].sum(), -R[R < 0].sum()
    h1 = tr.time < SPLIT
    return dict(n=len(R), perday=len(R) / len(days), avgR=R.mean(), t=tstat(R), win=(R > 0).mean(),
                PF=gp / gl if gl > 0 else np.inf, sharpe=daily.mean() / daily.std() * np.sqrt(252),
                sortino=daily.mean() / np.sqrt((neg ** 2).sum() / len(daily)) * np.sqrt(252) if len(neg) else np.inf,
                maxDD=dd, tail5=np.sort(R)[:max(1, len(R) // 20)].mean(), costR=tr.costR.mean(),
                H1=R[h1].mean() if h1.any() else np.nan, H2=R[~h1].mean() if (~h1).any() else np.nan)


def fmt(name, s):
    if s.get("n", 0) < 5:
        return f"  {name:28s} n={s.get('n', 0)}"
    return (f"  {name:28s} n={s['n']:5d} {s['perday']:4.1f}/日 avg={s['avgR']:+.3f}R t={s['t']:+5.1f} win={s['win']:.0%} "
            f"PF={s['PF']:.2f} Sh={s['sharpe']:+5.2f} So={s['sortino']:+5.2f} DD={s['maxDD']:+6.1f}R "
            f"尾5%={s['tail5']:+.2f} 成本={s['costR']:.3f}R | H1 {s['H1']:+.3f} H2 {s['H2']:+.3f}")


def stage3(df, months, preds, out):
    pr = lambda *a: print(*a, file=out)
    pr(f"\n=== 階段三：交易價值（停損 {STOP_K}σ̂、持有 {H} 根、一次一筆、點差每筆扣一次；R = 損益 / 停損距離）===")
    mon = df.month.to_numpy()
    test = np.isin(mon, months)
    days = pd.DatetimeIndex(sorted(set(df.index[test].normalize())))
    pr("(a) 門檻 B = b × σ̂ 掃描（b 未經樣本外挑選 → 看整條曲線，不是挑最好一格）")
    for k in ["base", "vol", "ent", "full", "B_gbm", "C_state"]:
        p = preds.get(k)
        if p is None:
            continue
        for b in (0.0, 0.1, 0.2, 0.3):
            tr = simulate(df, p, test, b)
            pr(fmt(f"{k:8s} b={b:.1f}", tstats(tr, days)))
    pr("\n(b) 預先登記設定 b=0.1：成本壓力、隨機方向、模組")
    for k in ["base", "full"]:
        p = preds[k]
        for cm in (1.0, 2.0, 3.0):
            pr(fmt(f"{k:8s} 成本×{cm:.0f}", tstats(simulate(df, p, test, 0.1, cost_mult=cm), days)))
        rr = [tstats(simulate(df, p, test, 0.1, rand=True), days)["avgR"] for _ in range(20)]
        pr(f"  {k:8s} 同進場時點隨機方向（20 次）avgR = {np.mean(rr):+.3f} ± {np.std(rr):.3f}")
    for k in ["cont_base", "cont_full", "mr_base", "mr_full"]:
        pr(fmt(k, tstats(simulate(df, preds[k], test, 0.1), days)))
    pr("  cont = 方向延續模組（MOM / TF 及其 × 熵、× 波動）；mr = 條件均值回歸模組（Z48 為價格參考，× 熵、× 波動）")

    pr("\n(c) 消融：full 去掉某一組後的 IC 與交易（b=0.1）")
    full = design(df, "full")
    for drop in ["H_T", "H_P", "TF", "I1", "×V", "JUMP"]:
        cols = [c for c in full.columns if drop not in c]
        X = full[cols].to_numpy()
        p = _wf_cols(df, X, months)
        m = test & np.isfinite(p) & np.isfinite(df.z.to_numpy())
        ic = np.corrcoef(rank(p[m]), rank(df.z.to_numpy()[m]))[0, 1]
        pr(fmt(f"−{drop:5s} IC={ic:+.4f}", tstats(simulate(df, p, test, 0.1), days)))

    pr("\n(d) 波動引擎用於風險控制：同一 full 訊號，比較停損 / 交易許可設定（b=0.1）")
    p = preds["full"]
    for sk in (1.0, 1.5, 2.0, 3.0):
        pr(fmt(f"停損 {sk}σ̂", tstats(simulate(df, p, test, 0.1, stop_k=sk), days)))
    hiJ = test & (df.JUMP.to_numpy() < np.nanquantile(df.JUMP.to_numpy()[~test], 0.8))
    pr(fmt("排除 JUMP 最高 20%", tstats(simulate(df, p, hiJ, 0.1), days)))
    cr = df.spread.to_numpy() * 0.01 / (df.sig_h.to_numpy() * df.close.to_numpy())
    hiC = test & (cr < np.nanquantile(cr[~test], 0.8))
    pr(fmt("排除 成本/σ̂ 最高 20%", tstats(simulate(df, p, hiC, 0.1), days)))


def _wf_cols(df, X, months, lam_frac=0.01):
    mon = df.month.to_numpy()
    z = df.z.to_numpy()
    fin = np.isfinite(X).all(1)
    p = np.full(len(df), np.nan)
    for m in months:
        tr = (mon < m) & fin & np.isfinite(z)
        te = (mon == m) & fin
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12
        beta = ols((X[tr] - mu) / sd, np.clip(z[tr], -5, 5), lam=lam_frac * tr.sum())
        p[te] = pred(beta, (X[te] - mu) / sd)
    return p


# ================================================================ main
def main():
    out = open(f"results_part54_h{H}.txt", "w")
    pr = lambda *a: print(*a, file=out)
    df = build()
    months = sorted(m for m in df.month.unique() if m >= pd.Period("2026-03", "M"))
    pr(f"第五十四部分：Entropy × Volatility 研究架構（M5，H={H} 根 = {H * 5} 分鐘）")
    pr(f"資料 {df.index[0]} ~ {df.index[-1]}，{len(df)} 根；walk-forward 測試月 {months[0]} ~ {months[-1]}")

    pr("\n=== Volatility Engine：未來 H 根 realized variance 預測（逐月 walk-forward，測試段）===")
    sh2, vt = vol_forecast(df, months)
    pr(vt.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    pr("  QLIKE 越小越好；har_tod = log-HAR（多尺度 RV + 跳躍）× 未來時段因子")
    df["sig_h"] = np.sqrt(sh2)
    df["z"] = df.y / df.sig_h
    mon = df.month.to_numpy()
    test = np.isin(mon, months) & np.isfinite(df.z.to_numpy())
    z = df.z.to_numpy()[test]
    pr(f"  z = y/σ̂ 測試段：std {z.std():.3f}，|z|>2 {np.mean(np.abs(z) > 2):.3f}（常態 0.046），|z|>3 {np.mean(np.abs(z) > 3):.4f}（常態 0.0027）")
    for q in (0.9, 0.99):
        cov = []
        for m in months:
            tr = (mon < m) & np.isfinite(df.z.to_numpy()) & np.isfinite(df.sig_h.to_numpy())
            qq = np.quantile(np.abs(df.z.to_numpy()[tr]), q) if tr.sum() > 1000 else np.quantile(np.abs(z), q)
            te = (mon == m) & np.isfinite(df.z.to_numpy())
            cov.append(np.mean(np.abs(df.z.to_numpy()[te]) <= qq))
        pr(f"  {q:.0%} 區間（分位數取自訓練段）測試覆蓋率：平均 {np.mean(cov):.3f}，逐月 " + " ".join(f"{c:.2f}" for c in cov))

    stage1(df, out)
    stage1_states(df, months, out)
    preds = walk_forward(df, months, KINDS + ["cont_base", "cont_full", "mr_base", "mr_full"])
    preds["B_gbm"] = model_b(df, months)
    preds["C_state"] = model_c(df, months)
    stage2(df, months, {k: v for k, v in preds.items() if not k.startswith(("cont", "mr"))}, out)
    stage3(df, months, preds, out)
    out.close()
    print(open(f"results_part54_h{H}.txt").read())


if __name__ == "__main__":
    main()
