"""第五十三部分：讓權益濾網對路徑不敏感 + ATR 熵上升條件能不能用。

擾動測試：停損 1.9 / 2.0 / 2.1 × ATR15 × 滑價 0.2 / 0.3 / 0.4 美元（9 組），RR 3，A+B+斜率 → 上限 8 → 濾網 → 防護（ATR比>2.2、單日 −15R）。
每個濾網版本在 9 組擾動下各跑一次：看平均的 年獲利/含浮虧MDD、最差的一組、以及各年份損益在 9 組間的離散度（越小 = 越不敏感）。
濾網版本：
  無        不用權益濾網
  現行      影子累積 R < 最近 50 筆均線 → 不做
  減半      < 50 筆均線 → 0.5 倍（不是不做）
  投票      均線 30/40/50/60/75 筆，站上幾條就下幾成（0~1 倍）
  投票(下限0.5) 同上，但最少 0.5 倍
  緩衝帶    跌破均線 −2R 才停、站回均線 +2R 才恢復
  天數      以「日」累積 R 對其 20 個交易日均線
ATR 熵（第五十一部分）：ATR熵_M15(20) 與 10 根前相比的上升幅度 > 訓練期 80% 分位 → 0.5 倍（或 0）。
"""
import numpy as np
import pandas as pd
import part49_safeguards as P
import part24_prev_bar_breakout as B

W, IDX = P.W, P.IDX
C = W.C
EVS = P.EV

# ---------- ATR 熵變化（每個事件，用前一根已完成的 M15）----------
pid = B.period_ids(15)
g = pd.DataFrame({"pid": pid, "H": W.H, "L": W.Lo, "C": C})
b = g.groupby("pid").agg(H=("H", "max"), L=("L", "min"), C=("C", "last"))
tr15 = np.maximum(b.H - b.L, np.maximum((b.H - b.C.shift(1)).abs(), (b.L - b.C.shift(1)).abs())).values


def norm_entropy_roll(x, w):
    out = np.full(len(x), np.nan)
    v = np.lib.stride_tricks.sliding_window_view(x, w)
    s = v.sum(1, keepdims=True)
    pp = np.where(s > 0, v / s, 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.nansum(np.where(pp > 0, pp * np.log(pp), 0), 1) / np.log(w)
    out[w - 1:] = h
    return out


ent = norm_entropy_roll(np.nan_to_num(tr15), 20)
dent = ent - np.r_[np.full(10, np.nan), ent[:-10]]
ev_i = np.array([e[0] for e in EVS])
ev_dent = dent[np.maximum(pid[ev_i] - 1, 0)]
ev_t = pd.DatetimeIndex(IDX[ev_i])
ENT_CUT = np.nanquantile(ev_dent[ev_t < "2025-01-01"], 0.8)


def simulate(sm, slip):
    rows = []
    for n, (i_, L, side, a_, vr) in enumerate(EVS):
        risk = sm * a_
        r = P.G.sim_detail(i_, L, side, L - side * risk, L + side * 3 * risk)
        if r is None:
            continue
        j, px, why = r
        net = (px - L) * side - P.SP[i_] - slip
        rows.append((IDX[i_], IDX[j], i_, j, side, L, net, net / risk, risk, vr, ev_dent[n]))
    x = pd.DataFrame(rows, columns=["進場時間", "出場時間", "i", "j", "dir", "entry", "net", "R", "risk", "volrel", "dent"])
    return P.cap_total(x.sort_values("進場時間").reset_index(drop=True), 8).reset_index(drop=True)


def shadow_state(x):
    """回傳每筆進場時：已出場影子單的累積 R，以及各長度均線（以出場順序計）。"""
    ex = x.sort_values("出場時間")
    cum = ex.R.cumsum().values
    et = ex.出場時間.values
    k = np.searchsorted(et, x.進場時間.values, side="right") - 1
    c_at = np.where(k >= 0, cum[np.maximum(k, 0)], 0.0)
    mas = {}
    for L in (30, 40, 50, 60, 75):
        ma = pd.Series(cum).rolling(L, min_periods=L).mean().values
        mas[L] = np.where(k >= L, ma[np.maximum(k, 0)], np.nan)
    return c_at, mas, k


def weights(x, kind):
    c, mas, k = shadow_state(x)
    if kind == "無":
        return np.ones(len(x))
    if kind == "現行":
        return np.where(np.isfinite(mas[50]) & (c < mas[50]), 0.0, 1.0)
    if kind == "減半":
        return np.where(np.isfinite(mas[50]) & (c < mas[50]), 0.5, 1.0)
    if kind.startswith("投票"):
        votes = np.mean([np.where(np.isfinite(m), (c >= m).astype(float), 1.0) for m in mas.values()], axis=0)
        return np.maximum(votes, 0.5) if "0.5" in kind else votes
    if kind == "緩衝帶":
        w = np.ones(len(x)); on = True
        # 依進場時間序檢查狀態（狀態只用已出場的影子單）
        for n in range(len(x)):
            m = mas[50][n]
            if np.isfinite(m):
                if on and c[n] < m - 2:
                    on = False
                elif not on and c[n] > m + 2:
                    on = True
            w[n] = 1.0 if on else 0.0
        return w
    if kind == "天數":
        day = x.出場時間.dt.normalize()
        dR = x.R.groupby(day).sum().sort_index()
        cum_d = dR.cumsum(); ma_d = cum_d.rolling(20, min_periods=20).mean()
        state = (cum_d >= ma_d) | ma_d.isna()
        # 進場那天之前（前一天收盤為止）的狀態
        prev = state.shift(1).fillna(True)
        s = prev.reindex(x.進場時間.dt.normalize(), method="ffill").fillna(True).values
        return np.where(s, 1.0, 0.0)
    raise ValueError(kind)


def apply_guards(x, w):
    x = x.copy(); x["w"] = w
    x = x[(x.w > 0) & ~(x.volrel > 2.2).values].reset_index(drop=True)
    keep, real, cur = [], [], None
    for r in x.itertuples():
        day = r.進場時間.normalize()
        if day != cur:
            cur, real = day, []                      # 只保留當天的已下單
        lost = sum(v for (te, v) in real if te <= r.進場時間)
        if lost <= -15:
            keep.append(False); continue
        keep.append(True); real.append((r.出場時間, r.R * r.w))
    return x[np.array(keep)].reset_index(drop=True)


def minute_mdd(df):
    n = len(IDX)
    D, E, R = np.zeros(n + 1), np.zeros(n + 1), np.zeros(n + 1)
    for r in df.itertuples():
        q = 2 * r.w
        D[r.i] += r.dir * q; D[r.j] -= r.dir * q
        E[r.i] += r.dir * r.entry * q; E[r.j] -= r.dir * r.entry * q
        R[r.j] += r.net * q
    D, E, R = np.cumsum(D)[:n], np.cumsum(E)[:n], np.cumsum(R)[:n]
    eq = R + C * D - E
    return (np.maximum.accumulate(eq) - eq).max()


FILTERS = ["無", "現行", "減半", "投票", "投票(下限0.5)", "緩衝帶", "天數"]
ENTS = [("", None), ("+熵減半", 0.5), ("+熵不做", 0.0)]
res = {}
for sm in (1.9, 2.0, 2.1):
    for slip in (0.2, 0.3, 0.4):
        x = simulate(sm, slip)
        print(f"擾動 停損 {sm} 滑價 {slip}：{len(x)} 筆影子", flush=True)
        for fk in FILTERS:
            w0 = weights(x, fk)
            for el, ew in ENTS:
                w = w0.copy()
                if ew is not None:
                    w = np.where(x.dent.values > ENT_CUT, w * ew, w)
                y = apply_guards(x, w)
                pnl = y.net * 2 * y.w
                yrs = (y.進場時間.max() - y.進場時間.min()).days / 365
                mm = minute_mdd(y)
                yr = {yy: pnl[(y.進場時間.dt.year == yy).values].sum() for yy in (2023, 2024, 2025, 2026)}
                trm = (y.進場時間 < "2025-01-01").values

                def rat(m):
                    s_ = pnl[m].groupby(y.進場時間.dt.date[m]).sum(); e = s_.cumsum(); return pnl[m].sum() / (e.cummax() - e).max()
                res.setdefault((fk, el), []).append(dict(tot=pnl.sum(), ratio=pnl.sum() / yrs / mm, mdd=mm, tr=rat(trm), va=rat(~trm), **{f"y{k}": v for k, v in yr.items()}))

lines = ["第五十三部分：權益濾網的穩健版本 + ATR 熵條件（9 組擾動：停損 1.9/2.0/2.1 × 滑價 0.2/0.3/0.4；0.02 口）",
         f"ATR 熵門檻（訓練期 80% 分位）= {ENT_CUT:+.3f}",
         f"{'版本':22s} {'平均總損益':>9s} {'平均MDD':>7s} {'平均比值':>7s} {'最差比值':>7s} | {'訓練(平均/最差)':>14s} {'驗證(平均/最差)':>14s} | 各年損益 平均 ± 擾動間標準差（2023 / 2024 / 2025 / 2026）"]
for (fk, el), rs in res.items():
    D = pd.DataFrame(rs)
    yrtxt = " / ".join(f"{D[f'y{y}'].mean():+6,.0f}±{D[f'y{y}'].std():4,.0f}" for y in (2023, 2024, 2025, 2026))
    lines.append(f"{fk + el:22s} {D.tot.mean():+9,.0f} {D.mdd.mean():7,.0f} {D.ratio.mean():7.2f} {D.ratio.min():7.2f} | "
                 f"{D.tr.mean():6.1f} / {D.tr.min():5.1f}  {D.va.mean():6.1f} / {D.va.min():5.1f} | {yrtxt}")
print("\n".join(lines))
open("results_part53.txt", "w").write("\n".join(lines) + "\n")
