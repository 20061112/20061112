"""第三十七部分：OU / 均值回歸 z 策略 — 均值算法 × 長度 × z 門檻比較，滾動前推，M1 執行。

均值算法（長度 L）
  ou     最近 L 根等權回歸 Δp = a + b·p₋₁ → μ* = -a/b、σ_eq = σ_ε/√(1-φ²)
  ou_ew  同上，但回歸用 EWMA 權重（span = L）
  sma    μ = SMA(L)，σ = 滾動標準差(L)
  ema    μ = EWMA(span L)，σ = EWMA 標準差(span L)
  z = (p - μ)/σ；|z| 首次 ≥ thr 那根收盤後進場，方向往 μ；目標 = 進場時的 μ（固定），停損 1.5 ATR14，最多 40 根。
週期 M5 / M15 / H1；L ∈ {50,100,200,400}；thr ∈ {1.5,2,2.5,3} → 192 組。

滾動前推：每月月初只用「已經出場」的過去交易（擴張視窗）幫每組打分數 = 平均R×√n（n ≥ 20），
  選第 1 名（Top1）或前 3 名合併（Top3），用在該月；3 月起都是樣本外。另外各均值算法各自選（看算法差異）。
M1 執行：同一批訊號改用 M1 K 棒走：進場 = 訊號 K 棒收盤後第一根 M1 開盤，停損用 M1 高低點，
  目標兩種：M1 收盤越過 μ（close）或 M1 高/低點碰到 μ 就以 μ 成交（touch，限價單）；
  時間上限 40 根原週期；點差用進場那根 M1；全組合同時最多 1 筆（不重疊）。
"""
import numpy as np
import pandas as pd
from load import load
from spring import resample, atr

TFS = {"5min": 5, "15min": 15, "1h": 60}
METHODS = ["ou", "ou_ew", "sma", "ema"]
LS = [50, 100, 200, 400]
THRS = [1.5, 2.0, 2.5, 3.0]
SPLIT = pd.Timestamp("2026-06-01")
SL, HOLD = 1.5, 40


def zscore(df, method, L):
    c = df.close
    if method in ("ou", "ou_ew"):
        p = c - c.iloc[0]
        x, y = p.shift(), p.diff()
        R = (lambda s: s.rolling(L).mean()) if method == "ou" else (lambda s: s.ewm(span=L, min_periods=L).mean())
        mx, my = R(x), R(y)
        sxx, sxy, syy = R(x * x) - mx * mx, R(x * y) - mx * my, R(y * y) - my * my
        b = sxy / sxx
        a = my - b * mx
        rv = (syy - b * sxy).clip(lower=1e-12)
        phi = 1 + b
        ok = (b < 0) & (phi > 0)
        mu = ((-a / b) + c.iloc[0]).where(ok)
        sd = np.sqrt(rv / (1 - phi ** 2)).where(ok)
    elif method == "sma":
        mu, sd = c.rolling(L).mean(), c.rolling(L).std()
    else:
        mu, sd = c.ewm(span=L, min_periods=L).mean(), c.ewm(span=L, min_periods=L).std()
    return ((c - mu) / sd).to_numpy(), mu.to_numpy()


def make_signals(z, thr):
    za = np.abs(z)
    prev = np.r_[np.nan, za[:-1]]
    i = np.where((za >= thr) & (prev < thr))[0]
    return i, -np.sign(z[i]).astype(int)


def sim_tf(df, A, mu, idx, dirs):
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    spr = df.spread.to_numpy(float) * 0.01
    n = len(c)
    rows = []
    busy = -1
    for t, d in zip(idx, dirs):
        if t + 1 >= n or t + 1 <= busy or np.isnan(A[t]) or np.isnan(mu[t]):
            continue
        e, risk, tgt = o[t + 1], SL * A[t], mu[t]
        if d * (tgt - e) <= 0:            # 開盤已經越過目標
            continue
        stop = e - d * risk
        out = None
        for j in range(t + 1, min(t + 1 + HOLD, n)):
            if (l[j] <= stop) if d == 1 else (h[j] >= stop):
                out, x = j, stop
                break
            if d * (c[j] - tgt) >= 0:
                out, x = j, c[j]
                break
        if out is None:
            out = min(t + HOLD, n - 1)
            x = c[out]
        busy = out
        rows.append((t, d, ((x - e) * d - spr[t + 1]) / risk, out))
    return rows


def sim_m1(m1, sigs, mode):
    """sigs: DataFrame(time_entry, dir, risk, tgt, tmax)，依時間排序；全組合同時最多 1 筆。"""
    T = m1.index.to_numpy()
    o, h, l, c = (m1[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    spr = m1.spread.to_numpy(float) * 0.01
    R = np.full(len(sigs), np.nan)
    ex = np.full(len(sigs), np.datetime64("NaT"), dtype="datetime64[ns]")
    busy = np.datetime64("1970-01-01")
    for q, s in enumerate(sigs.itertuples()):
        te = np.datetime64(s.t_entry)
        if te < busy:
            continue
        k0 = np.searchsorted(T, te)
        if k0 >= len(T) or T[k0] - te > np.timedelta64(30, "m"):
            continue
        k1 = np.searchsorted(T, np.datetime64(s.t_max))
        d, e = s.dir, o[k0]
        if d * (s.tgt - e) <= 0:
            continue
        stop = e - d * s.risk
        x, out = None, None
        for j in range(k0, min(k1, len(T))):
            if (l[j] <= stop) if d == 1 else (h[j] >= stop):
                x, out = stop, j
                break
            if mode == "touch":
                if (h[j] >= s.tgt) if d == 1 else (l[j] <= s.tgt):
                    x, out = s.tgt, j
                    break
            elif d * (c[j] - s.tgt) >= 0:
                x, out = c[j], j
                break
        if out is None:
            out = min(k1, len(T)) - 1
            x = c[out]
        R[q] = ((x - e) * d - spr[k0]) / s.risk
        ex[q] = T[out]
        busy = T[out] + np.timedelta64(1, "m")
    return R, ex


def summ(r, label):
    r = r[~np.isnan(r)]
    if len(r) == 0:
        return f"{label:<34} n=   0"
    pf = r[r > 0].sum() / -r[r < 0].sum() if (r < 0).any() else np.inf
    eq = np.cumsum(r)
    dd = (eq - np.maximum.accumulate(np.r_[0, eq])[1:]).min()
    t = r.mean() / r.std(ddof=1) * np.sqrt(len(r)) if len(r) > 2 else np.nan
    return (f"{label:<34} n={len(r):4d} avg={r.mean():+.3f}R t={t:+.1f} win={np.mean(r > 0):.0%} PF={pf:.2f} "
            f"總和={r.sum():+6.1f}R 最大回撤={dd:+.1f}R")


def main():
    m1 = load("data/XAUUSD_M1_2026.csv")
    lines = []
    P = lines.append
    trades = []
    meta = {}
    for tf, mins in TFS.items():
        df = resample(m1, tf)
        A = atr(df).to_numpy()
        for method in METHODS:
            for L in LS:
                z, mu = zscore(df, method, L)
                for thr in THRS:
                    idx, dirs = make_signals(z, thr)
                    cfg = f"{tf}|{method}|{L}|{thr}"
                    meta[cfg] = dict(tf=tf, method=method, L=L, thr=thr)
                    for t, d, r, out in sim_tf(df, A, mu, idx, dirs):
                        trades.append(dict(cfg=cfg, t_sig=df.index[t], t_entry=df.index[t] + pd.Timedelta(minutes=mins),
                                           t_exit=df.index[out] + pd.Timedelta(minutes=mins),
                                           t_max=df.index[t] + pd.Timedelta(minutes=mins * (HOLD + 1)),
                                           dir=d, risk=SL * A[t], tgt=mu[t], R=r))
    T = pd.DataFrame(trades)
    T.to_csv("part37_all_trades.csv.gz", index=False)

    # ---------- 1. 全期間（樣本內）比較 ----------
    g = T.groupby("cfg").R
    G = pd.DataFrame({"n": g.size(), "R": g.mean(), "sum": g.sum()})
    h1 = T[T.t_sig < SPLIT].groupby("cfg").R.mean()
    h2 = T[T.t_sig >= SPLIT].groupby("cfg").R.mean()
    G["R_H1"], G["R_H2"] = h1, h2
    G = G.join(pd.DataFrame.from_dict(meta, orient="index"))
    G.to_csv("part37_grid.csv")
    P("=" * 120)
    P("1. 樣本內：192 組（每組 R = 每筆平均 R，已扣點差）")
    P(f"   全部組合：平均 {G.R.mean():+.3f}R；為正 {np.mean(G.R > 0):.0%}；前後兩半都為正 {np.mean((G.R_H1 > 0) & (G.R_H2 > 0)):.0%}")
    for dim in ("method", "L", "thr", "tf"):
        P(f"\n   依 {dim}（各組平均 R 的平均｜兩半都正的比例｜平均筆數）")
        for k, s in G.groupby(dim):
            P(f"     {str(k):<7} 平均 {s.R.mean():+.3f}R  前半 {s.R_H1.mean():+.3f}  後半 {s.R_H2.mean():+.3f}  "
              f"兩半都正 {np.mean((s.R_H1 > 0) & (s.R_H2 > 0)):.0%}  平均筆數 {s.n.mean():.0f}")
    P("\n   均值算法 × 週期（平均 R）")
    P(G.pivot_table(index="method", columns="tf", values="R", aggfunc="mean").round(3).to_string())
    P("\n   均值算法 × 長度（平均 R）")
    P(G.pivot_table(index="method", columns="L", values="R", aggfunc="mean").round(3).to_string())
    P("\n   均值算法 × z 門檻（平均 R）")
    P(G.pivot_table(index="method", columns="thr", values="R", aggfunc="mean").round(3).to_string())
    P("\n   樣本內前 15 名（n ≥ 30，依 平均R×√n）")
    G["score"] = G.R * np.sqrt(G.n)
    P(G[G.n >= 30].sort_values("score", ascending=False).head(15)[["n", "R", "R_H1", "R_H2", "sum"]].round(3).to_string())

    # ---------- 2. 滾動前推 ----------
    months = pd.date_range("2026-03-01", "2026-10-01", freq="MS")
    P("\n" + "=" * 120)
    P("2. 滾動前推（每月月初只用已出場的過去交易打分數，分數 = 平均R×√n，n ≥ 20）")

    def pick(pool, m, k):
        past = pool[pool.t_exit < m]
        s = past.groupby("cfg").R.agg(["mean", "size"])
        s = s[s["size"] >= 20]
        s["score"] = s["mean"] * np.sqrt(s["size"])
        s = s[s.score > 0]
        return list(s.sort_values("score", ascending=False).index[:k])

    def walk(pool, k):
        sel, chosen = [], []
        for m in months:
            cfgs = pick(pool, m, k)
            nxt = m + pd.offsets.MonthBegin(1)
            cur = pool[(pool.cfg.isin(cfgs)) & (pool.t_sig >= m) & (pool.t_sig < nxt)]
            sel.append(cur)
            chosen.append((m.strftime("%Y-%m"), cfgs))
        return pd.concat(sel).sort_values("t_entry").reset_index(drop=True), chosen

    results = {}
    for lab, pool, k in [("全部 Top1", T, 1), ("全部 Top3", T, 3)] + \
                         [(f"{mt} Top1", T[T.cfg.str.contains(f"\\|{mt}\\|")], 1) for mt in METHODS] + \
                         [(f"{tf} Top1", T[T.cfg.str.startswith(tf + "|")], 1) for tf in TFS]:
        S, chosen = walk(pool, k)
        results[lab] = (S, chosen)
        P(summ(S.R.to_numpy(), f"  [原週期執行] {lab}"))
    P("\n  每月選到的組合（全部 Top1 / Top3）")
    for (mo, c1), (_, c3) in zip(results["全部 Top1"][1], results["全部 Top3"][1]):
        P(f"    {mo}: Top1 {c1}  Top3 {c3}")

    # ---------- 3. M1 執行 ----------
    P("\n" + "=" * 120)
    P("3. M1 執行（同一批前推訊號；全組合同時最多 1 筆；停損 / 時間上限同上）")
    for lab in ["全部 Top1", "全部 Top3"] + [f"{mt} Top1" for mt in METHODS]:
        S = results[lab][0]
        for mode in ("close", "touch"):
            R, ex = sim_m1(m1, S, mode)
            P(summ(R, f"  {lab} M1 {mode}"))
            if lab == "全部 Top1" and mode == "touch":
                S = S.assign(R_m1=R)
                S.to_csv("part37_wf_top1_trades.csv", index=False)
                P("    逐月（M1 touch）：")
                for mo, s in S.groupby(S.t_sig.dt.strftime("%Y-%m")):
                    r = s.R_m1.dropna()
                    P(f"      {mo}: n={len(r):3d} avg={r.mean():+.3f}R sum={r.sum():+6.1f}R")
                r = S.R_m1.dropna()
                P(f"    多/空：多 {r[S.dir == 1].mean():+.3f}R (n={(S.dir[r.index] == 1).sum()})  "
                  f"空 {r[S.dir == -1].mean():+.3f}R (n={(S.dir[r.index] == -1).sum()})")
    txt = "\n".join(lines)
    print(txt)
    open("results_part37.txt", "w").write(txt + "\n")


if __name__ == "__main__":
    main()
