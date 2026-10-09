"""第四十二部分：(2) 用更直接的順勢指標取代 OU z；(3) 多參數平均（集成）；(4) 隨機化檢定、PBO、Deflated Sharpe。

時間尺度：回看 70 / 100 / 130 / 200 小時；週期 M5 / M15 / H1（L = 小時 × 60 / 分鐘）。
指標（|指標| 從門檻下方穿越到上方時，下一根開盤進場）
  tstat  漂移 t 值 = mean(Δp)/std(Δp)·√L                     方向 = sign，門檻 2 / 2.5 / 3
  er     ER = (p_t − p_{t−L}) / Σ|Δp|                          方向 = sign，門檻 0.10 / 0.15 / 0.20
  ou     OU z（第 37 部分）                                     方向 = −sign(z)，門檻 2 / 2.5 / 3
出場（同第 39 部分基準）：停損 1.5 ATR、最多 40 根、週五最後一根平倉、週五 20 點後不開新單；點差 + 隔夜；R = 損益 / 初始停損；同一設定最多 1 單。
共 3 指標 × 3 週期 × 4 時間尺度 × 3 門檻 = 108 組。

(3) 集成：同一指標、同一門檻的 12 組（3 週期 × 4 時間尺度）每組權重 1/12 一起做；以及全部 108 組平均。
(4) 檢定
  a. 隨機方向：同樣的進場 K 棒，方向隨機（200 次）→ 方向有沒有價值
  b. 隨機時點：同樣筆數、隨機 K 棒進場，方向 = 過去 L 根漲跌（永遠順勢）（200 次）→ 「穿越門檻」的時點有沒有比「隨便什麼時候順勢進」好
  c. PBO（CSCV，108 組日損益，10 段、252 種切法）
  d. Deflated Sharpe（Bailey & López de Prado；試驗數 = 108，另以本系列研究粗估的總試驗數 1000 再算一次）
"""
import itertools
from statistics import NormalDist

import numpy as np
import pandas as pd
from load import load
from part37_wf import zscore
from part39_ou_vs_er import Frame, metr, daily, SWAP, SPLIT

HOURS = [70, 100, 130, 200]
THR = {"tstat": [2.0, 2.5, 3.0], "er": [0.10, 0.15, 0.20], "ou": [2.0, 2.5, 3.0]}
SL, HOLD = 1.5, 40
N_PERM = 200
rng = np.random.default_rng(42)
ND = NormalDist()


def indicator(F, kind, L):
    c = pd.Series(F.c)
    dp = c.diff()
    if kind == "tstat":
        s = dp.rolling(L).mean() / dp.rolling(L).std() * np.sqrt(L)
        return s.to_numpy(), 1
    if kind == "er":
        return ((c - c.shift(L)) / dp.abs().rolling(L).sum()).to_numpy(), 1
    z, _ = zscore(F.df, "ou", L)
    return z, -1


def entries(F, x, sgn, thr):
    a = np.abs(x)
    prev = np.r_[np.nan, a[:-1]]
    t = np.where((a >= thr) & (prev < thr))[0]
    t = t[(t + 1 < len(x)) & ~np.isnan(F.A[t])]
    k = t + 1
    ok = ~((F.dow[k] == 4) & (F.hour[k] >= 20))
    t = t[ok]
    return t, (sgn * np.sign(x[t])).astype(int)


def sim(F, ts, ds):
    n = len(F.c)
    rows = []
    busy = -1
    for t, d in zip(ts, ds):
        k = t + 1
        if k <= busy or d == 0:
            continue
        e, a = F.o[k], F.A[t]
        risk = SL * a
        stop = e - d * risk
        j_out, x = None, None
        for j in range(k, min(k + HOLD, n)):
            if (F.l[j] <= stop) if d == 1 else (F.h[j] >= stop):
                j_out, x = j, stop
                break
            if F.wk_last[j]:
                j_out, x = j, F.c[j]
                break
        if j_out is None:
            j_out = min(k + HOLD, n) - 1
            x = F.c[j_out]
        busy = j_out
        usd = (x - e) * d - F.spr[k] - SWAP * F.nights(k, j_out)
        rows.append((F.idx[j_out], usd / risk))
    T = pd.DataFrame(rows, columns=["xtime", "R"])
    T["xtime"] = pd.to_datetime(T.xtime)
    return T


def sharpe(s):
    return s.mean() / s.std() * np.sqrt(252) if s.std() > 0 else np.nan


def main():
    m1 = load("data/XAUUSD_M1_2026.csv")
    days = sorted(set(m1.index.date))
    frames = {tf: Frame(m1, tf) for tf in ("5min", "15min", "1h")}
    lines = []
    P = lines.append
    cfgs, books, sigs, D = [], {}, {}, {}
    for kind in THR:
        for tf, F in frames.items():
            for hrs in HOURS:
                L = int(hrs * 60 / F.mins)
                x, sgn = indicator(F, kind, L)
                for thr in THR[kind]:
                    cfg = (kind, tf, hrs, thr)
                    ts, ds = entries(F, x, sgn, thr)
                    T = sim(F, ts, ds)
                    cfgs.append(cfg)
                    books[cfg], sigs[cfg] = T, (ts, ds, L)
                    D[cfg] = daily(T, days)
    M = pd.DataFrame(D).astype(float)
    M.columns = ["|".join(map(str, c)) for c in cfgs]
    M.to_csv("part42_daily_matrix.csv")

    # ---------- (2) 指標比較 ----------
    P("=" * 130)
    P("(2) 指標比較：每個指標 36 組（3 週期 × 4 時間尺度 × 3 門檻）")
    rows = []
    for cfg in cfgs:
        m = metr(books[cfg].assign(R=books[cfg].R), days) if len(books[cfg]) >= 5 else dict(n=len(books[cfg]))
        rows.append(dict(kind=cfg[0], tf=cfg[1], hrs=cfg[2], thr=cfg[3], **m))
    G = pd.DataFrame(rows)
    G.to_csv("part42_grid.csv", index=False)
    P(G.groupby("kind")[["n", "avg", "Sharpe", "前Sharpe", "後Sharpe", "P/MDD"]].agg(["mean", "median"]).round(2).to_string())
    P("\n  Sharpe 為正的組數 / 前後兩半都為正的組數")
    for k, s in G.groupby("kind"):
        P(f"    {k:<6} {int((s.Sharpe > 0).sum())}/{len(s)}  兩半都正 {int(((s['前Sharpe'] > 0) & (s['後Sharpe'] > 0)).sum())}/{len(s)}")
    for dim in ("hrs", "tf", "thr"):
        P(f"\n  依 {dim}（平均 Sharpe）")
        P(G.pivot_table(index="kind", columns=dim, values="Sharpe", aggfunc="mean").round(2).to_string())
    P("\n  每個指標每天筆數（平均）：" + " ".join(f"{k} {s.n.mean() / len(days):.2f}" for k, s in G.groupby("kind")))
    # 訊號重疊：同一天同方向
    P("\n  訊號相似度（M15、100 小時、中間門檻）：兩指標每日 R 的相關係數")
    mid = {k: THR[k][1] for k in THR}
    keys = {k: f"{k}|15min|100|{mid[k]}" for k in THR}
    P("    " + "  ".join(f"{a}-{b} {M[keys[a]].corr(M[keys[b]]):+.2f}" for a, b in itertools.combinations(THR, 2)))

    # ---------- (3) 集成 ----------
    P("\n" + "=" * 130)
    P("(3) 集成：多組一起做（每組權重 1/N）vs 單組。Sharpe / 獲利回撤比 與比例縮放無關")

    def ens_stats(cols):
        s = M[cols].mean(axis=1)
        eq = s.cumsum().to_numpy()
        mdd = -(eq - np.maximum.accumulate(np.r_[0, eq])[1:]).min()
        first = [d for d in days if pd.Timestamp(d) < SPLIT]
        return dict(Sharpe=sharpe(s), 前=sharpe(s.loc[first]), 後=sharpe(s.drop(first)), PMDD=s.sum() / mdd if mdd > 0 else np.nan)

    for kind in THR:
        for thr in THR[kind]:
            cols = [c for c in M.columns if c.startswith(kind + "|") and c.endswith("|" + str(thr))]
            e = ens_stats(cols)
            singles = G[(G.kind == kind) & (G.thr == thr)].Sharpe
            P(f"  {kind:<6} thr={thr:<5} 集成 12 組：Sharpe {e['Sharpe']:+.2f}（前 {e['前']:+.2f} / 後 {e['後']:+.2f}）獲利/回撤 {e['PMDD']:5.1f}"
              f" | 單組 Sharpe 中位數 {singles.median():+.2f}、最好 {singles.max():+.2f}、最差 {singles.min():+.2f}")
        cols = [c for c in M.columns if c.startswith(kind + "|")]
        e = ens_stats(cols)
        P(f"  {kind:<6} 全部 36 組集成：Sharpe {e['Sharpe']:+.2f}（前 {e['前']:+.2f} / 後 {e['後']:+.2f}）獲利/回撤 {e['PMDD']:5.1f}")
    e = ens_stats(list(M.columns))
    P(f"  三種指標全部 108 組集成：Sharpe {e['Sharpe']:+.2f}（前 {e['前']:+.2f} / 後 {e['後']:+.2f}）獲利/回撤 {e['PMDD']:5.1f}")

    # ---------- (4a,b) 隨機化 ----------
    P("\n" + "=" * 130)
    P(f"(4a/4b) 隨機化檢定（每個指標取中間門檻的 12 組集成，{N_PERM} 次）")
    for kind in THR:
        sel = [c for c in cfgs if c[0] == kind and c[3] == mid[kind]]
        real = sharpe(M[["|".join(map(str, c)) for c in sel]].mean(axis=1))
        rd, rt = [], []
        for _ in range(N_PERM):
            dd, dt = {}, {}
            for c in sel:
                F = frames[c[1]]
                ts, ds, L = sigs[c]
                dd[c] = daily(sim(F, ts, rng.choice([-1, 1], size=len(ds))), days)
                valid = np.arange(L + 1, len(F.c) - 1)
                valid = valid[~((F.dow[valid + 1] == 4) & (F.hour[valid + 1] >= 20))]
                tr = np.sort(rng.choice(valid, size=len(ts), replace=False))
                dt[c] = daily(sim(F, tr, np.sign(F.c[tr] - F.c[tr - L]).astype(int)), days)
            rd.append(sharpe(pd.DataFrame(dd).mean(axis=1)))
            rt.append(sharpe(pd.DataFrame(dt).mean(axis=1)))
        rd, rt = np.array(rd), np.array(rt)
        P(f"  {kind:<6} 真實 Sharpe {real:+.2f} | 隨機方向：平均 {rd.mean():+.2f}、95% 分位 {np.quantile(rd, .95):+.2f}、"
          f"p = {np.mean(rd >= real):.3f} | 隨機時點但順勢：平均 {rt.mean():+.2f}、95% 分位 {np.quantile(rt, .95):+.2f}、"
          f"p = {np.mean(rt >= real):.3f}")

    # ---------- (4c) PBO ----------
    P("\n" + "=" * 130)
    S = 10
    blocks = np.array_split(np.arange(len(M)), S)
    sh = lambda X: X.mean() / X.std().replace(0, np.nan) * np.sqrt(252)
    logits, oos_best, oos_med = [], [], []
    for ins in itertools.combinations(range(S), S // 2):
        i_idx = np.concatenate([blocks[k] for k in ins])
        o_idx = np.concatenate([blocks[k] for k in range(S) if k not in ins])
        s_in, s_out = sh(M.iloc[i_idx]), sh(M.iloc[o_idx])
        best = s_in.idxmax()
        w = min(max(s_out.rank(pct=True)[best], 1e-6), 1 - 1e-6)
        logits.append(np.log(w / (1 - w)))
        oos_best.append(s_out[best])
        oos_med.append(s_out.median())
    logits, oos_best, oos_med = map(np.array, (logits, oos_best, oos_med))
    P(f"(4c) PBO（108 組，10 段、252 種切法）= {(logits <= 0).mean():.1%}")
    P(f"     樣本內最佳 → 樣本外 Sharpe：平均 {oos_best.mean():+.2f}、中位數 {np.median(oos_best):+.2f}、為負 {(oos_best < 0).mean():.0%}"
      f"；同切法下全部 108 組樣本外 Sharpe 中位數的平均 {oos_med.mean():+.2f}")

    # ---------- (4d) Deflated Sharpe ----------
    P("\n(4d) Deflated Sharpe（以日報酬計；SR 為未年化的日 Sharpe）")
    sr_all = (M.mean() / M.std().replace(0, np.nan)).dropna()
    P(f"  （{len(M.columns) - len(sr_all)} 組完全沒有交易，不計）")
    best = sr_all.idxmax()
    r = M[best]
    T_ = len(r)
    sk = ((r - r.mean()) ** 3).mean() / r.std(ddof=0) ** 3
    ku = ((r - r.mean()) ** 4).mean() / r.std(ddof=0) ** 4
    g = 0.5772156649
    for N in (len(sr_all), 1000):
        V = sr_all.var()
        sr0 = np.sqrt(V) * ((1 - g) * ND.inv_cdf(1 - 1 / N) + g * ND.inv_cdf(1 - 1 / (N * np.e)))
        sr = sr_all[best]
        dsr = ND.cdf((sr - sr0) * np.sqrt(T_ - 1) / np.sqrt(1 - sk * sr + (ku - 1) / 4 * sr ** 2))
        P(f"  試驗數 N={N:<5} 最佳組 {best}：年化 Sharpe {sr * np.sqrt(252):+.2f}；運氣門檻（預期最大值）{sr0 * np.sqrt(252):+.2f}；"
          f"DSR = {dsr:.3f}（> 0.95 才算顯著）")
    for lab, cols in (("108 組集成", list(M.columns)), ("tstat 36 組集成", [c for c in M.columns if c.startswith("tstat|")])):
        s = M[cols].mean(axis=1)
        sr = s.mean() / s.std()
        sk2 = ((s - s.mean()) ** 3).mean() / s.std(ddof=0) ** 3
        ku2 = ((s - s.mean()) ** 4).mean() / s.std(ddof=0) ** 4
        psr = ND.cdf(sr * np.sqrt(T_ - 1) / np.sqrt(1 - sk2 * sr + (ku2 - 1) / 4 * sr ** 2))
        P(f"  {lab}：年化 Sharpe {sr * np.sqrt(252):+.2f}；Probabilistic Sharpe（> 0 的機率，未扣多重測試）= {psr:.3f}")

    txt = "\n".join(lines)
    print(txt)
    open("results_part42.txt", "w").write(txt + "\n")


if __name__ == "__main__":
    main()
