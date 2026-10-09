"""第二十二部分 H：合成回復力指數 RFI + 雙向交易（高回復力做回歸、低回復力做動能）。

全部只用 IS（1/2~5/31）決定，OOS（6/1~10/8）只驗證：
  1. 量測挑選：IS1 與 IS2 的 Δβ 同向，且兩段 |Δβ| 都 ≥ 0.02；|ρ| > 0.6 的兩個量測只留 IS 較強者。
  2. RFI = 平均( 方向 × (IS 分位數 − 0.5) )，範圍 −0.5 ~ +0.5；高 = 回復力強。
  3. 交易（M5，下一根開盤進場，扣進場點差）：
       |D| ≥ d0 且 RFI ≥ IS 70% 分位 → 反向（回歸）
       |D| ≥ d0 且 RFI ≤ IS 30% 分位 → 順向（動能）
     出場：持有 H 根或停損 2 ATR；每一邊同時只持有一筆。
"""
import numpy as np
import pandas as pd
from part22_lib import load_tf, means, atr, CUT
from part22_beta import frame, state_beta, beta, FE
from part22_restore import NAMES

out = open("results_part22.txt", "a")
def P(s=""): print(s, flush=True); out.write(s + "\n"); out.flush()


def select(X, feats=FE, thr=0.02, rho_max=0.6):
    IS = X.per != "OOS"
    cand = []
    for f in feats:
        if f == "dist":                       # 距離是彈簧的「位置」，不是狀態；另外處理
            continue
        t = state_beta(X[IS], f)
        d1, d2 = t["IS1"][-1] - t["IS1"][0], t["IS2"][-1] - t["IS2"][0]
        if np.sign(d1) == np.sign(d2) and min(abs(d1), abs(d2)) >= thr:
            cand.append((f, np.sign(d1), min(abs(d1), abs(d2))))
    cand.sort(key=lambda x: -x[2])
    sub = X[IS].iloc[::6]
    keep = []
    for f, s, st in cand:
        if all(abs(sub[f].rank().corr(sub[g].rank())) <= rho_max for g, _, _ in keep):
            keep.append((f, s, st))
    return keep


def rfi(X, keep):
    IS = X.per != "OOS"
    parts = []
    for f, s, _ in keep:
        ref = np.sort(X.loc[IS, f].dropna().to_numpy())
        pct = pd.Series(np.searchsorted(ref, X[f].to_numpy()) / len(ref), X.index).where(X[f].notna())
        parts.append(s * (pct - 0.5))
    return pd.concat(parts, axis=1).mean(axis=1, skipna=True)


def simulate(d, A, D, R, d0, lo, hi, H=12, stop=2.0, legs=("mr", "mom")):
    o, h, l, c = (d[k].to_numpy() for k in ("open", "high", "low", "close"))
    spr = d.spread.to_numpy() / 100
    seg = d.seg.to_numpy(); a = A.to_numpy(); Dv = D.to_numpy(); Rv = R.to_numpy()
    n = len(c); trades = []
    for leg in legs:
        busy = -1
        for t in range(600, n - 1):
            if t <= busy or np.isnan(Dv[t]) or np.isnan(Rv[t]) or abs(Dv[t]) < d0:
                continue
            if leg == "mr" and Rv[t] < hi:
                continue
            if leg == "mom" and Rv[t] > lo:
                continue
            s = -np.sign(Dv[t]) if leg == "mr" else np.sign(Dv[t])
            e = t + 1; ep = o[e]; st = ep - s * stop * a[t]
            xp, xi = None, None
            for j in range(e, min(e + H, n)):
                if seg[j] != seg[t]:
                    xp, xi = c[j - 1], j - 1; break
                if (l[j] <= st) if s > 0 else (h[j] >= st):
                    xp, xi = (min(st, o[j]) if s > 0 else max(st, o[j])) if j > e else st, j; break
            if xp is None:
                xi = min(e + H - 1, n - 1); xp = c[xi]
            pnl = s * (xp - ep) - spr[e]
            trades.append(dict(time=d.index[t], leg=leg, side=int(s), D=Dv[t], rfi=Rv[t], atr=a[t], pnl=pnl, pnl_atr=pnl / a[t],
                               bars=xi - e + 1))
            busy = xi
    return pd.DataFrame(trades)


def stats(T):
    if len(T) == 0:
        return "   0 筆"
    p = T.pnl_atr
    pf = T.pnl[T.pnl > 0].sum() / max(-T.pnl[T.pnl <= 0].sum(), 1e-9)
    return (f"{len(T):5d} 筆  每筆 {p.mean():+.3f} ATR ({T.pnl.mean():+.2f} 美元)  勝率 {np.mean(p > 0):.0%}  PF {pf:.2f}  "
            f"t {p.mean() / p.std() * np.sqrt(len(p)):+.2f}")


def per_split(T):
    out = []
    for lab, m in (("IS1", T.time < pd.Timestamp("2026-04-01")), ("IS2", (T.time >= pd.Timestamp("2026-04-01")) & (T.time < CUT)), ("OOS", T.time >= CUT)):
        sub = T[m]
        out.append(f"{lab} {len(sub):4d}筆 {sub.pnl_atr.mean():+.3f}" if len(sub) else f"{lab}    0筆   n/a")
    return " | ".join(out)


if __name__ == "__main__":
    P("\n" + "=" * 120)
    P("第二十二部分 H：合成回復力指數 RFI（只用 IS 挑選與定權重）")
    d = load_tf("M5"); A = atr(d, 14); M = means(d)
    best = {}
    for mn in ("ema20", "ema50", "kama", "rvwap20", "linreg20"):
        X = frame(d, A, M, mn)
        keep = select(X)
        X["rfi"] = rfi(X, keep)
        t = state_beta(X, "rfi")
        IS = X.per != "OOS"
        e = np.nanquantile(X.loc[IS, "rfi"], [.3, .7])
        P(f"\n--- 均值 {mn}：選入 {len(keep)} 個量測 ---")
        P("  " + "、".join(f"{NAMES[f]}({'+' if s > 0 else '−'})" for f, s, _ in keep))
        for p in ("IS1", "IS2", "OOS"):
            P(f"  RFI 五分位 β {p}: " + " ".join(f"{v:+.3f}" for v in t[p]) + f"   Δβ {t[p][-1] - t[p][0]:+.3f}   "
              f"|D|≥1 反向12根 Q1 {t[p + '_fade'][0]:+.3f} / Q5 {t[p + '_fade'][-1]:+.3f}")
        best[mn] = (X, keep, e, np.mean([t["IS1"][-1] - t["IS1"][0], t["IS2"][-1] - t["IS2"][0]]))
    P("\nIS 平均 Δβ(RFI)：" + "  ".join(f"{k} {v[3]:+.3f}" for k, v in best.items()))
    pick = max(best, key=lambda k: best[k][3])
    P(f"→ 以 IS 選出的均值：{pick}")

    P("\n交易測試（每個均值都列出；d0 = 進場最小 |D|，H = 持有根數，停損 2 ATR）")
    rows = []
    for mn, (X, keep, e, _) in best.items():
        P(f"\n--- {mn}（RFI 30% / 70% 分位 = {e[0]:+.3f} / {e[1]:+.3f}） ---")
        for d0 in (1.0, 1.5, 2.0):
            for H in (6, 12, 24):
                T = simulate(d, A, X.D.reindex(d.index), X.rfi.reindex(d.index), d0, e[0], e[1], H=H)
                for leg, lab in (("mr", "回歸"), ("mom", "動能")):
                    S = T[T.leg == leg]
                    P(f"  d0 {d0:.1f} H {H:2d} {lab}  {per_split(S)}")
                    for per, m in (("IS", S.time < CUT), ("OOS", S.time >= CUT)):
                        sub = S[m]
                        rows.append(dict(mean=mn, d0=d0, H=H, leg=leg, per=per, n=len(sub), r=sub.pnl_atr.mean() if len(sub) else np.nan))
    G = pd.DataFrame(rows)
    G.to_csv("part22_rfi_grid.csv", index=False)
    P("\n參數格子總覽（9 組 d0×H 的平均每筆 ATR，及 OOS 為正的比例）")
    for (mn, leg), g in G.groupby(["mean", "leg"]):
        a, b = g[g.per == "IS"], g[g.per == "OOS"]
        P(f"  {mn:9s} {'回歸' if leg == 'mr' else '動能'}  IS {a.r.mean():+.3f}（正 {np.mean(a.r > 0):.0%}）  OOS {b.r.mean():+.3f}（正 {np.mean(b.r > 0):.0%}）")
