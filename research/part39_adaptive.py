"""第三十九部分：還有哪些固定參數可以自適應？

基準 = FINAL_STRATEGY_OW + 2R 保本、格子 1 美元、OW 篩選後去重、0.01 手最多 5 張。
A. 保本觸發：2R（R = IB 寬，隨 IB 大小變）vs 浮盈 ≥ k×ATR10 vs 兩者取大 / 取小
B. 停損上限：停損距離最多 k×ATR10（超過就用 進場 ∓ k×ATR，R 也跟著變小）
C. IB 寬度（÷ATR10）：太窄 / 太寬的 IB 不做
D. 部位大小：固定 0.01 手 vs 固定風險（每筆虧 1R = 同樣美元）vs 依 ATR 反比；比較 報酬÷回撤 與 日 Sharpe（與規模無關）
"""
import numpy as np
import pandas as pd
from part27_ow import data, run, pf, PER, prev_profiles, label_ow

out = open("results_part39.txt", "w")
def P(s=""): print(s); out.write(s + "\n"); out.flush()
SRC = {"M1": ("data/XAUUSD_M1_2026.csv", 1), "M5": ("data/XAUUSD_M5_2025_2026.csv", 5), "M15": ("data/XAUUSD_M15_2023_2026.csv", 15)}


def manage(r, ARR, bm, be_R=2.0, be_atr=None, be_mode="R", stop_atr=None):
    t, O, H, L, C, S, mins = ARR[r.day]
    s, e = r.side, r.lvl
    st = r.ibl if s == 1 else r.ibh
    if stop_atr is not None and s * (e - st) > stop_atr * r.atr:
        st = e - s * stop_atr * r.atr
    R = s * (e - st); best = e; be = False
    trig = {"R": be_R * R, "atr": (be_atr or 0) * r.atr, "max": max(be_R * R, (be_atr or 0) * r.atr),
            "min": min(be_R * R, (be_atr or 0) * r.atr), "none": np.inf}[be_mode]
    for k in range(r.i, len(C)):
        if (L[k] <= st) if s == 1 else (H[k] >= st):
            px = min(O[k], st) if s == 1 else max(O[k], st)
            return px, t[k] + pd.Timedelta(minutes=bm), R, s * (best - e) / R
        best = max(best, H[k]) if s == 1 else min(best, L[k])
        if not be and s * (best - e) >= trig:
            be = True; st = e
    return C[-1], t[-1] + pd.Timedelta(minutes=bm), R, s * (best - e) / R


def cap(T, n=5):
    keep, open_ = [], []
    for r in T.sort_values("t_in").itertuples():
        open_ = [x for x in open_ if x > r.t_in]
        if len(open_) < n:
            keep.append(r.Index); open_.append(r.exit_time)
    return T.loc[keep]


def build(X, ARR, bm, **kw):
    T = X.copy()
    ex = [manage(r, ARR, bm, **kw) for r in T.itertuples()]
    T["exit_px"] = [a[0] for a in ex]; T["exit_time"] = [a[1] for a in ex]; T["risk"] = [a[2] for a in ex]; T["mfe"] = [a[3] for a in ex]
    T["pnl"] = T.side * (T.exit_px - T.lvl) - T.spread; T["R"] = T.pnl / T.risk
    return T


def money(T, usd, alldays):
    """usd = 每筆美元損益（Series 對齊 T）；回傳 總、平倉回撤、報酬÷回撤、日 Sharpe。"""
    T = T.assign(u=usd).sort_values("exit_time"); eq = T.u.cumsum(); dd = (eq - eq.cummax()).min()
    dl = T.groupby("day").u.sum().reindex(alldays, fill_value=0)
    return T.u.sum(), dd, T.u.sum() / -dd, dl.mean() / dl.std() * np.sqrt(250)


def line(name, T, per, alldays):
    C5 = cap(T)
    tot, dd, rd, sh = money(C5, C5.pnl, alldays)
    cells = "  ".join(f"{n} {x.R.mean():+.2f}/{pf(x.R):.2f}" for n, a, b in per for x in [T[(T.day >= a) & (T.day <= b)]] if len(x))
    P(f"  {name:28s} {len(T):5d}筆  {cells} | 全 {T.R.mean():+.3f}R PF{pf(T.R):.2f} 大賺變虧{((T.mfe >= 2) & (T.R < -0.05)).sum():3d} | 0.01手5張 {tot:+6,.0f} 回撤{dd:+6,.0f} 報酬/回撤{rd:4.1f} Sharpe{sh:4.2f}")


for tf, (path, bm) in SRC.items():
    days, D, ARR = data(path, bm)
    X, _ = run(path, bm)
    X = X[X.ow & (X.day >= "2023-03-01")].sort_values(["t_in", "h"]).drop_duplicates(["t_in", "side"]).reset_index(drop=True)
    X["ibw"] = (X.ibh - X.ibl) / X.atr
    per = [p for p in PER if len(X[(X.day >= p[1]) & (X.day <= p[2])])]
    alldays = [d for d in days if d >= X.day.min()]
    base = build(X, ARR, bm)
    P("\n" + "=" * 150 + f"\n[{tf}]  每筆 IB 寬 中位 {X.ibw.median():.2f} ATR（10%~90%：{X.ibw.quantile(.1):.2f}~{X.ibw.quantile(.9):.2f}）；2R 中位 = {(2 * base.risk / X.atr).median():.2f} ATR")
    P(" A. 保本觸發")
    line("基準：2R", base, per, alldays)
    line("不保本", build(X, ARR, bm, be_mode="none"), per, alldays)
    for k in (0.2, 0.3, 0.5):
        line(f"浮盈 ≥ {k} ATR", build(X, ARR, bm, be_mode="atr", be_atr=k), per, alldays)
    line("max(2R, 0.3ATR)", build(X, ARR, bm, be_mode="max", be_atr=0.3), per, alldays)
    line("min(2R, 0.5ATR)", build(X, ARR, bm, be_mode="min", be_atr=0.5), per, alldays)
    P(" B. 停損上限（+2R 保本）")
    for k in (0.2, 0.3, 0.5):
        line(f"停損 ≤ {k} ATR", build(X, ARR, bm, stop_atr=k), per, alldays)
    P(" C. IB 寬度篩選（+2R 保本）")
    for lo, hi in [(0.05, 9), (0.08, 9), (0, 0.4), (0, 0.3), (0.08, 0.4)]:
        line(f"IB 寬 {lo}~{hi} ATR", base[(base.ibw >= lo) & (base.ibw <= hi)], per, alldays)
    P(" E. 組合")
    be5 = build(X, ARR, bm, be_mode="atr", be_atr=0.5)
    line("保本 0.5ATR + IB ≤ 0.4ATR", be5[be5.ibw <= 0.4], per, alldays)
    bm2 = build(X, ARR, bm, be_mode="min", be_atr=0.5)
    line("保本 min(2R,0.5ATR) + IB ≤ 0.4", bm2[bm2.ibw <= 0.4], per, alldays)
    line("2R 保本 + IB ≤ 0.4ATR", base[base.ibw <= 0.4], per, alldays)
    P(" D. 部位大小（基準交易、最多 5 張；報酬/回撤 與 Sharpe 與規模無關）")
    C5 = cap(base)
    for name, w in [("固定 0.01 手", pd.Series(1.0, C5.index)), ("固定風險：每筆 1R 同美元", C5.risk.median() / C5.risk),
                    ("依 ATR 反比", C5.atr.median() / C5.atr), ("固定風險、單筆最多 2 倍手數", (C5.risk.median() / C5.risk).clip(upper=2))]:
        tot, dd, rd, sh = money(C5, C5.pnl * w, alldays)
        yr = C5.assign(u=C5.pnl * w).groupby(C5.day.dt.year).u.sum()
        P(f"  {name:28s} 總 {tot:+6,.0f} 回撤 {dd:+6,.0f} 報酬/回撤 {rd:4.1f}  日Sharpe {sh:4.2f}  逐年 " +
          "  ".join(f"{y} {v:+,.0f}" for y, v in yr.items()) + f"  平均手數 {w.mean() * 0.01:.3f}")
out.close()
