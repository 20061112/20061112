"""第三十八部分：TPO 價格格子改成自適應（ATR 比例 / 價格比例 / 前日區間等分）vs 固定 1 美元。

前一天價值區（30 分字母、70%）的格子大小 g：
  固定 g 美元；ATR：g = ATR10 × p；價格：g = 前日收盤 × p；區間：g = 前日高低差 / n
其他規則同 FINAL_STRATEGY_OW + 2R 保本；OW 篩選後才去重（同根 K 棒同方向一筆）。
統計：每筆 R / PF、各期、0.01 手最多 5 張的美元與平倉回撤、前一天價值區寬度（÷ATR）。
"""
import numpy as np
import pandas as pd
from part27_ow import data, run, pf, PER
from tpo import poc_va

out = open("results_part38.txt", "w")
def P(s=""): print(s); out.write(s + "\n"); out.flush()
SRC = {"M1": ("data/XAUUSD_M1_2026.csv", 1), "M5": ("data/XAUUSD_M5_2025_2026.csv", 5), "M15": ("data/XAUUSD_M15_2023_2026.csv", 15)}


def profile(H, L, mins, g, va_pct=0.70):
    per = mins // 30
    lo0 = int(np.floor(L.min() / g)); n = int(np.floor(H.max() / g)) - lo0 + 1
    diff = np.zeros(n + 1)
    for k in np.unique(per):
        m = per == k
        diff[int(np.floor(L[m].min() / g)) - lo0] += 1; diff[int(np.floor(H[m].max() / g)) - lo0 + 1] -= 1
    cnt = np.cumsum(diff)[:n].astype(int)
    poc, vah, val = poc_va(np.arange(n), cnt, va_pct)
    return (lo0 + poc + 0.5) * g, (lo0 + vah + 1) * g, (lo0 + val) * g


def prev_profiles(days, D, ARR, kind, p):
    outp = {}
    for di in range(1, len(days)):
        t, O, H, L, C, S, mins = ARR[days[di - 1]]
        atr = D.atr10.get(days[di], np.nan)
        g = {"fix": p, "atr": atr * p, "px": C[-1] * p, "rng": (H.max() - L.min()) / p}[kind]
        if not np.isfinite(g) or g <= 0:
            continue
        outp[days[di]] = profile(H, L, mins, g) + (atr, g)
    return outp


def manage(r, ARR, bm):
    t, O, H, L, C, S, mins = ARR[r.day]
    s, e = r.side, r.lvl
    st = r.ibl if s == 1 else r.ibh
    R = s * (e - st); best = e; be = False
    for k in range(r.i, len(C)):
        if (L[k] <= st) if s == 1 else (H[k] >= st):
            px = min(O[k], st) if s == 1 else max(O[k], st)
            return px, t[k] + pd.Timedelta(minutes=bm)
        best = max(best, H[k]) if s == 1 else min(best, L[k])
        if not be and s * (best - e) / R >= 2:
            be = True; st = e
    return C[-1], t[-1] + pd.Timedelta(minutes=bm)


def cap(T, n=5):
    keep, open_ = [], []
    for r in T.sort_values("t_in").itertuples():
        open_ = [x for x in open_ if x > r.t_in]
        if len(open_) < n:
            keep.append(r.Index); open_.append(r.exit_time)
    return T.loc[keep]


GRIDS = [("固定 0.5 美元", "fix", 0.5), ("固定 1 美元（現行）", "fix", 1.0), ("固定 2 美元", "fix", 2.0), ("固定 3 美元", "fix", 3.0),
         ("ATR10 × 1%", "atr", 0.01), ("ATR10 × 2%", "atr", 0.02), ("ATR10 × 3%", "atr", 0.03), ("ATR10 × 5%", "atr", 0.05),
         ("價格 × 0.01%", "px", 1e-4), ("價格 × 0.02%", "px", 2e-4), ("價格 × 0.05%", "px", 5e-4),
         ("前日區間 / 30 格", "rng", 30), ("前日區間 / 60 格", "rng", 60)]

for tf, (path, bm) in SRC.items():
    days, D, ARR = data(path, bm)
    X, _ = run(path, bm)                                     # 全部訊號（未篩 OW）
    X = X[X.day >= "2023-03-01"]
    per = [p for p in PER if len(X[(X.day >= p[1]) & (X.day <= p[2])])]
    P("\n" + "=" * 150 + f"\n[{tf}]  格子 → 平均格子美元（首期/末期）、價值區寬÷ATR、筆數、各期 每筆R/PF、全期 每筆R/PF、0.01 手最多 5 張 美元 / 平倉回撤")
    for name, kind, p in GRIDS:
        PV = prev_profiles(days, D, ARR, kind, p)
        ok = np.array([(r.day in PV) and ((r.ibl > PV[r.day][1]) if r.side == 1 else (r.ibh < PV[r.day][2])) for r in X.itertuples()])
        T = X[ok].sort_values(["t_in", "h"]).drop_duplicates(["t_in", "side"]).copy()
        ex = [manage(r, ARR, bm) for r in T.itertuples()]
        T["exit_px"] = [a for a, b in ex]; T["exit_time"] = [b for a, b in ex]
        T["pnl"] = T.side * (T.exit_px - T.lvl) - T.spread; T["R"] = T.pnl / T.risk
        G = pd.DataFrame(PV, index=["poc", "vah", "val", "atr", "g"]).T
        G = G[G.index >= "2023-03-01"]
        wid = ((G.vah - G.val) / G.atr).median()
        C5 = cap(T).sort_values("exit_time"); eq = C5.pnl.cumsum()
        cells = "  ".join(f"{n} {x.R.mean():+.2f}/{pf(x.R):.2f}" for n, a, b in per for x in [T[(T.day >= a) & (T.day <= b)]] if len(x))
        P(f"  {name:18s} 格 {G.g.iloc[:20].mean():4.2f}/{G.g.iloc[-20:].mean():4.2f}  VA寬 {wid:.2f}ATR  {len(T):5d}筆  {cells}"
          f" | 全 {T.R.mean():+.3f}R PF{pf(T.R):.2f} 總{T.R.sum():+4.0f}R | 5張 {C5.pnl.sum():+6,.0f}美元 回撤{(eq - eq.cummax()).min():+5,.0f}")
out.close()
