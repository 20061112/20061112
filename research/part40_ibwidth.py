"""第四十部分：IB 寬度正視 —— 用不同的「尺」量 IB 寬，看它怎麼影響 OW + 2R 保本。

特徵（進場當下已知、與價格水準無關）：
  w_atr   IB 寬 ÷ ATR10
  w_rel   IB 寬 ÷ 前 20 天同一時段 IB 寬的中位數（今天這個時段比平常寬還是窄）
  w_va    IB 寬 ÷ 前一天價值區寬
  w_prng  IB 寬 ÷ 前一天高低差
  gap_w   IB 離價值區的距離（做多 = IB 低 − VAH）÷ IB 寬
  gap_atr 同上 ÷ ATR10
每個特徵切五等分：筆數、勝率、每筆 R、PF、每筆 損益÷ATR（與規模無關的「錢」）、停損率、MFE 中位；M15 另列各期每筆 R。
最後測幾個篩選：0.01 手最多 5 張的美元、回撤、日 Sharpe。
"""
import numpy as np
import pandas as pd
from part27_ow import data, run, pf, PER, prev_profiles

out = open("results_part40.txt", "w")
def P(s=""): print(s); out.write(s + "\n"); out.flush()
SRC = {"M1": ("data/XAUUSD_M1_2026.csv", 1), "M5": ("data/XAUUSD_M5_2025_2026.csv", 5), "M15": ("data/XAUUSD_M15_2023_2026.csv", 15)}
HS = np.arange(2, 11, 0.5)


def manage(r, ARR, bm):
    t, O, H, L, C, S, mins = ARR[r.day]
    s, e = r.side, r.lvl
    st = r.ibl if s == 1 else r.ibh
    R = s * (e - st); best = e; be = False
    for k in range(r.i, len(C)):
        if (L[k] <= st) if s == 1 else (H[k] >= st):
            px = min(O[k], st) if s == 1 else max(O[k], st)
            return px, t[k] + pd.Timedelta(minutes=bm), s * (best - e) / R, "停損" if not be else "保本"
        best = max(best, H[k]) if s == 1 else min(best, L[k])
        if not be and s * (best - e) / R >= 2:
            be = True; st = e
    return C[-1], t[-1] + pd.Timedelta(minutes=bm), s * (best - e) / R, "收盤"


def cap(T, n=5):
    keep, open_ = [], []
    for r in T.sort_values("t_in").itertuples():
        open_ = [x for x in open_ if x > r.t_in]
        if len(open_) < n:
            keep.append(r.Index); open_.append(r.exit_time)
    return T.loc[keep]


def ib_widths(days, ARR, bm):
    W = {}
    for d in days:
        t, O, H, L, C, S, mins = ARR[d]
        row = {}
        for h in HS:
            m = (mins >= h * 60) & (mins < h * 60 + 60)
            if m.sum() >= 60 / bm * 0.8:
                row[h] = H[m].max() - L[m].min()
        W[d] = row
    W = pd.DataFrame(W).T.sort_index()
    return W.rolling(20, min_periods=10).median().shift(1)


def table(T, col, lab, per, multi):
    q = pd.qcut(T[col], 5, duplicates="drop")
    P(f"\n  [{lab}]")
    for b, g in T.groupby(q, observed=True):
        cells = ("  " + "  ".join(f"{n} {x.R.mean():+.2f}" for n, a, c in per for x in [g[(g.day >= a) & (g.day <= c)]] if len(x))) if multi else ""
        P(f"   {str(b):18s} {len(g):5d}筆 勝{(g.R > 0.05).mean() * 100:3.0f}% 每筆{g.R.mean():+.3f}R PF{pf(g.R):.2f} 損益÷ATR{g.u_atr.mean() * 100:+5.1f}% "
          f"停損{(g.why == '停損').mean() * 100:3.0f}% MFE中位{g.mfe.median():.2f}R" + cells)


def money(T, alldays):
    C5 = cap(T).sort_values("exit_time"); eq = C5.pnl.cumsum()
    dl = C5.groupby("day").pnl.sum().reindex(alldays, fill_value=0)
    return f"5張 {C5.pnl.sum():+6,.0f}美元 回撤{(eq - eq.cummax()).min():+6,.0f} Sharpe{dl.mean() / dl.std() * np.sqrt(250):.2f}"


ALL = {}
for tf, (path, bm) in SRC.items():
    days, D, ARR = data(path, bm)
    PV = prev_profiles(days, ARR)
    REL = ib_widths(days, ARR, bm)
    X, _ = run(path, bm)
    T = X[X.ow & (X.day >= "2023-03-01")].sort_values(["t_in", "h"]).drop_duplicates(["t_in", "side"]).reset_index(drop=True)
    ex = [manage(r, ARR, bm) for r in T.itertuples()]
    T["exit_px"] = [a[0] for a in ex]; T["exit_time"] = [a[1] for a in ex]; T["mfe"] = [a[2] for a in ex]; T["why"] = [a[3] for a in ex]
    T["pnl"] = T.side * (T.exit_px - T.lvl) - T.spread; T["R"] = T.pnl / T.risk; T["u_atr"] = T.pnl / T.atr
    w = T.ibh - T.ibl
    di = {d: i for i, d in enumerate(days)}
    prev = [days[di[d] - 1] for d in T.day]
    T["w_atr"] = w / T.atr
    T["w_rel"] = [wi / REL.at[d, h] if (d in REL.index and h in REL.columns and REL.at[d, h] > 0) else np.nan for wi, d, h in zip(w, T.day, T.h)]
    T["w_va"] = [wi / (PV[d][1] - PV[d][2]) for wi, d in zip(w, T.day)]
    T["w_prng"] = [wi / (ARR[p][2].max() - ARR[p][3].min()) for wi, p in zip(w, prev)]
    gap = np.where(T.side == 1, T.ibl - [PV[d][1] for d in T.day], [PV[d][2] for d in T.day] - T.ibh)
    T["gap_w"] = gap / w; T["gap_atr"] = gap / T.atr
    T["sp_R"] = T.spread / T.risk
    ALL[tf] = (T, [d for d in days if d >= T.day.min()])
    per = [p for p in PER if len(T[(T.day >= p[1]) & (T.day <= p[2])])]
    P("\n" + "=" * 150 + f"\n[{tf} OW + 2R 保本] {len(T)} 筆；各特徵五等分（勝 = R > 0.05）" + ("；右側各期每筆 R" if tf == "M15" else ""))
    P(f"  相關（Spearman，與每筆 R / 與 損益÷ATR）：" + "  ".join(
        f"{c} {T[c].rank().corr(T.R.rank()):+.2f}/{T[c].rank().corr(T.u_atr.rank()):+.2f}" for c in ["w_atr", "w_rel", "w_va", "w_prng", "gap_w", "gap_atr"]))
    for col, lab in [("w_atr", "IB 寬 ÷ ATR10"), ("w_rel", "IB 寬 ÷ 前20天同時段中位"), ("w_va", "IB 寬 ÷ 前日價值區寬"),
                     ("w_prng", "IB 寬 ÷ 前日高低差"), ("gap_w", "IB 離價值區 ÷ IB 寬"), ("gap_atr", "IB 離價值區 ÷ ATR")]:
        table(T.dropna(subset=[col]), col, lab, per, tf == "M15")
    P(f"\n  機制：點差÷R 中位 最窄五分之一 {T[T.w_atr <= T.w_atr.quantile(.2)].sp_R.median() * 100:.1f}% vs 最寬 {T[T.w_atr >= T.w_atr.quantile(.8)].sp_R.median() * 100:.1f}%；"
      f"突破緩衝 0.05ATR ÷ R 中位 {(0.05 * T.atr / T.risk).median():.2f}")

P("\n" + "=" * 150 + "\n篩選測試（每格：筆數 / 每筆R / PF / 0.01手 5張美元 / 回撤 / Sharpe）；M15 另列各期每筆 R")
FILT = [("全部（現行）", lambda T: T.R == T.R),
        ("w_atr ≤ 0.4", lambda T: T.w_atr <= 0.4), ("w_atr ≤ 0.3", lambda T: T.w_atr <= 0.3), ("w_atr ≥ 0.08", lambda T: T.w_atr >= 0.08),
        ("w_rel ≤ 2", lambda T: T.w_rel <= 2), ("w_rel ≤ 1.5", lambda T: T.w_rel <= 1.5), ("w_rel ≥ 0.6", lambda T: T.w_rel >= 0.6),
        ("w_va ≤ 1", lambda T: T.w_va <= 1), ("w_va ≤ 0.6", lambda T: T.w_va <= 0.6),
        ("w_prng ≤ 0.4", lambda T: T.w_prng <= 0.4), ("w_prng ≤ 0.3", lambda T: T.w_prng <= 0.3),
        ("gap_w ≥ 0.25", lambda T: T.gap_w >= 0.25), ("gap_w ≥ 0.5", lambda T: T.gap_w >= 0.5), ("gap_atr ≥ 0.1", lambda T: T.gap_atr >= 0.1),
        ("w_rel ≥ 0.6 且 w_atr ≤ 0.4", lambda T: (T.w_rel >= 0.6) & (T.w_atr <= 0.4))]
for name, f in FILT:
    P(f"  {name}")
    for tf in ["M1", "M5", "M15"]:
        T, alldays = ALL[tf]; Y = T[f(T).fillna(False).astype(bool)]
        per = [p for p in PER if len(Y[(Y.day >= p[1]) & (Y.day <= p[2])])]
        cells = ("  " + "  ".join(f"{n} {x.R.mean():+.2f}" for n, a, c in per for x in [Y[(Y.day >= a) & (Y.day <= c)]] if len(x))) if tf == "M15" else ""
        P(f"     {tf:4s} {len(Y):5d}筆 {Y.R.mean():+.3f}R PF{pf(Y.R):.2f}  {money(Y, alldays)}" + cells)
out.close()
