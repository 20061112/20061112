"""第三十七部分：最終規則加入「浮盈到 2R → 停損移到進場價（保本）」後，重跑 M1 / M5 / M15 交易內容。

進場、停損、收盤出場同 FINAL_STRATEGY_OW；唯一新增：K 棒高（低）點使浮盈 ≥ 2R 後，下一根起停損 = 進場價。
部位：每筆 0.01 手（1 盎司，損益 = 美元）；另列「最多同時 5 張」。
輸出：results_part37.txt、part37_trades_{M1,M5,M15}.csv、part37_be.png
"""
import numpy as np
import pandas as pd
from part27_ow import data, pf

out = open("results_part37.txt", "w")
def P(s=""): print(s); out.write(s + "\n"); out.flush()
BE = 2.0
SRC = {"M1": ("part31_m1_trades.csv", "data/XAUUSD_M1_2026.csv", 1),
       "M5": ("part27_ow_M5.csv", "data/XAUUSD_M5_2025_2026.csv", 5),
       "M15": ("part27_ow_M15.csv", "data/XAUUSD_M15_2023_2026.csv", 15)}
PER = [("2023", "2023-03-01", "2023-12-31"), ("24H1", "2024-01-01", "2024-06-30"), ("24H2", "2024-07-01", "2024-12-31"),
       ("25H1", "2025-01-01", "2025-06-30"), ("25H2", "2025-07-01", "2025-12-31"), ("2026", "2026-01-01", "2026-12-31")]


def manage(r, ARR, bm):
    t, O, H, L, C, S, mins = ARR[r.day]
    s, e = r.side, r.lvl
    st = r.ibl if s == 1 else r.ibh
    R = s * (e - st); best = e; mfe = mae = 0.0; be = False
    for k in range(r.i, len(C)):
        if (L[k] <= st) if s == 1 else (H[k] >= st):
            px = min(O[k], st) if s == 1 else max(O[k], st)
            mae = max(mae, s * (e - px) / R)
            return px, t[k] + pd.Timedelta(minutes=bm), "保本" if be else "停損", mfe, mae
        best = max(best, H[k]) if s == 1 else min(best, L[k])
        mfe = s * (best - e) / R
        mae = max(mae, s * (e - (L[k] if s == 1 else H[k])) / R)
        if not be and mfe >= BE:
            be = True; st = e
    return C[-1], t[-1] + pd.Timedelta(minutes=bm), "收盤", mfe, mae


def cap(T, n):
    keep, open_ = [], []
    for r in T.sort_values("t_in").itertuples():
        open_ = [x for x in open_ if x > r.t_in]
        if len(open_) < n:
            keep.append(r.Index); open_.append(r.exit_time)
    return T.loc[keep]


def streak(x):
    return int(x.groupby((x != x.shift()).cumsum()).sum().max())


def usd_line(X):
    X = X.sort_values("exit_time"); eq = X.pnl.cumsum()
    dl = X.groupby("day").pnl.sum()
    return f"總 {X.pnl.sum():+7,.0f} 美元  平倉最大回撤 {(eq - eq.cummax()).min():+6,.0f}  最差單日 {dl.min():+5,.0f}  最好單日 {dl.max():+5,.0f}"


ALL = {}
for tf, (csv, path, bm) in SRC.items():
    T = pd.read_csv(csv, parse_dates=["day", "t_in", "exit_time"]).sort_values("t_in").reset_index(drop=True)
    T = T[T.day >= "2023-03-01"].copy()
    days, D, ARR = data(path, bm)
    old = T.R.copy()
    res = [manage(r, ARR, bm) for r in T.itertuples()]
    T["exit_px"] = [a[0] for a in res]; T["exit_time"] = [a[1] for a in res]; T["exit_reason"] = [a[2] for a in res]
    T["mfe_R"] = [a[3] for a in res]; T["mae_R"] = [a[4] for a in res]
    T["pnl"] = T.side * (T.exit_px - T.lvl) - T.spread; T["R"] = T.pnl / T.risk; T["R_old"] = old
    T["hold_h"] = (T.exit_time - T.t_in).dt.total_seconds() / 3600
    ALL[tf] = T
    r = T.R; w, l = r[r > 0], r[r <= 0]
    P("\n" + "=" * 120 + f"\n[{tf} OW + 2R 保本]  {T.day.min():%Y-%m-%d} ~ {T.day.max():%Y-%m-%d}")
    P(f"  交易 {len(T)} 筆（多 {(T.side == 1).sum()} / 空 {(T.side == -1).sum()}），有交易的日子 {T.day.nunique()} 天")
    P(f"  勝率 {(r > 0).mean() * 100:.1f}%（> +0.05R：{(r > 0.05).mean() * 100:.1f}%）  平均賺 {w.mean():+.2f}R / 平均賠 {l.mean():+.2f}R  每筆 {r.mean():+.3f}R（原 {old.mean():+.3f}R）  PF {pf(r):.2f}（原 {pf(old):.2f}）")
    P(f"  總 {r.sum():+.0f}R（原 {old.sum():+.0f}R）；出場：停損 {(T.exit_reason == '停損').mean() * 100:.0f}%、保本 {(T.exit_reason == '保本').mean() * 100:.0f}%、收盤 {(T.exit_reason == '收盤').mean() * 100:.0f}%")
    P(f"  浮盈曾 ≥ 2R 的單 {(T.mfe_R >= 2).sum()} 筆，其中以 < −0.05R 收場 {((T.mfe_R >= 2) & (r < -0.05)).sum()} 筆（原規則 {((T.mfe_R >= 2) & (old <= 0)).sum()} 筆）")
    P(f"  停損距離 中位 {T.risk.median():.1f} 美元；平均持倉 {T.hold_h.mean():.1f}h（停損 {T[T.exit_reason == '停損'].hold_h.mean():.1f}h / 保本 {T[T.exit_reason == '保本'].hold_h.mean():.1f}h / 收盤 {T[T.exit_reason == '收盤'].hold_h.mean():.1f}h）")
    P(f"  MFE 中位 {T.mfe_R.median():.2f}R；單筆最大 {r.max():+.1f}R；最長連虧 {streak((r <= 0).astype(int))} 筆（不含保本 {streak((r < -0.05).astype(int))} 筆）")
    P("  各期 每筆R/PF/筆數：" + "  ".join(f"{n} {x.R.mean():+.2f}/{pf(x.R):.2f}/{len(x)}" for n, a, b in PER for x in [T[(T.day >= a) & (T.day <= b)]] if len(x)))
    if tf == "M15":
        P("  逐年 0.01 手美元：" + "  ".join(f"{y} {g.pnl.sum():+,.0f}" for y, g in T.groupby(T.day.dt.year)))
    else:
        P("  逐月 R：" + "  ".join(f"{p.month}月 {g.R.sum():+.0f}" for p, g in T.groupby(T.day.dt.to_period("M"))))
    P(f"  0.01 手 不限張數：{usd_line(T)}")
    C5 = cap(T, 5)
    P(f"  0.01 手 最多 5 張（{len(C5)} 筆）：{usd_line(C5)}")
    T.to_csv(f"part37_trades_{tf}.csv", index=False)

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
for a, tf in zip(ax, ["M1", "M5", "M15"]):
    X = ALL[tf].sort_values("exit_time")
    a.plot(X.exit_time, (X.R_old * X.risk).cumsum(), color="#999", lw=1, label="original")
    a.plot(X.exit_time, X.pnl.cumsum(), color="#2e8b57", lw=1.4, label="2R breakeven")
    C5 = cap(ALL[tf], 5).sort_values("exit_time")
    a.plot(C5.exit_time, C5.pnl.cumsum(), color="#c9824a", lw=1.2, ls="--", label="2R BE, max 5 open")
    a.set_title(f"{tf} OW, 0.01 lot (USD)"); a.legend(); a.grid(alpha=.3); a.tick_params(axis="x", rotation=30)
plt.tight_layout(); plt.savefig("part37_be.png", dpi=110)
out.close()
