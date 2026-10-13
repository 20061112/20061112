"""第二十六部分：版本 A（IB 突破延續）在 2023/6 ~ 2025/12 的真正樣本外測試。
版本 A 的規則、參數都是在 2026/1~10 的資料上找到的；2023/6~2025/12 從來沒有用過。
三種門檻做法：
  凍結    直接用 final_A_live.json 的門檻（thr_ib 0.127 / tail 0.053 / ext 0.82），一個數字都不改
  前推    與原本相同的滾動前推：每月月初用「之前所有訊號」重算分位數（從 2023/8 開始）
  不篩    只有 IB 突破核心（不套 F1~F3）
另外重跑複雜度階梯 L0~L5（第二十一部分），看每條規則在樣本外有沒有貢獻。
單位：R（損益 ÷ 停損距離）與 PF 可跨年比較；美元/盎司 2023 年的價位約是 2026 年的一半，只供參考。
"""
import json
import numpy as np
import pandas as pd

OOS_END = pd.Timestamp("2026-01-01")
LIVE = json.load(open("final_A_live.json"))
out = open("results_part26.txt", "w")


def P(s=""):
    print(s)
    out.write(s + "\n")
    out.flush()


def pf(p):
    return p[p > 0].sum() / max(-p[p <= 0].sum(), 1e-9)


def dedup(Y):
    return Y.sort_values(["t_in", "h"]).drop_duplicates(["t_in", "side"]).sort_values("t_in")


def frozen(X, ti=LIVE["thr_ib_atr"], tt=LIVE["thr_tail_atr"], te=LIVE["thr_ext_letter"]):
    k = ~(X.aligned & (X.ib_atr < ti)) & (X["tail"] <= tt) & (X.ext_letter.fillna(0) <= te)
    return dedup(X[k])


def walk(X, q_tail=0.9, q_ext=0.9, q_ib=0.4, start=None):
    X = X.copy()
    X["ym"] = X.day.dt.to_period("M")
    parts = []
    for m in sorted(X.ym.unique())[2:]:
        past, cur = X[X.ym < m], X[X.ym == m]
        k = ~(cur.aligned & (cur.ib_atr < past.ib_atr.quantile(q_ib))) & (cur["tail"] <= past["tail"].quantile(q_tail)) \
            & (cur.ext_letter.fillna(0) <= past.ext_letter.quantile(q_ext))
        parts.append(cur[k])
    return dedup(pd.concat(parts))


def line(T):
    if len(T) == 0:
        return "n=0"
    p, R = T.pnl, T.R
    days = T.day.nunique()
    t = R.mean() / R.std() * np.sqrt(len(R))
    d = T.groupby("day").R.sum()
    td = d.mean() / d.std() * np.sqrt(len(d))           # 以「日」為單位的 t（同一天多筆高度相關，逐筆 t 會高估）
    return (f"n={len(T):5d}（{len(T) / max(days, 1):.1f}/有單日） 每筆 {R.mean():+.3f}R 勝率 {100 * (p > 0).mean():4.1f}% "
            f"PF {pf(p):.2f} PF(R) {pf(R):.2f} 累計 {R.sum():+7.1f}R 日均 {d.mean():+.2f}R (日t{td:+.1f})")


def by_year(T, lab):
    P(f"\n  {lab}")
    for y, g in T.groupby(T.day.dt.year):
        P(f"    {y}: {line(g)}")
    P(f"    樣本外 2023/6~2025/12: {line(T[T.day < OOS_END])}")
    P(f"    研究期 2026          : {line(T[T.day >= OOS_END])}")


def halfyear(T):
    h = T.day.dt.year.astype(str) + np.where(T.day.dt.month <= 6, "H1", "H2")
    g = T.groupby(h)
    return "  ".join(f"{k}:{v.R.mean():+.2f}/PF{pf(v.R):.2f}" for k, v in g)


A = pd.read_pickle("part26_signals_A.pkl")
L0 = pd.read_pickle("part26_signals_L0.pkl")
L1 = pd.read_pickle("part26_signals_L1.pkl")
for X in (A, L0, L1):
    X["day"] = pd.to_datetime(X.day)

P("第二十六部分：版本 A 在 2023/6 ~ 2025/12 的樣本外測試（規則與參數完全沿用，不做任何修改）")
P(f"訊號：{A.day.min().date()} ~ {A.day.max().date()}，版本 A 原始訊號 {len(A)} 筆（未篩、未去重）")
P(f"凍結門檻：thr_ib {LIVE['thr_ib_atr']:.3f} / thr_tail {LIVE['thr_tail_atr']:.3f} / thr_ext {LIVE['thr_ext_letter']:.3f}")

P("\n# 1. 版本 A")
TA_f = frozen(A)
TA_w = walk(A)
TA_0 = dedup(A)
by_year(TA_f, "A｜凍結門檻（2026 年算出的數字原封不動）")
by_year(TA_w, "A｜滾動前推門檻（原本的做法，從 2023/8 起）")
by_year(TA_0, "核心（M5 收盤確認、不做週一，不套 F1~F3）")
P("\n  半年度（每筆 R / PF）")
P(f"    A 凍結 : {halfyear(TA_f)}")
P(f"    A 前推 : {halfyear(TA_w)}")
P(f"    核心   : {halfyear(TA_0)}")

P("\n# 2. 複雜度階梯（滾動前推，每一步只多加一條規則）")
steps = [("L0 IB 觸價、停損另一側、收盤出（含週一）", L0, (1.0, 1.0, 0.0)),
         ("L1 + M5 收盤 + 0.05 ATR 確認", L1, (1.0, 1.0, 0.0)),
         ("L2 + 不做星期一", A, (1.0, 1.0, 0.0)),
         ("L3 + 順前日方向的小 IB 不做", A, (1.0, 1.0, 0.4)),
         ("L4 + 反向尾巴篩選", A, (0.9, 1.0, 0.4)),
         ("L5 + 反向極值字母（= 版本 A）", A, (0.9, 0.9, 0.4))]
for lab, X, (qt, qe, qi) in steps:
    Y = walk(X, qt, qe, qi)
    a, b = Y[Y.day < OOS_END], Y[Y.day >= OOS_END]
    P(f"  {lab:34s} 樣本外 {len(a):5d}筆 {a.R.mean():+.3f}R PF {pf(a.pnl):.2f} | 2026 {len(b):4d}筆 {b.R.mean():+.3f}R PF {pf(b.pnl):.2f}")

P("\n# 3. 被 F1~F3 刪掉的單（凍結門檻）：樣本外是不是真的比較差？")
kept = set(zip(TA_f.t_in, TA_f.side))
drop = TA_0[[(t, s) not in kept for t, s in zip(TA_0.t_in, TA_0.side)]]
for lab, T in (("保留", TA_f), ("刪掉", drop)):
    a, b = T[T.day < OOS_END], T[T.day >= OOS_END]
    P(f"  {lab}: 樣本外 {line(a)}")
    P(f"        2026   {line(b)}")
for f, cond in (("F1 順前日且小 IB", A.aligned & (A.ib_atr < LIVE["thr_ib_atr"])),
                ("F2 反向尾巴長", A["tail"] > LIVE["thr_tail_atr"]),
                ("F3 反向極值剛做出", A.ext_letter.fillna(0) > LIVE["thr_ext_letter"])):
    a = A[cond & (A.day < OOS_END)]
    b = A[~cond & (A.day < OOS_END)]
    P(f"  {f:18s}: 被這條刪的 樣本外 {a.R.mean():+.3f}R PF {pf(a.pnl):.2f} (n={len(a)}) vs 其餘 {b.R.mean():+.3f}R PF {pf(b.pnl):.2f}")

P("\n# 4. 分組（版本 A 凍結門檻，樣本外 2023/6~2025/12）")
O = TA_f[TA_f.day < OOS_END].copy()
O["IB起點"] = O.h.map(lambda h: f"{int(h):02d}:{int(round((h % 1) * 60)):02d}")
O["星期"] = O.day.dt.dayofweek.map(dict(enumerate("一二三四五")))
O["方向"] = O.side.map({1: "多", -1: "空"})
for g in ("方向", "星期", "exit_reason"):
    P(f"  [{g}] " + "  ".join(f"{k}: {v.R.mean():+.3f}R PF {pf(v.pnl):.2f} (n={len(v)})" for k, v in O.groupby(g)))
ib = O.groupby("IB起點").R.mean()
P(f"  [IB 起點] 每筆 R 為正的起點 {(ib > 0).sum()} / {len(ib)}：" + " ".join(f"{k}:{v:+.2f}" for k, v in ib.items()))
P(f"  停損距離中位數 {O.risk.median():.2f} 美元（{(O.risk / O.atr).median():.2f} ATR）；點差中位數 {O.spread.median():.3f}，點差 / 停損 {(O.spread / O.risk).median():.3f}R")
yrs = TA_f.groupby(TA_f.day.dt.year)
P("  逐年 點差/停損（R）：" + "  ".join(f"{y}:{(g.spread / g.risk).mean():.3f}" for y, g in yrs))
P("  逐年 未扣點差每筆 R：" + "  ".join(f"{y}:{((g.pnl + g.spread) / g.risk).mean():+.3f}" for y, g in yrs))

TA_f.to_csv("part26_A_frozen_trades.csv", index=False, float_format="%.4f")


def equity_chart():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 5))
    for T, lab, col in ((TA_f, "Version A, frozen thresholds", "#3b6ea8"), (TA_w, "Version A, walk-forward thresholds", "#c9824a"),
                        (TA_0, "Core only (no F1-F3 filters)", "#888888")):
        T = T.sort_values("exit_time")
        ax.plot(pd.to_datetime(T.exit_time), T.R.cumsum(), label=lab, color=col, lw=1.2)
    ax.axvline(OOS_END, color="k", ls="--", lw=1)
    ax.text(OOS_END, ax.get_ylim()[1] * 0.95, "  research period (2026) →", va="top")
    ax.set_ylabel("cumulative R (1R per trade)")
    ax.set_title("Version A out-of-sample: Jun 2023 – Dec 2025 never used in research")
    ax.legend(loc="upper left")
    ax.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig("part26_equity.png", dpi=110)


equity_chart()
out.close()

out = open("results_part26.txt", "a")
P("\n# 5. 階梯中樣本外最好的 L0 / L1，逐年與多空（滾動前推，L0/L1 沒有門檻，等於完全固定）")
P("   注意：這一段是「看過樣本外之後」的比較，只能當描述，不能當驗證。")
L1t = walk(L1, 0.9, 1.0, 0.0)          # L1 + 只用反向尾巴
for lab, T in (("L0 觸價、含週一", dedup(L0)), ("L1 M5 收盤確認、含週一", dedup(L1)), ("L1 + 反向尾巴（前推）", L1t), ("版本 A（凍結）", TA_f)):
    P(f"\n  {lab}")
    for y, g in T.groupby(T.day.dt.year):
        lo, sh = g[g.side == 1], g[g.side == -1]
        P(f"    {y}: {line(g)} | 多 {lo.R.mean():+.3f} 空 {sh.R.mean():+.3f}")
    o = T[T.day < OOS_END]
    P(f"    樣本外合計: {line(o)} | 多 {o[o.side == 1].R.mean():+.3f} 空 {o[o.side == -1].R.mean():+.3f}")
    mon = o.groupby(o.day.dt.dayofweek).R.mean()
    P("    星期 每筆 R：" + " ".join(f"{'一二三四五'[k]}:{v:+.3f}" for k, v in mon.items()))
out.close()
