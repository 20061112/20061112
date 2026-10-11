"""第六十部分：版本 A 用真 M1 擴大樣本外驗證（2024/9 ~ 2026/10）→ results_part60.txt、part60_trades.csv

資料：data/XAUUSD_M1_2024_2026feb.csv（2024/9/2 ~ 2026/2/11）接上 data/XAUUSD_M1_2026.csv（2026/1/2 ~ 10/8）；
      重疊的 39,803 根 M1 完全相同。合併檔寫到 data/pseudo/（不進 git）。
規則：final_A.py / FINAL_STRATEGY_A.md 原封不動；門檻照原本做法每月只用「過去」訊號滾動重算
      （前 2 個月 2024/9~10 當暖身，不交易）。
與第五十三部分的差別：用真 M1（不是 M5 展開的偽 M1），而且多了 2024/11 ~ 2025/6 這 8 個月從未測過的資料。
期間：2024Q4（11~12 月）、2025H1、2025H2、2026（開發期）。
"""
import os
import subprocess
import sys
import numpy as np
import pandas as pd

OUT = "data/pseudo/XAUUSD_M1_2024_2026.csv"
PERIODS = [("2024/11~12（樣本外）", "2024-11-01", "2025-01-01"), ("2025 上半年（樣本外）", "2025-01-01", "2025-07-01"),
           ("2025 下半年（樣本外）", "2025-07-01", "2026-01-01"), ("2026/1~10（開發期）", "2026-01-01", "2026-11-01")]


def build():
    from load import load
    a = load("data/XAUUSD_M1_2024_2026feb.csv")
    b = load("data/XAUUSD_M1_2026.csv")
    m = pd.concat([a, b[~b.index.isin(a.index)]]).sort_index()
    os.makedirs("data/pseudo", exist_ok=True)
    o = pd.DataFrame({"<DATE>": m.index.strftime("%Y.%m.%d"), "<TIME>": m.index.strftime("%H:%M:%S"),
                      "<OPEN>": m.open, "<HIGH>": m.high, "<LOW>": m.low, "<CLOSE>": m.close,
                      "<TICKVOL>": m.tickvol.astype(int), "<VOL>": 0, "<SPREAD>": m.spread.astype(int)})
    o.to_csv(OUT, sep="\t", index=False)
    return m


def block(S, lab, days_all):
    if len(S) < 3:
        return f"  {lab:24s} 無交易"
    p = S.pnl
    w, l = p[p > 0], p[p <= 0]
    eq = S.sort_values("exit_time").pnl.cumsum()
    dd = (eq - eq.cummax()).min()
    days = [d for d in days_all if S.day.min() <= d <= S.day.max() and d.dayofweek != 0]
    daily = pd.Series(0.0, index=days).add(S.groupby("day").pnl.sum(), fill_value=0)
    t = p.mean() / p.std(ddof=1) * np.sqrt(len(p))
    return (f"  {lab:24s} {len(S):4d}筆 {len(S) / max(len(days), 1):4.1f}筆/天 勝率 {len(w) / len(p):5.1%} "
            f"每筆 {p.mean():+6.2f}（{S.R.mean():+.2f}R, t={t:+.1f}） PF {w.sum() / max(-l.sum(), 1e-9):.2f} 總 {p.sum():+7.0f} "
            f"回撤 {dd:+6.0f} 日Sharpe {daily.mean() / daily.std() * np.sqrt(250):+.2f} 點差均 {S.spread.mean() if 'spread' in S else np.nan:.2f}")


def worker():
    from part14_deep import DAYS
    from part17_balance import gen, wf
    from final_A import walk_forward, CFG
    X = gen(**CFG)
    T, used = walk_forward(X)
    T.to_csv("part60_trades.csv", index=False)
    L = [f"版本 A（規則不變、滾動門檻）期間 {T.day.min().date()} ~ {T.day.max().date()}，每筆 1 盎司，美元，已扣點差"]
    for lab, a, b in PERIODS:
        L.append(block(T[(T.day >= a) & (T.day < b)], lab, DAYS))
    oos = T[(T.day >= "2024-11-01") & (T.day < "2026-01-01")]
    L.append(block(oos, "樣本外合計 2024/11~2025/12", DAYS))
    L.append("  多 / 空（樣本外）：" + "、".join(
        f"{'多' if s == 1 else '空'} {len(g)} 筆 PF {g.pnl[g.pnl > 0].sum() / -g.pnl[g.pnl <= 0].sum():.2f} 每筆 {g.pnl.mean():+.2f}"
        for s, g in oos.groupby("side")))
    L.append("\n  逐月（總損益 / PF / 筆數）：")
    for m, g in T.groupby(T.day.dt.to_period("M")):
        p = g.pnl
        L.append(f"    {m}  {len(g):3d}筆  總 {p.sum():+7.0f}  PF {p[p > 0].sum() / max(-p[p <= 0].sum(), 1e-9):5.2f}  每筆R {g.R.mean():+.2f}")
    L.append("\n  每月門檻（滾動）：" + "  ".join(f"{r.month}: ib {r.thr_ib:.3f} tail {r.thr_tail:.3f} ext {r.thr_ext:.2f}"
                                          for r in used.itertuples()))

    L.append("\n  複雜度階梯（每一步只多加一條規則，滾動門檻；每筆美元 / PF）：")
    steps = [("L0 只有 IB 突破（觸價）", ("touch", 0.0, True), (1.0, 1.0, 0.0)),
             ("L1 + M5 收盤確認", ("close5", 0.05, True), (1.0, 1.0, 0.0)),
             ("L2 + 不做週一", ("close5", 0.05, False), (1.0, 1.0, 0.0)),
             ("L3 + 小 IB 篩選", ("close5", 0.05, False), (1.0, 1.0, 0.4)),
             ("L5 = 版本 A（+ tail + ext）", ("close5", 0.05, False), (0.9, 0.9, 0.4))]
    for lab, (cf, bf, mon), (qt, qe, qi) in steps:
        Y = wf(gen(cf, bf, (2, 10), 30, mon), qt, qe, qi)
        cells = []
        for plab, a, b in PERIODS:
            g = Y[(Y.day >= a) & (Y.day < b)]
            p = g.pnl
            cells.append(f"{plab[:9]} {p.mean():+5.2f} PF {p[p > 0].sum() / -p[p <= 0].sum():.2f}")
        L.append(f"  {lab:26s} " + " | ".join(cells))
    print("\n".join(L))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker()
        raise SystemExit
    m = build()
    env = dict(os.environ, XAU_M1_PATH=OUT)
    r = subprocess.run([sys.executable, __file__, "--worker"], env=env, capture_output=True, text=True)
    if r.returncode:
        print(r.stderr)
        raise SystemExit(1)
    head = (f"第六十部分：版本 A 用真 M1 擴大樣本外驗證\n資料 {m.index[0]} ~ {m.index[-1]}，{len(m):,} 根 M1；"
            f"平均點差（美元）2024 {m.spread[m.index.year == 2024].mean() * .01:.2f}、2025 {m.spread[m.index.year == 2025].mean() * .01:.2f}、"
            f"2026 {m.spread[m.index.year == 2026].mean() * .01:.2f}\n")
    open("results_part60.txt", "w").write(head + r.stdout)
    print(head + r.stdout)
