"""第二十二部分 B：真正的樣本外 —— 版本 A 規則完全凍結，跑 2023~2025（這段資料在研究中從沒用過）。
M15：2023/3 ~ 2025/12（2023/1~2 當門檻暖身）。M5：2025/7 ~ 2025/12（2025/5~6 暖身）。2026 列出作為對照（已看過）。"""
import numpy as np, pandas as pd
from oos_engine import load_bars, signals, simulate, walk_forward, stats

out = open("results_part22.txt", "a")
def P(s=""): print(s); out.write(s + "\n"); out.flush()

def fmt(s):
    if not s.get("n"):
        return "沒有交易"
    return (f"{s['n']:4d}筆 {s['per_day']:.1f}/日 勝率 {s['win']:.0f}% 每筆 {s['avg']:+6.2f}（{s['avgR']:+.2f}R） PF {s['pf']:.2f}"
            f" | 0.5%/筆 {s['ret_pct']:+5.0f}% DD {s['dd_pct']:5.1f}% Sharpe {s['sharpe']:5.2f} 月正 {s['mon_pos']:.0f}%/{s['n_mon']} 連虧 {s['streak']} ATR中位 {s['atr']:.0f}")

days, D, ARR = load_bars("data/XAUUSD_M15_2023_2026.csv", 15)
X = simulate(signals(days, D, ARR, 15), ARR, 15)
T = walk_forward(X)
T.to_csv("oos_trades_M15.csv", index=False)

P("\n" + "=" * 120 + "\n第二十二部分 B：版本 A 凍結規則，M15（確認改用 M15 收盤，保真度檢查顯示比 M1 版略保守）")
for lab, a, b in [("2023（3~12 月）", "2023-01-01", "2023-12-31"), ("2024", "2024-01-01", "2024-12-31"), ("2025", "2025-01-01", "2025-12-31"),
                  ("2023~2025 合計（真正樣本外）", "2023-01-01", "2025-12-31"), ("2026 1~10 月（研究期，對照）", "2026-01-01", "2026-12-31")]:
    P(f"  {lab:28s} {fmt(stats(T[(T.day >= a) & (T.day <= b)], days))}")

P("\n  半年度")
for y in (2023, 2024, 2025, 2026):
    for h, (a, b) in enumerate([(f"{y}-01-01", f"{y}-06-30"), (f"{y}-07-01", f"{y}-12-31")]):
        x = T[(T.day >= a) & (T.day <= b)]
        if len(x):
            P(f"    {y} {'上' if h == 0 else '下'}半年  {fmt(stats(x, days))}")

m = T.groupby(T.day.dt.to_period("M")).agg(n=("pnl", "size"), R=("R", "sum"))
P("\n  逐月（每筆 0.5% 風險時的帳戶 %，= R 合計 × 0.5）")
for y in (2023, 2024, 2025, 2026):
    mm = m[m.index.year == y]
    P(f"    {y}: " + "  ".join(f"{k.month:02d}月 {v.R * 0.5:+5.1f}%" for k, v in mm.iterrows()))

P("\n  逐年拆解")
for y in (2023, 2024, 2025, 2026):
    x = T[T.day.dt.year == y]
    pf = lambda p: p[p > 0].sum() / max(-p[p <= 0].sum(), 1e-9)
    wd = "  ".join(f"{'一二三四五'[k]} PF{pf(g.pnl):.2f}" for k, g in x.groupby(x.day.dt.dayofweek))
    P(f"    {y}: 多 PF {pf(x[x.side == 1].pnl):.2f} / 空 PF {pf(x[x.side == -1].pnl):.2f} | {wd} | 停損比例 {(x.exit_reason == '停損').mean() * 100:.0f}%")

# M5（與 M1 版確認方式相同）2025/7~12
d5, D5, A5 = load_bars("data/XAUUSD_M5_2025_2026.csv", 5)
X5 = simulate(signals(d5, D5, A5, 5), A5, 5)
T5 = walk_forward(X5)
T5.to_csv("oos_trades_M5.csv", index=False)
P("\n  M5（確認方式與 M1 原版相同）")
P(f"    2025 7~12 月（真正樣本外）  {fmt(stats(T5[(T5.day >= '2025-07-01') & (T5.day <= '2025-12-31')], d5))}")
P(f"    2026 1~10 月（研究期對照）  {fmt(stats(T5[T5.day >= '2026-01-01'], d5))}")
out.close()
