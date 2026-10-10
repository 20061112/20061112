"""第二十二部分 A：保真度檢查 —— 同一段 2026/3~10/7，M1 原版 vs M5 / M15 K 棒重建版。"""
import pandas as pd
from oos_engine import load_bars, signals, simulate, walk_forward, stats

out = open("results_part22.txt", "w")
def P(s=""): print(s); out.write(s + "\n"); out.flush()

def fmt(s):
    return (f"{s['n']:4d}筆 {s['per_day']:.1f}/日 勝率 {s['win']:.0f}% 每筆 {s['avg']:+6.2f}（{s['avgR']:+.2f}R） PF {s['pf']:.2f}"
            f" 總 {s['total']:+6.0f} DD {s['dd']:5.0f} | 0.5%/筆 {s['ret_pct']:+5.0f}% DD {s['dd_pct']:5.1f}% Sharpe {s['sharpe']:.2f}")

P("第二十二部分 A：保真度（2026/3/3~10/7，門檻滾動前推，暖身 = 2026/1~2）")
m1 = pd.read_csv("final_A_trades.csv")
p = m1["損益_美元每盎司"]; P(f"  M1 原版（final_A_trades.csv）   {len(m1)}筆 每筆 {p.mean():+.2f}（{m1['損益_R'].mean():+.2f}R） PF {p[p > 0].sum() / -p[p <= 0].sum():.2f} 總 {p.sum():+.0f}")
for lab, path, bm in [("M5 重建", "data/XAUUSD_M5_2025_2026.csv", 5), ("M15 重建", "data/XAUUSD_M15_2023_2026.csv", 15)]:
    days, D, ARR = load_bars(path, bm)
    X = simulate(signals(days, D, ARR, bm), ARR, bm)
    X = X[(X.day >= "2026-01-01")]
    T = walk_forward(X); T = T[T.day <= "2026-10-07"]
    P(f"  {lab:28s} {fmt(stats(T, days))}")
    if bm == 5:
        T5 = T
# 逐筆對照 M5 vs M1：同一天、同一 IB 起點、同方向
m1["key"] = m1["交易日"].astype(str) + "|" + m1["IB起點"] + "|" + m1["方向"].map({"多": "1", "空": "-1"})
T5["key"] = T5.day.dt.strftime("%Y-%m-%d") + "|" + T5.h.map(lambda h: f"{int(h):02d}:{int(round((h % 1) * 60)):02d}") + "|" + T5.side.astype(str)
both = set(m1.key) & set(T5.key)
P(f"  M5 與 M1 逐筆重疊：M1 {len(m1)} 筆中 {len(both)} 筆在 M5 版也有（{len(both) / len(m1) * 100:.0f}%）")
out.close()
