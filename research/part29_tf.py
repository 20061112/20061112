"""第二十九部分：OW 在 M1 / M5 / M15 —— 互補嗎？搬到 M1 呢？
重疊期間 A：2025/7~2026/10（M5 vs M15）；B：2026/1~10/8（M1 vs M5 vs M15，M1 缺 4/3~4/14）。
規則同 FINAL_STRATEGY_OW（只換確認用的 K 棒週期；「縮小」= IB / 起點間隔 / 突破窗等比例縮小）。"""
import numpy as np, pandas as pd
from part27_ow import run

out = open("results_part29.txt", "w")
def P(s=""): print(s); out.write(s + "\n"); out.flush()
SRC = {"M15": ("data/XAUUSD_M15_2023_2026.csv", 15), "M5": ("data/XAUUSD_M5_2025_2026.csv", 5), "M1": ("data/XAUUSD_M1_2026.csv", 1)}
CFG = {"M15": ("M15", {}), "M5": ("M5", {}), "M1": ("M1", {}),
       "M5 縮1/3": ("M5", dict(ib_min=20, step=10, win=1)),
       "M1 縮1/3": ("M1", dict(ib_min=20, step=10, win=1)), "M1 縮1/6": ("M1", dict(ib_min=10, step=5, win=0.5))}
TR = {}
for k, (src, kw) in CFG.items():
    X, d = run(*SRC[src], **kw)
    TR[k] = X[X.ow].copy(); TR[k]["key"] = TR[k].day.dt.strftime("%Y-%m-%d") + "|" + TR[k].h.round(3).astype(str) + "|" + TR[k].side.astype(str)


def daily(T, a, b, days):
    return T[(T.day >= a) & (T.day <= b)].groupby("day").R.sum().reindex(days, fill_value=0)


def stats(s):
    eq = s.cumsum(); dd = (eq - eq.cummax()).min()
    return f"總 {s.sum():+5.0f}R 回撤 {dd:5.0f}R 總/回撤 {s.sum() / -dd:4.1f} 日Sharpe {s.mean() / s.std() * np.sqrt(250):.2f} 最差日 {s.min():+.1f}R"


for lab, a, b, names in [("A：2025/7~2026/10", "2025-07-01", "2026-10-09", ["M15", "M5", "M5 縮1/3"]),
                         ("B：2026/1~10/8", "2026-01-01", "2026-10-08", ["M15", "M5", "M1", "M5 縮1/3", "M1 縮1/3", "M1 縮1/6"])]:
    common = sorted(set.intersection(*[set(TR[n][(TR[n].day >= a) & (TR[n].day <= b)].day) | set() for n in names]) |
                    set(TR[names[0]][(TR[names[0]].day >= a) & (TR[names[0]].day <= b)].day))
    days = sorted(set(TR["M15"][(TR["M15"].day >= a) & (TR["M15"].day <= b)].day) | set(TR["M5"][(TR["M5"].day >= a) & (TR["M5"].day <= b)].day))
    if "M1" in names:   # 只用 M1 有資料的日子
        m1days = set(TR["M1"].day)
        days = [d for d in days if d >= pd.Timestamp("2026-01-01") and not (pd.Timestamp("2026-04-03") <= d <= pd.Timestamp("2026-04-14"))]
    P(f"\n=== 期間 {lab}（{len(days)} 個有訊號的交易日）===")
    S = {}
    for n in names:
        T = TR[n][(TR[n].day >= a) & (TR[n].day <= b) & TR[n].day.isin(days)]
        S[n] = daily(T, a, b, days)
        P(f"  {n:9s} {len(T):4d}筆 勝率 {(T.pnl > 0).mean() * 100:3.0f}% 每筆 {T.R.mean():+.2f}R | {stats(S[n])}")
    P("  日損益相關：" + "  ".join(f"{x}~{y} {S[x].corr(S[y]):+.2f}" for i, x in enumerate(names) for y in names[i + 1:]))
    base = names[0]
    for n in names[1:]:
        A_, B_ = TR[base], TR[n]
        A_ = A_[(A_.day >= a) & (A_.day <= b) & A_.day.isin(days)]; B_ = B_[(B_.day >= a) & (B_.day <= b) & B_.day.isin(days)]
        same = len(set(A_.key) & set(B_.key))
        P(f"  {base} vs {n}：同一天、同一 IB、同方向的重疊 {same} 筆（{base} {len(A_)} 筆的 {same / max(len(A_), 1) * 100:.0f}%）")
    for x, y in [(names[0], names[1])] + ([("M5", "M1")] if "M1" in names else []):
        comb = (S[x] + S[y]) / 2
        P(f"  合併：{x} 一半風險 + {y} 一半風險 → {stats(comb)}")
for k, T in TR.items():
    T.to_csv(f"part29_ow_{k.replace(' ', '_').replace('/', '-')}.csv", index=False)
out.close()
