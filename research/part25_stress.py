"""第二十五部分 (3)：A+B+C tp2 — 不做星期一、同時持倉數、滑價壓力測試、風險換算 → results_part25_stress.txt"""
import numpy as np
import pandas as pd
from part23_ma_cycle import load_m1
from part25_exits import CFGS, prep, entries, metrics

m1 = load_m1()
days = sorted(set(m1.index.date))
D = pd.read_csv("part25_ext_trades.csv", parse_dates=["time", "xtime"])
out = []
P = out.append
P("基準：" + str(metrics(D, days)))
P("不做星期一：" + str(metrics(D[D.dow != 0], days)))

# 同時持倉數
ev = pd.concat([pd.Series(1, index=D.time), pd.Series(-1, index=D.xtime)]).sort_index()
cc = ev.cumsum()
P(f"同時持倉：最大 {cc.max()} 筆、平均（有持倉時）{cc[cc > 0].mean():.2f} 筆")

# R 的美元大小 → 滑價壓力
cache = {}
risk = []
for name, (tf, L, M, S, a, cool) in CFGS.items():
    B = cache.setdefault(tf, prep(m1, tf))
    t, side = entries(B, L, M, S, a, cool)
    for k, s in zip(t, side):
        ext = B["l"][k - M + 1:k + 1].min() if s == 1 else B["h"][k - M + 1:k + 1].max()
        risk.append(dict(leg=name, time=B["idx"][k], Rusd=s * (B["o"][k + 1] - (ext - s * 0.1 * B["A"][k]))))
Rk = pd.DataFrame(risk)
D2 = D.merge(Rk, on=["leg", "time"], how="left")
P("每筆 R 的美元大小（每盎司）：" + D2.groupby("leg").Rusd.median().round(1).to_string().replace("\n", "  "))
for slip in (0.2, 0.5, 1.0):
    T = D2.assign(R=D2.R - slip / D2.Rusd)
    P(f"每筆再扣 ${slip} 滑價：" + str(metrics(T, days)))

P("\n風險換算（以 tp2 基準、每筆固定風險）")
m = metrics(D, days)
for r in (0.25, 0.5, 1.0):
    P(f"  每筆風險 {r}%：9 個月總報酬約 {m['總R'] * r:.0f}%（單利）、最大回撤約 {m['MDD'] * r:.1f}%、"
      f"最多同時 {cc.max()} 筆 = 同時承擔 {cc.max() * r:.1f}% 風險")
open("results_part25_stress.txt", "w").write("\n".join(out) + "\n")
print("\n".join(out))
