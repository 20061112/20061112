"""第二十九部分 (2)：整合後的最終版本（只保留前後兩半一致的選擇）→ results_part29_final.txt、part29_final_trades.csv

固定：週五最後一根平倉、不做星期一、掉期 $0.7/盎司/晚（週三 ×3）、停損上限 $100（拉近不跳單）、週五 20 點後不開新單
每組出場（前半段挑選，後半段驗證）：A tp1.5 最長 80 根、B tp2 最長 20 根、C tp1.5 最長 40 根
部位三種：1 單位 / 1R 且同向加 1 / 同向 2 單位 + 1R 加 1
MDD 兩種算法：逐筆（依出場時間）與每日合計
"""
import numpy as np
import pandas as pd
from part23_ma_cycle import load_m1, SPLIT
from part25_exits import prep, entries, metrics
from part25_ext import tf_er_done
from part27_sizing import sim
from part28_final import nights
from part29_optimize import LEGS, pmetrics

EXIT = {"A": (1.5, 80), "B": (2, 20), "C": (1.5, 40)}
m1 = load_m1()
days = sorted(set(m1.index.date))
cache, rows = {}, []
for leg, (tp, hold) in EXIT.items():
    tf, L, M, S, a, cool = LEGS[leg]
    B = cache.setdefault(tf, prep(m1, tf))
    if "h1" not in B:
        B["h1"], B["h4"] = tf_er_done(m1, B, "1h"), tf_er_done(m1, B, "4h")
    step = B["idx"][1] - B["idx"][0]
    t, side = entries(B, L, M, S, a, cool)
    for k, s in zip(t, side):
        ts = B["idx"][k] + step
        if ts.dayofweek == 0 or (ts.dayofweek == 4 and ts.hour >= 20):
            continue
        r = sim(B, k, s, M, hold // 2, tp=tp, cap_usd=100, cap_mode="cap", flat="week")
        if not r:
            continue
        nt = nights(B["idx"], k + 1, r["j"])
        sw = 0.7 * nt
        rows.append(dict(leg=leg, time=B["idx"][k], xtime=B["idx"][r["j"]], side=s, bars=r["j"] - k, nights=nt,
                         htf=bool((s * B["h1"][k] > 0) and (s * B["h4"][k] > 0)), added=r["added"], reason=r["reason"],
                         stop_usd=round(r["Rusd"], 2), R=r["R"] - sw / r["Rusd"],
                         addR=(r["addR"] - sw / r["Rusd"]) if r["added"] else 0.0,
                         usd=r["usd"] - sw, add_usd=(r["add_usd"] - sw) if r["added"] else 0.0))
T = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
out = []
P = out.append
P(f"訊號 {len(T)} 筆（{len(T) / len(days):.2f} 筆/天），過夜 {(T.nights > 0).mean():.0%}，出場原因：" +
  T.reason.value_counts().to_string().replace("\n", "、"))
res = []
for lab, x2, add in (("1 單位", False, False), ("1R 且同向加 1", False, True), ("同向 2 單位 + 1R 加 1", True, True)):
    w = np.where(T.htf & x2, 2.0, 1.0)
    wa = np.where(T.htf & add & T.added, 1.0, 0.0)
    R = T.R * w + T.addR * wa
    usd = T.usd * w + T.add_usd * wa
    pm, tm = pmetrics(T, R, days), metrics(T.assign(R=R), days)
    h = T.time < SPLIT
    res.append(dict(部位=lab, n=len(T), 每天=round(len(T) / len(days), 2), 平均單位=round(float((w + wa).mean()), 2),
                    勝率=tm["win"], PF=tm["PF"], 每筆R=round(float(R.mean()), 3), 總R=round(float(R.sum()), 1),
                    MDD逐筆=tm["MDD"], MDD每日=round(pm["全MDD"], 1), P_MDD=round(float(R.sum()) / tm["MDD"], 2),
                    Sharpe=round(pm["全Sharpe"], 2), 前半Sharpe=round(pm["前Sharpe"], 2), 後半Sharpe=round(pm["後Sharpe"], 2),
                    前半R=round(float(R[h].sum()), 1), 後半R=round(float(R[~h].sum()), 1),
                    每筆美元=round(float(usd.mean()), 2), 總美元=round(float(usd.sum())),
                    最大單筆虧損美元=round(float(usd.min()), 1)))
    T[f"R_{lab}"] = R
P(pd.DataFrame(res).set_index("部位").T.to_string())
P("\n逐月 R（1 單位 / 同向 2 + 1R 加 1）")
mo = T.groupby(T.xtime.dt.to_period("M"))[["R_1 單位", "R_同向 2 單位 + 1R 加 1"]].sum().round(1)
P(mo.T.to_string())
P("\n各組（1 單位）")
P(T.groupby("leg").agg(n=("R", "size"), 每筆R=("R", "mean"), 總R=("R", "sum"), 每筆美元=("usd", "mean"),
                       停損中位=("stop_usd", "median")).round(2).to_string())
T.to_csv("part29_final_trades.csv", index=False)
open("results_part29_final.txt", "w").write("\n".join(out) + "\n")
print("\n".join(out))
