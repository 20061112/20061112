"""第三十六部分：保守版最終系統的詳細統計與回撤（不加尾巴）。
規則：
  1. L1 核心：IB 02:00~10:30 每 30 分、長 60 分；IB 結束後 3 小時內，M5 收盤 > IB 高 + 0.05 ATR10 做多（< IB 低 − 0.05 ATR10 做空）；
     停損 IB 另一側；當天收盤平倉；沒有停利；星期一也做；同一根 M5 同方向只開一筆
  2. 120 天方向開關：每天開盤前，多單 / 空單分別看「這個系統自己」最近 120 個交易日的累計 R，< 0 的那一邊今天不開
  3. 疊單上限：同方向同時最多 3 筆（第 4 筆以後不開）
"""
import numpy as np
import pandas as pd
import part33_gate120_stats as p33
from part29_dir_filter import L1, dedup, pf
from part34_combo_stack import gate, cap

out = open("results_part36.txt", "w")
p33.out = out
P = p33.P

C = dedup(L1)
G = gate(C, C, 120)
F = cap(G, K=3)
P("第三十六部分：保守版最終系統（L1 核心 + 120 天方向開關 + 同方向最多 3 筆，不加尾巴）")
res = {}
for lab, T in (("最終系統（核心 + 開關 + 最多 3 筆）", F), ("對照：核心 + 開關（不限疊單）", G), ("對照：核心（不加任何濾網）", C)):
    res[lab] = p33.detailed(T, lab)

T, eq_cl, eq_mtm, dd_mtm = res["最終系統（核心 + 開關 + 最多 3 筆）"]
P("\n\n# 最終系統補充")
st = p33.gate_state(C).loc[C.day.min():C.day.max()]
for y, g in st.groupby(st.index.year):
    P(f"  {y}: 多單開放 {g['多開'].mean():.0%} 的交易日，空單開放 {g['空開'].mean():.0%}")
mdd = -dd_mtm.min()
P("\n  ## 每筆風險要設多少（以含浮動最大回撤反推，1 萬美元、不複利）")
for target in (10, 15, 20, 25):
    r = target / mdd
    P(f"    想把最大回撤控制在 {target}% → 每筆風險 {r:.2f}% → 全期總報酬約 {T.R.sum() * r:+.0f}%（年化約 {T.R.sum() * r / 3.2:+.0f}%）")
P("\n  ## 逐季")
q = T.groupby(T.day.dt.to_period("Q"))
P(pd.DataFrame({"筆數": q.size(), "每筆R": q.R.mean(), "總R": q.R.sum(), "PF": q.R.apply(pf),
                "多筆": q.side.apply(lambda s: (s == 1).sum()), "空筆": q.side.apply(lambda s: (s == -1).sum())}).round(2).to_string())
P("\n  ## IB 起點")
ib = T.groupby(T.h.map(lambda h: f"{int(h):02d}:{int(round(h % 1 * 60)):02d}"))
P(pd.DataFrame({"筆數": ib.size(), "每筆R": ib.R.mean(), "PF": ib.R.apply(pf)}).round(2).T.to_string())
P("\n  ## 星期")
wd = T.groupby(T.day.dt.dayofweek.map(dict(enumerate("一二三四五"))))
P(pd.DataFrame({"筆數": wd.size(), "每筆R": wd.R.mean(), "PF": wd.R.apply(pf)}).round(2).T.to_string())
P("\n  ## 出場與持倉")
P(f"    停損 {(T.exit_reason == '停損').mean():.0%} / 收盤 {(T.exit_reason == '收盤').mean():.0%}；收盤出場勝率 {(T[T.exit_reason == '收盤'].R > 0).mean():.0%}")
P(f"    平均持倉 {T.hold_h.mean():.1f} 小時；停損單 {T[T.exit_reason == '停損'].hold_h.mean():.1f}、收盤單 {T[T.exit_reason == '收盤'].hold_h.mean():.1f}")
P(f"    停損距離中位 {T.risk.median():.2f} 美元（{(T.risk / T.atr).median():.2f} ATR10）；點差 / 停損中位 {(T.spread / T.risk).median():.3f}R")
dn = T.groupby("day").size()
P(f"    每個有單的交易日筆數：中位 {dn.median():.0f}、平均 {dn.mean():.1f}、最多 {dn.max()}")
T.drop(columns=["ym"], errors="ignore").to_csv("part36_final_trades.csv", index=False, float_format="%.4f")


def chart():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1, 1]))
    cols = {"最終系統（核心 + 開關 + 最多 3 筆）": ("#3b6ea8", "final: core + 120d gate + max 3"),
            "對照：核心 + 開關（不限疊單）": ("#c9824a", "core + gate (no cap)"), "對照：核心（不加任何濾網）": ("#999999", "core only")}
    for lab, (Tx, ec, em, dm) in res.items():
        c, l = cols[lab]
        ax[0].plot(em.resample("1h").last().dropna(), color=c, lw=1.1, label=l)
        ax[1].plot(dm.resample("1h").min().dropna(), color=c, lw=.9, label=l)
    ax[0].set_ylabel("cumulative R (1R per trade)"); ax[0].legend(loc="upper left"); ax[0].grid(alpha=.3)
    ax[0].set_title("Conservative final system vs. references (incl. open positions)")
    ax[1].set_ylabel("drawdown (R)"); ax[1].grid(alpha=.3)
    mon = T.groupby(T.day.dt.to_period("M")).R.sum()
    x = mon.index.to_timestamp()
    ax[2].bar(x, mon.values, width=20, color=["#2e8b57" if v > 0 else "#c0392b" for v in mon.values])
    ax[2].set_ylabel("final system: monthly R"); ax[2].grid(alpha=.3)
    for a in ax:
        a.axvline(pd.Timestamp("2026-01-01"), color="k", ls="--", lw=.7)
    plt.tight_layout(); plt.savefig("part36_final.png", dpi=105)


chart()
out.close()
