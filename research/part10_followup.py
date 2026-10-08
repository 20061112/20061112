"""第十部分補充：候選訊號的穩定性檢查（讀 run_part10.py 輸出的事件表）。"""
import numpy as np, pandas as pd
from run_part10 import ALL, D, P, tstat, summ, OUT
import run_part10 as R
R.OUT = OUT = open("results_part10.txt", "a")


def auc(score, y):
    s = pd.Series(score).rank().values; y = np.asarray(y, bool)
    n1, n0 = y.sum(), (~y).sum()
    return (s[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


P("\n" + "=" * 100 + "\n補充檢查")
for s, E in ALL.items():
    P(f"\n[{s}]")
    # 1. 前日 POC 回補：剖面 z 是否比單純 ATR 距離更好？
    x = E.dropna(subset=["poc_touch", "poc_dist"])
    y = x.poc_touch.values > 0.5
    P("  碰前日 POC 的 AUC（越小越不會碰）：|剖面 z| %.3f  vs  |距離/ATR| %.3f  vs  |距離/IB區間| %.3f" %
      (1 - auc(x.z_prev.abs(), y), 1 - auc(x.poc_dist.abs(), y), 1 - auc((x.poc_dist * x.atr / x.rng).abs(), y)))
    # 2. overlap 依跨 VAH / VAL 分開；報酬方向 + = 往前日 VA 內
    prev = D.shift(1).reindex(E.index)
    hi = (E.vloc == "overlap") & (E.ibh > prev.vah) ; lo = (E.vloc == "overlap") & (E.ibl < prev.val) & ~hi
    for lab, m, sg in [("IB 跨 VAH", hi, -1), ("IB 跨 VAL", lo, 1), ("IB 在 VA 上方", E.vloc == "above_VA", -1), ("IB 在 VA 下方", E.vloc == "below_VA", 1)]:
        r = sg * E.ret_close[m]
        P(f"  {lab:10s} 收盤報酬（+ = 往前日價值區）{summ(r)} | H1 {summ(r[E.half == 'H1'])} | H2 {summ(r[E.half == 'H2'])}")
    # 3. 大 IB 突破延續 / LVN 突破延續
    for lab, m in [("大IB z>0.5", E.ib_z > 0.5), ("小IB z<-0.5", E.ib_z < -0.5),
                   ("LVN tree_z<-0.5", E.tree_z < -0.5), ("HVN tree_z>0.5", E.tree_z > 0.5),
                   ("大IB 且 非HVN", (E.ib_z > 0.5) & ~(E.tree_z > 0.5))]:
        r = E.brk_ret[m]
        P(f"  首破延續 {lab:16s} {summ(r)} | H1 {summ(r[E.half == 'H1'])} | H2 {summ(r[E.half == 'H2'])} | 勝率 {(r > 0).mean() * 100:.0f}%")
    # 4. IB z 預測當天剩餘波幅（ATR）— 與 IB/ATR 比
    c = E[["ib_z", "ib_atr", "rest_range"]].dropna().corr("spearman")
    P("  剩餘波幅(ATR) Spearman：ib_z %.2f, IB/ATR %.2f" % (c.loc["ib_z", "rest_range"], c.loc["ib_atr", "rest_range"]))
    # 5. 延伸目標：依 IB z 三組的延伸倍數分位數（單邊最大）
    for lab, m in [("小IB z<-0.5", E.ib_z < -0.5), ("中", E.ib_z.between(-0.5, 0.5)), ("大IB z>0.5", E.ib_z > 0.5)]:
        q = E.ext_max[m].quantile([.25, .5, .75]).round(2).tolist()
        P(f"  延伸倍數 {lab:12s} 25/50/75% = {q}  （IB 區間中位數 {E.rng[m].median():.1f}）")
OUT.close()
