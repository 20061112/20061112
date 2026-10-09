"""第二十二部分 J：RFI-U 能不能當版本 A（IB 突破）的過濾器？

版本 A 的每一筆單，在進場那根 M5 收盤時計算 EMA20 的 D 與 RFI-U（只用已收完的 K 棒）。
A 是順勢突破，進場方向幾乎都和 D 同向（順著偏離）→ 照 RFI 的邏輯，低 RFI（回復力弱）應該較好。
"""
import numpy as np
import pandas as pd
from part22_lib import load_tf, means, atr, CUT
from part22_beta import frame
from part22_final import universal_selection, rfi_u

out = open("results_part22.txt", "a")
def P(s=""): print(s, flush=True); out.write(s + "\n"); out.flush()
pf = lambda p: p[p > 0].sum() / -p[p <= 0].sum()

if __name__ == "__main__":
    keep = universal_selection(pd.read_csv("part22_beta.csv"))
    d = load_tf("M5"); A = atr(d, 14); M = means(d)
    X = frame(d, A, M, "ema20"); X["rfi"] = rfi_u(X, keep)
    T = pd.read_csv("final_A_trades.csv")
    t = pd.to_datetime(T["進場時間"]) - pd.Timedelta(minutes=4)          # 進場 = 該根 M5 最後一分鐘收盤
    pos = X.index.searchsorted(t, side="right") - 1
    T["rfi"], T["D"] = X.rfi.to_numpy()[pos], X.D.to_numpy()[pos]
    T["with_D"] = np.sign(T.D) == np.where(T["方向"] == "多", 1, -1)
    T["per"] = np.where(pd.to_datetime(T["交易日"]) < CUT, "IS（3~5月）", "OOS（6~10月）")
    lo, hi = np.nanquantile(X.loc[X.per != "OOS", "rfi"], [1 / 3, 2 / 3])
    T["g"] = np.where(T.rfi <= lo, "低 RFI（動能）", np.where(T.rfi >= hi, "高 RFI（回歸）", "中"))
    P("\n" + "=" * 120)
    P("第二十二部分 J：版本 A 的交易依進場時 RFI-U 分組（EMA20、M5）")
    P(f"  與 D 同方向的比例：{T.with_D.mean():.0%}")
    for per, s in T.groupby("per"):
        P(f"  {per} 全部 {len(s)} 筆  每筆 {s['損益_R'].mean():+.3f}R  PF {pf(s['損益_美元每盎司']):.2f}")
        for k, g in s.groupby("g"):
            P(f"    {k:12s} {len(g):4d} 筆  每筆 {g['損益_R'].mean():+.3f}R  PF {pf(g['損益_美元每盎司']):.2f}")
