"""第二十八部分 B：OW 搬到 M5 —— 參數不變 vs 參數等比例縮小（2025/7~2026/10，M5 資料）。
參數不變：IB 60 分、每 30 分起、突破窗 3 小時（只是確認改用 M5 收盤）。
縮小 1/3：IB 20 分、每 10 分起、突破窗 1 小時；縮小 1/2：IB 30 分、每 15 分起、突破窗 1.5 小時。
緩衝（0.05×ATR10）、前一天價值區、停損 IB 另一側、收盤出場都不變。"""
import pandas as pd
from part27_ow import run, per_cells, PER, pf

out = open("results_part28.txt", "a")
def P(s=""): print(s); out.write(s + "\n"); out.flush()
P5 = [p for p in PER if p[1] >= "2025-07-01"]
P("\n第二十八部分 B：搬到 M5（每格 = 每筆 R / PF / 勝率 / 每日筆數）")
X15, d15 = run("data/XAUUSD_M15_2023_2026.csv", 15)
P(f"  M15 參數不變（定案）        {per_cells(X15[X15.ow], d15, P5)}")
for lab, kw in [("M5 參數不變", {}), ("M5 縮小 1/2（IB30/每15分/窗1.5h）", dict(ib_min=30, step=15, win=1.5)),
                ("M5 縮小 1/3（IB20/每10分/窗1h）", dict(ib_min=20, step=10, win=1))]:
    X, d = run("data/XAUUSD_M5_2025_2026.csv", 5, **kw)
    T = X[X.ow & (X.day >= "2025-07-01")]
    hold = (T.exit_time - T.t_in).dt.total_seconds() / 3600
    P(f"  {lab:30s} {per_cells(T, d, P5)} | 合計 {len(T)}筆 每筆 {T.R.mean():+.3f}R 總 {T.R.sum():+.0f}R 停損距離中位 {T.risk.median():.1f} 美元 持倉 {hold.mean():.1f}h")

# 補充：同時持倉、R 回撤、日 Sharpe（同一段 2025/7~2026/10）
import numpy as np
P("\n  風險面比較（每筆 1R；回撤以 R 計）")
for lab, path, bm, kw in [("M15 參數不變", "data/XAUUSD_M15_2023_2026.csv", 15, {}), ("M5 參數不變", "data/XAUUSD_M5_2025_2026.csv", 5, {}),
                          ("M5 縮小 1/2", "data/XAUUSD_M5_2025_2026.csv", 5, dict(ib_min=30, step=15, win=1.5)),
                          ("M5 縮小 1/3", "data/XAUUSD_M5_2025_2026.csv", 5, dict(ib_min=20, step=10, win=1))]:
    X, d = run(path, bm, **kw)
    T = X[X.ow & (X.day >= "2025-07-01")].sort_values("exit_time")
    eq = T.R.cumsum(); dd = (eq - eq.cummax()).min()
    ev = sorted([(a, 1) for a in T.t_in] + [(b, -1) for b in T.exit_time], key=lambda x: (x[0], x[1])); cur = mx = 0
    for _, e in ev:
        cur += e; mx = max(mx, cur)
    dl = T.groupby("day").R.sum().reindex([x for x in d if x >= pd.Timestamp("2025-07-01")], fill_value=0)
    lose = (T.R <= 0).astype(int)
    P(f"    {lab:12s} 總 {T.R.sum():+5.0f}R  最大回撤 {dd:5.0f}R  總/回撤 {T.R.sum() / -dd:4.1f}  同時持倉最多 {mx:2d}  最差單日 {dl.min():+5.1f}R"
      f"  日 Sharpe {dl.mean() / dl.std() * np.sqrt(250):.2f}  最長連虧 {lose.groupby((lose != lose.shift()).cumsum()).sum().max()}")
out.close()
