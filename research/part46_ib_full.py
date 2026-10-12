"""第四十六部分（一）：把滾動 IB 版本 A（FINAL_STRATEGY_A.md）套到 2023/6~2026/10 的完整 M1 資料。
規則、參數完全不變；門檻每月月初只用過去資料重算（滾動前推）。輸出 part46_ib_trades.csv（每筆 1 盎司損益，已扣點差）。
注意：這套規則是在 2026 年資料上做出來的，2023~2025 對它而言是樣本外。"""
import tpo
tpo.load_m1.__defaults__ = ("data/XAUUSD_M1_2023_2026.csv",)
import final_A
from part17_balance import gen

X = gen(**final_A.CFG)
T, used = final_A.walk_forward(X)
T.to_csv("part46_ib_trades.csv", index=False)
used.to_csv("part46_ib_thresholds.csv", index=False)
print(len(T), T.day.min(), T.day.max(), T.pnl.sum())
