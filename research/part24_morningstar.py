"""第二十四部分：唯一兩段平均都為正的家族（晨星/夜星）細看：各定義 × 時框 × 多空，突破 2R 每筆 R、t 值、逐年。"""
import pandas as pd

R = pd.read_csv("part24_patterns.csv")
M = R[R.family == "晨星"]
print("晨星（多）/ 夜星（空）：突破進場、停損另一端、目標 2R、最多 12 根、扣點差。格式：每筆 R (t, 筆數)")
for tf in ("M5", "M15", "M30", "H1"):
    print(f"\n## {tf}")
    for _, r in M[M.tf == tf].iterrows():
        print(f"  {r.pattern:30s} {'晨星' if r.side > 0 else '夜星'}  發現 {r.R2_0:+.3f} (t{r.Rt2_0:+.1f}, {int(r.Rn2_0)})  驗證 {r.R2_1:+.3f} (t{r.Rt2_1:+.1f}, {int(r.Rn2_1)})  基準 {r.Rb2_0:+.3f}/{r.Rb2_1:+.3f} | {r.Ryear2}")
s = M.assign(both=(M.R2_0 > M.Rb2_0) & (M.R2_1 > M.Rb2_1))
print(f"\n兩段都勝過基準的 定義×時框×方向：{s.both.sum()} / {len(s)}")
print(s.groupby("tf").both.mean().round(2).to_string())
o = R[R.family != "晨星"].assign(both=lambda d: (d.R2_0 > d.Rb2_0) & (d.R2_1 > d.Rb2_1))
print(f"對照：其他所有型態兩段都勝過基準的比例 {o.both.mean():.2f}")
