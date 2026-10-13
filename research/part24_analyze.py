"""第二十四部分 B：彙整 part24_scan.csv（1-2-3 / 2B 全部組合）。"""
import numpy as np
import pandas as pd

df = pd.read_csv("part24_scan.csv")
df = df[(df.IS_n >= 30) & (df.OOS_n >= 30)]
pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 300)
print(f"有效組合 {len(df)}（樣本內、樣本外各至少 30 筆；同一時間只持有一筆）\n")

print("1) 各型態 × 週期：組合數、樣本內 / 樣本外平均 R、為正的比例、IS 與 OOS 的相關")
g = df.groupby(["pattern", "tf"])
print(pd.DataFrame({
    "組合數": g.size(),
    "IS平均R": g.IS_R.mean(), "OOS平均R": g.OOS_R.mean(),
    "IS>0比例": g.IS_R.apply(lambda x: (x > 0).mean()), "OOS>0比例": g.OOS_R.apply(lambda x: (x > 0).mean()),
    "IS顯著(t>2)": g.IS_t.apply(lambda x: (x > 2).mean()), "OOS顯著(t>2)": g.OOS_t.apply(lambda x: (x > 2).mean()),
    "IS_OOS相關": g.apply(lambda x: x.IS_R.corr(x.OOS_R)),
}).round(3).to_string())

print("\n2) 各演算法（波段點）平均 R")
print(df.pivot_table(index=["pattern", "swing"], columns="tf", values=["IS_R", "OOS_R"], aggfunc="mean")
      .round(3).to_string())

for col in ("entry", "stop", "tp", "f1", "f2", "trend"):
    print(f"\n3) 依「{col}」分組的平均 R")
    print(df.pivot_table(index=["pattern", col], values=["IS_R", "OOS_R", "IS_n"], aggfunc="mean")
          .round(3).to_string())

print("\n4) 多單 / 空單 平均 R（所有組合平均）")
print(df.groupby(["pattern", "tf"])[["long_R", "short_R"]].mean().round(3).to_string())

cols = ["pattern", "tf", "swing", "entry", "stop", "tp", "f1", "f2", "trend", "IS_n", "IS_win", "IS_R", "IS_t",
        "OOS_n", "OOS_win", "OOS_R", "OOS_t", "long_R", "short_R", "y2023", "y2024", "y2025", "y2026"]
print("\n5) 只用樣本內挑選：每個型態 × 週期 取 IS t 值最高的 5 組（IS 至少 100 筆），看樣本外")
sel = df[df.IS_n >= 100].sort_values("IS_t", ascending=False).groupby(["pattern", "tf"]).head(5)
print(sel[cols].sort_values(["pattern", "tf", "IS_t"], ascending=[True, True, False]).round(3).to_string(index=False))
print("\n   被選出的組合樣本外平均 R：")
print(sel.groupby(["pattern", "tf"]).OOS_R.mean().round(3).to_string())

print("\n6) 樣本內、樣本外都 t > 2 的組合")
both = df[(df.IS_t > 2) & (df.OOS_t > 2)].sort_values("OOS_t", ascending=False)
print(f"共 {len(both)} 組 / {len(df)}；隨機下的期望約 {len(df) * 0.023 ** 2:.1f} 組（若彼此獨立）")
print(both[cols].head(40).round(3).to_string(index=False))

print("\n7) 每年都為正的組合數（2023~2026 四年，IS、OOS 各至少 100 筆）")
yy = df[(df.IS_n >= 100) & (df.OOS_n >= 100)]
allpos = yy[(yy[["y2023", "y2024", "y2025", "y2026"]] > 0).all(axis=1)]
print(f"{len(allpos)} / {len(yy)}")
print(allpos.sort_values("OOS_t", ascending=False)[cols].head(30).round(3).to_string(index=False))
