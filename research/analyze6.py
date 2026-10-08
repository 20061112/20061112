import numpy as np
import pandas as pd

R = pd.read_pickle("scan6.pkl")
pd.set_option("display.width", 250)
q = R[(R.dev_n >= 30) & (R.test_n >= 20)]

print("== 各週期 × 壓縮方式：參數組平均（研究期 / 驗證期）==")
g = q.groupby(["tf", "comp"]).agg(組數=("dev_exp", "size"), 研究期=("dev_exp", "mean"), 研究正比例=("dev_exp", lambda x: (x > 0).mean()),
                                  驗證期=("test_exp", "mean"), 驗證正比例=("test_exp", lambda x: (x > 0).mean()))
print(g.round(3).to_string())

print("\n== 壓縮條件的淨效果：同一組其他參數下，(ma 或 sigma) 減 none ==")
key = ["tf", "ma_type", "lens", "box", "fan", "stop", "exit"]
none = R[R.comp == "none"].set_index(key)[["dev_exp", "test_exp"]]
for comp in ("ma", "sigma"):
    x = R[R.comp == comp].join(none, on=key, rsuffix="_none")
    x["d_dev"], x["d_test"] = x.dev_exp - x.dev_exp_none, x.test_exp - x.test_exp_none
    print(comp, x.groupby("tf")[["d_dev", "d_test"]].agg(["mean", lambda s: (s > 0).mean()]).round(3).to_string())

for col in ("exit", "stop", "fan", "box", "lens", "ma_type", "pct"):
    print(f"\n-- 依 {col}")
    print(q.groupby(["tf", col])[["dev_exp", "test_exp"]].mean().unstack("tf").round(3).to_string())

print("\n== 各週期研究期 t 值最高的 10 組，驗證期表現 ==")
for tf in ("M1", "M5", "M15"):
    top = q[q.tf == tf].sort_values("dev_t", ascending=False).head(10)
    print(tf, f"驗證期平均 {top.test_exp.mean():+.3f}R，正的比例 {(top.test_exp > 0).mean():.1f}")
    print(top[["ma_type", "lens", "comp", "pct", "box", "fan", "stop", "exit", "dev_n", "dev_exp", "dev_t", "test_n", "test_exp", "test_t"]].round(3).to_string(index=False))
