"""第三十四部分補充：三個一致因子的逐年檢查、相關性、組合（影子訊號，每筆 R）。
切點全部來自訓練期（2023~2024）的五分位：排除最差的那一個五分位。"""
import numpy as np
import pandas as pd

X = pd.read_pickle("part34_X.pkl")
X["year"] = pd.read_csv("final_m15_signals.csv", parse_dates=["進場時間"]).進場時間.dt.year.values
X["date"] = pd.read_csv("final_m15_signals.csv", parse_dates=["進場時間"]).進場時間.dt.date.values
tr = X[X.train]
c1 = np.nanquantile(tr["ATR15/ATR日"], 0.8)
c2 = np.nanquantile(tr["BB寬度百分位(M15)"], 0.8)
c3 = np.nanquantile(tr["日ER10"], 0.2)
F = {"F1 ATR15/ATR日 ≤ %.3f（不是最大的 20%%）" % c1: X["ATR15/ATR日"] <= c1,
     "F2 BB寬度百分位 ≤ %.2f（不是最寬的 20%%）" % c2: X["BB寬度百分位(M15)"] <= c2,
     "F3 日ER10 ≥ %.3f（不是最盤整的 20%%）" % c3: X["日ER10"] >= c3}
lines = ["", "=== 補充：三個一致因子（排除訓練期最差的五分位）===",
         f"相關係數：ATR15/ATR日 vs BB寬度 {X['ATR15/ATR日'].corr(X['BB寬度百分位(M15)']):.2f}；"
         f"ATR15/ATR日 vs 日ER10 {X['ATR15/ATR日'].corr(X['日ER10']):.2f}；BB寬度 vs 日ER10 {X['BB寬度百分位(M15)'].corr(X['日ER10']):.2f}"]


def line(m, lab):
    x = X[m.values if hasattr(m, "values") else m]
    d = x.groupby("date").R.sum(); e = d.cumsum(); mdd = (e.cummax() - e).max()
    yr = " / ".join(f"{x.R[x.year == y].mean():+.3f}" for y in (2023, 2024, 2025, 2026))
    rm = X[~(m.values if hasattr(m, "values") else m)]
    lines.append(f"  {lab:44s} 留 {len(x) / len(X):4.0%}  每筆 {x.R.mean():+.3f}R  總 {x.R.sum():+6.0f}R  MDD {mdd:4.0f}  總/MDD {x.R.sum() / mdd:4.1f} | "
                 f"訓練 {x.R[x.train].mean():+.3f} 驗證 {x.R[~x.train].mean():+.3f} | 逐年 {yr} | 被排除的每筆 {rm.R.mean():+.3f}")


line(pd.Series(True, X.index), "全部影子訊號")
for k, m in F.items():
    line(m, k)
ms = list(F.values())
line(ms[0] & ms[1], "F1 + F2")
line(ms[0] & ms[2], "F1 + F3")
line(ms[1] & ms[2], "F2 + F3")
line(ms[0] & ms[1] & ms[2], "F1 + F2 + F3")
score = sum(m.astype(int) for m in ms)
for s in (3, 2):
    line(score >= s, f"三選 {s} 以上")

lines.append("\n=== 時段 × 短期波動（ATR15/ATR日 是否 ≤ 中位數）每筆 R：訓練 / 驗證 ===")
med = np.nanmedian(tr["ATR15/ATR日"])
for a0, b0, nm in [(1, 8, "01-08 亞洲"), (8, 15, "08-15 倫敦"), (15, 22, "15-22 紐約")]:
    h = X.hour.between(a0, b0 - 1)
    lo, hi = h & (X["ATR15/ATR日"] <= med), h & (X["ATR15/ATR日"] > med)
    lines.append(f"  {nm}: 低波動 {X.R[lo & X.train].mean():+.3f} / {X.R[lo & ~X.train].mean():+.3f}（{lo.sum()}）   "
                 f"高波動 {X.R[hi & X.train].mean():+.3f} / {X.R[hi & ~X.train].mean():+.3f}（{hi.sum()}）")
print("\n".join(lines))
open("results_part34.txt", "a").write("\n".join(lines) + "\n")

# 套上權益濾網（50 筆均線，影子 = 過濾後的訊號流）後的結果
S = pd.read_csv("final_m15_signals.csv", parse_dates=["進場時間", "出場時間"])


def eqf(mask, look=50):
    s = S[mask.values].reset_index(drop=True)
    ex = s.sort_values("出場時間")
    cum = ex.R.cumsum().values
    ma = pd.Series(cum).rolling(look, min_periods=look).mean().values
    et = ex.出場時間.values
    take = []
    for t in s.進場時間.values:
        k = np.searchsorted(et, t, side="right") - 1
        take.append(not (k >= look and cum[k] < ma[k]))
    return s[np.array(take)]


out = ["", "=== 再套上權益濾網（50 筆均線）後 ==="]
for lab, m in [("最終版（無因子過濾）", pd.Series(True, X.index)), ("F1", ms[0]), ("F3", ms[2]), ("F1 + F3", ms[0] & ms[2]), ("F1 + F2 + F3", ms[0] & ms[1] & ms[2])]:
    x = eqf(m)
    d = x.groupby(x.進場時間.dt.date).R.sum(); e = d.cumsum(); mdd = (e.cummax() - e).max()
    yrs = (x.進場時間.max() - x.進場時間.min()).days / 365
    tr_ = x[x.進場時間 < "2025-01-01"]; te_ = x[x.進場時間 >= "2025-01-01"]
    def rat(z):
        dz = z.groupby(z.進場時間.dt.date).R.sum(); ez = dz.cumsum(); return z.R.sum() / (ez.cummax() - ez).max()
    out.append(f"  {lab:20s} {len(x):5d} 筆 每筆 {x.R.mean():+.3f}R 總 {x.R.sum():+6.0f}R MDD {mdd:4.0f} 年化/MDD {x.R.sum() / yrs / mdd:4.1f} | 訓練比值 {rat(tr_):4.1f} 驗證比值 {rat(te_):4.1f}")
print("\n".join(out))
open("results_part34.txt", "a").write("\n".join(out) + "\n")
