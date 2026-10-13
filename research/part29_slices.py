"""第二十九部分：定案設定在各時間切片中的相對排名。

沿用第二十八部分的 A 族（急跌買 960 組）日損益矩陣。把交易日切成：8 等分（約 5 個月）、每季、每年，
在每一片裡計算 960 組的日 Sharpe，看定案設定（門檻 2.5、停損 3、持有 120、span 20、z 1380）落在第幾百分位。
對照組：家族中位數、全期前 10 名的平均排名、「高原區」（門檻 2.5~3、持有 120~240、停損 3~6）的平均排名。
另做「留一片」：只用其他 7 片挑出最佳設定，看它在留下那一片的排名（沒有用到那一片的資料 → 沒有挑選偏誤）。
輸出：results_part29.txt、part29_slices.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

exec(open("part28_pbo.py", encoding="utf-8").read().split("def cscv")[0])   # 取得 M_all、FINAL（不跑 CSCV）

out = []


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    out.append(s)


MA = M_all[[c for c in M_all.columns if c[0] == "急跌買"]]
MA = MA.loc[:, MA.columns]
dates = pd.DatetimeIndex(MA.index)
full = MA.mean() / MA.std() * np.sqrt(252)
top10 = list(full.nlargest(10).index)
plateau = [c for c in MA.columns if c[1] in (2.5, 2.75, 3.0) and c[3] in (120, 240) and c[2] in (3.0, 4.0, 6.0)]


def sharpe(X):
    sd = X.std().replace(0, np.nan)
    return X.mean() / sd * np.sqrt(252)


def slice_table(groups, label):
    rows = []
    for key, idx in groups:
        X = MA.iloc[idx]
        s = sharpe(X)
        pct = s.rank(pct=True)
        rows.append(dict(切片=key, 天數=len(idx), 定案Sharpe=s[FINAL], 定案百分位=pct[FINAL], 定案總R=X[FINAL].sum(),
                         家族中位Sharpe=s.median(), 家族為正比例=(s > 0).mean(),
                         前10名平均百分位=pct[top10].mean(), 高原區平均百分位=pct[plateau].mean()))
    T = pd.DataFrame(rows)
    p(f"\n--- {label}")
    p(T.round(2).to_string(index=False))
    p(f"  定案：平均百分位 {T.定案百分位.mean():.0%}，中位 {T.定案百分位.median():.0%}，最差 {T.定案百分位.min():.0%}；"
      f"在前 50% 的切片 {(T.定案百分位 > .5).sum()}/{len(T)}、前 25% 的切片 {(T.定案百分位 > .75).sum()}/{len(T)}；"
      f"Sharpe 為正的切片 {(T.定案Sharpe > 0).sum()}/{len(T)}")
    return T


p("=" * 110)
p(f"第二十九部分：定案設定在各時間切片的相對排名（A 族 {MA.shape[1]} 組、{len(MA)} 個交易日）")
p(f"  高原區 = 門檻 2.5~3 × 持有 120~240 × 停損 3~6（共 {len(plateau)} 組）")
p("=" * 110)
n = len(MA)
eq = np.array_split(np.arange(n), 8)
T8 = slice_table([(f"{dates[b[0]]:%Y-%m}~{dates[b[-1]]:%Y-%m}", b) for b in eq], "8 等分")
q = dates.to_period("Q")
TQ = slice_table([(str(k), np.flatnonzero(q == k)) for k in q.unique() if (q == k).sum() >= 30], "每季")
y = dates.year
TY = slice_table([(str(k), np.flatnonzero(y == k)) for k in np.unique(y)], "每年")

# ---------------------------------------------------------------- 留一片
p("\n--- 留一片（8 等分）：只用其他 7 片挑全組最佳 / 高原區最佳 → 在留下那一片的百分位")
rows = []
for i, b in enumerate(eq):
    rest = np.concatenate([eq[j] for j in range(8) if j != i])
    s_rest = sharpe(MA.iloc[rest])
    s_hold = sharpe(MA.iloc[b])
    pct = s_hold.rank(pct=True)
    best = s_rest.idxmax()
    rows.append(dict(留下=f"{dates[b[0]]:%Y-%m}~{dates[b[-1]]:%Y-%m}", 其他片挑到=f"門檻{best[1]} 停損{best[2]} 持有{best[3]} span{best[4]} z{best[5]}",
                     是否為定案=best == FINAL, 留下片百分位=pct[best], 留下片Sharpe=s_hold[best], 定案在其他片排名=int((s_rest > s_rest[FINAL]).sum()) + 1))
L = pd.DataFrame(rows)
p(L.round(2).to_string(index=False))
p(f"  其他片挑到的設定在留下片：平均百分位 {L.留下片百分位.mean():.0%}，Sharpe 為正 {(L.留下片Sharpe > 0).sum()}/8；"
  f"其他 7 片挑到定案設定的次數 {L.是否為定案.sum()}/8；定案在其他 7 片的排名中位 {L.定案在其他片排名.median():.0f}")

fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
for a_, T, t in ((ax[0], T8, "8 equal slices"), (ax[1], TQ, "quarters")):
    x = np.arange(len(T))
    a_.bar(x, T.定案百分位 * 100, color=np.where(T.定案Sharpe > 0, "tab:green", "tab:red"), label="final config percentile")
    a_.plot(x, T.高原區平均百分位 * 100, "k.-", label="plateau avg percentile")
    a_.axhline(50, color="grey", ls="--", lw=0.8)
    a_.set_xticks(x, T.切片, rotation=60, fontsize=7)
    a_.set_ylim(0, 100)
    a_.set_ylabel("percentile among 960 configs")
    a_.set_title(t)
    a_.legend(fontsize=8)
fig.tight_layout()
fig.savefig("part29_slices.png", dpi=110)
open("results_part29.txt", "w").write("\n".join(out) + "\n")
