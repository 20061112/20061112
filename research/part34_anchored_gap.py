"""第三十四部分：使用者的「紅線」想法 —— 兩者從同一天起算（都設為 1），黃金跑到共同趨勢上方太遠就會被拉回、NAS 在下方會被拉上來。
資料：NAS100 M1 合成 D1 × XAUUSD M15 合成 D1，2023/10/2 ~ 2026/10/9。

定義（log 尺度）：從錨點日 a 起算 g = ln(XAU/XAU_a)、n = ln(NAS/NAS_a)；紅線 = (g + n) / 2；缺口 gap = g − n。
  黃金在紅線上方 gap/2、NAS 在紅線下方 gap/2，所以「兩邊都回到紅線」就是 gap 回到 0。
交易：gap > E → 空黃金、多 NAS（同名目）；gap < −E → 反過來；gap 回到 0 出場（不設停損，照原始想法）。E = 10% / 20% / 30%。
  每筆記錄：報酬（總名目 bp）、持有天數、持有期間最大浮虧、到資料結束時是否還沒收斂。
  隔夜成本：CFD 長期持有要付利息，壓力測試用總名目每年 3% 估。

重點檢查：錨點日換成 2023/10 ~ 2025/6 的每個月初，結果會不會完全不同（紅線的位置是錨點決定的）。
輸出：results_part34.txt、part34_anchor.png
"""
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from load import load

OUT = open("results_part34.txt", "w")
FIN = 0.03          # 隔夜利息壓力：總名目每年 3%


def P(*a):
    print(*a); print(*a, file=OUT)


n1 = pd.concat([load(f) for f in sorted(glob.glob("data/NAS100_M1_*.csv"))])
n1 = n1[~n1.index.duplicated(keep="last")].sort_index()
x15 = load("data/XAUUSD_M15_2023_2026.csv")
D = pd.concat({"n": n1.close.resample("1D").last(), "x": x15.close.resample("1D").last()}, axis=1, sort=True).dropna()


def trades(a, E):
    d = D[D.index >= a]
    gap = np.log(d.x / d.x.iloc[0]) - np.log(d.n / d.n.iloc[0])
    out, pos, i0 = [], 0, 0
    for i in range(len(d)):
        g = gap.iloc[i]
        if pos and pos * g >= 0:        # pos=−1 表示空黃金多 NAS（gap>E 時進），gap 回到 ≤0 出場
            out.append(close(d, gap, i0, i, pos, done=True)); pos = 0
        if not pos and abs(g) > E:
            pos, i0 = (-1 if g > 0 else 1), i
    if pos:
        out.append(close(d, gap, i0, len(d) - 1, pos, done=False))
    return pd.DataFrame(out), gap


def close(d, gap, i0, i, pos, done):
    seg = d.iloc[i0:i + 1]
    # pos=−1：空 XAU 多 NAS；每腳半名目
    path = (-pos * (seg.n / seg.n.iloc[0] - 1) + pos * (seg.x / seg.x.iloc[0] - 1)) / 2 * 1e4
    days = (seg.index[-1] - seg.index[0]).days
    return dict(entry=seg.index[0], exit=seg.index[-1], side="空金多NAS" if pos < 0 else "多金空NAS", gap_in=gap.iloc[i0],
                gap_worst=(gap.iloc[i0:i + 1].max() if pos < 0 else gap.iloc[i0:i + 1].min()), pnl=path.iloc[-1],
                mae=path.min(), days=days, fin=FIN * days / 365 * 1e4, done=done)


P(f"資料：D1 {len(D)} 根，{D.index[0].date()} ~ {D.index[-1].date()}")

P("\n## 1. 照圖上的錨點（2023/10/2）")
for E in [0.10, 0.20, 0.30]:
    T, gap = trades(D.index[0], E)
    P(f"  E={E:.0%}：缺口最大 {gap.max():+.0%}（{gap.idxmax().date()}）、最小 {gap.min():+.0%}")
    for _, t in T.iterrows():
        P(f"    {t.entry.date()} → {t.exit.date()} {t.side} 進場缺口 {t.gap_in:+.0%} 最差缺口 {t.gap_worst:+.0%}  "
          f"報酬 {t.pnl:+6.0f}bp（扣利息 {t.pnl - t.fin:+6.0f}）最大浮虧 {t.mae:+6.0f}bp 持有 {t.days} 天 {'已收斂' if t.done else '**到資料結束仍未收斂**'}")

P("\n## 2. 換錨點：2023/10 ~ 2025/6 每個月初起算，E = 20%")
rows = []
anchors = [D.index[D.index >= m][0] for m in pd.date_range("2023-10-01", "2025-06-01", freq="MS")]
for a in anchors:
    T, gap = trades(a, 0.20)
    end_gap = gap.iloc[-1]
    if len(T):
        rows.append(dict(anchor=a, n=len(T), pnl=T.pnl.sum(), pnl_fin=(T.pnl - T.fin).sum(), mae=T.mae.min(),
                         days=T.days.sum(), open=(~T.done).any(), end_gap=end_gap, max_gap=gap.abs().max()))
    else:
        rows.append(dict(anchor=a, n=0, pnl=0, pnl_fin=0, mae=0, days=0, open=False, end_gap=end_gap, max_gap=gap.abs().max()))
A = pd.DataFrame(rows)
for _, r in A.iterrows():
    P(f"  錨點 {r.anchor.date()}：缺口最大 {r.max_gap:4.0%} 目前 {r.end_gap:+4.0%}｜交易 {r.n} 筆 合計 {r.pnl:+6.0f}bp（扣利息 {r.pnl_fin:+6.0f}）"
      f" 最大浮虧 {r.mae:+6.0f}bp 持有 {r.days:4.0f} 天{'  仍未收斂' if r.open else ''}")
P(f"  合計為正的錨點 {(A.pnl_fin > 0).sum()}/{len(A)}；有沒收斂部位的 {A.open.sum()}/{len(A)}；"
  f"最大浮虧中位數 {A.mae.median():+.0f}bp、最差 {A.mae.min():+.0f}bp")

P("\n## 3. 錨點決定紅線：同一天（2026/2/2）用不同錨點看，黃金在紅線上方多少？")
day = D.index[D.index >= "2026-02-02"][0]
for a in anchors[::3]:
    d = D[D.index >= a]
    g = np.log(d.x / d.x.iloc[0]) - np.log(d.n / d.n.iloc[0])
    P(f"  錨點 {a.date()}：{day.date()} 缺口 {g.loc[day]:+.0%}（黃金在紅線上方 {g.loc[day] / 2:+.0%}）")

fig, ax = plt.subplots(2, 1, figsize=(11, 7))
for a in anchors[::4]:
    d = D[D.index >= a]
    g = np.log(d.x / d.x.iloc[0]) - np.log(d.n / d.n.iloc[0])
    ax[0].plot(g * 100, lw=1, label=f"anchor {a.date()}")
ax[0].axhline(0, color="k", lw=.5); ax[0].axhline(20, color="r", lw=.5, ls="--"); ax[0].axhline(-20, color="r", lw=.5, ls="--")
ax[0].legend(fontsize=7); ax[0].set_title("Gap ln(XAU/XAU_a) - ln(NAS/NAS_a), % -- depends on the anchor date")
d = D
g, n_ = np.log(d.x / d.x.iloc[0]), np.log(d.n / d.n.iloc[0])
ax[1].plot(np.exp(g), lw=1, label="xau"); ax[1].plot(np.exp(n_), lw=1, label="nas")
ax[1].plot(np.exp((g + n_) / 2), color="r", lw=2, label="red line = geometric mean")
ax[1].legend(fontsize=8); ax[1].set_title("Anchor 2023-10-02 (as in the chart)")
plt.tight_layout(); plt.savefig("part34_anchor.png", dpi=110)
