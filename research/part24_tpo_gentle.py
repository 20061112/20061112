"""第二十四部分 B：最原本的 IB 突破 + 經典 TPO 情境，溫和篩選（只刪一個最差類別）。

基準 BO：part24_bo_base.csv（M15，IB 60 分每 30 分 02:00~10:30、M15 收盤確認 +0.05ATR、IB 另一側停損、收盤出場、不篩、含週一）。
前一天 TPO：30 分一個字母、1 美元一格、70% 價值區（tpo.poc_va）。今天的發展中剖面：到進場那根為止。
情境（全部是類別，順交易方向調整）：
  open_va   今天開盤 vs 前一天價值區：+1 在交易方向的價值區外、0 價值區內、−1 在反方向外
  lvl_va    進場價 vs 前一天價值區（同上）
  ib_va     IB vs 前一天價值區：out_with（整個 IB 在交易方向的價值區外）/ overlap / inside / out_against
  poc_dir   進場價在前一天 POC 的哪一邊：toward（往 POC 走，還沒到）/ away（已經在 POC 另一邊、往外走）
  dpoc      今天發展中 POC 相對 IB 中點：with（價值往交易方向移）/ against
  dva_ovl   今天發展中價值區與前一天價值區重疊：high（> 50%，平衡）/ low（不平衡）
挑選：只用 2023~2024，類別內至少 150 筆，2023 和 2024 都比全體差 ≥ 0.05R 的類別才刪；2025H1、25H2、2026 驗證。
"""
import numpy as np
import pandas as pd
from oos_engine import load_bars
from tpo import poc_va

out = open("results_part24.txt", "a")
def P(s=""): print(s); out.write(s + "\n"); out.flush()

days, D, ARR = load_bars("data/XAUUSD_M15_2023_2026.csv", 15)
BO = pd.read_csv("part24_bo_base.csv", parse_dates=["day", "t_in", "exit_time"])


def profile(H, L, mins, upto=None):
    if upto is not None:
        H, L, mins = H[:upto], L[:upto], mins[:upto]
    per = mins // 30
    lo0 = np.floor(L.min()); n = int(np.floor(H.max()) - lo0) + 1
    diff = np.zeros(n + 1)
    for k in np.unique(per):
        m = per == k
        diff[int(np.floor(L[m].min()) - lo0)] += 1; diff[int(np.floor(H[m].max()) - lo0) + 1] -= 1
    cnt = np.cumsum(diff)[:n].astype(int)
    pr = lo0 + np.arange(n)
    poc, vah, val = poc_va(pr, cnt)
    return poc + 0.5, vah + 1, val


PREV = {}
for di, d in enumerate(days[1:], 1):
    t, O, H, L, C, S, mins = ARR[days[di - 1]]
    PREV[d] = profile(H, L, mins)

rows = []
for r in BO.itertuples():
    if r.day not in PREV:
        rows.append({}); continue
    poc, vah, val = PREV[r.day]
    t, O, H, L, C, S, mins = ARR[r.day]
    s = r.side
    def loc(x):
        if x > vah: return s
        if x < val: return -s
        return 0
    o = loc(O[0]); lv = loc(r.lvl)
    ib_hi, ib_lo = r.ibh, r.ibl
    if (s == 1 and ib_lo > vah) or (s == -1 and ib_hi < val): ibva = "out_with"
    elif (s == 1 and ib_hi < val) or (s == -1 and ib_lo > vah): ibva = "out_against"
    elif ib_lo >= val and ib_hi <= vah: ibva = "inside"
    else: ibva = "overlap"
    pocd = "toward" if s * (poc - r.lvl) > 0 else "away"
    k = r.i                                   # 進場後第一根；剖面用到確認那根（含）
    dpoc, dvah, dval = profile(H, L, mins, k)
    dp = "with" if s * (dpoc - (ib_hi + ib_lo) / 2) > 0 else "against"
    ovl = max(0, min(dvah, vah) - max(dval, val)) / max(dvah - dval, 1e-9)
    rows.append(dict(open_va=o, lvl_va=lv, ib_va=ibva, poc_dir=pocd, dpoc=dp, dva_ovl="high" if ovl > 0.5 else "low"))
F = pd.DataFrame(rows, index=BO.index)
X = pd.concat([BO, F], axis=1).dropna(subset=["ib_va"])
X["win"] = (X.pnl > 0).astype(int)
PER = [("2023", "2023-03-01", "2023-12-31"), ("2024", "2024-01-01", "2024-12-31"), ("25H1", "2025-01-01", "2025-06-30"),
       ("25H2", "2025-07-01", "2025-12-31"), ("2026", "2026-01-01", "2026-12-31")]
pf = lambda p: p[p > 0].sum() / max(-p[p <= 0].sum(), 1e-9)

P("\n" + "=" * 120 + "\n第二十四部分 B：原本的 IB 突破 + 前一天 TPO 情境（每格 = 每筆 R / 勝率（筆數））")
drops = []
for f in ["open_va", "lvl_va", "ib_va", "poc_dir", "dpoc", "dva_ovl"]:
    P(f"\n  {f}")
    for c in sorted(X[f].astype(str).unique()):
        cells = []
        for n, a, b in PER:
            x = X[(X[f].astype(str) == c) & (X.day >= a) & (X.day <= b)]
            cells.append(f"{n} {x.R.mean():+.2f}/{x.win.mean() * 100:3.0f}%({len(x)})" if len(x) else f"{n}   -  ")
        P(f"    {c:12s} " + "  ".join(cells))
        d23 = X[(X[f].astype(str) == c) & (X.day.dt.year == 2023)]; d24 = X[(X[f].astype(str) == c) & (X.day.dt.year == 2024)]
        a23 = X[X.day.dt.year == 2023].R.mean(); a24 = X[X.day.dt.year == 2024].R.mean()
        if len(d23) + len(d24) >= 150 and d23.R.mean() <= a23 - 0.05 and d24.R.mean() <= a24 - 0.05:
            drops.append((f, c))
P(f"\n  設計期（2023~2024）挑出的刪除類別：{drops if drops else '無'}")
keep = pd.Series(True, index=X.index)
for f, c in drops:
    keep &= X[f].astype(str) != c
for lab, m in [("原本 IB 突破（不篩）", X.R == X.R), ("溫和篩選（刪上述類別）", keep)]:
    P(f"  {lab:24s} " + "  ".join(f"{n} {x.R.mean():+.2f}R/PF{pf(x.pnl):.2f}/勝{x.win.mean() * 100:.0f}%({len(x)})"
                                   for n, a, b in PER for x in [X[m & (X.day >= a) & (X.day <= b)]]))
X.to_csv("part24_bo_tpo.csv", index=False)
out.close()
