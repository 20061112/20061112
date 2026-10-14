"""第三十四部分：反向尾巴 × 120 天開關怎麼配合，以及疊單上限。
A. 篩選組合（L1 核心訊號，門檻都滾動前推）：
   尾巴分位 q ∈ 不用 / 0.95 / 0.9 / 0.8 / 0.7 × 開關 N ∈ 不用 / 60 / 120 / 180 / 250
   開關的「績效來源」兩種：(自) 用篩選後策略自己的單算最近 N 天 R；(核) 用沒篩選的 L1 核心的單算（樣本多、雜訊少）
B. 疊單上限（套在 L1 + 尾巴 0.9 + 120 天開關）：
   K = 同方向「同時持有」最多幾筆；M = 同方向「每天新開」最多幾筆。超過就不開（保留最早的）。
   因為疊單會放大回撤，另列「把風險調到平倉最大回撤 = 15% 時」的總報酬，公平比較。
PF 以 R 計。樣本外 = 2023/6~2025/12。
"""
import numpy as np
import pandas as pd
from part29_dir_filter import L1, walk, dedup, OOS_END, pf
from part32_side_gate import DAYS

out = open("results_part34.txt", "w")


def P(s=""):
    print(s)
    out.write(s + "\n")
    out.flush()


CORE = dedup(L1)


def gate(T, ref, N):
    """ref 的某一邊最近 N 個交易日累計 R < 0 → T 那一邊今天不開。"""
    if N is None:
        return T
    keep = np.ones(len(T), bool)
    for s in (1, -1):
        daily = ref[ref.side == s].groupby("day").R.sum().reindex(DAYS, fill_value=0)
        v = daily.rolling(N, min_periods=N).sum().shift(1).reindex(T.day).to_numpy()
        keep &= ~((T.side.to_numpy() == s) & (v < 0))
    return T[keep]


def dd_closed(T):
    eq = T.sort_values("exit_time").R.cumsum()
    return (eq - eq.cummax()).min()


def row(T):
    o, i = T[T.day < OOS_END], T[T.day >= OOS_END]
    yrs = [pf(g.R) for _, g in T.groupby(T.day.dt.year)]
    dd = dd_closed(T)
    return dict(n=len(T), pf=pf(T.R), pf_o=pf(o.R), pf_i=pf(i.R), worst=min(yrs), tot=T.R.sum(), dd=dd, rdd=T.R.sum() / -dd,
                ret15=T.R.sum() * 15 / -dd, years=" ".join(f"{x:.2f}" for x in yrs))


P("第三十四部分：尾巴 × 開關的配合，以及疊單上限（L1 核心訊號，PF 以 R 計）")
P("ret15 = 把每筆風險調到「平倉最大回撤剛好 15%」時，全期總報酬（%，不複利）——可以公平比較不同筆數 / 回撤的設定")

P("\n# A. 尾巴分位 × 開關 N（開關用策略自己的單）")
rows = []
for q in (None, 0.95, 0.9, 0.8, 0.7):
    T = CORE if q is None else walk(L1, q, 1.0, 0.0)
    for N in (None, 60, 120, 180, 250):
        for src in (("自", "核") if N else ("-",)):
            ref = T if src in ("自", "-") else CORE
            r = row(gate(T, ref, N))
            rows.append(dict(tq="不用" if q is None else f"{q:.2f}", N="不用" if N is None else N, src=src, **r))
R = pd.DataFrame(rows)
R.to_csv("part34_combo.csv", index=False, float_format="%.3f")
for _, r in R.iterrows():
    P(f"  尾巴 {r.tq:>4s} 開關 {str(r.N):>4s}{'(' + r.src + ')' if r.src != '-' else '    '} | {r.n:5d} 筆 PF {r.pf:.2f} | 樣本外 {r.pf_o:.2f} 2026 {r.pf_i:.2f} "
      f"最差年 {r.worst:.2f}（{r.years}）| 總 {r.tot:+5.0f}R 回撤 {r.dd:5.0f}R 總/回撤 {r.rdd:4.1f} | ret15 {r.ret15:+5.0f}%")

P("\n  熱圖：全期 PF（開關用自己的單）")
piv = R[R.src.isin(["自", "-"])].pivot_table(index="tq", columns="N", values="pf", aggfunc="first")
P(piv.round(2).to_string())
P("\n  熱圖：ret15（%）")
piv = R[R.src.isin(["自", "-"])].pivot_table(index="tq", columns="N", values="ret15", aggfunc="first")
P(piv.round(0).to_string())
P("\n  熱圖：最差一年 PF")
piv = R[R.src.isin(["自", "-"])].pivot_table(index="tq", columns="N", values="worst", aggfunc="first")
P(piv.round(2).to_string())


# ---------------- B. 疊單上限 ----------------
def cap(T, K=None, M=None):
    T = T.sort_values(["entry_time", "h"]).reset_index(drop=True)
    keep = np.zeros(len(T), bool)
    open_exits = {1: [], -1: []}
    count_day = {}
    for j, r in enumerate(T.itertuples()):
        s = r.side
        open_exits[s] = [e for e in open_exits[s] if e > r.entry_time]
        cd = count_day.get((r.day, s), 0)
        if (K is not None and len(open_exits[s]) >= K) or (M is not None and cd >= M):
            continue
        keep[j] = True
        open_exits[s].append(r.exit_time)
        count_day[(r.day, s)] = cd + 1
    return T[keep]


BASE = gate(walk(L1, 0.9, 1.0, 0.0), walk(L1, 0.9, 1.0, 0.0), 120)
P("\n\n# B. 疊單上限（L1 + 尾巴 0.9 + 120 天開關）")
rows = []
for K in (1, 2, 3, 4, 6, None):
    for M in (1, 2, 3, 5, None):
        Y = cap(BASE, K, M)
        r = row(Y)
        rows.append(dict(K="不限" if K is None else K, M="不限" if M is None else M, **r))
        P(f"  同時最多 {("不限" if K is None else str(K)):>4s} 筆 / 每天最多 {("不限" if M is None else str(M)):>4s} 筆 | {r['n']:5d} 筆 PF {r['pf']:.2f} | 樣本外 {r['pf_o']:.2f} 2026 {r['pf_i']:.2f} 最差年 {r['worst']:.2f} "
          f"| 總 {r['tot']:+5.0f}R 回撤 {r['dd']:5.0f}R 總/回撤 {r['rdd']:4.1f} | ret15 {r['ret15']:+5.0f}%（每筆風險 {15 / -r['dd']:.2f}%）")
B = pd.DataFrame(rows)
B.to_csv("part34_stack.csv", index=False, float_format="%.3f")
P("\n  熱圖：ret15（%）— 列 = 同時最多 K 筆，欄 = 每天最多 M 筆")
P(B.pivot_table(index="K", columns="M", values="ret15", aggfunc="first").round(0).to_string())
P("\n  熱圖：總損益 / 回撤")
P(B.pivot_table(index="K", columns="M", values="rdd", aggfunc="first").round(1).to_string())
out.close()
