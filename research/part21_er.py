"""第二十一部分 B：當天 ER（Kaufman 效率比）能否融入版本 A；C：複雜度階梯（每條規則各貢獻多少）。

ER（進場當下，只用進場前的 M1 收盤）：
  er_day   今天到目前：|現價 − 開盤| / Σ|逐分鐘變動|（0 = 來回、1 = 單邊）
  der_day  帶方向：方向 ×（現價 − 開盤）/ Σ|變動|（+ = 今天到目前的走勢和這筆單同方向）
  der_60   帶方向，最近 60 分鐘
  der_180  帶方向，最近 180 分鐘
事先登記：用 3~5 月（樣本內）五分位找最差一組並刪除，看 6~10 月（樣本外）。
"""
import numpy as np, pandas as pd
from part14_deep import ARR
from part17_balance import gen, wf, score

out = open("results_part21.txt", "a")
def P(s=""): print(s); out.write(s + "\n"); out.flush()
CUT = pd.Timestamp("2026-06-01")


def er_feats(r):
    t, O, H, L, C, S = ARR[r.day]
    i = r.i                                  # 進場後第一根（close5：訊號那根已收完）
    c = C[:i]
    if len(c) < 30:
        return {}
    path = np.abs(np.diff(c)).sum()
    f = dict(er_day=abs(c[-1] - O[0]) / path, der_day=r.side * (c[-1] - O[0]) / path)
    for n in (60, 180):
        w = c[-n:]; p = np.abs(np.diff(w)).sum()
        f[f"der_{n}"] = r.side * (w[-1] - w[0]) / p if p > 0 else 0
    return f


def pf(p):
    return p[p > 0].sum() / -p[p <= 0].sum()


def show(X, lab):
    a, b = X[X.day < CUT], X[X.day >= CUT]
    return (f"{lab:34s} 3~5月 {len(a):3d}筆 每筆 {a.pnl.mean():+6.2f} PF {pf(a.pnl):.2f} | 6~10月 {len(b):3d}筆 每筆 {b.pnl.mean():+6.2f} PF {pf(b.pnl):.2f}"
            f" | 全部 PF {pf(X.pnl):.2f} 總 {X.pnl.sum():+6.0f}")


X = gen("close5", 0.05, (2, 10), 30, False)
A = wf(X, 0.9, 0.9).reset_index(drop=True)
F = pd.DataFrame([er_feats(r) for r in A.itertuples()], index=A.index)
A = pd.concat([A, F], axis=1)

P("\n" + "=" * 120 + "\n第二十一部分 B：當天 ER（版本 A 的交易，五分位分界取 3~5 月）")
for f in F.columns:
    e = np.unique(np.nanquantile(A[A.day < CUT][f], [0, .2, .4, .6, .8, 1])); e[0], e[-1] = -np.inf, np.inf
    g = pd.cut(A[f], e)
    cells = []
    for c in g.cat.categories:
        a, b = A[(g == c) & (A.day < CUT)].pnl, A[(g == c) & (A.day >= CUT)].pnl
        cells.append(f"≤{c.right:+.2f}: {a.mean():+5.1f}|{b.mean():+5.1f}")
    P(f"  {f:8s} " + "  ".join(cells))
P("\n  事先登記：刪掉 3~5 月最差的五分位")
P("  " + show(A, "版本 A"))
for f in F.columns:
    IS = A[A.day < CUT]
    e = np.unique(np.nanquantile(IS[f], [0, .2, .4, .6, .8, 1])); e[0], e[-1] = -np.inf, np.inf
    m = IS.groupby(pd.cut(IS[f], e), observed=True).pnl.mean(); w = m.idxmin()
    keep = ~((A[f] > w.left) & (A[f] <= w.right))
    P("  " + show(A[keep], f"刪 {f} ∈ ({w.left:+.2f},{w.right:+.2f}]"))

P("\n" + "=" * 120 + "\n第二十一部分 C：複雜度階梯（每一步只多加一條規則；滾動前推 3~10 月）")
steps = [
    ("L0 IB 60 分每 30 分、觸價、停損另一側、收盤出", ("touch", 0.0, True), (1.0, 1.0, 0.0)),
    ("L1 + M5 收盤 + 0.05 ATR 確認", ("close5", 0.05, True), (1.0, 1.0, 0.0)),
    ("L2 + 不做星期一", ("close5", 0.05, False), (1.0, 1.0, 0.0)),
    ("L3 + 順前日方向的小 IB 不做", ("close5", 0.05, False), (1.0, 1.0, 0.4)),
    ("L4 + 反向尾巴篩選", ("close5", 0.05, False), (0.9, 1.0, 0.4)),
    ("L5 + 反向極值字母篩選（= 版本 A）", ("close5", 0.05, False), (0.9, 0.9, 0.4)),
]
for lab, (cf, bf, mon), (qt, qe, qi) in steps:
    Y = wf(gen(cf, bf, (2, 10), 30, mon), qt, qe, qi)
    s = score(Y)
    P("  " + show(Y, lab) + f" Sharpe {s['sharpe']:.2f} 總/DD {s['ret_dd']:.1f}")
out.close()
