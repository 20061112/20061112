"""第三十七部分：固定 0.01 口（= 1 盎司，金價每動 1 美元 = 1 美元損益）、允許疊單，結果會怎樣？
比較：最終系統（同方向最多 3 筆）/ 核心 + 開關（不限疊單）/ 只有核心。全部用美元計，逐分鐘含浮動。
重點：固定口數時，每筆的風險 = 停損距離（美元），會隨金價、波動變大；所以美元結果主要由 2025~2026 決定。
"""
import numpy as np
import pandas as pd
from load import load
from part29_dir_filter import L1, dedup, pf
from part34_combo_stack import gate, cap

out = open("results_part37.txt", "w")


def P(s=""):
    print(s)
    out.write(s + "\n")
    out.flush()


m1 = load("data/XAUUSD_M1_2023_2026.csv")
tm = m1.index.values
H, L = m1.high.to_numpy(), m1.low.to_numpy()


def usd_curves(T):
    open_arr = np.zeros(len(tm))
    cnt = np.zeros(len(tm) + 1)
    for r in T.itertuples():
        a = np.searchsorted(tm, np.datetime64(r.entry_time), side="right")
        b = np.searchsorted(tm, np.datetime64(r.exit_time), side="left")
        if b > a:
            open_arr[a:b] += np.maximum(r.side * ((L if r.side == 1 else H)[a:b] - r.lvl) - r.spread, -r.risk - r.spread)
            cnt[a] += 1
            cnt[b] -= 1
    n = np.cumsum(cnt)[:-1]
    act = n > 0
    op = pd.Series(open_arr[act], index=tm[act])
    real = T.groupby("exit_time").pnl.sum().sort_index()
    ix = op.index.union(real.index)
    eq = real.reindex(ix, fill_value=0).cumsum() + op.reindex(ix, fill_value=0)
    return real.cumsum(), eq, op, pd.Series(n[act], index=tm[act])


C = dedup(L1)
G = gate(C, C, 120)
F = cap(G, K=3)
P("第三十七部分：固定 0.01 口（1 盎司）每筆，美元損益（含點差）")
rows = []
for lab, T in (("最終系統（最多 3 筆）", F), ("核心 + 開關（不限疊單）", G), ("只有核心（不限疊单）".replace("单", "單"), C)):
    ec, em, op, n = usd_curves(T)
    ddc, ddm = (ec - ec.cummax()).min(), (em - em.cummax()).min()
    daily = T.groupby("day").pnl.sum()
    P(f"\n# {lab}")
    P(f"  筆數 {len(T):,}；總損益 {T.pnl.sum():+,.0f} 美元；PF（美元）{pf(T.pnl):.2f}（以 R 計 {pf(T.R):.2f}）")
    P(f"  最大回撤 平倉 {ddc:,.0f} / 含浮動 {ddm:,.0f} 美元（含浮動發生於 {(em - em.cummax()).idxmin():%Y-%m-%d}）")
    P(f"  帳戶同時最大浮虧 {op.min():,.0f} 美元（{op.idxmin():%Y-%m-%d %H:%M}，當時 {int(n[op.idxmin()])} 筆）；同時最多 {int(n.max())} 筆")
    P(f"  最差單日 {daily.min():+,.0f}、最好單日 {daily.max():+,.0f} 美元")
    P(f"  需要的帳戶大小（讓含浮動最大回撤 = 10% / 20% / 30%）：{-ddm / .10:,.0f} / {-ddm / .20:,.0f} / {-ddm / .30:,.0f} 美元")
    P(f"     對應總報酬：{T.pnl.sum() / (-ddm / .10):+.0%} / {T.pnl.sum() / (-ddm / .20):+.0%} / {T.pnl.sum() / (-ddm / .30):+.0%}")
    P("  逐年（美元）：")
    for y, g in T.groupby(T.day.dt.year):
        e2 = g.sort_values("exit_time").pnl.cumsum()
        P(f"    {y}: {len(g):5d} 筆 損益 {g.pnl.sum():+8,.0f} PF {pf(g.pnl):.2f} | 平均停損距離 {g.risk.mean():5.1f} 美元 | 年內平倉最大回撤 {(e2 - e2.cummax()).min():8,.0f} | 最差單日 {g.groupby('day').pnl.sum().min():+7,.0f}")
    rows.append(dict(lab=lab, n=len(T), tot=T.pnl.sum(), pf=pf(T.pnl), ddm=ddm, worst_open=op.min(), maxn=int(n.max())))

P("\n# 總結")
for r in rows:
    P(f"  {r['lab']:22s} {r['n']:5d} 筆 總 {r['tot']:+8,.0f} 美元 PF {r['pf']:.2f} 含浮動回撤 {r['ddm']:8,.0f} 總/回撤 {r['tot'] / -r['ddm']:.1f} 同時最多 {r['maxn']} 筆、同時浮虧 {r['worst_open']:,.0f}")
out.close()
