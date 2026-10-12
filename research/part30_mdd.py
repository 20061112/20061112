"""第三十部分補充：最大回撤（MDD）拆解。

A+B 過濾的 M15 突破，停損 2 ATR15、RR 3，持倉上限 N = 1 / 8 / 不限，每單 1 盎司。
三種算法：
  日結   ：每天把「當天進場的單」損益加總，用日權益曲線算（第二十八~三十部分表格用的就是這個）
  逐筆   ：依出場時間逐筆累加已實現損益
  逐分鐘 ：含未平倉浮動損益，每分鐘用 M1 收盤價重估（最接近帳戶實際看到的）
另列：回撤開始 / 谷底 / 回到新高的日期、持續天數、前 5 大回撤、谷底那段的成因。
"""
import numpy as np
import pandas as pd
import part29_rr_grid as G
from part30_max_positions import cap_total

IDX = G.IDX
C = G.W.C
pos_of = pd.Series(np.arange(len(IDX)), IDX)
evs = [e for e, k in zip(G.ev, G.mA & G.mB) if k]
lines = []


def p(s=""):
    print(s)
    lines.append(s)


def sim_with_px(evs, s, t):
    rows = []
    for i, L, side, a in evs:
        risk = s * a
        r = G.sim_detail(i, L, side, L - side * risk, L + side * t * risk)
        if r is None:
            continue
        j, px, why = r
        rows.append((IDX[i], IDX[j], side, L, G.W.SP[i], (px - L) * side - G.W.SP[i], why, i, j))
    T = pd.DataFrame(rows, columns=["進場時間", "出場時間", "dir", "entry", "sp", "usd", "why", "i", "j"])
    T["train"] = T.進場時間 < "2025-01-01"
    return T.sort_values("進場時間").reset_index(drop=True)


def episodes(eq, top=5):
    """回傳前 top 大的回撤段：(高點時間, 谷底時間, 回到高點時間, 幅度)。"""
    peak = eq.cummax()
    dd = peak - eq
    out = []
    used = pd.Series(False, eq.index)
    for _ in range(top):
        x = dd[~used]
        if x.empty or x.max() <= 0:
            break
        tr = x.idxmax()
        pk = eq[:tr].idxmax()
        after = eq[tr:]
        rec = after[after >= eq[pk]]
        end = rec.index[0] if len(rec) else None
        out.append((pk, tr, end, dd[tr]))
        used[pk:(end if end is not None else eq.index[-1])] = True
    return out


def minute_equity(T):
    n = len(IDX)
    D = np.zeros(n + 1)
    E = np.zeros(n + 1)
    R = np.zeros(n + 1)
    for r in T.itertuples():
        D[r.i] += r.dir; D[r.j] -= r.dir
        E[r.i] += r.dir * r.entry; E[r.j] -= r.dir * r.entry
        R[r.j] += r.usd
        R[r.i] -= 0  # 點差在出場時隨已實現損益一起計
    D, E, R = np.cumsum(D)[:n], np.cumsum(E)[:n], np.cumsum(R)[:n]
    eq = R + C * D - E
    keep = (D != 0) | (np.diff(np.r_[0, R]) != 0)
    return pd.Series(eq, IDX)[keep]


T0 = sim_with_px(evs, 2.0, 3.0)
for n in (1, 8, None):
    x = cap_total(T0, n)
    lab = f"上限 {n}" if n else "不限"
    daily = x.groupby(x.進場時間.dt.date).usd.sum()
    deq = daily.cumsum()
    tq = x.sort_values("出場時間").set_index("出場時間").usd.cumsum()
    meq = minute_equity(x)
    month = x.groupby(x.進場時間.dt.to_period("M")).usd.sum()
    p(f"=== {lab}（{len(x)} 筆，每天 {len(x) / x.進場時間.dt.date.nunique():.1f} 筆，總獲利 {x.usd.sum():+,.0f}，平均每月 {month.mean():+,.0f}）===")
    for nm, eq in (("日結", deq), ("逐筆", tq), ("逐分鐘(含浮動)", meq)):
        dd = (eq.cummax() - eq)
        p(f"  {nm:12s} MDD {dd.max():7,.0f} 美元（= {dd.max() / month.mean():.1f} 個月平均獲利）")
    p("  前 5 大回撤（日結）：")
    for pk, tr, end, amt in episodes(deq):
        dur = (pd.Timestamp(end) - pd.Timestamp(pk)).days if end is not None else None
        seg = x[(x.進場時間.dt.date > pk) & (x.進場時間.dt.date <= tr)]
        p(f"    {amt:6,.0f}：高點 {pk} → 谷底 {tr}（{(pd.Timestamp(tr) - pd.Timestamp(pk)).days} 天）→ "
          f"{'回到新高 ' + str(end) + f'（共 {dur} 天）' if end is not None else '尚未回到新高'}；"
          f"期間 {len(seg)} 筆，勝率 {(seg.usd > 0).mean():.0%}，多單 {(seg.dir > 0).mean():.0%}")
    worst = daily.nsmallest(3)
    p("  最差 3 天：" + "，".join(f"{k} {v:+,.0f}（{len(x[x.進場時間.dt.date == k])} 筆）" for k, v in worst.items()))
    p("  逐年 MDD（日結）：" + "，".join(
        f"{y} {((deq[[d.year == y for d in deq.index]].cummax() - deq[[d.year == y for d in deq.index]]).max()):,.0f}"
        for y in (2023, 2024, 2025, 2026)))
    p()
    if n == 8:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True, gridspec_kw=dict(height_ratios=[2, 1, 1]))
            mi = pd.DatetimeIndex(meq.index)
            ax[0].plot(mi, meq.values, lw=.6, label="minute (incl. open P&L)")
            ax[0].plot(pd.DatetimeIndex(deq.index), deq.values, lw=1, label="daily")
            ax[0].set_ylabel("cum USD (1 oz/trade)"); ax[0].legend(); ax[0].grid(alpha=.3)
            ax[0].set_title("A+B M15 breakout, stop 2 ATR15, RR 3, max 8 positions")
            ax[1].fill_between(mi, -(meq.cummax() - meq).values, 0, color="tab:red", alpha=.5)
            ax[1].set_ylabel("drawdown USD"); ax[1].grid(alpha=.3)
            px = G.W.B.daily.C
            ax[2].plot(px.index, px.values, color="goldenrod"); ax[2].set_ylabel("XAUUSD"); ax[2].grid(alpha=.3)
            plt.tight_layout(); plt.savefig("part30_mdd.png", dpi=110)
        except ImportError:
            pass
open("results_part30_mdd.txt", "w").write("\n".join(lines) + "\n")
