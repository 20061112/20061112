"""第三十四部分：每筆固定 0.01 手（= 1 盎司，損益 = 每盎司美元）+ 最大同時持倉限制。
M1 OW 2026（part31_m1_trades.csv）與 M15 OW 2023~2026（part27_ow_M15.csv）。
上限規則：同時持倉已達上限就不開新單（先到先做）；也測「同方向」上限。
M1 另算逐分鐘浮動權益（K 棒最差價）。保證金以 1:100 槓桿估：0.01 手 ≈ 金價 × 1 盎司 / 100。"""
import numpy as np, pandas as pd
from part27_ow import data

out = open("results_part34.txt", "w")
def P(s=""): print(s); out.write(s + "\n"); out.flush()


def cap(T, n, same_dir=False):
    keep, open_ = [], []
    for r in T.sort_values("t_in").itertuples():
        open_ = [o for o in open_ if o[0] > r.t_in]
        cnt = sum(1 for o in open_ if (o[1] == r.side or not same_dir))
        if cnt < n:
            keep.append(r.Index); open_.append((r.exit_time, r.side))
    return T.loc[keep]


def mtm(T, ARR):
    rows = []
    for n, r in enumerate(T.itertuples()):
        t, O, H, L, C, S, mins = ARR[r.day]
        j = int(np.searchsorted(t, r.exit_time - pd.Timedelta(minutes=1)))
        if j > r.i:
            rows.append(pd.Series(r.side * ((L if r.side == 1 else H)[r.i:j] - r.lvl), index=t[r.i:j], name=n))
    U = pd.concat(rows, axis=1).sort_index()
    op = U.sum(axis=1, min_count=1).fillna(0); nopen = U.notna().sum(axis=1)
    real = T.groupby("exit_time").pnl.sum().sort_index()
    ix = op.index.union(real.index)
    eq = real.reindex(ix, fill_value=0).cumsum() + op.reindex(ix, fill_value=0)
    return eq, op, nopen


def line(T, lab, ARR=None):
    T = T.sort_values("exit_time")
    eq = T.pnl.cumsum(); dd = (eq - eq.cummax()).min()
    dl = T.groupby("day").pnl.sum()
    ev = sorted([(a, 1) for a in T.t_in] + [(b, -1) for b in T.exit_time], key=lambda z: (z[0], z[1])); cur = mx = 0
    for _, e in ev:
        cur += e; mx = max(mx, cur)
    s = (f"  {lab:24s} {len(T):4d}筆 總 {T.pnl.sum():+7.0f} 美元  平倉回撤 {dd:6.0f}  最差單日 {dl.min():+5.0f}  最好單日 {dl.max():+6.0f}"
         f"  最多同時 {mx:2d} 張（{mx * 0.01:.2f} 手）  單筆最大虧 {T.pnl.min():+5.0f}")
    if ARR is not None:
        e, op, nopen = mtm(T, ARR)
        s += f"  含浮動回撤 {(e - e.cummax()).min():6.0f}  同時浮虧最大 {op.min():+5.0f}"
    P(s)


for lab, csv, src, per in [("M1 OW 2026", "part31_m1_trades.csv", ("data/XAUUSD_M1_2026.csv", 1), None),
                           ("M15 OW 2023/3~2026/10", "part27_ow_M15.csv", ("data/XAUUSD_M15_2023_2026.csv", 15), True)]:
    T = pd.read_csv(csv, parse_dates=["day", "t_in", "exit_time"]).sort_values("t_in").reset_index(drop=True)
    T = T[T.day >= "2023-03-01"]
    days, D, ARR = data(*src)
    P("\n" + "=" * 150 + f"\n[{lab}]  每筆 0.01 手 = 1 盎司，損益單位：美元")
    P(f"  每筆風險（停損距離 × 1 盎司）：中位 {T.risk.median():.0f} 美元，90% {T.risk.quantile(.9):.0f}，最大 {T.risk.max():.0f}；"
      f"每筆平均 {T.pnl.mean():+.2f} 美元，勝率 {(T.pnl > 0).mean() * 100:.0f}%")
    A = ARR if src[1] == 1 else None
    line(T, "不限", A)
    for n in (3, 5, 8):
        line(cap(T, n), f"最多同時 {n} 張", A)
    for n in (3, 5):
        line(cap(T, n, same_dir=True), f"同方向最多 {n} 張", A)
    if per:
        P("  逐期（不限）：" + "  ".join(f"{y}: {g.pnl.sum():+.0f} 美元/{len(g)}筆（回撤 {(g.pnl.cumsum() - g.pnl.cumsum().cummax()).min():.0f}）"
                                     for y, g in T.groupby(T.day.dt.year)))
        P("  逐期（同時最多 5 張）：" + "  ".join(f"{y}: {g.pnl.sum():+.0f}" for y, g in cap(T, 5).groupby(cap(T, 5).day.dt.year)))
    price = D.close.reindex(T.day).median()
    P(f"  保證金估計（1:100）：0.01 手 ≈ {price / 100:.0f} 美元；同時 13 張 ≈ {13 * price / 100:.0f} 美元")
out.close()
