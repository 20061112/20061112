"""第十七部分 B：風險控制實驗（目前最佳系統、滾動前推門檻，3~10 月）。

1. 浮盈保護：浮盈達 k×R 後，停損移到 進場 + x×R（x 可為負 = 只保護一部分）。
2. 總風險上限：每筆風險 r% 權益，同時持倉的總風險超過上限就不開新單（與「限制筆數」比較）。
6. 暫停開關：
   a. 權益回撤超過 X（美元/盎司）就暫停，直到「影子交易」（照樣記錄但不下單）的權益創新高才恢復；
   b. 最近 N 筆的 PF < 1 就暫停，影子交易的最近 N 筆 PF 回到 ≥ 1 才恢復。
"""
import numpy as np
import pandas as pd
from part14_deep import ARR, SPLIT
from final_system import raw_signals, walk_forward, simulate
from part17_balance import score

out = open("results_part17_risk.txt", "w")
def P(s=""): print(s); out.write(s + "\n"); out.flush()

X = simulate(raw_signals())
WF, _ = walk_forward(X)
WF = WF.sort_values("t_in").reset_index(drop=True)
base = score(WF)


def fmt(s):
    return (f"{s['per_day']:.1f}/天 每筆 {s['avg']:+6.2f}（{s['avgR']:+.2f}R） PF {s['pf']:.2f} 勝率 {s['win']:.0f}% 總 {s['total']:+6.0f}"
            f" DD {s['dd']:5.0f} 總/DD {s['ret_dd']:4.1f} Sharpe {s['sharpe']:.2f} 月PF>1 {s['mon_ok']:.0f}% 去前10天 {s['ex10']:+.0f}")


P("\n" + "=" * 120 + "\n第十七部分 B：風險控制（目前最佳系統，滾動前推 3~10 月）")
P(f"  基準 {fmt(base)}")

# ---------- 1. 浮盈保護 ----------
def protect(T, k, x):
    rows = []
    for r in T.itertuples():
        t, O, H, L, C, S = ARR[r.day]
        side, lvl, R = r.side, r.lvl, r.risk
        stop = r.ibl if side == 1 else r.ibh; moved = False
        px, j = C[-1], len(C) - 1
        for kk in range(r.i, len(C)):
            if (L[kk] <= stop) if side == 1 else (H[kk] >= stop):
                px = min(O[kk], stop) if side == 1 else max(O[kk], stop); j = kk; break
            fav = side * ((H[kk] if side == 1 else L[kk]) - lvl)
            if not moved and fav >= k * R:
                stop = lvl + side * x * R; moved = True
        rows.append(side * (px - lvl) - r.spread)
    Y = T.copy(); Y["pnl"] = rows; Y["R"] = Y.pnl / Y.risk
    return Y


P("\n1. 浮盈保護（浮盈 ≥ k×R 後停損移到 進場 + x×R）")
for k in (1.0, 1.5, 2.0, 3.0):
    for x in (-0.5, 0.0, 0.5):
        if x >= k:
            continue
        Y = protect(WF, k, x)
        P(f"  k={k:<3} x={x:+.1f}  {fmt(score(Y))}")

# ---------- 2. 總風險上限 ----------
def sizing(T, r, cap_risk=None, cap_n=None, start=10000.0):
    T = T.sort_values("t_in"); eq = peak = start; mdd = 0.0; open_ = []; n = 0
    closes = []
    events = []
    for x in T.itertuples():
        # 先結算已經出場的
        still = []
        for (et, rr, pnl_usd) in open_:
            if et <= x.t_in:
                eq += pnl_usd; peak = max(peak, eq); mdd = min(mdd, eq / peak - 1)
            else:
                still.append((et, rr, pnl_usd))
        open_ = still
        cur_risk = sum(rr for _, rr, _ in open_)
        if cap_risk is not None and cur_risk + r > cap_risk + 1e-9:
            continue
        if cap_n is not None and len(open_) >= cap_n:
            continue
        lot = max(0.01, round(eq * r / 100 / (x.risk * 100), 2))
        open_.append((x.exit_time, r, lot * 100 * x.pnl)); n += 1
    for _, _, pnl_usd in sorted(open_):
        eq += pnl_usd; peak = max(peak, eq); mdd = min(mdd, eq / peak - 1)
    return eq, mdd, n


P("\n2. 總風險上限（起始 10,000 美元；同時持倉的風險總和）")
for r in (0.25, 0.5):
    for lab, kw in [("無上限", {}), ("總風險 ≤ 1%", dict(cap_risk=1.0)), ("總風險 ≤ 1.5%", dict(cap_risk=1.5)),
                    ("總風險 ≤ 2%", dict(cap_risk=2.0)), ("總風險 ≤ 3%", dict(cap_risk=3.0)), ("最多 3 筆", dict(cap_n=3))]:
        eq, mdd, n = sizing(WF, r, **kw)
        P(f"  每筆 {r}%  {lab:12s} 下單 {n:3d} 筆  期末 {eq:9,.0f}（{(eq / 1e4 - 1) * 100:+5.0f}%） 最大回撤 {mdd * 100:5.1f}%  報酬/回撤 {(eq / 1e4 - 1) / -mdd:5.1f}")

# ---------- 6. 暫停開關 ----------
def kill(T, mode, th, n=30):
    T = T.sort_values("t_in").reset_index(drop=True)
    shadow_eq, shadow_peak, live_eq, live_peak = 0.0, 0.0, 0.0, 0.0
    paused = False; taken = []; closed = []          # closed: (exit_time, pnl)
    hist = []
    for x in T.itertuples():
        done = [c for c in closed if c[0] <= x.t_in]; closed = [c for c in closed if c[0] > x.t_in]
        for _, p_, live in sorted(done):
            shadow_eq += p_; shadow_peak = max(shadow_peak, shadow_eq); hist.append(p_)
            if live:
                live_eq += p_; live_peak = max(live_peak, live_eq)
        if mode == "dd":
            if not paused and live_eq - live_peak < -th:
                paused = True
            elif paused and shadow_eq >= shadow_peak - 1e-9 and len(done):
                paused = False; live_peak = live_eq
        else:
            last = np.array(hist[-n:])
            pf = last[last > 0].sum() / max(-last[last <= 0].sum(), 1e-9) if len(last) >= n else 9
            paused = pf < th
        live = not paused
        closed.append((x.exit_time, x.pnl, live))
        if live:
            taken.append(x.Index)
    return T.loc[taken]


P("\n6. 暫停開關")
for th in (200, 300, 400):
    Y = kill(WF, "dd", th)
    P(f"  回撤 > {th} 暫停、影子權益創新高恢復   {fmt(score(Y))}")
for n in (20, 30, 50):
    Y = kill(WF, "pf", 1.0, n)
    P(f"  最近 {n} 筆 PF < 1 暫停             {fmt(score(Y))}")
out.close()
