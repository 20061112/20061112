"""第十八部分：版本 A 的 (1) 停損縮緊、(2) 浮虧加倉 / 浮盈加碼。

交易集合 = 版本 A（滾動前推門檻，2026/3~10/7），只改出場 / 加倉規則。
(1) 停損縮緊
    cap_usd   停損距離上限（美元）：IB 另一側比上限遠時，改成 進場價 ∓ 上限
    cap_atr   停損距離上限（ATR10 倍數）
    skip      停損距離超過上限的單直接不做
    mid       停損放 IB 中點
(2) 加倉（同一筆單內，第二單位在觸發價成交，停損與出場跟原單相同）
    add_loss  浮虧達 a × R 時加一單位（攤平）
    add_win   浮盈達 a × R 時加一單位（加碼），可選加碼後把整體停損移到原單成本
單位：美元/盎司（每單位 1 盎司）。資金模擬：每筆「初始」風險 0.5%（加倉部位另外承擔風險）。
"""
import numpy as np
import pandas as pd
from part14_deep import ARR, DAYS
from part17_balance import gen, score
from final_A import CFG, walk_forward

out = open("results_part18.txt", "w")
def P(s=""): print(s); out.write(s + "\n"); out.flush()

T, _ = walk_forward(gen(**CFG))


def run(T, stop_mode="opp", cap=None, skip=None, add=None, add_a=0.5, add_be=False):
    rows = []
    for r in T.itertuples():
        t, O, H, L, C, S = ARR[r.day]
        side, e = r.side, r.lvl
        st = r.ibl if side == 1 else r.ibh
        if stop_mode == "mid":
            st = (r.ibh + r.ibl) / 2
            if side * (e - st) <= 0:
                st = r.ibl if side == 1 else r.ibh
        risk0 = side * (e - st)
        if skip is not None and risk0 > skip(r):
            continue
        if cap is not None and risk0 > cap(r):
            st = e - side * cap(r)
        risk = side * (e - st)
        units = [(e, r.spread)]          # (成交價, 成本)
        added = False
        px, j = C[-1], len(C) - 1
        mae_tot = 0.0
        for k in range(r.i, len(C)):
            lo, hi = L[k], H[k]
            if add == "loss" and not added:      # 攤平觸發價在停損之前，同根 K 棒先算加倉（保守）
                trig = e - side * add_a * risk
                if (lo <= trig) if side == 1 else (hi >= trig):
                    units.append((trig, S[k])); added = True
            if (lo <= st) if side == 1 else (hi >= st):
                px = min(O[k], st) if side == 1 else max(O[k], st); j = k; break
            if add == "win" and not added:
                trig = e + side * add_a * risk
                if (hi >= trig) if side == 1 else (lo <= trig):
                    units.append((trig, S[k])); added = True
                    if add == "win" and add_be:
                        st = e
            worst = sum(side * ((lo if side == 1 else hi) - u) for u, _ in units)
            mae_tot = min(mae_tot, worst)
        pnl = sum(side * (px - u) - c for u, c in units)
        max_risk = sum(side * (u - st) for u, _ in units) if add == "loss" else risk * len(units) if add else risk
        rows.append(dict(day=r.day, t_in=r.t_in, exit_time=t[j], pnl=pnl, risk=risk, max_risk=max(max_risk, risk),
                         R=pnl / risk, units=len(units), mae=-mae_tot, side=side, h=r.h,
                         hold_h=(t[j] - r.t_in).total_seconds() / 3600))
    return pd.DataFrame(rows)


def sizing(X, r=0.5, start=10000.0):
    """每筆以初始風險 r% 決定手數（加倉部位用同樣手數），平倉結算、不複利。回傳 (總報酬%, 最大回撤%)。"""
    X = X.sort_values("exit_time")
    pnl_usd = start * r / 100 / X.risk * X.pnl
    eq = start + pnl_usd.cumsum()
    return (eq.iloc[-1] / start - 1) * 100, ((eq - eq.cummax()) / eq.cummax()).min() * 100


def line(lab, X):
    s = score(X); ret, dd = sizing(X)
    P(f"  {lab:30s} {len(X):4d}筆 每筆 {s['avg']:+6.2f}（{X.R.mean():+.2f}R） PF {s['pf']:.2f} 勝率 {s['win']:.0f}%"
      f" 總 {s['total']:+6.0f} 平倉DD {s['dd']:5.0f} | 停損距離 中位 {X.risk.median():5.1f} 最大 {X.risk.max():5.1f}"
      f" 單筆最大浮虧 {X.mae.max():6.1f} | 0.5%/筆：{ret:+5.0f}% DD {dd:5.1f}%")


P("第十八部分：版本 A 的停損縮緊與加倉（滾動前推 3~10 月）")
base = run(T)
line("基準（IB 另一側）", base)
P(f"\n  停損距離分佈：中位 {base.risk.median():.1f}，75% {base.risk.quantile(.75):.1f}，90% {base.risk.quantile(.9):.1f}，"
  f"95% {base.risk.quantile(.95):.1f}，最大 {base.risk.max():.1f}（> 50 美元的單 {(base.risk > 50).sum()} 筆）")
for lab, g in [("停損距離 ≤ 30", base.risk <= 30), ("30~50", (base.risk > 30) & (base.risk <= 50)), ("> 50", base.risk > 50)]:
    x = base[g]
    P(f"    {lab:12s} {len(x):3d}筆 每筆 {x.pnl.mean():+6.2f}（{x.R.mean():+.2f}R） PF {x.pnl[x.pnl > 0].sum() / -x.pnl[x.pnl <= 0].sum():.2f}")

P("\n(1) 停損縮緊")
for c in (30, 40, 50, 60):
    line(f"停損上限 {c} 美元", run(T, cap=lambda r, c=c: c))
for c in (0.25, 0.3, 0.4):
    line(f"停損上限 {c} ATR", run(T, cap=lambda r, c=c: c * r.atr))
for c in (40, 50, 60):
    line(f"停損 > {c} 美元就不做", run(T, skip=lambda r, c=c: c))
line("停損放 IB 中點", run(T, stop_mode="mid"))

P("\n(2) 加倉")
for a in (0.3, 0.5, 0.7):
    line(f"浮虧 {a}R 加一單位（攤平）", run(T, add="loss", add_a=a))
for a in (1.0, 2.0):
    line(f"浮盈 {a}R 加一單位", run(T, add="win", add_a=a))
    line(f"浮盈 {a}R 加一單位 + 停損移成本", run(T, add="win", add_a=a, add_be=True))
P("  說明：加倉的「0.5%/筆」以初始風險定手數，加倉部位用同樣手數，所以加倉版每筆實際承擔的風險較大。")

# 攤平為什麼不行：浮虧到 a×R 之後，最後賺錢的比例
P("\n  浮虧到某個程度之後的結局（基準版逐筆路徑）")
for a in (0.3, 0.5, 0.7, 0.9):
    reach = base[base.mae >= a * base.risk]
    P(f"    浮虧曾達 {a}R：{len(reach):3d} 筆（{len(reach) / len(base) * 100:.0f}%），最後賺錢 {(reach.pnl > 0).mean() * 100:.0f}%，"
      f"平均 {reach.R.mean():+.2f}R；從觸發點起算的期望 {(reach.R + a).mean():+.2f}R（攤平那一單位的期望）")

# ---- 補充：IB 中點停損的成本壓力、與上限 50 美元合併 ----
P("\n(3) IB 中點停損的成本壓力（每筆額外扣 c 美元，含加倉前的基準對照）")
mid = run(T, stop_mode="mid")
for c in (0, 0.5, 1.0, 2.0):
    for lab, X in [("基準 IB 另一側", base), ("IB 中點", mid)]:
        Y = X.copy(); Y["pnl"] -= c; Y["R"] = Y.pnl / Y.risk
        ret, dd = sizing(Y)
        P(f"  +{c:<3} {lab:14s} PF {Y.pnl[Y.pnl > 0].sum() / -Y.pnl[Y.pnl <= 0].sum():.2f} 每筆 {Y.R.mean():+.2f}R  0.5%/筆 {ret:+5.0f}% DD {dd:5.1f}%")
for h_ in ("IS", "OOS"):
    pass
mm = mid.groupby(mid.day.dt.to_period("M")).pnl.agg(["sum", "count"])
P("  IB 中點 逐月：" + "  ".join(f"{k.strftime('%m')}月 {v['sum']:+.0f}" for k, v in mm.iterrows()))
line("IB 中點 + 上限 50 美元", run(T, stop_mode="mid", cap=lambda r: 50))
out.close()
