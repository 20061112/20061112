"""第二十二部分 C：樣本外失效診斷 —— 複雜度階梯逐期、市場型態（日內效率、紐約反轉）逐期。"""
import numpy as np, pandas as pd
from oos_engine import load_bars, signals, simulate, walk_forward, stats

out = open("results_part22.txt", "a")
def P(s=""): print(s); out.write(s + "\n"); out.flush()
days, D, ARR = load_bars("data/XAUUSD_M15_2023_2026.csv", 15)
PER = [("2023", "2023-03-01", "2023-12-31"), ("2024", "2024-01-01", "2024-12-31"), ("2025H1", "2025-01-01", "2025-06-30"),
       ("2025H2", "2025-07-01", "2025-12-31"), ("2026", "2026-01-01", "2026-12-31")]
pf = lambda p: p[p > 0].sum() / max(-p[p <= 0].sum(), 1e-9)

P("\n" + "=" * 120 + "\n第二十二部分 C：複雜度階梯逐期（M15；每格 = PF / 每筆 R）")
steps = [("L0 觸價、不篩、含週一", dict(confirm="touch", buf=0.0, monday=True), (1, 1, 0)),
         ("L1 + 收盤確認 + 0.05ATR", dict(monday=True), (1, 1, 0)),
         ("L2 + 不做週一", dict(), (1, 1, 0)),
         ("L3 + 小 IB 篩選", dict(), (1, 1, 0.4)),
         ("L4 + 尾巴", dict(), (0.9, 1, 0.4)),
         ("L5 + 反向極值（= A）", dict(), (0.9, 0.9, 0.4))]
cache = {}
for lab, kw, (qt, qe, qi) in steps:
    key = tuple(sorted(kw.items()))
    if key not in cache:
        cache[key] = simulate(signals(days, D, ARR, 15, **kw), ARR, 15)
    T = walk_forward(cache[key], qt, qe, qi)
    cells = [f"{n}: {pf(x.pnl):.2f}/{x.R.mean():+.2f}" for n, a, b in PER for x in [T[(T.day >= a) & (T.day <= b)]]]
    P(f"  {lab:24s} " + "  ".join(cells))

P("\n停損改 IB 中點（版本 A 其他不變）")
A = walk_forward(cache[()])
Am = walk_forward(simulate(signals(days, D, ARR, 15), ARR, 15, stop_mode="mid"))
for lab, T in [("IB 另一側", A), ("IB 中點", Am)]:
    P(f"  {lab:10s} " + "  ".join(f"{n}: {pf(x.pnl):.2f}/{x.R.mean():+.2f}" for n, a, b in PER for x in [T[(T.day >= a) & (T.day <= b)]]))

P("\n市場型態逐期（交易日，不含週一）")
rows = []
for d in days:
    if d.dayofweek == 0 or d < pd.Timestamp("2023-03-01"):
        continue
    t, O, H, L, C, S, mins = ARR[d]
    k11 = np.searchsorted(mins, 11 * 60); k1530 = np.searchsorted(mins, 15 * 60 + 30)
    if k1530 >= len(C):
        continue
    rng = H.max() - L.min(); atr = D.atr10[d]
    am, ny = np.sign(C[k11 - 1] - O[0]), np.sign(C[-1] - C[k1530 - 1])
    asia_rng = H[:k11].max() - L[:k11].min()
    rows.append(dict(day=d, eff=abs(C[-1] - O[0]) / rng, rng_atr=rng / atr, rev=int(am != 0 and ny == -am),
                     am_share=asia_rng / rng, ny_share=(H[k1530:].max() - L[k1530:].min()) / rng,
                     ret=C[-1] - O[0], atr=atr))
M = pd.DataFrame(rows).set_index("day")
for n, a, b in PER:
    x = M[(M.index >= a) & (M.index <= b)]
    P(f"  {n:7s} 日內效率 {x.eff.mean():.2f}  紐約反轉 {x.rev.mean() * 100:.0f}%  01~11點波幅占全日 {x.am_share.mean():.2f}"
      f"  15:30後波幅占全日 {x.ny_share.mean():.2f}  日波幅/ATR {x.rng_atr.mean():.2f}  ATR {x.atr.median():.0f}  日均漲跌 {x.ret.mean():+.1f}")
M.to_csv("oos_regime_days.csv")
out.close()
