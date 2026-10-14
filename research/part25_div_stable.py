"""背離順勢單（H1/M15，D+E 盤面過濾）逐年穩定度與相對無背離 base 的差異。輸出 results_part25_div_stable.txt"""
import numpy as np
from part25_regime_drift import cont_trades
from part22_divergence import get_frames, nonoverlap, SPLIT
from part24_common import tstat
F = get_frames()
for tf in ("H1", "M15"):
    A, U = cont_trades(F[tf])
    for title, T in (("C1S2", A), ("統一出場", U)):
        print(f"\n## {tf} {title}，D+E 盤面過濾（不重疊）")
        print(f"  {'條件':6s} {'研究':>14s} {'驗證':>14s}   2023   2024   2025   2026  正年數  多/空(全期)       對 base 差(Welch t)")
        base = None
        for cnd in ["base", "push", "speed", "cci", "rsi", "macd", "stoch", "obv"]:
            x = T[(T.cond == cnd) & (T.side == T.reg) & (T.side == T.ema)]
            x = nonoverlap(x)
            a, b = x[x.entry_time < SPLIT].pnl_R, x[x.entry_time >= SPLIT].pnl_R
            yr = x.groupby(x.entry_time.dt.year).pnl_R.mean().reindex([2023, 2024, 2025, 2026])
            if cnd == "base":
                base = x.pnl_R
                diff = ""
            else:
                d = x.pnl_R.mean() - base.mean()
                se = np.sqrt(x.pnl_R.var() / len(x) + base.var() / len(base))
                diff = f"{d:+.3f} (t{d / se:+.1f})"
            print(f"  {cnd:6s} {a.mean():+.3f}({len(a):3d},t{tstat(a):+.1f}) {b.mean():+.3f}({len(b):3d},t{tstat(b):+.1f}) "
                  + " ".join(f"{v:+.2f}" for v in yr) + f"   {(yr > 0).sum()}/4   "
                  f"{x[x.side == 1].pnl_R.mean():+.2f}/{x[x.side == -1].pnl_R.mean():+.2f}({(x.side == -1).sum()})   {diff}")
