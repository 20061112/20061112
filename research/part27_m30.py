"""第二十七部分：M30（由 M15 合成）重跑第二十五、二十六部分的背離順勢單。
參數一律沿用（以 K 棒數計：EMA200、最長 32 根、擺動點 K=3、第二推最多 60 根），不針對 M30 調整。
輸出：python3 part27_m30.py > results_part27_m30.txt
"""
import numpy as np
import pandas as pd
from load import load
from backtest import resample
from part22_divergence import nonoverlap, SPLIT
from part24_common import tstat
from part25_regime_drift import cont_trades
import part26_corr

M30 = resample(load("data/XAUUSD_M15_2023_2026.csv"), "30min")


def stable_table(T, title):
    print(f"\n## M30 {title}（不重疊）")
    print(f"  {'條件':6s} {'過濾':5s} {'研究':>17s} {'驗證':>17s}   2023   2024   2025   2026  正年數  多/空(空筆數)")
    for cnd in ["base", "push", "speed", "cci", "rsi", "macd", "stoch", "obv"]:
        x0 = T[T.cond == cnd]
        for fn, m in (("無", np.ones(len(x0), bool)), ("D+E", (x0.side == x0.reg) & (x0.side == x0.ema))):
            x = nonoverlap(x0[m])
            a, b = x[x.entry_time < SPLIT].pnl_R, x[x.entry_time >= SPLIT].pnl_R
            yr = x.groupby(x.entry_time.dt.year).pnl_R.mean().reindex([2023, 2024, 2025, 2026])
            print(f"  {cnd:6s} {fn:5s} {a.mean():+.3f}({len(a):4d},t{tstat(a):+.1f}) {b.mean():+.3f}({len(b):4d},t{tstat(b):+.1f}) "
                  + " ".join(f"{v:+.2f}" for v in yr) + f"   {(yr > 0).sum()}/4   "
                  f"{x[x.side == 1].pnl_R.mean():+.2f}/{x[x.side == -1].pnl_R.mean():+.2f}({(x.side == -1).sum()})")


if __name__ == "__main__":
    A, U = cont_trades(M30)
    stable_table(A, "C1S2（突破訊號棒進場、停損回檔低點、2R）")
    stable_table(U, "統一出場（1.5 ATR / 3 ATR）")
    print(f"\n{'=' * 100}\n## M30 獲利 vs 超額、六個版本相關性（D+E、C1S2）")
    part26_corr.main(M30)
