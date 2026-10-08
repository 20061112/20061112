"""凍結參數盲測：strategy.py 的參數在看到 2/2~6/28 資料之前就已決定（commit 於前一輪），這裡一個都不改。"""
import numpy as np
import pandas as pd
from load import load
from backtest import backtest_bars, backtest_m1, stats
from strategy import CFG, filtered_setups

m15 = load("data/XAUUSD_M15_full.csv")
m1 = load("data/XAUUSD_M1_full.csv")
BLIND_END = pd.Timestamp("2026-06-29")

st = filtered_setups(m15)
tr = backtest_bars(m15, CFG, st)


def line(s):
    return (f"n={s['n']:4d} 勝率={s['win']:.2f} 期望={s['expR']:+.3f}R t={s['t']:+.2f} "
            f"PF={s['pf']:.2f} 累計={s['totR']:+.1f}R 最大回撤={s['maxddR']:.1f}R")


if __name__ == "__main__":
    # 先確認 M15 K 棒保守模擬 vs M1 逐根模擬 在重疊期間差多少
    ov = tr[tr.entry_time >= pd.Timestamp("2026-06-29 06:00")]
    from strategy import run
    print("重疊期間 6/29~10/8：M15 保守模擬", line(stats(ov)))
    print("重疊期間 6/29~10/8：M1 逐根模擬  ", line(stats(run(m1))))
    b = tr[tr.entry_time < BLIND_END]
    print("\n== 盲測 2/2~6/28（參數凍結）==\n ", line(stats(b)))
    print(b.groupby(b.entry_time.dt.to_period("M")).pnl_R.agg(n="size", 平均R="mean", 合計R="sum").round(2).to_string())
    r = b.pnl_R.to_numpy()
    bs = np.array([np.random.default_rng(i).choice(r, len(r)).mean() for i in range(3000)])
    print(f"  bootstrap 95% 區間 [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]  出場 {b.exit.value_counts().to_dict()}")
    print("\n== 全期間 2/2~10/8 ==\n ", line(stats(tr)))
