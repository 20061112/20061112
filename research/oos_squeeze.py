"""第十部分補充：第六部分「均線密集突破（squeeze_breakout）」的凍結參數樣本外測試。
資料：data/XAUUSD_M1_2023_2026.csv（2023/6/1 起）。研究只用過 2026/6/29 以後的 M1，
因此 2026/6/29 以前全部是樣本外。參數沿用 run_part6.py，一個都沒改。
"""
import numpy as np
import pandas as pd
from load import load
from squeeze_breakout import SqCfg, run
from oos_frozen import line

OOS_END = pd.Timestamp("2026-06-29")
m1 = load("data/XAUUSD_M1_2023_2026.csv")


def report(name, T):
    oos, ins = T[T.entry_time < OOS_END], T[T.entry_time >= OOS_END]
    yr = T.groupby(T.entry_time.dt.year).pnl_R.agg(["mean", "size"])
    print(f"\n### {name}")
    print(f"  樣本外 2023/6~2026/6 : {line(oos)}")
    print(f"  研究期 2026/6/29~10  : {line(ins)}")
    print("  逐年: " + "  ".join(f"{y}: {r['mean']:+.3f} ({int(r['size'])})" for y, r in yr.iterrows()))
    gross = (oos.pnl_usd + m1.spread.reindex(oos.entry_time).to_numpy() * 0.01) / oos.R_usd
    print(f"  樣本外中位 R ${oos.R_usd.median():.2f}，扣點差前期望 {gross.mean():+.3f}R")
    return oos


if __name__ == "__main__":
    res = {}
    for mt in ("ema", "sma"):
        for comp in ("ma", "none"):
            for stop in ("box", "mid"):
                cfg = SqCfg(ma_type=mt, ma_lens=(5, 10, 20), comp=comp, pct=0.2, box=20, fan=True, stop=stop, exit="3R", hold=240)
                T = run(m1, cfg)
                res[(mt, comp, stop)] = report(f"M1 {mt} 壓縮={comp} 停損={stop}（5/10/20、下20%、箱體20、發散、3R）", T)
    print("\n== 壓縮效果（樣本外，ma − none）==")
    for mt in ("ema", "sma"):
        for stop in ("box", "mid"):
            a, b = res[(mt, "ma", stop)].pnl_R.mean(), res[(mt, "none", stop)].pnl_R.mean()
            print(f"  {mt} {stop}: 均線密集 {a:+.3f}R  單純突破 {b:+.3f}R  差 {a - b:+.3f}R")
