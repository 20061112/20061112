"""第六部分：M1 中心參數的細節（掃描結果見 scan6.py / analyze6.py）。"""
import pandas as pd
from load import load
from backtest import stats
from squeeze_breakout import SqCfg, run

m1 = load("data/XAUUSD_M1_full.csv")
SPLIT = pd.Timestamp("2026-08-15")
days = m1.index.normalize().nunique()
for mt in ("ema", "sma"):
    for comp in ("ma", "none"):
        for stop in ("box", "mid"):
            cfg = SqCfg(ma_type=mt, ma_lens=(5, 10, 20), comp=comp, pct=0.2, box=20, fan=True, stop=stop, exit="3R", hold=240)
            tr = run(m1, cfg)
            a, b, s = stats(tr[tr.entry_time < SPLIT]), stats(tr[tr.entry_time >= SPLIT]), stats(tr)
            print(f"{mt} 壓縮={comp:4s} 停損={stop:3s} 研究 n={a['n']} {a['expR']:+.3f}R | 驗證 n={b['n']} {b['expR']:+.3f}R | "
                  f"全期 {s['n'] / days:.1f}/日 {s['expR']:+.3f}R t={s['t']:.2f} PF={s['pf']:.2f} 中位R=${tr.R_usd.median():.2f}")
tr = run(m1, SqCfg(ma_type="ema", ma_lens=(5, 10, 20), comp="ma", pct=0.2, box=20, fan=True, stop="box", exit="3R", hold=240))
print("\nEMA 中心參數逐月")
print(tr.groupby(tr.entry_time.dt.to_period("M")).pnl_R.agg(["size", "mean", "sum"]).round(2).T.to_string())
