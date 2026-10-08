"""參數掃描：只用樣本內 (IS) 挑參數，再看樣本外 (OOS) 表現。"""
import itertools, sys
import numpy as np
import pandas as pd
from load import load
from backtest import Cfg, resample, find_setups, backtest, stats

IS_END = pd.Timestamp("2026-09-01")
HOLD = {"1min": 60, "5min": 24, "15min": 16}

m1 = load("data/XAUUSD_M1_full.csv")
rows = []
for tf in ("1min", "5min", "15min"):
    d = resample(m1, tf)
    for swing, tl in itertools.product((10, 20, 40), (20, 40, 60)):
        base = Cfg(pattern="both", swing=swing, trend_len=tl, trend_atr=3.0)
        st = find_setups(d, base)
        for pat, ta, entry, tp, buf in itertools.product(("star", "engulf", "both"), (3, 5, 7, 9),
                                                         ("close", "break"), (1.0, 1.5, 2.0, 3.0), (0.1, 0.5)):
            cfg = Cfg(pattern=pat, swing=swing, trend_len=tl, trend_atr=ta, entry=entry, tp=tp, buf=buf, hold=HOLD[tf])
            tr = backtest(d, cfg, st)
            if len(tr) == 0:
                continue
            isx, oos = tr[tr.entry_time < IS_END], tr[tr.entry_time >= IS_END]
            a, b = stats(isx), stats(oos)
            rows.append(dict(tf=tf, **{k: v for k, v in cfg.__dict__.items() if k != "hold"},
                             **{f"is_{k}": v for k, v in a.items()}, **{f"oos_{k}": v for k, v in b.items()}))
    print(tf, "done", file=sys.stderr)
R = pd.DataFrame(rows)
R.to_pickle("scan.pkl")
print(len(R), "configs")
