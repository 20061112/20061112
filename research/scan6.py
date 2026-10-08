"""第六部分參數掃描（三個週期平行跑）。研究期 / 驗證期：M15 以 6/1 切，M1/M5 以 8/15 切。"""
import itertools, sys
from multiprocessing import Pool
import pandas as pd
from load import load
from backtest import resample, stats
from squeeze_breakout import SqCfg, prep, run

SPLIT = {"M15": pd.Timestamp("2026-06-01"), "M5": pd.Timestamp("2026-08-15"), "M1": pd.Timestamp("2026-08-15")}


def data(tf):
    if tf == "M15":
        return load("data/XAUUSD_M15_full.csv")
    m1 = load("data/XAUUSD_M1_full.csv")
    return m1 if tf == "M1" else resample(m1, "5min")


def scan(tf):
    d = data(tf)
    hold = {"M15": 32, "M5": 72, "M1": 240}[tf]
    rows = []
    for mt, lens, comp, pct, box, fan in itertools.product(("ema", "sma"), ((5, 10, 20), (10, 20, 50), (20, 50, 100)),
                                                            ("ma", "sigma", "none"), (0.1, 0.2, 0.3), (10, 20, 40), (True, False)):
        if comp == "none" and pct != 0.2:
            continue
        base = SqCfg(ma_type=mt, ma_lens=lens, comp=comp, pct=pct, box=box, fan=fan, hold=hold)
        P = prep(d, base)
        for stop, ex in itertools.product(("box", "mid"), ("1.5R", "2R", "3R", "trail")):
            cfg = SqCfg(**{**base.__dict__, "stop": stop, "exit": ex})
            tr = run(d, cfg, P)
            if len(tr) == 0:
                continue
            a, b = stats(tr[tr.entry_time < SPLIT[tf]]), stats(tr[tr.entry_time >= SPLIT[tf]])
            rows.append(dict(tf=tf, ma_type=mt, lens=str(lens), comp=comp, pct=pct, box=box, fan=fan, stop=stop, exit=ex,
                             dev_n=a["n"], dev_exp=a["expR"], dev_t=a["t"], test_n=b["n"], test_exp=b["expR"], test_t=b["t"],
                             test_pf=b["pf"]))
    print(tf, "done", len(rows), file=sys.stderr)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    with Pool(3) as p:
        R = pd.concat(p.map(scan, ["M15", "M5", "M1"]))
    R.to_pickle("scan6.pkl")
    print(len(R))
