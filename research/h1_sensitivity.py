"""H1 鄰近參數敏感度 + 拆解（重現：python3 h1_sensitivity.py）"""
import itertools
import pandas as pd
from load import load
from h1_strategy import BASE, run_h1
from backtest import Cfg, stats

d = load("data/XAUUSD_M15_full.csv")
F = [("2026-02-01", "2026-04-01"), ("2026-04-01", "2026-06-01"), ("2026-06-01", "2026-08-01"), ("2026-08-01", "2026-10-09")]
folds = lambda tr: [tr[(tr.entry_time >= a) & (tr.entry_time < z)].pnl_R.mean() for a, z in F]
rows = []
for bl, bk, rn, tp in itertools.product((14, 20, 30), (1.5, 2.0, 2.5), (2, 3, 5), (1.5, 2.0, 3.0)):
    tr = run_h1(d, Cfg(**{**BASE.__dict__, "tp": tp}), bb_len=bl, bb_k=bk, roc_n=rn)
    s, fm = stats(tr), folds(tr)
    rows.append(dict(bb_len=bl, bb_k=bk, roc_n=rn, tp=tp, n=s["n"], exp=s["expR"], folds_pos=sum(x > 0 for x in fm)))
R = pd.DataFrame(rows)
print(f"81 組鄰近參數：期望 > 0 的比例 {(R.exp > 0).mean():.2f}，中位期望 {R.exp.median():+.3f}R，4 段全正的比例 {(R.folds_pos == 4).mean():.2f}")
for nm, kw in (("只要外軌收盤", dict(require_accel=False)), ("只要最後加速", dict(require_outside=False)), ("兩者 (H1)", {})):
    tr = run_h1(d, **kw); s = stats(tr)
    print(f"{nm:10s} n={s['n']:3d} 期望={s['expR']:+.3f}R t={s['t']:+.2f} 各段 " + " ".join(f"{x:+.2f}" for x in folds(tr)))
tr = run_h1(d)
tr.to_csv("h1_trades.csv", index=False)
print("多單", (tr.side == 1).sum(), f"{tr[tr.side == 1].pnl_R.mean():+.3f}R；空單", (tr.side == -1).sum(), f"{tr[tr.side == -1].pnl_R.mean():+.3f}R")
