"""M1 全部型態候選，換不同停利/持有時間，看是否有任何出場方式能讓基準轉正。"""
import itertools
import pandas as pd
from load import load
from backtest import find_setups, _arrays
from mr_features import simulate
from m1_momentum import CAND

m1 = load("data/XAUUSD_M1_full.csv")
o, h, l, c, _, spr = _arrays(m1)
setups = [x for x in find_setups(m1, CAND) if x[0] >= 300]
SPLIT = pd.Timestamp("2026-08-15")
rows = []
for tp, hold, buf in itertools.product((0.75, 1.0, 1.5, 2.0, 3.0), (15, 30, 60), (0.1, 0.5)):
    res = {True: [], False: []}
    for t, s, kind, ext, A, trend in setups:
        stop = ext - s * buf * A
        r = simulate(o, h, l, c, spr, t, s, stop, h[t] if s == 1 else l[t], lambda ep, R: ep + s * tp * R, 3, hold)
        if r:
            res[m1.index[r[0]] < SPLIT].append(r[4])
    rows.append(dict(停利R=tp, 持有分=hold, 停損緩衝ATR=buf, 研究期=round(pd.Series(res[True]).mean(), 3),
                     驗證期=round(pd.Series(res[False]).mean(), 3)))
print(pd.DataFrame(rows).sort_values("研究期", ascending=False).to_string(index=False))
