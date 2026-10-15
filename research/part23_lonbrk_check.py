"""第二十三部分 S5 補充：LONBRK 依 ρ 分組的差異是否顯著（t、逐年、置換檢定）。"""
import numpy as np
import pandas as pd
from load import load
from hawkes import trading_clock, day_scale, make_events, online_em, to_bars
from part22_fade import trailing_terciles, OOS

m = load("data/XAUUSD_M1_2023_2026.csv")
day_id, minute, t0, days = trading_clock(m)
sc = day_scale(m, day_id)
L = pd.read_csv("oos_lonbrk_trades.csv", parse_dates=["entry_time"])
L = L[L.entry_time >= "2023-06-01"]
pos = m.index.get_indexer(L.entry_time - pd.Timedelta(minutes=1))
L = L[pos >= 0].copy(); pos = pos[pos >= 0]
r = L.pnl_R.values; yr = L.entry_time.dt.year.values; oos = (L.entry_time >= OOS).values
rng = np.random.default_rng(0)
for kind, beta, hl in [("step_1", 0.05, 480.0), ("step_2", 0.1, 1440.0)]:
    T, S, B = make_events(m, day_id, t0, sc, kind); T = T.astype(np.float64)
    rho = online_em(T, S, day_id[B], beta, hl)[0]
    v = to_bars(np.log(rho), B, len(m))
    lo, hi = trailing_terciles(v, day_id, len(days))
    g = np.where(v[pos] >= hi[pos], "hi", np.where(v[pos] <= lo[pos], "lo", "mid"))
    print(f"\n{kind} β={beta} hl={int(hl)}")
    print(pd.DataFrame({"y": yr, "g": g, "r": r}).pivot_table(index="y", columns="g", values="r", aggfunc=["mean", "count"]).round(3).to_string())
    for per, pm in (("全期", np.ones(len(r), bool)), ("樣本內", ~oos), ("樣本外", oos)):
        a, b = r[pm & (g == "hi")], r[pm & (g == "lo")]
        d = a.mean() - b.mean(); t = d / np.sqrt(a.var() / len(a) + b.var() / len(b))
        # 置換：打亂分組標籤
        rr = r[pm & (g != "mid")]; gg = g[pm & (g != "mid")]
        perm = [(lambda p: rr[p == "hi"].mean() - rr[p == "lo"].mean())(rng.permutation(gg)) for _ in range(5000)]
        print(f"  {per}: ρ高 {a.mean():+.3f}R (n={len(a)})  ρ低 {b.mean():+.3f}R (n={len(b)})  差 {d:+.3f}  t={t:+.2f}  置換 p={np.mean(np.array(perm) >= d):.3f}")
