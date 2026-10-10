"""第二十三部分 B：乾淨的設計 / 驗證切分。
設計期 2023/3~2024/12：每個狀態特徵用設計期找「最好組 / 最差組」，
  只保留 2023、2024 兩年都「最好組勝率 > 最差組勝率」且差距 ≥ 3 個百分點的特徵；
  規則 = 進場時落在任一保留特徵的「最差組」就不做（也測：落在 ≥2 個最差組才不做）。
驗證期 2025H1、2025H2、2026：完全不參與挑選。"""
import numpy as np, pandas as pd
out = open("results_part23.txt", "a")
def P(s=""): print(s); out.write(s + "\n"); out.flush()
X = pd.read_csv("part23_trades_states.csv", parse_dates=["day", "t_in", "exit_time"])
BINS = {"wd": None, "ib_h": [1, 3, 5, 7, 9, 11], "delay": [-1, 15, 45, 90, 240], "k_today": [0, 1, 3, 6, 99],
        "n_same": [-1, 0, 2, 5, 99], "n_opp": [-1, 0, 1, 3, 99], "n_stopped": [-1, 0, 1, 3, 99],
        "pos_day": [-.01, .6, .8, .95, 1.01], "ib_pos": [-.01, .3, .5, .7, 1.01], "letters": [0, 12, 20, 28, 99],
        "otf": [-1, 0, 1, 3, 99], "rf": [-9, -0.2, 0.2, 0.5, 9], "er_day": [-.01, .15, .3, .45, 1.01],
        "aligned": None, "prev_eff": [-.01, .25, .5, .75, 1.01], "prev_cl": [-.01, .3, .6, .85, 1.01],
        "open_vs_prev": None, "streak": [-99, -2, -1, 0, 1, 2, 99]}
DES = X[X.day < "2025-01-01"]
kept = {}
P("\n" + "=" * 100 + "\n第二十三部分 B：只用 2023~2024 挑選，2025~2026 驗證")
for f, b in BINS.items():
    g = (X[f] if b is None else pd.cut(X[f], b)).astype(str)
    gd = g[DES.index]
    w = DES.groupby(gd).win.mean(); n = DES.groupby(gd).size(); w = w[n >= 80]
    if len(w) < 2: continue
    best, worst = w.idxmax(), w.idxmin()
    d23 = DES[(DES.day.dt.year == 2023) & (gd == best)].win.mean() - DES[(DES.day.dt.year == 2023) & (gd == worst)].win.mean()
    d24 = DES[(DES.day.dt.year == 2024) & (gd == best)].win.mean() - DES[(DES.day.dt.year == 2024) & (gd == worst)].win.mean()
    ok = d23 >= .03 and d24 >= .03
    P(f"  {f:13s} 最差組 {worst:14s}（設計期 {w[worst] * 100:.0f}% vs 最好 {w[best] * 100:.0f}%）2023 {d23 * 100:+5.1f}  2024 {d24 * 100:+5.1f}  {'→ 保留' if ok else ''}")
    if ok: kept[f] = (g, worst)
bad = pd.DataFrame({f: (g == w).astype(int) for f, (g, w) in kept.items()})
X["n_bad"] = bad.sum(axis=1)
P(f"\n  保留的特徵（{len(kept)} 個）：" + "、".join(f"{f}={w}" for f, (g, w) in kept.items()))
pf = lambda p: p[p > 0].sum() / max(-p[p <= 0].sum(), 1e-9)
PER = [("設計 2023", "2023-03-01", "2023-12-31"), ("設計 2024", "2024-01-01", "2024-12-31"), ("驗證 25H1", "2025-01-01", "2025-06-30"),
       ("驗證 25H2", "2025-07-01", "2025-12-31"), ("驗證 2026", "2026-01-01", "2026-12-31")]
for lab, m in [("不篩", X.n_bad >= 0), ("壞狀態 = 0 才做", X.n_bad == 0), ("壞狀態 ≤ 1 才做", X.n_bad <= 1)]:
    P(f"\n  [{lab}]")
    for n, a, b in PER:
        x = X[m & (X.day >= a) & (X.day <= b)]
        P(f"    {n:9s} {len(x):5d}筆 勝率 {x.win.mean() * 100:4.1f}% 每筆 {x.R.mean():+.2f}R PF {pf(x.pnl):.2f}")
P("\n  依壞狀態數（驗證期 2025~2026 合計）")
V = X[X.day >= "2025-01-01"]
for k, x in V.groupby("n_bad"):
    P(f"    {k} 個壞狀態：{len(x):5d}筆 勝率 {x.win.mean() * 100:4.1f}% 每筆 {x.R.mean():+.2f}R PF {pf(x.pnl):.2f}")
out.close()
