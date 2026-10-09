"""第十二部分：依序優化（只用 1~5 月選參數，6~10 月驗證）。結果 → results_part12.txt"""
import itertools, sys
import numpy as np, pandas as pd
from part12_strategy import backtest, stats, split_stats, SPLIT

OUT = open("results_part12.txt", "w")
pd.set_option("display.width", 220)


def P(*a):
    s = " ".join(str(x) for x in a); print(s); OUT.write(s + "\n")


COLS = ["n", "per_day", "win", "avg", "avg_w", "avg_l", "pf", "total", "maxdd", "ret_dd"]


def row(T, **cfg):
    a, b = split_stats(T, "")
    r = dict(cfg)
    for k in COLS:
        r["is_" + k] = a.get(k); r["oos_" + k] = b.get(k)
    return r


def grid(name, space, base):
    rows = []
    keys = list(space)
    for vals in itertools.product(*space.values()):
        cfg = dict(base); cfg.update(dict(zip(keys, vals)))
        rows.append(row(backtest(**cfg), **{k: str(v) for k, v in zip(keys, vals)}))
    R = pd.DataFrame(rows)
    show = keys + ["is_n", "is_pf", "is_avg", "is_maxdd", "oos_n", "oos_pf", "oos_avg", "oos_maxdd"]
    P(f"\n=== {name} ===  （is = 1~5 月樣本內，oos = 6~10 月樣本外；金額 = 美元/盎司 = 0.01 手美元）")
    P(R[show].sort_values("is_pf", ascending=False).round(2).to_string(index=False))
    return R


P("單位：美元/盎司（下 0.01 手 = 1 盎司時就是美元）。成本：每筆一個點差。")


# ---------- 步驟 1：單一部位 ----------
H_ALL, H_2_10 = tuple(range(1, 11)), tuple(range(2, 11))
R1 = grid("步驟 1：單一淨部位（反手與否、時段、IB 長度）",
          dict(reverse=[True, False], hours=[H_ALL, H_2_10], ib_min=[30, 60]), dict(win=3))
OUT.flush()

# 步驟 1 依樣本內 PF 取前兩名當基底
BASES = [dict(win=3, reverse=False, hours=H_2_10, ib_min=60), dict(win=3, reverse=True, hours=H_2_10, ib_min=30)]
for b in BASES:
    tag = f"reverse={b['reverse']} ib={b['ib_min']}"
    R2 = grid(f"步驟 2：停損 / 保本 / 時間窗（基底 {tag}）",
              dict(stop=["opp", "mid", 0.75], be=[None, 0.5, 1.0, 2.0], win=[2, 3, 5]), b)
    best = R2.sort_values("is_pf", ascending=False).iloc[0]
    b2 = dict(b); b2.update(stop=best["stop"] if best["stop"] in ("opp", "mid") else float(best["stop"]),
                           be=None if best["be"] == "None" else float(best["be"]), win=int(best["win"]))
    P(f"  → 樣本內最佳：stop={b2['stop']} be={b2['be']} win={b2['win']}")
    R3 = grid(f"步驟 3：方向過濾（基底 {tag} + 步驟 2 最佳）", dict(flt=[None, "prevday", "trend24"]), b2)
OUT.close()
