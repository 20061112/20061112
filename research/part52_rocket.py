"""第五十二部分 B：Entropy / Volatility 對版本 A（IB 突破系統）有沒有增量價值 —— A/B/C/D 消融。

版本 A：見 FINAL_STRATEGY_A.md（門檻滾動前推，3/3 ~ 10/8，735 筆）。
特徵在進場那根 M1（訊號棒，進場價 = 其收盤）收盤時計算，只用到當下以前的資料：
  波動組（Volatility）：V_seas、V_roll、J60
  熵組（Entropy）     ：H_trans、MI、p_cont、ex_cont（磚塊流，最後一個已完成事件）、H_dir、PE（M1）
  路徑組              ：ER60、tER60（帶方向：最近 60 分鐘走勢與這筆單同向為 +）、tdrift（最近 60 塊淨方向 × 單方向）
事先登記的規則（和第 21 部分相同）：
  1. 每個特徵用 3~5 月（IS）分五分位，刪掉 IS 每筆最差的一組；6~10 月（OOS）只驗證。
  2. B / C 組各自「在 IS 內」挑出改善最大的一個特徵（挑選只看 IS），D = B + C 兩條一起用。
  3. D'：事先指定的主線 H_trans × V_seas 三分位 3×3 格，刪 IS 最差的一格（不挑特徵）。
  4. 每條規則和「隨機刪掉同樣多筆」比較（2,000 次），報告 OOS 勝過隨機的比例。
  5. 交互作用：pnl_R ~ H + V + H×V 用 IS 估計，看 OOS 預測前後半的差距。
另外：對 A + ER（第 21 部分的 er_day 篩選）重做一次，看熵 / 波動是否在 ER 之上還有增量；
     以及波動特徵當「風險尺度」：能不能預測這筆單的最大浮虧（MAE，ATR 單位）與停損率。
"""
import numpy as np
import pandas as pd
from tpo import load_m1, build_days
from entropy_vol import time_features
from part14_deep import ARR
from part17_balance import gen
from final_A import walk_forward, CFG
from part52_rangebar import build_events

out = open("results_part52.txt", "a")
def P(s=""): print(s); out.write(s + "\n"); out.flush()
CUT = pd.Timestamp("2026-06-01")
RNG = np.random.default_rng(23)

VOLF = ["V_seas", "V_roll", "J60"]
ENTF = ["H_trans", "MI", "p_cont", "ex_cont", "H_dir", "PE"]
PATHF = ["ER60", "tER60", "tdrift"]


def pf(p):
    return p[p > 0].sum() / max(-p[p <= 0].sum(), 1e-9)


def show(X, lab):
    a, b = X[X.day < CUT], X[X.day >= CUT]
    return (f"{lab:40s} 3~5月 {len(a):3d}筆 每筆 {a.pnl.mean():+6.2f} PF {pf(a.pnl):.2f} | 6~10月 {len(b):3d}筆 每筆 {b.pnl.mean():+6.2f} PF {pf(b.pnl):.2f}"
            f" | 全部 PF {pf(X.pnl):.2f} 總 {X.pnl.sum():+6.0f}")


def attach(A, d, E, TF):
    pos = d.index.get_indexer(A.t_in)
    assert (pos >= 0).all()
    for c in VOLF + ["ER60", "sER60", "PE"]:
        A[c] = TF[c].values[pos]
    A["tER60"] = A.side * A.sER60
    k = np.searchsorted(E.i.values, pos, side="right") - 1      # 最後一個「在進場棒或之前」完成的磚塊事件
    ok = k >= 0
    for c in ["H_trans", "MI", "p_cont", "ex_cont", "H_dir", "drift"]:
        A[c] = np.where(ok, E[c].values[np.clip(k, 0, None)], np.nan)
    A["tdrift"] = A.side * A.drift
    # 交易結果：R 與最大浮虧（ATR 單位）
    A["mae_atr"] = A.mae / A.atr
    return A


def worst_quintile(A, f):
    IS = A[A.day < CUT]
    e = np.unique(np.nanquantile(IS[f], [0, .2, .4, .6, .8, 1])); e[0], e[-1] = -np.inf, np.inf
    m = IS.groupby(pd.cut(IS[f], e), observed=True).pnl.mean(); w = m.idxmin()
    drop = (A[f] > w.left) & (A[f] <= w.right)
    return drop.fillna(False).values, w


def rand_pct(A, drop):
    """OOS：策略刪除後的每筆期望值，勝過多少比例的『隨機刪掉同樣多筆』。"""
    o = (A.day >= CUT).values; p = A.pnl.values[o]; k = int(drop[o].sum()); n = len(p)
    real = p[~drop[o]].mean()
    sims = np.array([np.delete(p, RNG.choice(n, k, replace=False)).mean() for _ in range(2000)])
    return (sims < real).mean() * 100


def is_gain(A, drop):
    IS = (A.day < CUT).values
    return pf(A.pnl.values[IS & ~drop]) - pf(A.pnl.values[IS])


def ablation(A, lab):
    P(f"\n  —— 基準：{lab} ——")
    P("  " + show(A, lab))
    P("  單一特徵（刪 IS 最差五分位）                                                                                         OOS 勝過隨機")
    res = {}
    for f in VOLF + ENTF + PATHF:
        drop, w = worst_quintile(A, f)
        res[f] = drop
        P("  " + show(A[~drop], f"刪 {f} ∈ ({w.left:+.3f},{w.right:+.3f}]") + f" | {rand_pct(A, drop):5.1f}%")
    pick_v = max(VOLF, key=lambda f: is_gain(A, res[f]))
    pick_e = max(ENTF, key=lambda f: is_gain(A, res[f]))
    P(f"\n  A/B/C/D（B、C 各自只用 IS 挑特徵：波動挑到 {pick_v}，熵挑到 {pick_e}）")
    rows = [("A 原始", np.zeros(len(A), bool)), (f"B + 波動（{pick_v}）", res[pick_v]),
            (f"C + 熵（{pick_e}）", res[pick_e]), ("D + 熵 + 波動", res[pick_v] | res[pick_e])]
    # D'：事先指定的 H_trans × V_seas 3×3
    IS = A[A.day < CUT]
    qh = np.nanquantile(IS.H_trans, [1 / 3, 2 / 3]); qv = np.nanquantile(IS.V_seas, [1 / 3, 2 / 3])
    gh = np.digitize(A.H_trans, qh); gv = np.digitize(A.V_seas, qv)
    cell = gh * 3 + gv
    m = pd.Series(A.pnl.values[(A.day < CUT).values]).groupby(cell[(A.day < CUT).values]).mean()
    worst = m.idxmin()
    rows.append((f"D' H_trans×V_seas 刪最差格（H{'低中高'[worst // 3]}×V{'低中高'[worst % 3]}）", cell == worst))
    for lab2, drop in rows:
        tail = f" | {rand_pct(A, drop):5.1f}%" if drop.any() else ""
        P("  " + show(A[~drop], lab2) + tail)
    P("  H_trans × V_seas 3×3 每筆（美元）IS | OOS：")
    for h in range(3):
        cells = []
        for v in range(3):
            a = A[(cell == h * 3 + v) & (A.day < CUT).values].pnl; b = A[(cell == h * 3 + v) & (A.day >= CUT).values].pnl
            cells.append(f"V{'低中高'[v]}: {a.mean():+6.1f}({len(a):3d})|{b.mean():+6.1f}({len(b):3d})")
        P(f"    H{'低中高'[h]}  " + "   ".join(cells))
    return res


def interaction(A):
    P("\n  交互作用迴歸（pnl_R ~ H + V + H×V，IS 估計；OOS 依預測值分兩半比較）")
    IS, OS = A[A.day < CUT], A[A.day >= CUT]
    for hf in ("H_trans", "p_cont", "ex_cont", "PE"):
        for vf in ("V_seas", "V_roll"):
            mu, sd = IS[[hf, vf]].mean(), IS[[hf, vf]].std()
            def X(Z):
                z = ((Z[[hf, vf]] - mu) / sd).values
                return np.c_[np.ones(len(z)), z, z[:, 0] * z[:, 1]]
            m = IS[[hf, vf, "R"]].notna().all(axis=1)
            w, *_ = np.linalg.lstsq(X(IS[m]), IS.R[m].values, rcond=None)
            res = IS.R[m].values - X(IS[m]) @ w
            se = np.sqrt(np.diag(np.linalg.inv(X(IS[m]).T @ X(IS[m])) * res.var()))
            po = X(OS) @ w; hi = po > np.nanmedian(po)
            P(f"  {hf:8s}×{vf:7s} 係數 H {w[1]:+.3f}(t{w[1] / se[1]:+.1f}) V {w[2]:+.3f}(t{w[2] / se[2]:+.1f}) H×V {w[3]:+.3f}(t{w[3] / se[3]:+.1f})"
                  f" | OOS 預測高半 每筆 R {OS.R[hi].mean():+.3f} PF {pf(OS.pnl[hi]):.2f}  低半 {OS.R[~hi].mean():+.3f} PF {pf(OS.pnl[~hi]):.2f}")


def risk_use(A):
    P("\n  波動特徵當風險尺度：五分位（分界取 IS）→ 最大浮虧 MAE（ATR 單位）中位數、停損距離 / ATR、R 的標準差；IS | OOS")
    for f in ("V_seas", "V_roll", "J60", "H_trans"):
        IS = A[A.day < CUT]
        e = np.unique(np.nanquantile(IS[f], [0, .2, .4, .6, .8, 1])); e[0], e[-1] = -np.inf, np.inf
        g = pd.cut(A[f], e); cells = []
        for c in g.cat.categories:
            a, b = A[(g == c) & (A.day < CUT)], A[(g == c) & (A.day >= CUT)]
            cells.append(f"Q≤{c.right:+.2f}: MAE {a.mae_atr.median():.3f}|{b.mae_atr.median():.3f} sdR {a.R.std():.2f}|{b.R.std():.2f}")
        P(f"  {f:8s} " + "  ".join(cells))
    o = A.dropna(subset=["V_seas"])
    P(f"  相關（全期）：V_seas vs MAE/ATR {np.corrcoef(o.V_seas, o.mae_atr)[0, 1]:+.3f}；V_seas vs 停損距離/ATR {np.corrcoef(o.V_seas, o.risk / o.atr)[0, 1]:+.3f}；"
      f"V_seas vs |R| {np.corrcoef(o.V_seas, o.R.abs())[0, 1]:+.3f}")


def er_day(A):
    v = []
    for r in A.itertuples():
        t, O, H, L, C, S = ARR[r.day]
        c = C[:r.i]
        path = np.abs(np.diff(c)).sum()
        v.append(abs(c[-1] - O[0]) / path if len(c) >= 30 and path > 0 else np.nan)
    return np.array(v)


if __name__ == "__main__":
    d = load_m1(); D, _ = build_days(d)
    _, E, *_ = build_events(d, D)
    TF = time_features(d)
    A, _ = walk_forward(gen(**CFG))
    A = attach(A, d, E, TF)
    P("\n" + "=" * 140 + "\n第五十二部分 B：Entropy / Volatility 對版本 A 的增量價值（A/B/C/D 消融，IS = 3~5 月、OOS = 6~10 月）")
    P(f"  缺值：" + "  ".join(f"{c} {A[c].isna().sum()}" for c in VOLF + ENTF + PATHF))
    P("  特徵與 R 的相關（IS | OOS）：" + "  ".join(
        f"{c} {A[A.day < CUT][[c, 'R']].corr().iloc[0, 1]:+.2f}|{A[A.day >= CUT][[c, 'R']].corr().iloc[0, 1]:+.2f}" for c in VOLF + ENTF + PATHF))
    ablation(A, "版本 A")
    interaction(A)
    risk_use(A)

    # A + ER：第 21 部分的 er_day 篩選（刪 IS 最高五分位，門檻取 3~5 月）
    A["er_day"] = er_day(A)
    thr = A[A.day < CUT].er_day.quantile(0.8)
    AE = A[~(A.er_day > thr)].reset_index(drop=True)
    P(f"\n  在 A + ER（er_day ≤ {thr:.3f}）之上再加熵 / 波動：")
    ablation(AE, "版本 A + ER")
    A.to_csv("part52_A_features.csv", index=False)
    out.close()
