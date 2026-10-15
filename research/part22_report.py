"""第二十二部分 B 的彙整：讀 part22_fade.py 的輸出，印出 results_part22_B.txt。"""
import numpy as np
import pandas as pd
from load import load
from hawkes import trading_clock

SIG = "part22_signals.pkl"


def main():
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 300)
    D = pd.read_csv("part22_fade_grid.csv")
    print("=" * 100)
    print("B. ρ 高時反向做短爆發 — 報酬單位 = 前一天 M1 平均高低差；已扣點差；樣本內 2023/6~2024/12、樣本外 2025/1~2026/10")
    print("=" * 100)

    print("\n1. 反彈力（不分 ρ，全部爆發反向做）：gross = 未扣點差、net = 已扣；t 以交易日為群集")
    U = D[(D.ind == "rev hl=480") & (D.z0 == 2.0)]
    U = U.pivot_table(index=["kind", "W", "H"], columns="per", values=["n", "gross", "all", "t_all"])
    U.columns = [f"{a}_{b}" for a, b in U.columns]
    print(U[["n_IS", "gross_IS", "all_IS", "t_all_IS", "n_OOS", "gross_OOS", "all_OOS", "t_all_OOS"]].round(3).to_string())

    print("\n2. ρ 分組的差異（ρ 高 1/3 − ρ 低 1/3 的反向報酬）在整個參數網格上的分布（兩組各 ≥ 30 筆）")
    G = D[(D.n_hi >= 30) & (D.n_lo >= 30)]
    P = G.pivot_table(index=["kind", "W", "z0", "H", "ind"], columns="per",
                      values=["diff", "t_diff", "hi", "lo", "all", "n_hi"]).dropna()
    P.columns = [f"{a}_{b}" for a, b in P.columns]
    P["est"] = np.where(P.index.get_level_values("ind").str.startswith("hawkes"), "Hawkes ρ", "反向比例")
    for est, g in P.groupby("est"):
        print(f"  {est}：{len(g)} 組 | 差異平均 樣本內 {g.diff_IS.mean():+.3f} / 樣本外 {g.diff_OOS.mean():+.3f} | "
              f"差異 > 0 比例 {(g.diff_IS > 0).mean():.2f} / {(g.diff_OOS > 0).mean():.2f} | "
              f"t > 2 比例 {(g.t_diff_IS > 2).mean():.3f} / {(g.t_diff_OOS > 2).mean():.3f}（純運氣約 0.025） | "
              f"樣本內外差異的相關 {g.diff_IS.corr(g.diff_OOS):+.2f} | ρ 高組扣成本後 > 0：{(g.hi_IS > 0).mean():.2f} / {(g.hi_OOS > 0).mean():.2f}")
    H = P[P.est == "Hawkes ρ"]
    print("\n  依事件種類 / 估計參數 / 持有時間平均（Hawkes）：")
    for lvl in ("kind", "ind", "H"):
        print(H.groupby(level=lvl)[["hi_IS", "lo_IS", "diff_IS", "hi_OOS", "lo_OOS", "diff_OOS"]].mean().round(3).to_string())
        print()

    print("3. 用樣本內挑「ρ 高 − ρ 低」t 值最大的 10 組 Hawkes 參數，看樣本外")
    top = H.sort_values("t_diff_IS", ascending=False).head(10)
    print(top[["n_hi_IS", "hi_IS", "lo_IS", "t_diff_IS", "n_hi_OOS", "hi_OOS", "lo_OOS", "t_diff_OOS"]].round(3).to_string())
    print(f"  → 這 10 組樣本外差異平均 {top.diff_OOS.mean():+.3f}，ρ 高組平均 {top.hi_OOS.mean():+.3f}")

    print("\n4. 事先指定的代表組（step_2、W=30、z0=2、H=30、Hawkes β=0.1 半衰期 480 分）逐年")
    sig = pd.read_pickle(SIG)
    m = load("data/XAUUSD_M1_2023_2026.csv")
    day_id, minute, t0, days = trading_clock(m)
    for key, ind in (((("step_2", 30, 2.0, 30)), "hawkes b=0.1 hl=480"),
                     ((("step_1", 30, 2.0, 30)), "hawkes b=0.05 hl=1440")):
        idx, r, gross, inds = sig[key]
        v, lo, hi = inds[ind]
        yr = days[day_id[idx]].year
        T = pd.DataFrame({"year": yr, "r": r, "g": np.where(v >= hi, "ρ高", np.where(v <= lo, "ρ低", "中"))})
        print(f"  {key} / {ind}")
        tab = T.pivot_table(index="year", columns="g", values="r", aggfunc=["mean", "count"])
        print(tab.round(3).to_string())
        print()


if __name__ == "__main__":
    main()
