"""第二十二部分：FadeEA v1 可移植的邏輯，套到版本 A（IB 突破）上，並先做版本 A 的 2023~2025 樣本外。

資料：data/XAUUSD_M1_2023_2026.csv（2023/6/1 ~ 2026/10/8）。版本 A 的規則與分位數一個都不改，
門檻照舊每月月初只用過去的訊號重算（滾動前推）。2023/6 ~ 2025/12 版本 A 從沒看過。
單位一律用 R（每筆固定風險），因為 2023 與 2026 的波動差 2~6 倍，美元/盎司不能跨年比。

從 FadeEA 借來測的概念（都只用進場當下已知的資訊）：
  1. ER_8h：最後一根已收完的 M5，ER(96)。FADE 只在 ER_8h < 0.18 時做反轉。
  2. layer：同一天、同方向、進場前已經開出的版本 A 單數（FADE 的 layer = 35 分鐘內同方向訊號數，越深倉位越大）。
  3. 前一天方向倍數：FADE 順前一天 0.5 倍、逆前一天 1.5 倍。
  4. 滾動勝率倍數：FADE 最近 100 筆（已平倉）勝率 < 40% 時倉位 ×2。
  5. 風險上限：FADE 限制「未平倉部位 Σ(單位 × 停損距離)」。這裡用同時持有的 R 數上限。
"""
import sys
import numpy as np
import pandas as pd
import tpo

tpo.load_m1.__defaults__ = ("data/XAUUSD_M1_2023_2026.csv",)
from part17_balance import gen          # noqa: E402  （必須在換資料之後才 import）
from part14_deep import d as M1          # noqa: E402
from m1_momentum import efficiency       # noqa: E402

Q_IB, Q_TAIL, Q_EXT = 0.4, 0.9, 0.9
CFG = dict(confirm="close5", buf=0.05, hours=(2, 10), step=30, monday=False)
PERIODS = [("2023/6~2024", "2023-06-01", "2025-01-01"), ("2025", "2025-01-01", "2026-01-01"),
           ("2026/1~10", "2026-01-01", "2026-12-31")]


def walk_forward(X):
    """同 final_A.walk_forward。"""
    X = X.copy(); X["ym"] = X.day.dt.to_period("M")
    parts = []
    for m in sorted(X.ym.unique())[2:]:
        past, cur = X[X.ym < m], X[X.ym == m]
        ti, tt, te = past.ib_atr.quantile(Q_IB), past["tail"].quantile(Q_TAIL), past.ext_letter.quantile(Q_EXT)
        keep = ~(cur.aligned & (cur.ib_atr < ti)) & (cur["tail"] <= tt) & (cur.ext_letter.fillna(0) <= te)
        parts.append(cur[keep])
    Y = pd.concat(parts).sort_values(["t_in", "h"]).drop_duplicates(["t_in", "side"])
    return Y.sort_values("t_in").reset_index(drop=True)


def line(R, label):
    R = np.asarray(R, float)
    if len(R) < 2:
        return f"  {label:<22} n={len(R):5d}"
    w, l = R[R > 0].sum(), -R[R <= 0].sum()
    t = R.mean() / R.std(ddof=1) * np.sqrt(len(R))
    return f"  {label:<22} n={len(R):5d} 每筆 {R.mean():+.3f}R t={t:+5.2f} 勝率 {np.mean(R > 0):.2f} PF {w / max(l, 1e-9):.2f}"


def by_period(T, col="R", w=None):
    out = []
    for name, a, b in PERIODS:
        m = (T.day >= a) & (T.day < b)
        x = T.loc[m, col] * (1 if w is None else T.loc[m, w])
        out.append(line(x, name))
    return out


def max_dd(R):
    eq = np.cumsum(R); return (eq - np.maximum.accumulate(np.r_[0, eq][1:])).min()


def main():
    out = []
    X = gen(**CFG)
    T = walk_forward(X)
    T = T[T.R.notna() & (T.risk > 0)].reset_index(drop=True)

    out.append("## 0. 版本 A 原封不動（滾動前推門檻），每筆固定 1R")
    out += by_period(T)
    for y, g in T.groupby(T.day.dt.year):
        out.append(line(g.R, f"{y} 年"))
    out.append(f"  中位停損距離（美元）：" + "  ".join(f"{y}: {g.risk.median():.1f}" for y, g in T.groupby(T.day.dt.year)))
    out.append(f"  中位點差 / 停損：" + "  ".join(f"{y}: {(g.spread / g.risk).median():.3f}" for y, g in T.groupby(T.day.dt.year)))

    # ---- FADE 概念 1：ER_8h（最後一根已收 M5 的 ER(96)）----
    m5 = M1.close.resample("5min", label="right", closed="left").last().dropna()
    er8 = efficiency(m5, 96)
    T["er8"] = er8.reindex(T.t_in + pd.Timedelta(minutes=1), method="ffill").values
    out.append("\n## 1. ER_8h 分組（FADE：< 0.18 才做反轉）；分位數用全期，只看兩段方向是否一致")
    edges = T.er8.quantile([0, .2, .4, .6, .8, 1]).values
    T["er8_q"] = pd.cut(T.er8, edges, labels=["Q1低", "Q2", "Q3", "Q4", "Q5高"], include_lowest=True)
    out.append(f"  分界：{np.round(edges, 3)}")
    for q, g in T.groupby("er8_q", observed=True):
        out.append(f" {q}"); out += by_period(g)
    out.append(" FADE 原門檻 ER_8h < 0.18"); out += by_period(T[T.er8 < 0.18])
    out.append(" ER_8h >= 0.18"); out += by_period(T[T.er8 >= 0.18])

    # ---- 概念 2：layer（當天同方向第幾筆）----
    T["layer"] = T.groupby(["day", "side"]).cumcount() + 1
    out.append("\n## 2. layer = 當天同方向第幾筆（FADE：越深倉位越大）")
    for a, b in [(1, 1), (2, 2), (3, 4), (5, 99)]:
        out.append(f" 第 {a}~{b} 筆"); out += by_period(T[T.layer.between(a, b)])

    # ---- 概念 3：前一天方向 ----
    out.append("\n## 3. 前一天方向（A 已刪掉「順前日 + 小 IB」，這裡看剩下的）")
    out.append(" 順前一天"); out += by_period(T[T.aligned])
    out.append(" 逆前一天"); out += by_period(T[~T.aligned])

    # ---- 概念 4：滾動勝率（只用進場前已平倉的 100 筆）----
    ex = T.sort_values("exit_time")
    ex_t, ex_w = ex.exit_time.values, (ex.R > 0).values.astype(float)
    cw = np.r_[0, np.cumsum(ex_w)]
    k = np.searchsorted(ex_t, T.t_in.values, side="right")
    T["wr100"] = np.where(k >= 100, (cw[k] - cw[np.maximum(k - 100, 0)]) / 100, np.nan)
    out.append("\n## 4. 進場前最近 100 筆已平倉的勝率（FADE：< 0.40 → 倉位 ×2）")
    for a, b, lab in [(0, .40, "< 0.40"), (.40, .47, "0.40~0.47"), (.47, .53, "0.47~0.53"), (.53, 1.01, ">= 0.53")]:
        out.append(f" 勝率 {lab}"); out += by_period(T[(T.wr100 >= a) & (T.wr100 < b)])

    # ---- 概念 5：同時持有上限（以 R 計，每筆 1R）----
    out.append("\n## 5. 同時持有上限（FADE 的風險上限；超過就不開新單）")
    out.append("  上限      " + " | ".join(f"{n:<11}" for n, _, _ in PERIODS) + " | 全期 總R / 平倉最大回撤R")
    for cap in [99, 8, 5, 3, 2]:
        keep, opened = [], []
        for r in T.itertuples():
            opened = [e for e in opened if e > r.t_in]
            if len(opened) < cap:
                keep.append(r.Index); opened.append(r.exit_time)
        K = T.loc[keep].sort_values("exit_time")
        cells = []
        for name, a, b in PERIODS:
            g = K[(K.day >= a) & (K.day < b)]
            cells.append(f"{g.R.mean():+.3f} n{len(g):<5}")
        out.append(f"  {cap if cap < 99 else '無':<8} " + " | ".join(cells) + f" | {K.R.sum():+.0f} / {max_dd(K.R.values):.0f}")

    T.to_csv("part22_trades.csv", index=False)
    txt = "\n".join(out)
    print(txt)
    open("results_part22.txt", "w").write(txt + "\n")


if __name__ == "__main__":
    sys.exit(main())
