"""第二十九部分：M30 紅綠不平衡（動能）當版本 A 的方向過濾。
事先登記（看結果前就決定）：
  主規則  M30 最近 25 根已收完的 K 棒，陽 ≥ 15（60%）→ 只做多；陰 ≥ 15 → 只做空；都沒有 → 多空都可以做
          （= 只刪「逆著 M30 多數色」的突破單）
  嚴格版  都沒有（中性）時也不做
  動能版  用 M30 最近 25 根的漲跌方向取代顏色（逆著漲跌方向的單不做）
套用在：版本 A（凍結門檻）、版本 A（滾動前推）、L1 核心（M5 收盤確認、含週一）+ 反向尾巴。
只用進場當下「已經收完」的 M30 K 棒，沒有未來資料。期間：樣本外 2023/6~2025/12、研究期 2026。
另附鄰近參數（N 15~40、比例 55%~70%）看是不是只在 25 / 60% 才有效。
"""
import json
import numpy as np
import pandas as pd
from load import load

OOS_END = pd.Timestamp("2026-01-01")
LIVE = json.load(open("final_A_live.json"))
out = None


def P(s=""):
    print(s)
    if out:
        out.write(s + "\n")
        out.flush()


def pf(p):
    return p[p > 0].sum() / max(-p[p <= 0].sum(), 1e-9)


def dedup(Y):
    return Y.sort_values(["t_in", "h"]).drop_duplicates(["t_in", "side"]).sort_values("t_in")


def frozen(X):
    k = ~(X.aligned & (X.ib_atr < LIVE["thr_ib_atr"])) & (X["tail"] <= LIVE["thr_tail_atr"]) & (X.ext_letter.fillna(0) <= LIVE["thr_ext_letter"])
    return dedup(X[k])


def walk(X, q_tail=0.9, q_ext=0.9, q_ib=0.4):
    X = X.copy()
    X["ym"] = X.day.dt.to_period("M")
    parts = []
    for m in sorted(X.ym.unique())[2:]:
        past, cur = X[X.ym < m], X[X.ym == m]
        k = ~(cur.aligned & (cur.ib_atr < past.ib_atr.quantile(q_ib))) & (cur["tail"] <= past["tail"].quantile(q_tail)) \
            & (cur.ext_letter.fillna(0) <= past.ext_letter.quantile(q_ext))
        parts.append(cur[k])
    return dedup(pd.concat(parts))


def line(T):
    if len(T) < 5:
        return f"n={len(T)}"
    d = T.groupby("day").R.sum()
    td = d.mean() / d.std() * np.sqrt(len(d))
    return f"n={len(T):5d} 每筆 {T.R.mean():+.3f}R PF {pf(T.pnl):.2f} 勝率 {100 * (T.pnl > 0).mean():4.1f}% 累計 {T.R.sum():+7.1f}R 日t {td:+.1f}"


# ---------- M30 特徵（只用已收完的 K 棒）----------
m1 = load("data/XAUUSD_M1_2023_2026.csv")
m30 = m1.resample("30min").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
d30 = np.sign(m30.close - m30.open)
end30 = (m30.index + pd.Timedelta(minutes=30)).values


def m30_feats(T, N):
    g = (d30 > 0).rolling(N).sum().to_numpy()
    r = (d30 < 0).rolling(N).sum().to_numpy()
    mom = (m30.close - m30.close.shift(N)).to_numpy()
    # 進場時間 t_in 是 M1 K 棒的開始時間，進場在它收盤 → 可用的 M30 = 結束時間 ≤ t_in + 1 分鐘
    pos = np.searchsorted(end30, (pd.to_datetime(T.t_in) + pd.Timedelta(minutes=1)).values, side="right") - 1
    return g[pos], r[pos], mom[pos]


def apply(T, N=25, frac=0.6, mode="主規則"):
    g, r, mom = m30_feats(T, N)
    k = int(np.ceil(frac * N - 1e-9))
    up, dn = g >= k, r >= k
    s = T.side.to_numpy()
    if mode == "主規則":
        keep = ~((s == 1) & dn) & ~((s == -1) & up)
    elif mode == "嚴格":
        keep = ((s == 1) & up) | ((s == -1) & dn)
    elif mode == "動能":
        keep = np.sign(mom) * s >= 0
    elif mode == "反向":            # 對照：只做逆著多數色的單
        keep = ((s == 1) & dn) | ((s == -1) & up)
    return T[keep]


A = pd.read_pickle("part26_signals_A.pkl")
L1 = pd.read_pickle("part26_signals_L1.pkl")
for X in (A, L1):
    X["day"] = pd.to_datetime(X.day)
BASES = {"版本 A（凍結門檻）": frozen(A), "版本 A（滾動前推）": walk(A), "L1 核心 + 反向尾巴（前推）": walk(L1, 0.9, 1.0, 0.0)}

if __name__ == "__main__":
    out = open("results_part29.txt", "w")
    P("第二十九部分：M30 紅綠不平衡當方向過濾（事先登記：N=25、多數 ≥ 60%、逆向的單不做）")
    for name, T in BASES.items():
        P(f"\n# {name}")
        for mode in ("不加過濾", "主規則", "嚴格", "動能", "反向"):
            Y = T if mode == "不加過濾" else apply(T, mode=mode)
            o, i = Y[Y.day < OOS_END], Y[Y.day >= OOS_END]
            P(f"  {mode:6s} 樣本外 {line(o)}")
            P(f"  {'':6s} 2026   {line(i)}")
            if mode in ("不加過濾", "主規則"):
                for y, g in Y.groupby(Y.day.dt.year):
                    lo, sh = g[g.side == 1], g[g.side == -1]
                    P(f"           {y}: {line(g)} | 多 {lo.R.mean():+.3f} (n={len(lo)}) 空 {sh.R.mean():+.3f} (n={len(sh)})")

    P("\n# 被主規則刪掉的單（逆著 M30 多數色）本身表現")
    for name, T in BASES.items():
        kept = apply(T)
        drop = T[~T.index.isin(kept.index)]
        o, i = drop[drop.day < OOS_END], drop[drop.day >= OOS_END]
        P(f"  {name:22s} 刪掉的：樣本外 {line(o)} | 2026 {line(i)}")
        for side, lab in ((1, "多"), (-1, "空")):
            P(f"      {lab}：保留 {kept[(kept.side == side) & (kept.day < OOS_END)].R.mean():+.3f}R / 刪掉 {drop[(drop.side == side) & (drop.day < OOS_END)].R.mean():+.3f}R（樣本外）")

    P("\n# 鄰近參數（主規則，版本 A 凍結）：每筆 R 樣本外 / 2026，括號 = 樣本外保留比例")
    T = BASES["版本 A（凍結門檻）"]
    FR = (0.55, 0.6, 0.65, 0.7)
    P("   N   " + "  ".join(f"{f:>24.0%}" for f in FR))
    for N in (10, 15, 20, 25, 30, 40, 50):
        cells = []
        for f in FR:
            Y = apply(T, N, f)
            o, i = Y[Y.day < OOS_END], Y[Y.day >= OOS_END]
            cells.append(f"{o.R.mean():+.3f}/{i.R.mean():+.3f}({len(o) / (T.day < OOS_END).sum():.0%})")
        P(f"  {N:3d} " + "  ".join(f"{c:>24s}" for c in cells))
    P("   對照：不加過濾 樣本外 {:+.3f} / 2026 {:+.3f}".format(T[T.day < OOS_END].R.mean(), T[T.day >= OOS_END].R.mean()))
    out.close()
