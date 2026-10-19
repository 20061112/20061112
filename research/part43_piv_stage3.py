"""第四十三部分 第三階段：濾網 —— 固定 P（K 線 10 分、01:00~04:00、前一根高低觸價、停損 2×ATR、RR 3、收盤平倉）。研究期 only。→ part43_piv_stage3.csv
(1) 日線方向的定義（這其實是策略的核心）：SMA 長度 × 斜率回看、只看均線上下、n 日動能、不分方向
(2) B 條件、前一天效率 peff、ATR10/ATR60、同時持倉上限、影子權益濾網（XAU 的 50 筆均線）
門檻只用研究期。"""
import numpy as np, pandas as pd
from piv_core import run, summ, direction, period, RESEARCH, HALVES, D

BASE = dict(k=10, mode=1, hours=(1, 4))


def cap_total(X, nmax):
    """同時持倉最多 nmax 筆（依進場時間，已滿就略過）。"""
    X = X.sort_values(["d", "ent"]).reset_index(drop=True)
    keep = np.zeros(len(X), bool)
    for d, g in X.groupby("d"):
        open_x = []
        for i, r in g.iterrows():
            open_x = [x for x in open_x if x > r.ent]
            if len(open_x) < nmax:
                keep[i] = True
                open_x.append(r.xm)
    return X[keep]


def equity_filter(X, look=50):
    """XAU 第 2.3 節：影子累積 R < 最近 look 筆的均線 → 不下單。"""
    X = X.copy()
    X["tin"] = X.day + pd.to_timedelta(X.ent, unit="m")
    X["tout"] = X.day + pd.to_timedelta(X.xm, unit="m")
    X = X.sort_values("tin").reset_index(drop=True)
    ex = X.sort_values("tout")
    cum = ex.R.cumsum().to_numpy()
    ma = pd.Series(cum).rolling(look, min_periods=look).mean().to_numpy()
    kk = np.searchsorted(ex.tout.values, X.tin.values, side="right") - 1
    take = np.array([not (k >= look and cum[k] < ma[k]) for k in kk])
    return X[take]


def row(lab, X):
    return dict(var=lab, **summ(X))


if __name__ == "__main__":
    rows = []
    for n in (20, 30, 50, 100, 150, 200):
        for sl in (5, 10, 20):
            rows.append(row(f"方向 SMA{n} 斜率{sl}", run(**BASE, dirs=direction(n, sl))))
    for n in (20, 50, 100, 200):
        rows.append(row(f"方向 只看收盤>SMA{n}", run(**BASE, dirs=direction(n, rule="price"))))
    for n in (5, 10, 20, 60):
        rows.append(row(f"方向 {n} 日動能", run(**BASE, dirs=direction(n, rule="mom"))))
    rows.append(row("不分方向", run(k=10, mode=2, hours=(1, 4))))
    X = run(**BASE, dirs=direction())
    X = X.join(D[["peff", "volr"]], on="day")
    for b in (0.1, 0.17, 0.3, 0.5):
        rows.append(row(f"B 條件 > −{b}", run(**BASE, bthr=b)))
    R0 = period(X, *RESEARCH)
    for f in ("peff", "volr"):
        for q in (0.5, 0.67, 0.75, 0.9):
            thr = R0[f].quantile(q)
            rows.append(row(f"{f} ≤ q{q} ({thr:.2f})", X[X[f] <= thr]))
            rows.append(row(f"{f} > q{q} ({thr:.2f})（被濾掉的）", X[X[f] > thr]))
    for nmax in (2, 4, 8):
        rows.append(row(f"同時持倉 ≤ {nmax}", cap_total(X, nmax)))
    for look in (20, 50, 100):
        rows.append(row(f"權益濾網 {look} 筆", equity_filter(X, look)))
    T = pd.DataFrame(rows)
    T.to_csv("part43_piv_stage3.csv", index=False)
    pd.set_option("display.width", 250)
    print(T.round(3).to_string(index=False))
