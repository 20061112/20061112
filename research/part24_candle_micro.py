"""第二十四部分：裸 K 微觀結構 —— 影線 / 實體 / 全長的各種比例，以及兩兩配合。

單根 K 棒（R = 高−低，B = |收−開|，U = 上影 = 高−max(開,收)，D = 下影 = min(開,收)−低，d = 陽 +1 / 陰 −1）：
  body_r   B/R                    實體佔全長
  up_r     U/R                    上影佔全長
  dn_r     D/R                    下影佔全長
  imb      (D−U)/R                下影 − 上影（>0 = 下方被拒絕）
  lr_ud    log((D+e)/(U+e))       下影 / 上影
  up_body  log((U+e)/(B+e))       上影 / 實體
  dn_body  log((D+e)/(B+e))       下影 / 實體
  clv      (2C−H−L)/R             收盤在 K 棒中的位置（−1 最低、+1 最高）
  bodypos  (O+C−H−L)/R            實體中心位置
  sbody    (C−O)/R                帶方向的實體比例
  R_atr, B_atr, U_atr, D_atr      全長 / 實體 / 上影 / 下影 ÷ ATR20（大小）
  sR_atr, sB_atr                  上兩者 × d（帶方向）
與前一根的關係：
  sR_prev  log(R/R前) × d         sB_prev  log(B/B前) × d
  cpos_prev  收盤在前一根高低的位置（−1~+1 以外 = 突破）
  engulf   反向吞噬（+1 陽吞陰、−1 陰吞陽）   inside / outside（無方向，× d）
  gap      (開 − 前收)/ATR
位置：rank20  收盤在近 20 根的排名（第二十二部分）
e = 0.02 × ATR，避免除以 0。

驗證：2023~2024 訓練（決定方向、分箱門檻、選格子、擬合權重），2025~2026 測試；前瞻 h 根收盤報酬 / ATR，不重疊樣本。
"""
import warnings
import numpy as np
import pandas as pd
from load import load
from part22_stalin_rank import resample, tstat

warnings.filterwarnings("ignore", category=RuntimeWarning)
HS = [1, 4, 16]
TRAIN = [2023, 2024]
PAIRS = [("body_r", "imb"), ("up_r", "dn_r"), ("body_r", "clv"), ("imb", "R_atr"),
         ("clv", "R_atr"), ("imb", "rank20"), ("sbody", "rank20"), ("up_body", "dn_body"),
         ("sbody", "sR_prev"), ("clv", "cpos_prev")]


def candle_features(df):
    o, h, l, c = (df[k] for k in ("open", "high", "low", "close"))
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(20).mean().shift(1)                      # 用到前一根為止，避免當根大小污染分母
    R = (h - l).where(h > l)
    B = (c - o).abs()
    U = h - np.maximum(o, c)
    D = np.minimum(o, c) - l
    d = np.sign(c - o)
    e = 0.02 * atr
    f = pd.DataFrame(index=df.index)
    f["body_r"], f["up_r"], f["dn_r"] = B / R, U / R, D / R
    f["imb"] = (D - U) / R
    f["lr_ud"] = np.log((D + e) / (U + e))
    f["up_body"] = np.log((U + e) / (B + e))
    f["dn_body"] = np.log((D + e) / (B + e))
    f["clv"] = (2 * c - h - l) / R
    f["bodypos"] = (o + c - h - l) / R
    f["sbody"] = (c - o) / R
    f["R_atr"], f["B_atr"], f["U_atr"], f["D_atr"] = R / atr, B / atr, U / atr, D / atr
    f["sR_atr"], f["sB_atr"] = f["R_atr"] * d, f["B_atr"] * d
    Rp, Bp, hp, lp, op, cp = R.shift(), B.shift(), h.shift(), l.shift(), o.shift(), c.shift()
    f["sR_prev"] = np.log(R / Rp) * d
    f["sB_prev"] = np.log((B + e) / (Bp + e)) * d
    f["cpos_prev"] = ((2 * c - hp - lp) / (hp - lp)).clip(-3, 3)
    dp = np.sign(cp - op)
    eng = (np.maximum(o, c) >= np.maximum(op, cp)) & (np.minimum(o, c) <= np.minimum(op, cp)) & (d * dp < 0)
    f["engulf"] = eng * d
    f["inside"] = ((h <= hp) & (l >= lp)) * d
    f["outside"] = ((h > hp) & (l < lp)) * d
    f["gap"] = (o - cp) / atr
    cw = np.lib.stride_tricks.sliding_window_view(c.to_numpy(), 20)
    f["rank20"] = np.r_[np.full(19, np.nan), (cw[:, :-1] < cw[:, -1:]).sum(1) / 19]
    feats = list(f.columns)
    for k in HS:
        f[f"fwd{k}"] = (c.shift(-k) - c) / atr
    f["cost"] = df["spread"] * 0.01 / atr
    f["year"] = df.index.year
    return f.replace([np.inf, -np.inf], np.nan).dropna(), feats


def q_edge(x, y, q=5):
    b = pd.qcut(x.rank(method="first"), q, labels=False)
    return (y[b == q - 1].mean() - y[b == 0].mean()) / 2


def single_grid(tf, f, feats):
    rows = []
    for h in HS:
        g = f.iloc[::h]
        tr, te = g[g.year.isin(TRAIN)], g[~g.year.isin(TRAIN)]
        for ft in feats:
            ic_tr = tr[ft].rank().corr(tr[f"fwd{h}"].rank())
            ic_te = te[ft].rank().corr(te[f"fwd{h}"].rank())
            yrs = [s[ft].rank().corr(s[f"fwd{h}"].rank()) for _, s in g.groupby("year")]
            rows.append({"tf": tf, "h": h, "feat": ft, "IC_train": ic_tr, "t_train": ic_tr * np.sqrt(len(tr)),
                         "IC_test": ic_te, "t_test": ic_te * np.sqrt(len(te)),
                         "edge_test": np.sign(ic_tr) * q_edge(te[ft], te[f"fwd{h}"]),
                         "cost": g["cost"].median(),
                         "yrs_same": int(sum(np.sign(v) == np.sign(ic_tr) for v in yrs))})
    return rows


def pair_grid(tf, f, a, b, h, nb=4):
    g = f.iloc[::h]
    tr, te = g[g.year.isin(TRAIN)], g[~g.year.isin(TRAIN)]
    def edges(x):
        e = np.unique(np.quantile(tr[x], np.linspace(0, 1, nb + 1)))
        e[0], e[-1] = -np.inf, np.inf
        return e
    ea, eb = edges(a), edges(b)
    out = {}
    for name, s in (("train", tr), ("test", te)):
        y = s[f"fwd{h}"] - s[f"fwd{h}"].mean()                # 扣掉該期漂移
        ca = pd.cut(s[a], ea, labels=False)
        cb = pd.cut(s[b], eb, labels=False)
        out[name] = y.groupby([ca, cb]).agg(["mean", "count", tstat])
    m = out["train"].join(out["test"], lsuffix="_tr", rsuffix="_te").dropna()
    min_tr = min(100, max(20, len(tr) // 64))                 # H1 長 h 樣本少，門檻跟著降
    m = m[(m.count_tr >= min_tr) & (m.count_te >= min_tr // 2)]
    cell_corr = m["mean_tr"].rank().corr(m["mean_te"].rank())
    best = m.sort_values("mean_tr")
    picks = pd.concat([best.head(2).assign(side=-1), best.tail(2).assign(side=1)])
    picks_test = (picks.side * picks.mean_te)
    return {"tf": tf, "pair": f"{a} × {b}", "h": h, "cells": len(m), "cell_corr": cell_corr,
            "pick_train": (picks.side * picks.mean_tr).mean(), "pick_test": picks_test.mean(),
            "pick_test_min_t": (picks.side * picks.tstat_te).min(),
            "cost": g["cost"].median()}


def linear_model(tf, f, feats):
    rows = []
    lag = f[feats].shift(1).add_suffix("_l1")
    lag2 = f[feats].shift(2).add_suffix("_l2")
    full = f.join(lag).join(lag2).dropna()
    for cols, name in ((feats, "單根"), (feats + list(lag.columns) + list(lag2.columns), "三根（含前 1、2 根）")):
        for h in HS:
            g = full.iloc[::h]
            tr, te = g[g.year.isin(TRAIN)], g[~g.year.isin(TRAIN)]
            mu, sd = tr[cols].mean(), tr[cols].std()
            Xtr = ((tr[cols] - mu) / sd).clip(-5, 5).to_numpy()
            Xte = ((te[cols] - mu) / sd).clip(-5, 5).to_numpy()
            lam = 0.1 * len(tr)                                  # ridge
            beta = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(len(cols)), Xtr.T @ (tr[f"fwd{h}"] - tr[f"fwd{h}"].mean()))
            p = pd.Series(Xte @ beta, index=te.index)
            ic = p.rank().corr(te[f"fwd{h}"].rank())
            rows.append({"tf": tf, "model": name, "h": h, "n_feat": len(cols), "IC_test": ic,
                         "t_test": ic * np.sqrt(len(te)), "edge_top_bottom_20%": q_edge(p, te[f"fwd{h}"]),
                         "edge_top_bottom_5%": q_edge(p, te[f"fwd{h}"], 20), "cost": g["cost"].median(),
                         **{y: p[te.year == y].rank().corr(te.loc[te.year == y, f"fwd{h}"].rank())
                            for y in sorted(te.year.unique())}})
    return rows


def pinbar_ref(tf, f):
    """傳統 pinbar 當對照：影線 ≥ 2/3 全長、實體 ≤ 1/3，出現在近 20 根極端位置，反向做"""
    rows = []
    for h in [4, 16]:
        g = f.iloc[::h]
        for name, m, sg in (("看多 pin（下影長、在低檔）", (g.dn_r >= 2 / 3) & (g.body_r <= 1 / 3) & (g.rank20 <= 0.2), 1),
                            ("看空 pin（上影長、在高檔）", (g.up_r >= 2 / 3) & (g.body_r <= 1 / 3) & (g.rank20 >= 0.8), -1)):
            y = sg * (g[f"fwd{h}"] - g.groupby("year")[f"fwd{h}"].transform("mean"))
            s = y[m]
            rows.append({"tf": tf, "rule": name, "h": h, "n": len(s), "mean": s.mean(), "t": tstat(s),
                         **s.groupby(g.year[m]).mean().to_dict()})
    return rows


def main():
    m1 = load("data/XAUUSD_M1_2023_2026.csv")
    m15 = load("data/XAUUSD_M15_2023_2026.csv")
    tfs = {"M5": resample(m1, "5min"), "M15": m15, "H1": resample(m15, "1h")}
    single, pairs, lin, pins = [], [], [], []
    for tf, df in tfs.items():
        f, feats = candle_features(df)
        single += single_grid(tf, f, feats)
        for a, b in PAIRS:
            for h in HS:
                pairs.append(pair_grid(tf, f, a, b, h))
        lin += linear_model(tf, f, feats)
        pins += pinbar_ref(tf, f)
    pd.set_option("display.width", 250)
    S = pd.DataFrame(single)
    S.to_csv("part24_single.csv", index=False)
    print(f"=== A. 單一比例：{len(S)} 組（3 時框 × {S.feat.nunique()} 特徵 × {len(HS)} 種 h），訓練 2023~24 → 測試 2025~26 ===")
    print("edge_test = 依訓練方向，最高/最低 1/5 一多一空，測試期每筆平均優勢（ATR）")
    print(S.reindex(S.t_train.abs().sort_values(ascending=False).index).head(25).round(3).to_string(index=False))
    sel = S[S.t_train.abs() > 2]
    same = np.sign(sel.IC_test) == np.sign(sel.IC_train)
    print(f"\n訓練 |t|>2：{len(sel)} / {len(S)}（隨機預期 ≈ {0.0455 * len(S):.0f}）；測試同方向 {same.sum()}；"
          f"測試也 |t|>2 同方向 {(same & (sel.t_test.abs() > 2)).sum()}；測試優勢 > 點差 {(sel.edge_test > sel.cost).sum()}")
    print("\n各時框 / h 的 |IC| 平均（訓練 → 測試）")
    print(S.assign(a_tr=S.IC_train.abs(), a_te=S.IC_test.abs()).groupby(["tf", "h"])[["a_tr", "a_te"]].mean().round(4).to_string())

    P = pd.DataFrame(pairs)
    P.to_csv("part24_pairs.csv", index=False)
    print("\n=== B. 兩兩配合（各 4 分箱 = 16 格，門檻用訓練期）===")
    print("cell_corr = 訓練 vs 測試 各格平均的排名相關（1 = 格子的好壞完全重現）；pick = 訓練最好 2 格做多 + 最差 2 格做空，平均優勢（ATR）")
    print(P.round(3).to_string(index=False))
    print(f"\ncell_corr 平均 {P.cell_corr.mean():.3f}；> 0.5 的組合 {(P.cell_corr > 0.5).sum()} / {len(P)}；"
          f"pick_test > 點差 {(P.pick_test > P.cost).sum()} / {len(P)}")

    print("\n=== C. 全部比例合成（ridge，訓練擬合 → 測試）===")
    print(pd.DataFrame(lin).round(3).to_string(index=False))

    print("\n=== D. 對照：傳統 pinbar（扣漂移，ATR，正 = 照 pinbar 方向賺）===")
    print(pd.DataFrame(pins).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
