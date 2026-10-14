"""第二十三部分：多種根數 N + 「整段」排名形狀（不只看最新一根）。

每根 K 棒取最近 N 根收盤，換成段內排名序列 r_0..r_{N-1}（0 = 最低），再算整段的形狀特徵：
  last      : 最新一根的排名 /(N-1)（第二十二部分的 rank）
  stalin    : (創新高次數 − 創新低次數) / N（Stalin sort 存活數差）
  rho       : 排名 vs 時間的 Spearman 相關（整段單調程度；+1 = 每根都比前一根高）
  q_slope   : 最後 1/4 的平均排名 − 最前 1/4 的平均排名（段尾相對段頭）
  hump      : 中間 1/2 平均排名 − 頭尾 1/4 平均（>0 = 倒 V／先漲後跌，<0 = V 形）
  ext_pos   : (最高點位置 − 最低點位置)/(N-1)（>0 = 低點在前高點在後，類 Aroon）
  rough     : 排名序列相鄰差的平均絕對值 /N（段內來回洗的程度）
形狀樣板：段切 4 塊，每塊平均排名分 低/中/高 → 81 種代碼。

嚴格切開：2023~2024 = 訓練（選方向、選樣板、擬合權重），2025~2026 = 測試（只看一次）。
前瞻報酬以 ATR20 為單位，t 值用不重疊樣本（每 h 根取一次）。
"""
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as swv
from load import load
from part22_stalin_rank import resample, tstat

NS = [10, 20, 40, 60, 120]
HS = [4, 16]
FEATS = ["last", "stalin", "rho", "q_slope", "hump", "ext_pos", "rough"]
TRAIN = [2023, 2024]


def seg_features(c, n):
    w = swv(c, n).astype(np.float64)
    r = w.argsort(1).argsort(1).astype(np.float32)        # 段內排名
    t = np.arange(n, dtype=np.float32)
    q = max(n // 4, 1)
    runmax = np.maximum.accumulate(w, 1)
    runmin = np.minimum.accumulate(w, 1)
    up = (w[:, 1:] >= runmax[:, :-1]).sum(1)
    dn = (w[:, 1:] <= runmin[:, :-1]).sum(1)
    tc = t - t.mean()
    rc = r - r.mean(1, keepdims=True)
    out = {
        "last": r[:, -1] / (n - 1),
        "stalin": (up - dn) / n,
        "rho": (rc * tc).sum(1) / np.sqrt((rc ** 2).sum(1) * (tc ** 2).sum()),
        "q_slope": (r[:, -q:].mean(1) - r[:, :q].mean(1)) / (n - 1),
        "hump": (r[:, q:n - q].mean(1) - 0.5 * (r[:, :q].mean(1) + r[:, -q:].mean(1))) / (n - 1),
        "ext_pos": (w.argmax(1) - w.argmin(1)) / (n - 1),
        "rough": np.abs(np.diff(r, axis=1)).mean(1) / n,
    }
    blocks = np.array_split(np.arange(n), 4)
    lv = [np.digitize(r[:, b].mean(1) / (n - 1), [1 / 3, 2 / 3]) for b in blocks]
    out["code"] = lv[0] * 27 + lv[1] * 9 + lv[2] * 3 + lv[3]
    pad = n - 1
    return {k: np.r_[np.full(pad, np.nan), v] for k, v in out.items()}


def base(df):
    tr = pd.concat([df["high"] - df["low"], (df["high"] - df["close"].shift()).abs(),
                    (df["low"] - df["close"].shift()).abs()], axis=1).max(axis=1)
    b = pd.DataFrame(index=df.index)
    b["atr"] = tr.rolling(20).mean()
    for h in HS:
        b[f"fwd{h}"] = (df["close"].shift(-h) - df["close"]) / b["atr"]
    b["cost"] = df["spread"] * 0.01 / b["atr"]
    b["year"] = df.index.year
    return b


def code_str(k):
    k = int(k)
    return "".join("LMH"[d] for d in (k // 27, k // 9 % 3, k // 3 % 3, k % 3))


def q_spread(x, y):
    """(最高 1/5 − 最低 1/5) 的前瞻報酬差 / 2 = 一多一空每筆的平均優勢"""
    try:
        b = pd.qcut(x.rank(method="first"), 5, labels=False)
    except ValueError:
        return np.nan
    return (y[b == 4].mean() - y[b == 0].mean()) / 2


def main():
    m1 = load("data/XAUUSD_M1_2023_2026.csv")
    m15 = load("data/XAUUSD_M15_2023_2026.csv")
    tfs = {"M5": resample(m1, "5min"), "M15": m15, "H1": resample(m15, "1h")}
    grid, ridge_rows, tmpl_rows = [], [], []
    for tf, df in tfs.items():
        b = base(df)
        c = df["close"].to_numpy()
        cost = b["cost"].median()
        for n in NS:
            f = b.join(pd.DataFrame(seg_features(c, n), index=df.index)).dropna()
            for h in HS:
                g = f.iloc[::h]
                tr, te = g[g.year.isin(TRAIN)], g[~g.year.isin(TRAIN)]
                for ft in FEATS:
                    ic_tr = tr[ft].rank().corr(tr[f"fwd{h}"].rank())
                    ic_te = te[ft].rank().corr(te[f"fwd{h}"].rank())
                    sgn = np.sign(ic_tr)
                    yrs = {y: s[ft].rank().corr(s[f"fwd{h}"].rank()) for y, s in g.groupby("year")}
                    grid.append({"tf": tf, "N": n, "h": h, "feat": ft,
                                 "IC_train": ic_tr, "t_train": ic_tr * np.sqrt(len(tr)),
                                 "IC_test": ic_te, "t_test": ic_te * np.sqrt(len(te)),
                                 "edge_test": sgn * q_spread(te[ft], te[f"fwd{h}"]), "cost": cost,
                                 "yrs_same_sign": sum(np.sign(v) == np.sign(ic_tr) for v in yrs.values()),
                                 "n_years": len(yrs)})

                # 整段形狀合起來：訓練期擬合線性權重（標準化），測試期看 IC 與一多一空優勢
                X = tr[FEATS]
                mu, sd = X.mean(), X.std()
                A = np.c_[np.ones(len(tr)), ((X - mu) / sd).to_numpy()]
                beta = np.linalg.lstsq(A, tr[f"fwd{h}"].to_numpy(), rcond=None)[0]
                pred = pd.Series(((te[FEATS] - mu) / sd).to_numpy() @ beta[1:], index=te.index)
                ic = pred.rank().corr(te[f"fwd{h}"].rank())
                ridge_rows.append({"tf": tf, "N": n, "h": h, "IC_test": ic, "t_test": ic * np.sqrt(len(te)),
                                   "edge_test": q_spread(pred, te[f"fwd{h}"]), "cost": cost,
                                   **{y: pred[te.year == y].rank().corr(te.loc[te.year == y, f"fwd{h}"].rank())
                                      for y in sorted(te.year.unique())}})

                # 形狀樣板：訓練期每種代碼（≥ 150 筆）扣漂移後的平均，取最好 / 最差各 3 個到測試期
                if n in (20, 40) and h == 16:
                    dtr = tr[f"fwd{h}"] - tr[f"fwd{h}"].mean()
                    dte = te[f"fwd{h}"] - te[f"fwd{h}"].mean()
                    st = dtr.groupby(tr["code"]).agg(["mean", "count"])
                    st = st[st["count"] >= 150].sort_values("mean")
                    for side, codes in (("short", st.index[:3]), ("long", st.index[-3:])):
                        for k in codes:
                            s = dte[te["code"] == k]
                            sg = 1 if side == "long" else -1
                            tmpl_rows.append({"tf": tf, "N": n, "code": code_str(k), "side": side,
                                              "train_mean": sg * st.loc[k, "mean"], "n_train": int(st.loc[k, "count"]),
                                              "test_mean": sg * s.mean(), "t_test": sg * tstat(s), "n_test": len(s),
                                              "cost": cost})
    pd.set_option("display.width", 250)
    G = pd.DataFrame(grid)
    print(f"共 {len(G)} 組（3 時框 × {len(NS)} 種 N × {len(HS)} 種 h × {len(FEATS)} 特徵）；訓練 2023~2024、測試 2025~2026")
    print("edge_test = 依訓練期 IC 方向做一多一空（前後各 1/5）時，測試期每筆平均優勢（ATR）；cost = 點差中位數（ATR）\n")
    print("=== 訓練期 |t| 最大的 20 組，到測試期的表現 ===")
    top = G.reindex(G["t_train"].abs().sort_values(ascending=False).index).head(20)
    print(top.round(3).to_string(index=False))
    print(f"\n訓練期 |t|>2 的組數：{(G.t_train.abs() > 2).sum()} / {len(G)}（純隨機預期約 {0.0455 * len(G):.0f}）")
    sel = G[G.t_train.abs() > 2]
    same = (np.sign(sel.IC_test) == np.sign(sel.IC_train))
    print(f"其中測試期方向相同：{same.sum()} / {len(sel)}；測試期也 |t|>2 且同方向：{(same & (sel.t_test.abs() > 2)).sum()}")
    print(f"測試期優勢 > 點差：{(sel.edge_test > sel.cost).sum()} / {len(sel)}")
    print("\n=== 每個特徵：所有時框/N/h 平均（看哪一類有沒有系統性方向）===")
    print(G.groupby("feat")[["IC_train", "IC_test"]].mean().round(4).to_string())
    print("\n=== 每個 N：所有特徵 |IC| 平均 ===")
    print(G.assign(a_tr=G.IC_train.abs(), a_te=G.IC_test.abs()).groupby(["tf", "N"])[["a_tr", "a_te"]].mean().round(4).unstack(0).to_string())
    print("\n=== 整段形狀 7 特徵合成（訓練期線性擬合 → 測試期）===")
    print(pd.DataFrame(ridge_rows).round(3).to_string(index=False))
    print("\n=== 形狀樣板（4 塊 × 低/中/高）：訓練期最好/最差各 3 個 → 測試期（fwd16，扣漂移，ATR，正 = 該方向賺）===")
    print(pd.DataFrame(tmpl_rows).round(3).to_string(index=False))
    G.to_csv("part23_grid.csv", index=False)


if __name__ == "__main__":
    main()
