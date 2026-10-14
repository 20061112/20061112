"""第二十二部分：滾動 N 根的 Stalin sort 存活數 + 排名，是否有預測力。

特徵（只用到 t 收盤為止的資料）：
  rank      : 目前收盤在最近 N 根收盤中的排名，0 = 最低、N-1 = 最高（≈ 收盤版 Stochastic / PercentRank）
  stalin_up : 把最近 N 根收盤做 Stalin sort（只保留 >= 目前最大值的元素）後剩幾根 = 「創新高的次數」
  stalin_dn : 反方向（只保留 <= 目前最小值）= 「創新低的次數」
  stalin_net = stalin_up - stalin_dn（趨勢品質）
前瞻：h 根後收盤 - 目前收盤，以 ATR20 為單位；t 值用每 h 根取一次（不重疊）的樣本。
"""
import sys
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as swv
from load import load

N = 20
HS = [1, 4, 8, 16]


def resample(df, rule):
    o = df.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "spread": "mean"})
    return o.dropna()


def features(df, n=N):
    c = df["close"].to_numpy()
    w = swv(c, n)                                    # 第 i 列 = c[i : i+n]，最後一個元素是 t = i+n-1
    last = w[:, -1:]
    rank = (w[:, :-1] < last).sum(1) + 0.5 * (w[:, :-1] == last).sum(1)
    runmax = np.maximum.accumulate(w, axis=1)
    runmin = np.minimum.accumulate(w, axis=1)
    up = 1 + (w[:, 1:] >= runmax[:, :-1]).sum(1)
    dn = 1 + (w[:, 1:] <= runmin[:, :-1]).sum(1)
    pad = np.full(n - 1, np.nan)
    f = pd.DataFrame(index=df.index)
    f["rank"] = np.r_[pad, rank]
    f["stalin_up"] = np.r_[pad, up]
    f["stalin_dn"] = np.r_[pad, dn]
    f["stalin_net"] = f["stalin_up"] - f["stalin_dn"]
    f["last_survives"] = np.r_[pad, (w[:, -1] >= runmax[:, -2]).astype(float)]
    tr = pd.concat([df["high"] - df["low"], (df["high"] - df["close"].shift()).abs(),
                    (df["low"] - df["close"].shift()).abs()], axis=1).max(axis=1)
    f["atr"] = tr.rolling(20).mean()
    f["er"] = (df["close"] - df["close"].shift(n - 1)).abs() / df["close"].diff().abs().rolling(n - 1).sum()
    f["er_signed"] = np.sign(df["close"] - df["close"].shift(n - 1)) * f["er"]
    for h in HS:
        f[f"fwd{h}"] = (df["close"].shift(-h) - df["close"]) / f["atr"]
    f["cost"] = df["spread"] * 0.01 / f["atr"]          # 點差（point = 0.01）換成 ATR 單位
    f["year"] = df.index.year
    return f.dropna()


def spearman(a, b):
    return a.rank().corr(b.rank())


def tstat(x):
    x = x.dropna()
    return x.mean() / (x.std() / np.sqrt(len(x))) if len(x) > 2 else np.nan


def bucket_table(f, col, bins, h, label):
    g = f.iloc[::h]                                   # 不重疊樣本
    b = pd.cut(g[col], bins, include_lowest=True)
    rows = []
    for k, sub in g.groupby(b, observed=True):
        r = {"bucket": str(k), "n": len(sub), f"mean_fwd{h}": sub[f"fwd{h}"].mean(), "t": tstat(sub[f"fwd{h}"])}
        for y, s2 in sub.groupby("year"):
            r[y] = s2[f"fwd{h}"].mean()
        rows.append(r)
    print(f"\n[{label}] {col} → fwd{h}（ATR）")
    print(pd.DataFrame(rows).round(3).to_string(index=False))


def random_walk_baseline(n=N, sims=200000, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((sims, n)).cumsum(1)
    runmax = np.maximum.accumulate(w, axis=1)
    up = 1 + (w[:, 1:] >= runmax[:, :-1]).sum(1)
    return pd.Series(up).value_counts(normalize=True).sort_index()


def run(df, label):
    f = features(df)
    print(f"\n==================== {label}：{f.index[0]} ~ {f.index[-1]}，{len(f):,} 根 ====================")
    print("點差中位數（ATR）:", round(f["cost"].median(), 3))
    print("\nstalin_up 分佈：實際 vs 隨機漫步")
    rw = random_walk_baseline()
    act = f["stalin_up"].value_counts(normalize=True).sort_index()
    print(pd.DataFrame({"actual": act, "random_walk": rw}).fillna(0).head(14).round(3).T.to_string())
    print("\n特徵相關（Spearman）:")
    print(f[["rank", "stalin_net", "er_signed"]].rank().corr().round(2).to_string())

    for h in HS:
        print(f"\n--- h = {h}：與前瞻報酬的 Spearman IC（逐年）---")
        rows = []
        for y, s in f.iloc[::h].groupby("year"):
            rows.append({"year": y, **{c: spearman(s[c], s[f"fwd{h}"])
                                     for c in ["rank", "stalin_net", "stalin_up", "stalin_dn", "er_signed"]}})
        print(pd.DataFrame(rows).round(3).to_string(index=False))

    bucket_table(f, "rank", [-0.1, 1, 4, 9.5, 14.5, 17.9, 19.1], 4, label)
    bucket_table(f, "stalin_net", [-20, -6, -3, -1, 1, 3, 6, 20], 4, label)
    bucket_table(f, "stalin_net", [-20, -6, -3, -1, 1, 3, 6, 20], 16, label)

    # 組合：排名在頂端時，用 stalin 分「一路新高（趨勢）」vs「剛突破（之前盤整/下跌）」
    for h in [4, 16]:
        g = f.iloc[::h]
        print(f"\n[{label}] 組合（fwd{h}，ATR）：排名頂/底 × stalin 存活數")
        rows = []
        for name, m in [
            ("rank=19 & up>=7（一路創高）", (g["rank"] >= 19) & (g["stalin_up"] >= 7)),
            ("rank=19 & up<=3（剛突破）", (g["rank"] >= 19) & (g["stalin_up"] <= 3)),
            ("rank=0 & dn>=7（一路創低）", (g["rank"] <= 0) & (g["stalin_dn"] >= 7)),
            ("rank=0 & dn<=3（剛跌破）", (g["rank"] <= 0) & (g["stalin_dn"] <= 3)),
        ]:
            s = g.loc[m, f"fwd{h}"]
            rows.append({"cond": name, "n": len(s), "mean": s.mean(), "t": tstat(s),
                         **{y: v for y, v in s.groupby(g.loc[m, "year"]).mean().items()}})
        print(pd.DataFrame(rows).round(3).to_string(index=False))

    # 增量資訊：fwd16 對 rank、stalin_net、er_signed 的回歸（標準化，不重疊）
    g = f.iloc[::16]
    X = g[["rank", "stalin_net", "er_signed"]]
    X = (X - X.mean()) / X.std()
    X.insert(0, "const", 1.0)
    y = g["fwd16"].to_numpy()
    beta, *_ = np.linalg.lstsq(X.to_numpy(), y, rcond=None)
    resid = y - X.to_numpy() @ beta
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X) * resid.var(ddof=X.shape[1])))
    print(f"\n[{label}] fwd16 ~ rank + stalin_net + er_signed（每 1 個標準差，ATR）")
    print(pd.DataFrame({"beta": beta, "t": beta / se}, index=X.columns).round(4).to_string())


if __name__ == "__main__":
    m15 = load("data/XAUUSD_M15_2023_2026.csv")
    run(m15, "M15")
    run(resample(m15, "1h"), "H1")
    if "--m5" in sys.argv:
        m1 = load("data/XAUUSD_M1_2023_2026.csv")
        run(resample(m1, "5min"), "M5")
