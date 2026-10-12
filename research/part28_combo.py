"""第二十八部分（續）：過濾組合 + 限制疊單。

候選條件（由 part28_m15_filters.py 訓練期 2023/6~2024/12 挑出，驗證期不調整）：
  A 順 50 日均線（做多需前日收盤在 50 日均線上，做空需在下方）
  B 今天沒有反向走太多：day_move > −0.17 ATRd（訓練期最差的五分之一）
  C 順 M15 EMA50（前一根 M15 收盤在 EMA50 同側）
  D 不做 18~22 點
疊單限制：同方向同時最多 N 筆（N = 1 / 3 / 無限制）。
"""
import itertools
import numpy as np
import pandas as pd

T = pd.read_pickle("part28_features.pkl").sort_values("進場時間").reset_index(drop=True)
OUT = "results_part28.txt"
lines = ["", "=== 3. 過濾組合（訓練 2023/6~2024/12 / 驗證 2025/1~2026/10）==="]
FLT = {"A": T.trend50 == 1, "B": T.day_move > -0.17, "C": T.ema_al == 1, "D": T.hour < 18}


def cap(df, n):
    """同方向同時最多 n 筆（依進場時間順序，已有 n 筆未出場就略過）。"""
    if n is None:
        return df
    keep = []
    open_ = {1: [], -1: []}
    for r in df.itertuples():
        lst = [x for x in open_[r.dir] if x > r.進場時間]
        if len(lst) < n:
            lst.append(r.出場時間)
            keep.append(r.Index)
        open_[r.dir] = lst
    return df.loc[keep]


def stats(x):
    if len(x) < 30:
        return "n<30"
    dd = x.groupby(x.進場時間.dt.date).usd.sum()
    eq = dd.cumsum()
    days = x.進場時間.dt.date.nunique()
    gp, gl = x.usd[x.usd > 0].sum(), -x.usd[x.usd < 0].sum()
    return (f"{len(x) / days:5.1f}筆/天 {x.R.mean():+.3f}R {x.usd.mean():+.2f}$ PF {gp / gl:.2f} "
            f"日t {dd.mean() / dd.std() * np.sqrt(len(dd)):+.2f} 總 {x.usd.sum():+7,.0f} 回撤 {(eq.cummax() - eq).max():6,.0f}")


def yearly(x):
    return " ".join(f"{x.R[x.進場時間.dt.year == y].mean():+.3f}" for y in (2023, 2024, 2025, 2026))


res = []
for r in range(0, 5):
    for combo in itertools.combinations("ABCD", r):
        m = np.ones(len(T), bool)
        for c in combo:
            m &= FLT[c].values
        x = T[m]
        res.append(("".join(combo) or "無", x))
lines.append(f"{'條件':6s} | {'訓練':78s} | 驗證")
for name, x in res:
    tr, te = x[x.train], x[~x.train]
    lines.append(f"{name:6s} | {stats(tr):78s} | {stats(te)}")

lines.append("\n=== 4. 加上疊單限制（同方向同時最多 N 筆）===")
for name in ("無", "A", "AB", "ABC"):
    x = dict(res)[name]
    for n in (None, 3, 1):
        y = cap(x, n)
        tr, te = y[y.train], y[~y.train]
        lines.append(f"  {name:4s} N={str(n) if n else '∞':2s} | 訓練 {stats(tr)} | 驗證 {stats(te)} | 逐年R {yearly(y)}")

best = cap(dict(res)["AB"], 1)  # 訓練期 R 與日 t 最好的組合
best.drop(columns=["i"]).to_csv("part28_m15_filtered_trades.csv", index=False, encoding="utf-8-sig")
lines.append(f"\nAB + 同方向最多 1 筆 的逐筆明細：part28_m15_filtered_trades.csv（{len(best)} 筆）")
y = best.groupby(best.進場時間.dt.year).agg(筆數=("usd", "size"), 每筆美元=("usd", "mean"), 總計=("usd", "sum"), 每筆R=("R", "mean"))
for k, v in y.iterrows():
    lines.append(f"  {k}: {int(v.筆數):5d} 筆  每筆 {v.每筆美元:+.2f}$  {v.每筆R:+.3f}R  總計 {v.總計:+8,.0f}$")
m = best.groupby(best.進場時間.dt.to_period("M")).usd.sum()
lines.append(f"  正報酬月份 {(m > 0).sum()}/{len(m)}；滑價每筆 +0.1$ 後每筆 {best.usd.mean() - 0.1:+.2f}$、+0.3$ 後 {best.usd.mean() - 0.3:+.2f}$")
print("\n".join(lines))
open(OUT, "a").write("\n".join(lines) + "\n")
