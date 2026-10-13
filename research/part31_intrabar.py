"""第三十一部分：大週期急跌買改用 M1 提前進場（盤中即時 z）。

第三十部分的大週期版本要等 M5/M15/... K 棒收盤才判斷。這裡改成：每根 M1 收盤時，把「正在形成的大週期 K 棒」
當作收在目前價格，即時算大週期的 v 與 z：
  v_即時 = (1−α) × 上一根已收大週期 K 棒的 EWMA + α × (目前價 − 上一根已收收盤) ÷ 上一根已收的 ATR60
  z_即時 = (v_即時 − 已收 K 棒 v 的滾動平均) ÷ 已收 K 棒 v 的滾動標準差（都用上一根為止的資料，沒有偷看）
z_即時 首次 ≤ −門檻 → 下一分鐘開盤做多。停損、持有、出場與第三十部分完全相同（停損 = 大週期 ATR60 × 倍數，M1 逐根出場）。
比較：同一組參數，「收盤才進」vs「M1 提前進」。
輸出：results_part31.txt、part31_grid.csv、part31_equity.png
"""
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

exec(open("part30_tf_corr.py", encoding="utf-8").read().split("rows, trades = [], {}")[0])   # 取得 m、tf_signals、run、st 等

TFS2 = (3, 5, 15, 30, 60)
out = []


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    out.append(s)


def intrabar_signals(k):
    b = m.resample(f"{k}min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    c = b.close
    pc = c.shift()
    tr = pd.concat([b.high - b.low, (b.high - pc).abs(), (b.low - pc).abs()], axis=1).max(axis=1)
    atr = tr.rolling(60).mean()
    gap = b.index.to_series().diff() > pd.Timedelta(minutes=k + 5 if k < 60 else 180)
    alpha = 2 / 21
    v = c.diff().where(~gap).ewm(span=20, adjust=False).mean() / atr
    zw = max(1380 // k, 240)
    mu, sd = v.rolling(zw).mean(), v.rolling(zw).std()
    # 已收大週期 K 棒的狀態，對齊到「下一根大週期 K 棒」的每一分鐘
    st_ = pd.DataFrame({"ema_raw": (v * atr), "close": c, "atr": atr, "mu": mu, "sd": sd})
    st_.index = b.index + pd.Timedelta(minutes=k)                    # 這些數值在該 K 棒收盤後才可用
    S = st_.reindex(m.index, method="ffill")
    ema_now = (1 - alpha) * S.ema_raw + alpha * (m.close - S.close)  # 把目前價當作形成中 K 棒的收盤
    v_now = ema_now / S.atr
    z_now = (v_now - S.mu) / S.sd
    close_time = m.index + pd.Timedelta(minutes=1)                    # M1 收盤後下一分鐘進場
    return pd.DataFrame({"z": z_now.to_numpy(), "atr": S.atr.to_numpy()}, index=close_time)


rows, trades = [], {}
for k in TFS2:
    S_close = tf_signals(k)
    S_intra = intrabar_signals(k)
    for thr, sl, hold in itertools.product(THRS, SLS, HOLDS):
        for mode, S in (("收盤才進", S_close), ("M1 提前進", S_intra)):
            T = run(S, thr, sl, hold)
            trades[(k, thr, sl, hold, mode)] = T
            rows.append(dict(週期=f"M{k}", k=k, 門檻=thr, 停損=sl, 持有分=hold, 方式=mode, **st(T)))
    print("done", k)
G = pd.DataFrame(rows)
G.round(4).to_csv("part31_grid.csv", index=False, encoding="utf-8-sig")

p("=" * 120)
p("第三十一部分：大週期急跌買，收盤才進 vs 用 M1 提前進（盤中即時 z）；參數與第三十部分相同（12 組 / 週期）")
p("=" * 120)
p("\n--- 每個週期：12 組平均（每筆 R、樣本內 / 樣本外、樣本外為正比例、筆數）")
agg = G.groupby(["週期", "方式"], sort=False).agg(全期=("R", "mean"), IS=("IS_R", "mean"), OOS=("OOS_R", "mean"),
                                                   OOS為正=("OOS_R", lambda s: (s > 0).mean()), 筆數=("n", "mean"), 日Sharpe=("日Sharpe", "mean"))
p(agg.round(3).to_string())

p("\n--- 同一組參數，M1 提前進 − 收盤才進 的差（每筆 R），12 組中變好的比例")
P = G.pivot_table(index=["週期", "門檻", "停損", "持有分"], columns="方式", values=["R", "IS_R", "OOS_R"])
for k in TFS2:
    sub = P.loc[f"M{k}"]
    d = sub[("R", "M1 提前進")] - sub[("R", "收盤才進")]
    di = sub[("IS_R", "M1 提前進")] - sub[("IS_R", "收盤才進")]
    do = sub[("OOS_R", "M1 提前進")] - sub[("OOS_R", "收盤才進")]
    p(f"  M{k:<3d} 全期差 {d.mean():+.3f}R（變好 {(d > 0).mean():.0%}）| 樣本內差 {di.mean():+.3f}（{(di > 0).mean():.0%}）| 樣本外差 {do.mean():+.3f}（{(do > 0).mean():.0%}）")

p("\n--- 第三十部分代表設定（樣本內 t 最高）兩種方式對照")
reps = {3: (2.5, 6.0, 1440), 5: (3.0, 6.0, 480), 15: (2.5, 6.0, 1440), 30: (2.5, 3.0, 1440), 60: (2.5, 3.0, 1440)}
cols = ["方式", "n", "R", "bp", "t", "勝率", "IS_R", "OOS_R", "OOS_t", "日Sharpe", "DD", "y2023", "y2024", "y2025", "y2026"]
for k, (thr, sl, hold) in reps.items():
    sub = G[(G.k == k) & (G.門檻 == thr) & (G.停損 == sl) & (G.持有分 == hold)]
    p(f"  M{k} 門檻 {thr} 停損 {sl} 持有 {hold} 分")
    p(sub[cols].round(3).to_string(index=False))

p("\n--- M1 提前進：每個週期依樣本內 t 重新挑代表設定 → 樣本外")
for k in TFS2:
    r = G[(G.k == k) & (G.方式 == "M1 提前進")].sort_values("IS_t", ascending=False).iloc[0]
    p(f"  M{k:<3d} 門檻 {r.門檻} 停損 {r.停損} 持有 {int(r.持有分)}：n={int(r.n)} 每筆 {r.R:+.3f}R IS {r.IS_R:+.3f}（t {r.IS_t:+.2f}）→ OOS {r.OOS_R:+.3f}（t {r.OOS_t:+.2f}）")

# 與 M1 定案的相關性（M1 提前進的代表設定）
m1T = run(tf_signals(1), 2.5, 3.0, 120)
days = pd.Index(np.unique(tday[ts.searchsorted(WARM):]))
D = pd.DataFrame({"M1": m1T.groupby("day").R.sum().reindex(days).fillna(0)})
for k, (thr, sl, hold) in reps.items():
    for mode in ("收盤才進", "M1 提前進"):
        D[f"M{k} {mode}"] = trades[(k, thr, sl, hold, mode)].groupby("day").R.sum().reindex(days).fillna(0)
p("\n--- 與 M1 定案的日損益相關（代表設定）")
p(D.corr()["M1"].drop("M1").round(2).to_string())

fig, ax = plt.subplots(1, len(TFS2), figsize=(18, 3.8), sharey=False)
for a_, k in zip(ax, TFS2):
    for mode, col in (("收盤才進", "tab:gray"), ("M1 提前進", "tab:green")):
        a_.plot(D.index, D[f"M{k} {mode}"].cumsum(), color=col, label="bar close" if mode == "收盤才進" else "M1 early")
    a_.axvline(SPLIT, color="k", ls="--", lw=0.8)
    a_.axhline(0, color="grey", lw=0.6)
    a_.set_title(f"M{k} rep config: cum R")
    a_.legend(fontsize=7)
fig.tight_layout()
fig.savefig("part31_equity.png", dpi=110)
open("results_part31.txt", "w").write("\n".join(out) + "\n")
