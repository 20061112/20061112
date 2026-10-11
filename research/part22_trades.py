"""第二十二部分：H1「短陡長陡同向」共振的逐筆交易。
參數沿用 part22_angle.py 的 H1 8/64、短 ≥ 0.8、長 ≥ 0.8 百分位、抱 8 根，不再調整。
訊號在第 t 根 H1 收完時成立 → 第 t+1 根開盤進場 → 進場後第 8 根收盤出場（持有 8 小時）。
持倉中不重複進場。損益扣點差（進場當根 spread × 0.01）。輸出 part22_trades.csv、part22_trades.png。
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from load import load
from part22_angle import build, quant_rank

HOLD = 8


def trades(h1):
    X = build(h1, 8, 64, 16)
    rs, rl = quant_rank(X.angs.abs()), quant_rank(X.angl.abs())
    sig = (rs >= .8) & (rl >= .8) & (np.sign(X.angs) == np.sign(X.angl))
    d = h1.loc[X.index]
    o, hi, lo, c = (d[k].to_numpy() for k in ("open", "high", "low", "close"))
    pos_sig = np.flatnonzero(sig.to_numpy())
    rows, free = [], 0
    for p in pos_sig:
        e = p + 1
        x = e + HOLD - 1
        if e < free or x >= len(d):
            continue
        s = 1 if X.angs.iat[p] > 0 else -1
        atr = X.atr.iat[p]
        cost = d.spread.iat[e] * 0.01
        pnl = s * (c[x] - o[e]) - cost
        mfe = (hi[e:x + 1].max() - o[e]) if s > 0 else (o[e] - lo[e:x + 1].min())
        mae = (o[e] - lo[e:x + 1].min()) if s > 0 else (hi[e:x + 1].max() - o[e])
        rows.append(dict(signal_time=d.index[p], entry_time=d.index[e], exit_time=d.index[x] + pd.Timedelta("1h"),
                         dir="多" if s > 0 else "空", entry=o[e], exit=c[x], atr=round(atr, 2),
                         ang_short=round(X.angs.iat[p], 1), ang_long=round(X.angl.iat[p], 1),
                         pnl_usd=round(pnl, 2), pnl_atr=round(pnl / atr, 3),
                         mfe_atr=round(mfe / atr, 3), mae_atr=round(mae / atr, 3), cost_usd=round(cost, 2)))
        free = x + 1
    return pd.DataFrame(rows)


def stats(T, label):
    r = T.pnl_atr
    eq = r.cumsum()
    gp, gl = r[r > 0].sum(), -r[r < 0].sum()
    print(f"{label:<10s} 筆數 {len(r):4d}  勝率 {(r > 0).mean():.2f}  平均 {r.mean():+.3f} ATR  t {r.mean() / r.std() * np.sqrt(len(r)):+.2f}  "
          f"PF {gp / gl:.2f}  平均賺 {r[r > 0].mean():+.2f} / 平均賠 {r[r < 0].mean():+.2f}  累計 {r.sum():+.1f} ATR  "
          f"最大回撤 {(eq.cummax() - eq).max():.1f} ATR  最大連虧 {max_streak(r < 0)}  美元合計 {T.pnl_usd.sum():+.0f}")


def max_streak(b):
    m = k = 0
    for v in b:
        k = k + 1 if v else 0
        m = max(m, k)
    return m


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    m15 = load("data/XAUUSD_M15_2023_2026.csv")
    h1 = m15.resample("1h", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "tickvol": "sum", "spread": "max"}).dropna()
    T = trades(h1)
    T.to_csv("part22_trades.csv", index=False)
    print("H1 共振（短 8 / 長 64，雙 ≥ 0.8 百分位，同向）：下一根開盤進、持有 8 小時收盤出，扣點差\n")
    stats(T, "全部")
    for k, g in T.groupby("dir"):
        stats(g, k)
    for y, g in T.groupby(T.entry_time.dt.year):
        stats(g, str(y))
    print("\n逐月（ATR 合計 / 筆數）")
    m = T.groupby(T.entry_time.dt.to_period("M")).pnl_atr.agg(["sum", "count"])
    print(m.round(2).T.to_string())
    print(f"正月 {(m['sum'] > 0).sum()}/{len(m)}")
    print("\n單筆分布（ATR）：", T.pnl_atr.quantile([.05, .25, .5, .75, .95]).round(2).to_dict())
    print("MFE 中位", T.mfe_atr.median().round(2), " MAE 中位", T.mae_atr.median().round(2),
          " MAE ≥ 1 ATR 比例", (T.mae_atr >= 1).mean().round(2))
    print("前 10 大賺 / 賠占總和：", round(T.pnl_atr.nlargest(10).sum(), 1), round(T.pnl_atr.nsmallest(10).sum(), 1),
          "去掉最大 10 筆後平均", round(T.pnl_atr.drop(T.pnl_atr.nlargest(10).index).mean(), 3))
    print("進場時段（伺服器時間）平均 ATR / 筆數：")
    print(T.groupby(T.entry_time.dt.hour).pnl_atr.agg(["mean", "count"]).round(2).T.to_string())
    print("\n最近 15 筆：")
    print(T.tail(15)[["entry_time", "dir", "entry", "exit", "pnl_usd", "pnl_atr", "mfe_atr", "mae_atr"]].to_string(index=False))

    fig, ax = plt.subplots(2, 1, figsize=(11, 7), gridspec_kw={"height_ratios": [2, 1]})
    for k, g in T.groupby("dir"):
        ax[0].plot(g.entry_time, g.pnl_atr.cumsum(), label=f"{'long' if k == '多' else 'short'} only", lw=1)
    ax[0].plot(T.entry_time, T.pnl_atr.cumsum(), label="all", lw=2, color="k")
    ax[0].set_title("H1 short/long angle resonance - cumulative ATR (spread deducted)")
    ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].bar(m.index.to_timestamp(), m["sum"], width=20, color=np.where(m["sum"] > 0, "tab:green", "tab:red"))
    ax[1].set_title("monthly ATR"); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig("part22_trades.png", dpi=110)
