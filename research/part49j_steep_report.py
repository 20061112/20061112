"""第四十九部分 j：「很陡順勢」完整交易數據。

主設定（49g / 49h 建議區間的中間，事先固定）
  M15；每根收盤往回擷取 L = 48 根（12 小時）
  tanθ = |c_t − c_{t−48}| / (ATR14 · √48)；cosθ = 1 / √(1 + tan²θ)
  z = (cosθ − EWMA_1000(cosθ)) / EWMSTD_1000(cosθ)
  z ≤ −2 → 下一根開盤順著這 48 根的淨方向進場
  停損 1.5 × ATR14；沒停損就持有 48 根（12 小時）收盤出場；扣點差；持倉中不再進場
輸出：part49j_trades.csv、part49j_stats.txt、part49j_equity.png、part49j_angles.csv（角度分佈）
"""
import numpy as np
import pandas as pd
from load import load
from part49_friction import resample, features, tst
from part49e_mua_detail import backtest, stats
from part49g_zwin_m5 import z_sma, z_ewm

TF, L, W, THR = "15min", 48, 1000, -2.0


def signal(F, zf=z_ewm, w=W, thr=THR):
    z = zf(F.cos, w)
    return z, (z <= thr) & F.a.notna() & F.atr.notna() & (F.dir.fillna(0) != 0)


def one(T, base=None):
    if len(T) < 5:
        return f"n={len(T)}"
    m, t, _ = tst(T.R.to_numpy())
    p = T.pnl
    eq = p.cumsum()
    dd = (eq - np.maximum.accumulate(np.maximum(eq, 0))).min()
    ex = f" 超額 {m - base:+.2f}" if base is not None else ""
    return f"n={len(T):3d} {m:+.3f}R(t={t:+.1f}){ex} PF {p[p > 0].sum() / -p[p < 0].sum():.2f} 總 {p.sum():+6.0f} DD {dd:5.0f}"


def main():
    m1 = load("data/XAUUSD_M1_2026.csv")
    df = resample(m1, TF)
    F = features(df, L)
    z, sig = signal(F)
    T = backtest(df, F, sig, L)
    tan = np.sqrt(1 / F.cos ** 2 - 1)
    T["tan"] = tan.reindex(T.signal_time).to_numpy()
    T["theta_deg"] = np.degrees(np.arctan(T.tan))
    T["cos"] = F.cos.reindex(T.signal_time).to_numpy()
    T["z"] = z.reindex(T.signal_time).to_numpy()
    T.to_csv("part49j_trades.csv", index=False, float_format="%.4f")

    lines = ["第四十九部分 j：很陡順勢 —— M15 L48，EWMA z(cosθ)（W=1000）≤ −2", ""]
    lines.append(f"訊號 K 棒數: {int(sig.sum())}（佔 {sig.mean():.1%}），實際進場 {len(T)} 筆")
    eq, dd = stats(T, lines)
    p = T.pnl.sort_values(ascending=False)
    lines.append(f"最好 5 / 10 筆佔總損益: {p[:5].sum() / T.pnl.sum():.0%} / {p[:10].sum() / T.pnl.sum():.0%}；拿掉最好 10 筆剩 {T.pnl.sum() - p[:10].sum():+.0f}")
    valid = F.a.notna() & F.atr.notna() & (F.dir.fillna(0) != 0)
    B = backtest(df, F, valid, L)
    base = B.R.mean()
    pb = B.pnl.sort_values(ascending=False)
    lines.append(f"基準（每根都順勢、同樣出場）: {one(B)}；最好 10 筆佔 {pb[:10].sum() / B.pnl.sum():.0%}")

    # R 分佈
    bins = [-np.inf, -0.9, 0, 1, 2, 3, 5, 10, np.inf]
    labs = ["停損(≈−1R)", "−0.9~0", "0~1R", "1~2R", "2~3R", "3~5R", "5~10R", ">10R"]
    cnt = pd.cut(T.R, bins, labels=labs).value_counts().reindex(labs)
    lines += ["", "每筆 R 分佈:"] + [f"  {k:10s} {v:3d} 筆 ({v / len(T):.0%})" for k, v in cnt.items()]
    # 每月
    lines += ["", "每月（依進場時間）:", "  月份     筆數 勝率    總損益   每筆R   最大單筆"]
    for mth, s in T.groupby(T.entry_time.dt.to_period("M")):
        lines.append(f"  {mth}  {len(s):3d}  {np.mean(s.pnl > 0):4.0%} {s.pnl.sum():+8.1f}  {s.R.mean():+.3f}  {s.pnl.max():+7.1f}")
    # 星期 / 時段
    lines += ["", "進場星期:"]
    wd = ["一", "二", "三", "四", "五"]
    for k, s in T.groupby(T.entry_time.dt.dayofweek):
        lines.append(f"  週{wd[k] if k < 5 else k}  n={len(s):3d} 勝率 {np.mean(s.pnl > 0):.0%} 每筆 {s.R.mean():+.3f}R 總 {s.pnl.sum():+.0f}")
    lines += ["", "進場小時（伺服器時間）:"]
    hb = pd.cut(T.entry_time.dt.hour, [-1, 7, 13, 18, 23], labels=["0-7 亞洲", "8-13 歐洲", "14-18 美盤前段", "19-23 美盤後段"])
    for k, s in T.groupby(hb, observed=True):
        lines.append(f"  {k:12s} n={len(s):3d} 勝率 {np.mean(s.pnl > 0):.0%} 每筆 {s.R.mean():+.3f}R 總 {s.pnl.sum():+.0f}")
    # 角度
    allt = tan[valid]
    lines += ["", "角度（tanθ = 淨變動 / (ATR·√48)；θ 以度表示）:",
              f"  全部 K 棒：θ 中位數 {np.degrees(np.arctan(allt.median())):.1f}°，90% 分位 {np.degrees(np.arctan(allt.quantile(0.9))):.1f}°",
              f"  進場訊號：θ 中位數 {T.theta_deg.median():.1f}°（範圍 {T.theta_deg.min():.1f}° ~ {T.theta_deg.max():.1f}°），cosθ 中位數 {T['cos'].median():.3f}"]
    tb = pd.qcut(T.theta_deg, 3, labels=["較緩", "中", "最陡"])
    for k, s in T.groupby(tb, observed=True):
        lines.append(f"  訊號內 θ {k}（{s.theta_deg.min():.0f}°~{s.theta_deg.max():.0f}°）n={len(s)} 每筆 {s.R.mean():+.3f}R 勝率 {np.mean(s.pnl > 0):.0%}")
    ang = pd.DataFrame({"theta_deg": np.degrees(np.arctan(allt)).round(1)})
    ang.to_csv("part49j_angles.csv", index=False)
    # 資金
    lines += ["", "資金（1 萬美元起、不複利、每筆固定風險）:"]
    for r in (0.0025, 0.005, 0.01):
        units = r * 10000 / T.risk
        pnl = T.pnl * units
        e = pnl.cumsum()
        d = (e - np.maximum.accumulate(np.maximum(e, 0))).min()
        lines.append(f"  每筆風險 {r:.2%}: 總報酬 {e.iloc[-1] / 100:+.0f}%，最大回撤（平倉）{d / 100:.1f}%")
    # 鄰近參數
    lines += ["", "鄰近參數（同樣出場）:"]
    for LL in (40, 48, 56, 64, 96):
        FF = features(df, LL)
        vv = FF.a.notna() & FF.atr.notna() & (FF.dir.fillna(0) != 0)
        bb = backtest(df, FF, vv, LL).R.mean()
        for zn, zf in (("EWMA", z_ewm), ("SMA ", z_sma)):
            for w in (500, 1000):
                for thr in (-1.5, -2.0):
                    _, sg = signal(FF, zf, w, thr)
                    mark = " ← 主設定" if (LL, zn.strip(), w, thr) == (L, "EWMA", W, THR) else ""
                    lines.append(f"  L={LL:3d} {zn} W={w:4d} z≤{thr:+.1f}: {one(backtest(df, FF, sg, LL), bb)}{mark}")
    txt = "\n".join(lines)
    print(txt)
    with open("part49j_stats.txt", "w") as fh:
        fh.write(txt + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "WenQuanYi Zen Hei"
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(3, 1, figsize=(11, 9), gridspec_kw={"height_ratios": [3, 1.3, 1.5]})
    ax[0].plot(T.exit_time, eq, color="#2a6fdb", lw=1.6, label=f"很陡順勢（{len(T)} 筆，每筆 1 盎司）")
    ax[0].plot(B.exit_time, B.pnl.cumsum(), color="#999", lw=1, label=f"基準：全部順勢（{len(B)} 筆）")
    ax[0].set_title("M15 L48  EWMA z(cosθ) ≤ -2 順勢：累積損益（美元/盎司）")
    ax[0].legend(loc="upper left")
    ax[0].grid(alpha=0.3)
    ax[1].fill_between(T.exit_time, dd, 0, color="#d9534f", alpha=0.6, step="post")
    ax[1].set_title("回撤（平倉後）")
    ax[1].grid(alpha=0.3)
    ax[2].hist(T.R, bins=40, color="#5b9", edgecolor="white")
    ax[2].axvline(0, color="k", lw=0.8)
    ax[2].set_title("每筆 R 分佈")
    fig.tight_layout()
    fig.savefig("part49j_equity.png", dpi=110)


if __name__ == "__main__":
    main()
