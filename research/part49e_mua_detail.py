"""第四十九部分 e：M15 L48「z(ln μ + ln|a|) ≤ −2 → 順勢」的詳細交易數據。

規則（和 49c 相同）
  M15；每根收盤往回擷取 L = 48 根（12 小時）
  μ = 1/ER；a = 二次回歸曲率 2·b2·L²/ATR14；g = ln μ + ln|a|（= ln(μ·|a|)）
  z = (g − 過去 500 根平均) / 過去 500 根標準差
  z ≤ −2（很滑、曲率很小 = 走得乾淨又直）→ 下一根開盤順著這段 48 根的淨方向進場
  停損 1.5 × ATR14（M15）；沒停損就持有 48 根（12 小時）收盤出場；扣點差；不重疊（持倉中不再進場）
輸出：part49e_trades.csv、part49e_stats.txt、part49e_equity.png、敏感度（門檻 / W / L）
"""
import numpy as np
import pandas as pd
from load import load
from part49_friction import SPLIT, resample, features, tst

TF, L, W, THR, SL = "15min", 48, 500, -2.0, 1.5


def zs(s, w):
    return (s - s.rolling(w).mean()) / s.rolling(w).std()


def signal(F, w=W, thr=THR):
    g = np.log(F.mu) + np.log(F.a.abs().clip(lower=1e-6))
    z = zs(g, w)
    return z, (z <= thr) & (F.dir.fillna(0) != 0)


def backtest(df, F, sig, L):
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    spr = df.spread.to_numpy(float) * 0.01
    A, d_all = F.atr.to_numpy(), F.dir.to_numpy()
    idx = df.index
    n = len(c)
    rows, busy = [], -1
    for t in np.where(sig.to_numpy())[0]:
        if t + 1 >= n or t + 1 <= busy:
            continue
        d = int(d_all[t])
        e = o[t + 1]
        risk = SL * A[t]
        stop = e - d * risk
        last = min(t + L, n - 1)
        x, out, why = None, last, "時間"
        mfe = mae = 0.0
        for j in range(t + 1, last + 1):
            adv = (l[j] - e) * d if d == 1 else (e - h[j])
            fav = (h[j] - e) if d == 1 else (e - l[j])
            mae, mfe = min(mae, adv), max(mfe, fav)
            if (l[j] <= stop) if d == 1 else (h[j] >= stop):
                x, out, why = stop, j, "停損"
                mae = -risk
                break
        if x is None:
            x = c[out]
        busy = out
        pnl = (x - e) * d - spr[t + 1]
        rows.append(dict(signal_time=idx[t], entry_time=idx[t + 1], exit_time=idx[out] + pd.Timedelta(TF),
                         dir="多" if d == 1 else "空", entry=e, exit=x, stop=stop, risk=risk,
                         atr=A[t], er=F.er.iat[t], a=F.a.iat[t], exit_reason=why,
                         bars=out - t, pnl=pnl, R=pnl / risk, mfe=mfe, mae=mae,
                         mfe_R=mfe / risk, mae_R=mae / risk))
    return pd.DataFrame(rows)


def stats(T, lines):
    p, R = T.pnl.to_numpy(), T.R.to_numpy()
    eq = p.cumsum()
    dd = eq - np.maximum.accumulate(np.r_[0, eq])[1:]
    win = p > 0
    pf = p[win].sum() / -p[~win].sum()
    m, t, _ = tst(R)
    streak = lambda b: max((len(s) for s in "".join("1" if x else "0" for x in b).split("0")), default=0)
    hours = (T.exit_time - T.entry_time).dt.total_seconds() / 3600
    lines += [
        f"期間: {T.entry_time.min():%Y-%m-%d} ~ {T.exit_time.max():%Y-%m-%d}",
        f"交易筆數: {len(T)}（多 {np.sum(T.dir == '多')} / 空 {np.sum(T.dir == '空')}）",
        f"交易週數 ≈ {(T.entry_time.max() - T.entry_time.min()).days / 7:.1f}，平均每週 {len(T) / ((T.entry_time.max() - T.entry_time.min()).days / 7):.1f} 筆",
        f"勝率: {win.mean():.1%}",
        f"平均賺 / 平均賠（美元/盎司）: {p[win].mean():+.2f} / {p[~win].mean():+.2f}，賺賠比 {p[win].mean() / -p[~win].mean():.2f}",
        f"每筆期望: {p.mean():+.2f} 美元/盎司，{m:+.3f}R（t={t:+.1f}），R 中位數 {np.median(R):+.2f}",
        f"PF: {pf:.2f}",
        f"總損益（每筆 1 盎司）: {p.sum():+.1f}；總 R: {R.sum():+.1f}",
        f"最大回撤（平倉後）: {dd.min():.1f} 美元/盎司（{(dd / T.risk.mean()).min():.1f} 個平均 R）",
        f"總損益 / 最大回撤: {p.sum() / -dd.min():.1f}",
        f"最長連虧 / 連勝: {streak(~win)} / {streak(win)}",
        f"最大單筆賺 / 賠: {p.max():+.1f} / {p.min():+.1f}（{R.max():+.2f}R / {R.min():+.2f}R）",
        f"出場：停損 {np.mean(T.exit_reason == '停損'):.0%} / 時間到 {np.mean(T.exit_reason == '時間'):.0%}",
        f"停損單 平均 R {T.R[T.exit_reason == '停損'].mean():+.2f}；時間出場 平均 R {T.R[T.exit_reason == '時間'].mean():+.2f}",
        f"平均持倉: {hours.mean():.1f} 小時（中位數 {hours.median():.1f}）；贏單 {hours[win].mean():.1f} / 輸單 {hours[~win].mean():.1f}",
        f"停損距離 中位數: {T.risk.median():.1f} 美元（1.5 ATR），範圍 {T.risk.min():.1f} ~ {T.risk.max():.1f}",
        f"MFE 中位數 {T.mfe_R.median():.2f}R；MAE 中位數 {T.mae_R.median():.2f}R",
        f"輸單中曾浮盈 ≥ 1R 的比例: {np.mean(T.mfe_R[~win] >= 1):.0%}",
        f"贏單中曾浮虧 ≥ 0.5R 的比例: {np.mean(T.mae_R[win] <= -0.5):.0%}",
        f"訊號當下 ER 中位數 {T.er.median():.2f}，|a| 中位數 {T.a.abs().median():.2f} ATR",
    ]
    for nm, msk in (("前半 1/2~5/31", T.entry_time < SPLIT), ("後半 6/1~10/8", T.entry_time >= SPLIT),
                    ("多單", T.dir == "多"), ("空單", T.dir == "空")):
        s = T[msk]
        pp = s.pnl.to_numpy()
        mm, tt, _ = tst(s.R.to_numpy())
        lines.append(f"  {nm:14s} n={len(s):3d} 勝率 {np.mean(pp > 0):.0%} 每筆 {pp.mean():+6.2f} 美元 {mm:+.3f}R(t={tt:+.1f}) "
                     f"PF {pp[pp > 0].sum() / -pp[pp < 0].sum():.2f} 總 {pp.sum():+.0f}")
    return eq, dd


def main():
    m1 = load("data/XAUUSD_M1_2026.csv")
    df = resample(m1, TF)
    F = features(df, L)
    z, sig = signal(F)
    T = backtest(df, F, sig, L)
    T.to_csv("part49e_trades.csv", index=False, float_format="%.4f")
    lines = ["第四十九部分 e：M15 L48  z(ln μ + ln|a|) ≤ −2 → 順勢", ""]
    lines.append(f"訊號 K 棒數: {int(sig.sum())}（佔 {sig.mean():.1%}），實際進場 {len(T)} 筆（持倉中的訊號略過）")
    eq, dd = stats(T, lines)
    # 基準
    allsig = (F.dir.fillna(0) != 0) & z.notna()
    B = backtest(df, F, allsig, L)
    mb, tb, _ = tst(B.R.to_numpy())
    lines.append(f"基準（每根都順勢、同樣出場）: n={len(B)} {B.pnl.mean():+.2f} 美元 {mb:+.3f}R(t={tb:+.1f}) 總 {B.pnl.sum():+.0f}")
    # 每月
    lines += ["", "每月（依進場時間）:", "  月份    筆數 勝率   總損益   每筆R"]
    for mth, s in T.groupby(T.entry_time.dt.to_period("M")):
        lines.append(f"  {mth}  {len(s):3d}  {np.mean(s.pnl > 0):4.0%} {s.pnl.sum():+8.1f}  {s.R.mean():+.3f}")
    # 進場時段
    lines += ["", "進場小時（伺服器時間）分組:"]
    hb = pd.cut(T.entry_time.dt.hour, [-1, 7, 13, 18, 23], labels=["0-7", "8-13", "14-18", "19-23"])
    for k, s in T.groupby(hb, observed=True):
        lines.append(f"  {k:6s} n={len(s):3d} 勝率 {np.mean(s.pnl > 0):.0%} 每筆 {s.R.mean():+.3f}R 總 {s.pnl.sum():+.0f}")
    # 敏感度
    lines += ["", "敏感度（每筆 R / t / 筆數；其他參數不變）:"]
    for thr in (-1.0, -1.5, -2.0, -2.5):
        _, sg = signal(F, W, thr)
        S = backtest(df, F, sg, L)
        mm, tt, _ = tst(S.R.to_numpy())
        lines.append(f"  門檻 z ≤ {thr:+.1f}: n={len(S):3d} {mm:+.3f}R(t={tt:+.1f}) PF {S.pnl[S.pnl > 0].sum() / -S.pnl[S.pnl < 0].sum():.2f}")
    for w in (250, 500, 1000, 2000):
        _, sg = signal(F, w)
        S = backtest(df, F, sg, L)
        mm, tt, _ = tst(S.R.to_numpy())
        lines.append(f"  z 視窗 W = {w:4d}: n={len(S):3d} {mm:+.3f}R(t={tt:+.1f})")
    for LL in (32, 40, 48, 56, 64, 96):
        FF = features(df, LL)
        _, sg = signal(FF)
        S = backtest(df, FF, sg, LL)
        Bb = backtest(df, FF, (FF.dir.fillna(0) != 0) & FF.a.notna() & FF.atr.notna(), LL)
        mm, tt, _ = tst(S.R.to_numpy())
        lines.append(f"  L = {LL:3d}: n={len(S):3d} {mm:+.3f}R(t={tt:+.1f})  基準全部順勢 {Bb.R.mean():+.3f}R")
    txt = "\n".join(lines)
    print(txt)
    with open("part49e_stats.txt", "w") as fh:
        fh.write(txt + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "WenQuanYi Zen Hei"
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(3, 1, figsize=(11, 9), gridspec_kw={"height_ratios": [3, 1.3, 1.5]})
    ax[0].plot(T.exit_time, eq, color="#2a6fdb", lw=1.6, label="本策略（每筆 1 盎司）")
    ax[0].plot(B.exit_time, B.pnl.cumsum(), color="#999", lw=1, label="基準：全部順勢")
    ax[0].axvline(SPLIT, color="#c33", ls="--", lw=0.8)
    ax[0].set_title("M15 L48  z(ln μ+ln|a|) ≤ -2 順勢：累積損益（美元/盎司）")
    ax[0].legend(loc="upper left")
    ax[0].grid(alpha=0.3)
    ax[1].fill_between(T.exit_time, dd, 0, color="#d9534f", alpha=0.6, step="post")
    ax[1].set_title("回撤（平倉後）")
    ax[1].grid(alpha=0.3)
    ax[2].hist(T.R, bins=40, color="#5b9", edgecolor="white")
    ax[2].axvline(0, color="k", lw=0.8)
    ax[2].set_title("每筆 R 分佈")
    fig.tight_layout()
    fig.savefig("part49e_equity.png", dpi=110)


if __name__ == "__main__":
    main()
