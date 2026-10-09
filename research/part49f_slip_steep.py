"""第四十九部分 f：M15「滑」與「很陡」順勢的交易穩健性（同 49e 的出場與統計）。

  滑   ：z(ln μ) ≤ thr（μ = 1/ER，W = 500）
  很陡 ：z(cosθ) ≤ thr（tanθ = |淨變動| / (ATR·√L)）
  進出場同 49e：下一根開盤順這段方向；停損 1.5 ATR；L 根收盤出；扣點差；不重疊。
主設定 L = 48、thr = −1（49c 的分組），另列 thr = −2；敏感度：門檻、W、L，以及每個 L 的基準（全部順勢）。
"""
import numpy as np
import pandas as pd
from load import load
from part49_friction import resample, features, tst
from part49e_mua_detail import backtest, stats

TF = "15min"


def zs(s, w):
    return (s - s.rolling(w).mean()) / s.rolling(w).std()


VARS = {"滑 z(ln μ)": lambda F: np.log(F.mu), "很陡 z(cosθ)": lambda F: F.cos}


def sig(F, f, w=500, thr=-1.0):
    return (zs(f(F), w) <= thr) & (F.dir.fillna(0) != 0)


def one(S):
    if len(S) < 5:
        return f"n={len(S):3d}（太少）"
    mm, tt, _ = tst(S.R.to_numpy())
    p = S.pnl
    return f"n={len(S):3d} {mm:+.3f}R(t={tt:+.1f}) PF {p[p > 0].sum() / -p[p < 0].sum():.2f} 總 {p.sum():+6.0f}"


def main():
    m1 = load("data/XAUUSD_M1_2026.csv")
    df = resample(m1, TF)
    Fs = {L: features(df, L) for L in (32, 40, 48, 56, 64, 96)}
    base = {L: backtest(df, F, (F.dir.fillna(0) != 0) & F.a.notna() & F.atr.notna(), L) for L, F in Fs.items()}
    lines = ["第四十九部分 f：M15「滑」「很陡」順勢的穩健性", ""]
    F = Fs[48]
    for name, f in VARS.items():
        for thr in (-1.0, -2.0):
            T = backtest(df, F, sig(F, f, thr=thr), 48)
            T.to_csv(f"part49f_trades_{'slip' if '滑' in name else 'steep'}_{int(-thr)}.csv", index=False, float_format="%.4f")
            lines.append(f"===== {name} ≤ {thr:+.0f}，L = 48 =====")
            stats(T, lines)
            p = T.pnl.sort_values(ascending=False)
            lines.append(f"最好 10 筆佔總損益 {p[:10].sum() / T.pnl.sum():.0%}；拿掉最好 10 筆剩 {T.pnl.sum() - p[:10].sum():+.0f}")
            mon = T.groupby(T.entry_time.dt.to_period("M")).pnl.sum()
            lines.append("每月總損益: " + "  ".join(f"{k.month}月 {v:+.0f}" for k, v in mon.items()))
            lines.append(f"基準（全部順勢）: {one(base[48])}")
            lines.append("")
        lines.append(f"----- {name} 敏感度 -----")
        for thr in (-0.5, -1.0, -1.5, -2.0, -2.5):
            lines.append(f"  L=48 門檻 {thr:+.1f}: {one(backtest(df, F, sig(F, f, thr=thr), 48))}")
        for w in (250, 500, 1000, 2000):
            lines.append(f"  L=48 門檻 -1 W={w:4d}: {one(backtest(df, F, sig(F, f, w=w), 48))}")
        for L, FF in Fs.items():
            r1 = backtest(df, FF, sig(FF, f, thr=-1.0), L)
            r2 = backtest(df, FF, sig(FF, f, thr=-2.0), L)
            lines.append(f"  L={L:3d}  門檻-1: {one(r1)} | 門檻-2: {one(r2)} | 基準 {base[L].R.mean():+.3f}R")
        lines.append("")
    txt = "\n".join(lines)
    print(txt)
    with open("part49f_stats.txt", "w") as fh:
        fh.write(txt + "\n")


if __name__ == "__main__":
    main()
