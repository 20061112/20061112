"""第四十九部分 g：z 的視窗長度（SMA vs EWMA）以及 M5 是否適用。

變數：很陡 z(cosθ)、滑 z(ln μ)、z(ln μ + ln|a|)
z 寫法：SMA z = (x − rolling mean_W) / rolling std_W；EWMA z = (x − ewm mean) / ewm std（span = W）
W：50、100、150、250、500、1000
門檻：很陡 / μ·|a| 用 ≤ −2 與 ≤ −1；滑用 ≤ −1（≤ −2 幾乎不會發生）
M15：L = 48；M5：L = 48、96、144、192（144 = 12 小時，和 M15 L48 同樣時間）
出場同 49e：下一根開盤順勢、1.5 ATR 停損、L 根收盤出、扣點差、不重疊。
每格報：筆數、每筆 R、t、超額（− 同 TF 同 L 全部順勢）、前後兩半。
"""
import numpy as np
import pandas as pd
from load import load
from part49_friction import SPLIT, resample, features, tst
from part49e_mua_detail import backtest

CASES = [("15min", 48), ("5min", 48), ("5min", 96), ("5min", 144), ("5min", 192)]
WS = [50, 100, 150, 250, 500, 1000]
VARS = [("很陡", lambda F: F.cos, (-1.0, -2.0)),
        ("滑", lambda F: np.log(F.mu), (-1.0,)),
        ("μ|a|", lambda F: np.log(F.mu) + np.log(F.a.abs().clip(lower=1e-6)), (-1.0, -2.0))]


def z_sma(x, w):
    return (x - x.rolling(w).mean()) / x.rolling(w).std()


def z_ewm(x, w):
    return (x - x.ewm(span=w, adjust=False).mean()) / x.ewm(span=w, adjust=False).std()


def cell(T, base):
    if len(T) < 10:
        return f"n={len(T):3d}".ljust(46)
    m, t, _ = tst(T.R.to_numpy())
    h1 = T.R[T.entry_time < SPLIT].mean()
    h2 = T.R[T.entry_time >= SPLIT].mean()
    return f"n={len(T):3d} {m:+.2f}R t={t:+.1f} 超{m - base:+.2f} ({h1:+.2f}/{h2:+.2f})"


def main():
    m1 = load("data/XAUUSD_M1_2026.csv")
    out = ["第四十九部分 g：z 視窗（SMA / EWMA）與 M5", "每格：筆數 每筆R t 超額（前半/後半）", ""]
    summary = []
    for tf, L in CASES:
        df = resample(m1, tf)
        F = features(df, L)
        valid = F.a.notna() & F.atr.notna() & (F.dir.fillna(0) != 0)
        B = backtest(df, F, valid, L)
        base = B.R.mean()
        out.append(f"===== {tf} L={L}  基準 全部順勢 n={len(B)} {base:+.3f}R =====")
        for name, f, thrs in VARS:
            x = f(F)
            for thr in thrs:
                out.append(f"  [{name} ≤ {thr:+.0f}]")
                for zn, zf in (("SMA ", z_sma), ("EWMA", z_ewm)):
                    for w in WS:
                        T = backtest(df, F, valid & (zf(x, w) <= thr), L)
                        s = cell(T, base)
                        out.append(f"    {zn} W={w:4d}  {s}")
                        if len(T) >= 10:
                            summary.append((tf, L, name, thr, zn.strip(), w, T.R.mean() - base, len(T)))
        out.append("")
    S = pd.DataFrame(summary, columns=["tf", "L", "var", "thr", "z", "W", "excess", "n"])
    out.append("===== 超額 > 0 的比例（每個 TF/L × 變數，跨 W 與 SMA/EWMA）=====")
    g = S.groupby(["tf", "L", "var", "thr"]).excess.agg(["count", lambda v: (v > 0).mean(), "mean", "min", "max"])
    g.columns = ["格數", "超額>0比例", "平均超額", "最小", "最大"]
    out.append(g.round(3).to_string())
    out.append("")
    out.append("===== SMA vs EWMA 平均超額（全部格子）=====")
    out.append(S.groupby(["tf", "z"]).excess.mean().round(3).to_string())
    out.append("")
    out.append("===== 依 W 的平均超額（全部變數）=====")
    out.append(S.pivot_table(index=["tf", "L"], columns="W", values="excess", aggfunc="mean").round(3).to_string())
    txt = "\n".join(out)
    print(txt)
    with open("results_part49g.txt", "w") as fh:
        fh.write(txt + "\n")
    S.to_csv("part49g_grid.csv", index=False, float_format="%.4f")


if __name__ == "__main__":
    main()
