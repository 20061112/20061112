"""第四十九部分 k：很陡順勢放到 M1。

只需要 cosθ，不算二次回歸（M1 L=720 用滑動視窗會吃掉好幾 GB 記憶體）。
  tanθ = |c − c_{−L}| / (ATR14 · √L)，cosθ = 1/√(1+tan²θ)
  z = SMA 或 EWMA z-score（W 根 M1）
  z ≤ thr → 下一根開盤順勢；停損 1.5 ATR14（M1）；L 根收盤出；扣點差；不重疊
L：48（48 分鐘）、144（2.4 小時）、240（4 小時）、720（12 小時，和 M15 L48 同時間）
W：1000（約 17 小時）、5000（約 3.5 天）、15000（約 10 天，和 M15 W=1000 同時間）
另報：點差佔停損距離（R）的比例，以及同 L 的「全部順勢」基準。
"""
import numpy as np
import pandas as pd
from load import load
from part49_friction import SPLIT, atr, tst
from part49e_mua_detail import backtest
from part49g_zwin_m5 import z_sma, z_ewm

LS = [48, 144, 240, 720]
WS = [1000, 5000, 15000]


def feats(df, L):
    c = df.close
    A = atr(df)
    net = c - c.shift(L)
    tan = net.abs() / (A * np.sqrt(L))
    F = pd.DataFrame(index=df.index)
    F["atr"], F["dir"], F["cos"] = A, np.sign(net), 1 / np.sqrt(1 + tan * tan)
    F["er"], F["a"] = np.nan, np.nan
    return F


def cell(T, base):
    if len(T) < 10:
        return f"n={len(T):4d}"
    m, t, _ = tst(T.R.to_numpy())
    p = T.pnl
    h1, h2 = T.R[T.entry_time < SPLIT].mean(), T.R[T.entry_time >= SPLIT].mean()
    return (f"n={len(T):4d} {m:+.3f}R t={t:+.1f} 超{m - base:+.3f} PF{p[p > 0].sum() / -p[p < 0].sum():.2f} "
            f"總{p.sum():+6.0f} ({h1:+.2f}/{h2:+.2f})")


def main():
    df = load("data/XAUUSD_M1_2026.csv")
    spr = df.spread * 0.01
    out = ["第四十九部分 k：很陡順勢放到 M1", ""]
    rows = []
    for L in LS:
        F = feats(df, L)
        valid = F.atr.notna() & (F.dir.fillna(0) != 0) & F.cos.notna()
        B = backtest(df, F, valid, L)
        base = B.R.mean()
        cost = (spr / (1.5 * F.atr)).median()
        out.append(f"===== M1 L={L}（{L / 60:.1f} 小時）  點差 ≈ {cost:.0%} R  基準 全部順勢 {cell(B, base)} =====")
        for zn, zf in (("EWMA", z_ewm), ("SMA ", z_sma)):
            for w in WS:
                z = zf(F.cos, w)
                for thr in (-1.5, -2.0):
                    T = backtest(df, F, valid & (z <= thr), L)
                    out.append(f"  {zn} W={w:5d} z≤{thr:+.1f}  {cell(T, base)}")
                    if len(T) >= 10:
                        rows.append((L, zn.strip(), w, thr, T.R.mean(), T.R.mean() - base, len(T), T.pnl.sum()))
        out.append("")
    S = pd.DataFrame(rows, columns=["L", "z", "W", "thr", "R", "excess", "n", "total"])
    out.append("===== 各 L 彙總（12 格：2 種 z × 3 個 W × 2 個門檻）=====")
    g = S.groupby("L").agg(格數=("R", "size"), R正比例=("R", lambda v: (v > 0).mean()), 平均R=("R", "mean"),
                           超額正比例=("excess", lambda v: (v > 0).mean()), 平均超額=("excess", "mean"),
                           最差超額=("excess", "min"), 平均筆數=("n", "mean"))
    out.append(g.round(3).to_string())
    txt = "\n".join(out)
    print(txt)
    with open("results_part49k.txt", "w") as fh:
        fh.write(txt + "\n")


if __name__ == "__main__":
    main()
