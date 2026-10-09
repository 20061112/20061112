"""第四十九部分 k2：M1 版本的停損太小？改用 M15 的 ATR 當停損再比一次。

49k 用 M1 的 ATR14（14 分鐘）× 1.5 當停損，但持有長達 12 小時 → 停損只有幾美元，
R 被少數「停損很小、後來大漲」的單撐大，和美元損益、PF 不一致。
這裡把停損換成「最近一根已收完的 M15 ATR14 × 1.5」（和 M15 版本一樣大），角度的 ATR 也一起換成 M15 ATR，
其他不變。看 PF、每筆美元、總損益，以及每筆 R。
"""
import numpy as np
import pandas as pd
from load import load
from part49_friction import SPLIT, resample, atr, tst
from part49e_mua_detail import backtest
from part49g_zwin_m5 import z_sma, z_ewm

LS = [240, 720]
WS = [1000, 5000, 15000]


def m15_atr_on_m1(df):
    m15 = resample(df, "15min")
    a = atr(m15)
    a.index = a.index + pd.Timedelta("15min")      # 該根 M15 收完之後才可用
    return a.reindex(df.index, method="ffill")


def feats(df, L, A):
    c = df.close
    net = c - c.shift(L)
    tan = net.abs() / (A * np.sqrt(L / 15))         # 用 M15 單位的 √根數
    F = pd.DataFrame(index=df.index)
    F["atr"], F["dir"], F["cos"] = A, np.sign(net), 1 / np.sqrt(1 + tan * tan)
    F["er"], F["a"] = np.nan, np.nan
    return F


def cell(T, base_usd):
    if len(T) < 10:
        return f"n={len(T):4d}"
    m, t, _ = tst(T.R.to_numpy())
    p = T.pnl
    eq = p.cumsum()
    dd = (eq - np.maximum.accumulate(np.maximum(eq, 0))).min()
    return (f"n={len(T):4d} {m:+.3f}R(t={t:+.1f}) 每筆 {p.mean():+5.2f} 美元（超{p.mean() - base_usd:+.2f}） PF{p[p > 0].sum() / -p[p < 0].sum():.2f} "
            f"總{p.sum():+6.0f} DD{dd:5.0f} 前/後 {T.pnl[T.entry_time < SPLIT].mean():+.2f}/{T.pnl[T.entry_time >= SPLIT].mean():+.2f}")


def main():
    df = load("data/XAUUSD_M1_2026.csv")
    A = m15_atr_on_m1(df)
    out = ["第四十九部分 k2：M1 版本改用 M15 ATR 停損", "超額改用「每筆美元 − 基準每筆美元」", ""]
    rows = []
    for L in LS:
        F = feats(df, L, A)
        valid = F.atr.notna() & (F.dir.fillna(0) != 0) & F.cos.notna()
        B = backtest(df, F, valid, L)
        bu = B.pnl.mean()
        out.append(f"===== M1 L={L}（{L / 60:.0f} 小時）  基準 全部順勢 {cell(B, bu)} =====")
        for zn, zf in (("EWMA", z_ewm), ("SMA ", z_sma)):
            for w in WS:
                z = zf(F.cos, w)
                for thr in (-1.5, -2.0):
                    T = backtest(df, F, valid & (z <= thr), L)
                    out.append(f"  {zn} W={w:5d} z≤{thr:+.1f}  {cell(T, bu)}")
                    if len(T) >= 10:
                        p = T.pnl
                        rows.append((L, p.mean() - bu, p[p > 0].sum() / -p[p < 0].sum(), len(T)))
        out.append("")
    S = pd.DataFrame(rows, columns=["L", "excess_usd", "pf", "n"])
    out.append("===== 彙總（每 L 12 格）=====")
    out.append(S.groupby("L").agg(超額正比例=("excess_usd", lambda v: (v > 0).mean()), 平均超額美元=("excess_usd", "mean"),
                                  最差=("excess_usd", "min"), 平均PF=("pf", "mean"), 平均筆數=("n", "mean")).round(2).to_string())
    txt = "\n".join(out)
    print(txt)
    with open("results_part49k2.txt", "w") as fh:
        fh.write(txt + "\n")


if __name__ == "__main__":
    main()
