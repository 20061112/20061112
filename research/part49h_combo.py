"""第四十九部分 h：很陡 vs z(μ|a|)，以及兩者怎麼搭配（M15）。

訊號（都是順勢，出場同 49e）
  陡   ：z(cosθ) ≤ −2
  μa   ：z(ln μ + ln|a|) ≤ −2
  AND  ：兩個同時成立
  OR   ：任一成立
  只陡 / 只μa：一個成立、另一個不成立
z：SMA 與 EWMA，W = 500、1000；L = 40、48、56、64、96。另列兩訊號 K 棒的重疊率與相關。
"""
import numpy as np
import pandas as pd
from load import load
from part49_friction import SPLIT, resample, features, tst
from part49e_mua_detail import backtest
from part49g_zwin_m5 import z_sma, z_ewm

LS = [40, 48, 56, 64, 96]


def cell(T, base):
    if len(T) < 10:
        return f"n={len(T):3d}".ljust(40)
    m, t, _ = tst(T.R.to_numpy())
    p = T.pnl
    eq = p.cumsum()
    dd = (eq - np.maximum.accumulate(np.maximum(eq, 0))).min()
    h1 = T.R[T.entry_time < SPLIT].mean()
    h2 = T.R[T.entry_time >= SPLIT].mean()
    return (f"n={len(T):3d} {m:+.2f}R t={t:+.1f} 超{m - base:+.2f} PF{p[p > 0].sum() / -p[p < 0].sum():.2f} "
            f"總{p.sum():+5.0f} DD{dd:5.0f} ({h1:+.2f}/{h2:+.2f})")


def main():
    df = resample(load("data/XAUUSD_M1_2026.csv"), "15min")
    out = ["第四十九部分 h：很陡 vs z(μ|a|) 與搭配（M15，門檻 −2）", ""]
    rows = []
    for L in LS:
        F = features(df, L)
        valid = F.a.notna() & F.atr.notna() & (F.dir.fillna(0) != 0)
        B = backtest(df, F, valid, L)
        base = B.R.mean()
        out.append(f"===== L={L}  基準 全部順勢 n={len(B)} {base:+.3f}R 總{B.pnl.sum():+.0f} =====")
        cos = F.cos
        mua = np.log(F.mu) + np.log(F.a.abs().clip(lower=1e-6))
        for zn, zf in (("SMA", z_sma), ("EWMA", z_ewm)):
            for w in (500, 1000):
                s1 = valid & (zf(cos, w) <= -2)
                s2 = valid & (zf(mua, w) <= -2)
                ov = (s1 & s2).sum() / max(1, (s1 | s2).sum())
                out.append(f"  [{zn} W={w}] 陡 {int(s1.sum())} 根、μa {int(s2.sum())} 根、重疊 {ov:.0%}，"
                           f"z 相關 {zf(cos, w).corr(zf(mua, w)):+.2f}")
                for name, s in (("陡", s1), ("μa", s2), ("AND", s1 & s2), ("OR", s1 | s2),
                                ("只陡", s1 & ~s2), ("只μa", s2 & ~s1)):
                    T = backtest(df, F, s, L)
                    out.append(f"    {name:4s} {cell(T, base)}")
                    if len(T) >= 10:
                        rows.append((L, zn, w, name, T.R.mean() - base, len(T), T.pnl.sum()))
        out.append("")
    S = pd.DataFrame(rows, columns=["L", "z", "W", "sig", "excess", "n", "total"])
    out.append("===== 各訊號跨 L × z × W（20 格）=====")
    g = S.groupby("sig").agg(格數=("excess", "size"), 超額正比例=("excess", lambda v: (v > 0).mean()),
                             平均超額=("excess", "mean"), 最小=("excess", "min"), 平均筆數=("n", "mean"),
                             平均總損益=("total", "mean"))
    out.append(g.round(3).to_string())
    txt = "\n".join(out)
    print(txt)
    with open("results_part49h.txt", "w") as fh:
        fh.write(txt + "\n")


if __name__ == "__main__":
    main()
