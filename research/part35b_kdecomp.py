"""第三十五部分 B：zf 的極值到底是 k 撐大的還是 x 撐大的？換 k 的定義比較。

k 版本
  ln1/ER     原始 ln(1/ER)（ER→0 時發散）
  ln1/ER≥.1  ln(1/max(ER, 0.1))（上限 2.3）
  1-ER       有界 0~1
  k=1        只看偏離 x（純 z-score 均值回歸，當對照組）
  x 極值+ER低 不相乘：|z(x)| ≥ thr 且 ER ≤ 0.2（把 k 當「盤整」濾網而不是乘數）
其他設定同 part35_spring.py（x = log 版，z = raw / std，進場 E0 / E1）。
"""
import numpy as np
import pandas as pd
from load import load
from spring import resample, features
from part35_spring import signals, fwd, simulate, fmt_R, tstat, HS, SPLIT


def zcols(F, f, W=100):
    sd = f.rolling(W).std()
    for zn, z in (("std", (f - f.rolling(W).mean()) / sd), ("raw", f / sd)):
        v = z.diff().ewm(span=3, adjust=False).mean()
        F[f"zz_{zn}"], F[f"zz_{zn}_v"], F[f"zz_{zn}_v1"], F[f"zz_{zn}_a"] = z, v, z.diff(), v.diff()


def main():
    m1 = load("data/XAUUSD_M1_2026.csv")
    lines = []
    P = lines.append
    for tf in ("5min", "15min", "1h"):
        df = resample(m1, tf)
        F = features(df)
        c, A = df.close.to_numpy(), F.atr.to_numpy()
        drift = {h: np.nanmean((np.r_[c[h:], [np.nan] * h] - c) / A) for h in HS}
        half_all = np.where(df.index < SPLIT, "H1", "H2")
        x = F.x_log
        ks = {"ln1/ER": F.k, "ln1/ER≥.1": np.log(1 / F.er.clip(lower=0.1)), "1-ER": 1 - F.er,
              "k=1": pd.Series(1.0, index=F.index), "x極值+ER低": None}
        P(f"\n=== {tf} ===")
        P(f"{'k':<11}{'z':<4}{'thr':>4} {'mode':<3} {'ER中位':>6} {'|x|中位':>7}"
          + "".join(f"{'k=' + str(h):>15}" for h in (5, 10, 20, 40)) + "   交易")
        thrs = (2.0, 2.5) if tf == "1h" else (2.5, 3.0)
        for kn, k in ks.items():
            zcols(F, -(k if k is not None else 1.0) * x)
            for zn in ("raw", "std"):
                for thr in thrs:
                    for mode in ("E0", "E1"):
                        sig = signals(F, f"zz_{zn}", thr, mode)
                        if k is None:
                            sig = sig[F.er.to_numpy()[sig.i] <= 0.2].reset_index(drop=True)
                        if len(sig) < 10:
                            continue
                        FR = fwd(df, F, sig, drift)
                        R = simulate(df, F, sig)
                        st = [tstat(FR[h].to_numpy()) for h in (5, 10, 20, 40)]
                        P(f"{kn:<11}{zn:<4}{thr:>4.1f} {mode:<3} {F.er.to_numpy()[sig.i].mean():>6.3f} "
                          f"{np.abs(x.to_numpy()[sig.i]).mean():>7.2f}"
                          + "".join(f"{m:>+8.3f}({t:>+5.1f})" for m, t, _ in st)
                          + "  " + fmt_R(R, half_all[sig.i]))
    txt = "\n".join(lines)
    print(txt)
    open("results_part35b.txt", "w").write(txt + "\n")


if __name__ == "__main__":
    main()
