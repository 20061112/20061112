"""第四十九部分 d：和 ER 不同的「角度」能不能加進來用。

案例：M15 L48、M5 L48、M5 L144。基準 = 每根都順著這段 L 根的淨方向（下一根開盤進、1.5 ATR 停損、L 根出、扣點差、不重疊）。
候選（都只用當根收盤以前的資料）
  big    大週期同向：往回 4L 根的淨方向是否和這段同向（同向 / 反向）
  turn   轉折角：後半段角度 − 前半段角度（順著段方向；θ = atan(淨變動 / (ATR·√(L/2)))）
         > 0 = 越走越陡（加速轉進），< 0 = 越走越平（轉彎）
  pos    位置：收盤在這段高低區間的位置（順著段方向，1 = 在段的盡頭極值，0 = 回到起點那側）
  steep  對照：z(cosθ) ≤ −1（第 49c 的「很陡」）
先列每個候選和 ER 的 rank 相關（越接近 0 越「和 ER 不同」），再分組看順勢交易，並和基準比。
最後把候選加在「很陡」上面，看能不能讓「很陡」三組都更穩。
"""
import numpy as np
import pandas as pd
from load import load
from part49_friction import SPLIT, resample, features, simulate, tst

CASES = [("15min", 48), ("5min", 48), ("5min", 144)]
W = 500


def fmt(R, T, base):
    if len(R) < 5:
        return f"n={len(R):4d}"
    m, t, _ = tst(R)
    h1, h2 = R[T < SPLIT], R[T >= SPLIT]
    return f"n={len(R):4d} {m:+.3f}R(t={t:+.1f}) 超額{m - base:+.3f} H1 {h1.mean():+.3f} H2 {h2.mean():+.3f}"


def run(df, F, msk, dirn, L):
    ii = np.where(msk & (dirn != 0))[0]
    return simulate(df, F, ii, dirn[ii].astype(int), L)


def main():
    m1 = load("data/XAUUSD_M1_2026.csv")
    out = ["第四十九部分 d：和 ER 不同的角度", ""]
    for tf, L in CASES:
        df = resample(m1, tf)
        F = features(df, L)
        c, A = df.close, F.atr
        dirn = F.dir.to_numpy()
        h = L // 2
        th1 = np.arctan((c.shift(h) - c.shift(L)) / (A * np.sqrt(h)))
        th2 = np.arctan((c - c.shift(h)) / (A * np.sqrt(h)))
        turn = (th2 - th1) * F.dir
        hi, lo = df.high.rolling(L).max(), df.low.rolling(L).min()
        pos = ((c - lo) / (hi - lo)).where(F.dir > 0, (hi - c) / (hi - lo))
        big = np.sign(c - c.shift(4 * L)) * F.dir
        zcos = (F.cos - F.cos.rolling(W).mean()) / F.cos.rolling(W).std()
        ok = (turn.notna() & pos.notna() & big.notna() & zcos.notna()).to_numpy()
        Rb, Tb = run(df, F, ok, dirn, L)
        base = Rb.mean()
        out.append(f"===== {tf} L={L}  基準 全部順勢 {fmt(Rb, Tb, base)}")
        er = F.er[ok].rank()
        out.append(f"  和 ER 的相關：big {big[ok].rank().corr(er):+.2f}  turn {turn[ok].rank().corr(er):+.2f}  "
                   f"pos {pos[ok].rank().corr(er):+.2f}  zcos {zcos[ok].rank().corr(er):+.2f}")
        t, p, b, s = turn.to_numpy(), pos.to_numpy(), big.to_numpy(), zcos.to_numpy() <= -1
        groups = [
            ("big 同向", b > 0), ("big 反向", b < 0),
            ("turn 越走越平 (<−0.3)", t < -0.3), ("turn 中", (t >= -0.3) & (t <= 0.3)), ("turn 越走越陡 (>0.3)", t > 0.3),
            ("pos 回檔 (<0.6)", p < 0.6), ("pos 中 (0.6~0.9)", (p >= 0.6) & (p < 0.9)), ("pos 極值 (≥0.9)", p >= 0.9),
            ("很陡", s), ("很陡 + big 同向", s & (b > 0)), ("很陡 + big 反向", s & (b < 0)),
            ("很陡 + pos<0.9", s & (p < 0.9)), ("很陡 + pos≥0.9", s & (p >= 0.9)),
            ("big 同向 + pos<0.6", (b > 0) & (p < 0.6)),
        ]
        for name, g in groups:
            R, T = run(df, F, ok & g, dirn, L)
            out.append(f"  {name:22s} 佔比 {(ok & g).sum() / ok.sum():5.1%} | {fmt(R, T, base)}")
        out.append("")
    txt = "\n".join(out)
    print(txt)
    with open("results_part49d.txt", "w") as fh:
        fh.write(txt + "\n")


if __name__ == "__main__":
    main()
