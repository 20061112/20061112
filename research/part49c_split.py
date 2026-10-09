"""第四十九部分 c：把 μ 和 cosθ 拆開測（M15 L48、M5 L48、M5 L144 = 和 M15 L48 同樣 12 小時）。

拆解：tanθ = |淨變動|/(ATR·√L) = ER × P，其中 P = 路徑長 / (ATR·√L)
  → cosθ 裡和 ER 重疊的部分是 ER，剩下的是 P（這段來回走了多少路 / 正常波動，「活躍度」）。
  所以 cosθ 單獨測一次、再用 P 測「去掉 ER 之後的 cosθ」。
變數（都做滾動 z，W = 500；μ、P、|a| 右偏先取 ln）
  z_mu   = z(ln μ) = z(−ln ER)     大 = 黏
  z_cos  = z(cosθ)                 大 = 平
  z_P    = z(ln P)                 大 = 來回走很多路（相對 ATR）
  z_a    = z(a_rel)                大 = 順勢加速，小 = 減速
  z_mua  = z(ln μ + ln|a|)         μ·|a|（f 去掉 cosθ）
二維：μ 三區（z_mu ≤ −1 滑、中、≥ 1 黏）× a 方向（a_rel > 0 加速 / ≤ 0 減速）
      μ 三區 × cosθ 三區（z_cos ≤ −1 陡、中、≥ 1 平）
每格：順勢、逆勢交易（下一根開盤、1.5 ATR 停損、L 根出、扣點差、不重疊），前後兩半，基準 = 全部順勢。
"""
import numpy as np
import pandas as pd
from load import load
from part49_friction import SPLIT, resample, features, simulate, tst

CASES = [("15min", 48), ("5min", 48), ("5min", 144)]
W = 500
BINS = [(-np.inf, -2), (-2, -1), (-1, 1), (1, 2), (2, np.inf)]


def zs(s):
    return (s - s.rolling(W).mean()) / s.rolling(W).std()


def fmt(R, T):
    if len(R) < 5:
        return f"n={len(R):4d}            "
    m, t, _ = tst(R)
    h1, h2 = R[T < SPLIT], R[T >= SPLIT]
    f = lambda x: f"{x.mean():+.3f}" if len(x) else "  nan "
    return f"n={len(R):4d} {m:+.3f}R(t={t:+.1f}) H1 {f(h1)} H2 {f(h2)}"


def trades(df, F, msk, dirn, L):
    ii = np.where(msk & (dirn != 0))[0]
    Rf, Tf = simulate(df, F, ii, dirn[ii].astype(int), L)
    Rr, Tr = simulate(df, F, ii, -dirn[ii].astype(int), L)
    return fmt(Rf, Tf), fmt(Rr, Tr)


def main():
    m1 = load("data/XAUUSD_M1_2026.csv")
    out = ["第四十九部分 c：μ、cosθ 拆開測", ""]
    for tf, L in CASES:
        df = resample(m1, tf)
        F = features(df, L)
        c = df.close
        path = c.diff().abs().rolling(L).sum()
        P = path / (F.atr * np.sqrt(L))
        Z = pd.DataFrame({
            "z_mu": zs(np.log(F.mu)), "z_cos": zs(F.cos), "z_P": zs(np.log(P)),
            "z_a": zs(F.a_rel), "z_mua": zs(np.log(F.mu) + np.log(F.a.abs().clip(lower=1e-6))),
        })
        dirn = F.dir.to_numpy()
        ok = Z.notna().all(axis=1).to_numpy()
        bf, _ = trades(df, F, ok, dirn, L)
        out.append(f"===== {tf} L={L}   基準 全部順勢 {bf}")
        zz = Z[ok]
        r = lambda a, b: zz[a].rank().corr(zz[b].rank())
        out.append(f"  相關：mu~cos {r('z_mu','z_cos'):+.2f}  mu~P {r('z_mu','z_P'):+.2f}  "
                   f"cos~P {r('z_cos','z_P'):+.2f}  mu~a {r('z_mu','z_a'):+.2f}  cos~a {r('z_cos','z_a'):+.2f}")
        for v in Z.columns:
            out.append(f"  [{v}] 區間       佔比  | 順勢交易                              | 逆勢交易")
            z = Z[v].to_numpy()
            for lo, hi in BINS:
                msk = ok & (z > lo) & (z <= hi)
                fo, fa = trades(df, F, msk, dirn, L)
                out.append(f"    ({lo:+},{hi:+}] {msk[ok].mean():5.1%} | {fo} | {fa}")
        zm, za, zc = Z.z_mu.to_numpy(), F.a_rel.to_numpy(), Z.z_cos.to_numpy()
        mus = (("滑 z_mu≤−1", zm <= -1), ("中", (zm > -1) & (zm < 1)), ("黏 z_mu≥1", zm >= 1))
        out.append("  [二維] μ × a 方向")
        for mn, mm in mus:
            for an, am in (("加速", za > 0), ("減速", za <= 0)):
                msk = ok & mm & am
                fo, fa = trades(df, F, msk, dirn, L)
                out.append(f"    {mn:10s} {an} {msk[ok].mean():5.1%} | 順 {fo} | 逆 {fa}")
        out.append("  [二維] μ × cosθ")
        for mn, mm in mus:
            for cn, cm in (("陡 z_cos≤−1", zc <= -1), ("中", (zc > -1) & (zc < 1)), ("平 z_cos≥1", zc >= 1)):
                msk = ok & mm & cm
                if msk.sum() == 0:
                    out.append(f"    {mn:10s} {cn:10s}  0")
                    continue
                fo, fa = trades(df, F, msk, dirn, L)
                out.append(f"    {mn:10s} {cn:10s} {msk[ok].mean():5.1%} | 順 {fo} | 逆 {fa}")
        out.append("")
    txt = "\n".join(out)
    print(txt)
    with open("results_part49c.txt", "w") as fh:
        fh.write(txt + "\n")


if __name__ == "__main__":
    main()
