"""第四十九部分 b：把 f 做成 z-score，看偏大 / 偏小時這段行情之後是什麼狀態。

z = (f − 過去 W 根平均) / 過去 W 根標準差（W = 500，只用當根以前的資料）
另測 z_log：先取 ln(f) 再做 z（f 右偏，ln 後比較對稱，兩邊尾巴才有意義）。
分組：z ≤ −2、−2~−1、−1~1、1~2、≥ 2。
每組看：cont（順段方向報酬，ATR）、absmv（之後走多遠）、er_nx（下一段 ER，越高 = 越趨勢）、
       交易順勢 / 逆勢（下一根開盤進、1.5 ATR 停損、H 根出、扣點差），並和「全部順勢」基準比。
"""
import numpy as np
import pandas as pd
from load import load
from part49_friction import TFS, LS, SPLIT, resample, features, outcomes, simulate, tst

W = 500
BINS = [(-np.inf, -2), (-2, -1), (-1, 1), (1, 2), (2, np.inf)]


def zs(s):
    return (s - s.rolling(W).mean()) / s.rolling(W).std()


def fmt(R, T):
    if len(R) < 5:
        return f"n={len(R):4d}"
    m, t, _ = tst(R)
    h1, h2 = R[T < SPLIT], R[T >= SPLIT]
    return (f"n={len(R):4d} {m:+.3f}R(t={t:+.1f}) H1 {h1.mean() if len(h1) else np.nan:+.3f} "
            f"H2 {h2.mean() if len(h2) else np.nan:+.3f}")


def main():
    m1 = load("data/XAUUSD_M1_2026.csv")
    out = ["第四十九部分 b：f 的 z-score（W=500）", ""]
    for tf in TFS:
        df = resample(m1, tf)
        for L in LS:
            F = features(df, L)
            O = outcomes(df, F, L)
            dirn = F.dir.to_numpy()
            base_i = np.where(F.f.notna() & (dirn != 0))[0]
            Rb, Tb = simulate(df, F, base_i, dirn[base_i].astype(int), L)
            out.append(f"===== {tf} L={L}   基準 全部順勢 {fmt(Rb, Tb)}")
            for zn, z in (("z", zs(F.f)), ("z_log", zs(np.log(F.f.clip(lower=1e-6))))):
                out.append(f"  [{zn}]  區間      佔比   cont        absmv  er_nx | 順勢交易 | 逆勢交易")
                zz = z.to_numpy()
                for lo, hi in BINS:
                    msk = (zz > lo) & (zz <= hi) & (dirn != 0)
                    ii = np.where(msk)[0]
                    if len(ii) == 0:
                        out.append(f"    ({lo:+},{hi:+}] 0")
                        continue
                    idx = ii[::L]  # 不重疊取樣看分佈
                    Os = O.iloc[idx]
                    m, t, _ = tst(Os.cont.to_numpy())
                    Rf, Tf = simulate(df, F, ii, dirn[ii].astype(int), L)
                    Rr, Tr = simulate(df, F, ii, -dirn[ii].astype(int), L)
                    out.append(f"    ({lo:+},{hi:+}] {msk.mean():5.1%} {m:+.2f}(t={t:+.1f}) "
                               f"{Os.absmv.mean():5.2f} {Os.er_nx.mean():.3f} | {fmt(Rf, Tf)} | {fmt(Rr, Tr)}")
            out.append("")
    txt = "\n".join(out)
    print(txt)
    with open("results_part49b.txt", "w") as fh:
        fh.write(txt + "\n")


if __name__ == "__main__":
    main()
