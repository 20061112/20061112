"""第四十九部分 i：反過來 —— z(cosθ) 偏大（很平）時逆勢，可不可以？

cosθ ≤ 1 而且左偏，z 往上的尾巴很短，先列各門檻實際出現的比例。
訊號：z(cosθ) ≥ thr（thr = 1、1.5、2），以及對照 z(μ|a|) ≥ 2。
方向：逆勢 = 和這 L 根的淨方向相反；順勢一起列出當對照。出場同 49e。
M15，L = 40 / 48 / 56 / 64 / 96 × SMA / EWMA × W = 500 / 1000。
"""
import numpy as np
import pandas as pd
from load import load
from part49_friction import SPLIT, resample, features, tst
from part49e_mua_detail import backtest
from part49g_zwin_m5 import z_sma, z_ewm

LS = [40, 48, 56, 64, 96]


def run(df, F, sig, L, flip):
    G = F.copy()
    if flip:
        G["dir"] = -G["dir"]
    return backtest(df, G, sig, L)


def main():
    df = resample(load("data/XAUUSD_M1_2026.csv"), "15min")
    out = ["第四十九部分 i：z 偏大（很平）逆勢？（M15）", ""]
    rows = []
    for L in LS:
        F = features(df, L)
        valid = F.a.notna() & F.atr.notna() & (F.dir.fillna(0) != 0)
        base_f = backtest(df, F, valid, L).R.mean()
        base_r = run(df, F, valid, L, True).R.mean()
        out.append(f"===== L={L}  全部順勢 {base_f:+.3f}R / 全部逆勢 {base_r:+.3f}R =====")
        mua = np.log(F.mu) + np.log(F.a.abs().clip(lower=1e-6))
        for zn, zf in (("SMA", z_sma), ("EWMA", z_ewm)):
            for w in (500, 1000):
                zc, zm = zf(F.cos, w), zf(mua, w)
                share = "、".join(f"≥{t}: {(zc[valid] >= t).mean():.1%}" for t in (1, 1.5, 2))
                out.append(f"  [{zn} W={w}] z(cosθ) 佔比 {share}")
                for name, s in (("很平 z≥1", zc >= 1), ("很平 z≥1.5", zc >= 1.5), ("很平 z≥2", zc >= 2),
                                ("μa z≥2", zm >= 2)):
                    s = valid & s
                    parts = []
                    for lab, flip, base in (("逆", True, base_r), ("順", False, base_f)):
                        T = run(df, F, s, L, flip)
                        if len(T) < 10:
                            parts.append(f"{lab} n={len(T):3d}".ljust(44))
                            continue
                        m, t, _ = tst(T.R.to_numpy())
                        h1 = T.R[T.entry_time < SPLIT].mean()
                        h2 = T.R[T.entry_time >= SPLIT].mean()
                        p = T.pnl
                        parts.append(f"{lab} n={len(T):3d} {m:+.2f}R t={t:+.1f} PF{p[p > 0].sum() / -p[p < 0].sum():.2f} ({h1:+.2f}/{h2:+.2f})")
                        rows.append((L, zn, w, name, lab, m, m - base, len(T)))
                    out.append(f"    {name:10s} " + " | ".join(parts))
        out.append("")
    S = pd.DataFrame(rows, columns=["L", "z", "W", "sig", "side", "R", "excess", "n"])
    out.append("===== 跨 L × z × W 彙總（R > 0 比例、平均 R、相對同方向全部進場的超額）=====")
    g = S.groupby(["sig", "side"]).agg(格數=("R", "size"), R正比例=("R", lambda v: (v > 0).mean()),
                                       平均R=("R", "mean"), 最差R=("R", "min"), 平均超額=("excess", "mean"),
                                       平均筆數=("n", "mean"))
    out.append(g.round(3).to_string())
    txt = "\n".join(out)
    print(txt)
    with open("results_part49i.txt", "w") as fh:
        fh.write(txt + "\n")


if __name__ == "__main__":
    main()
