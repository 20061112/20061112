"""第三十六部分：用 OU 過程（f = -kx 的隨機版本）估彈簧，加上「彈性限度」判斷彈簧是否斷掉。

OU / AR(1)：在最近 W 根上回歸 Δp_t = a + b·p_{t-1} + ε
  k = θ = -b（每根）            彈簧強度，由資料估出來
  t_b = b 的 t 值                彈簧是否存在（Dickey-Fuller 5% 臨界值約 -2.86）
  μ* = -a/b                      彈簧的平衡點
  σ_eq = σ_ε / sqrt(1-φ²)，φ=1+b  平衡分佈的標準差
  z = (p - μ*) / σ_eq            偏離（已經是 z-score，不需要再標準化）
  f = -θ·z；半衰期 HL = ln2 / -ln(φ)
彈性限度：把 z 當成彈簧振子，m·v² + θ·z² = θ·A²
  v = z 的速度（EMA3），A = sqrt(z² + m·v²/θ) = 照目前動能還會擺到多遠；A 超過限度 L → 視為彈簧會斷
  m ∈ {1, 相對σ}（第 35 部分 m=相對σ 分得最開）
進場：|z| 首次 ≥ thr（E0），或之後 z 的速度轉向（E1）；方向 = 往 μ*。
出場：收盤碰到進場時的 μ*、停損 1.5 ATR、或持有上限（固定 40 根 / c×HL）；扣點差；不重疊。
另外 [C]：第 35 部分架構下 k 的不同寫法（1/ER、ln(1/ER)、1/ln(1/ER)）。
"""
import numpy as np
import pandas as pd
from load import load
from spring import resample, features
from part35_spring import signals, fwd, simulate, fmt_R, tstat, HS, SPLIT

TFS = ["5min", "15min", "1h"]
SL = 1.5


def ou(df, W):
    p = df.close - df.close.iloc[0]
    x, y = p.shift(), p.diff()
    mx, my = x.rolling(W).mean(), y.rolling(W).mean()
    sxx = (x * x).rolling(W).mean() - mx * mx
    sxy = (x * y).rolling(W).mean() - mx * my
    syy = (y * y).rolling(W).mean() - my * my
    b = sxy / sxx
    a = my - b * mx
    rv = (syy - b * sxy).clip(lower=1e-12) * W / (W - 2)
    t = b / np.sqrt(rv / (W * sxx))
    phi = 1 + b
    ok = (b < 0) & (phi > 0)
    out = pd.DataFrame(index=df.index)
    out["theta"] = (-b).where(ok)
    out["t"] = t
    out["mu"] = (-a / b).where(ok) + df.close.iloc[0]
    out["seq"] = np.sqrt(rv / (1 - phi ** 2)).where(ok)
    out["z"] = (df.close - out.mu) / out.seq
    out["hl"] = (np.log(2) / -np.log(phi)).where(ok)
    v = out.z.diff().ewm(span=3, adjust=False).mean()
    out["v"] = v
    return out


def ou_signals(O, thr, mode, L=10):
    z, v = O.z.to_numpy(), O.v.to_numpy()
    out = []
    armed, side, t0 = False, 0, 0
    for t in range(1, len(z)):
        if np.isnan(z[t]) or np.isnan(z[t - 1]):
            armed = False
            continue
        if not armed:
            if abs(z[t]) >= thr > abs(z[t - 1]):
                armed, side, t0 = True, int(np.sign(z[t])), t
                if mode == "E0":
                    out.append((t, -side))
                    armed = False
            continue
        if side * z[t] < 1.0 or t - t0 > L:
            armed = False
        elif side * v[t] < 0:
            out.append((t, -side))
            armed = False
    return pd.DataFrame(out, columns=["i", "dir"])


def sim_ou(df, F, O, sig, hold):
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    spr = df.spread.to_numpy(float) * 0.01
    A, MU = F.atr.to_numpy(), O.mu.to_numpy()
    n = len(c)
    R = np.full(len(sig), np.nan)
    bars = np.full(len(sig), np.nan)
    busy = -1
    for q, (t, d) in enumerate(zip(sig.i, sig.dir)):
        if t + 1 >= n or t + 1 <= busy:
            continue
        e, risk, tgt = o[t + 1], SL * A[t], MU[t]
        stop = e - d * risk
        H = int(hold[q])
        out = None
        for j in range(t + 1, min(t + 1 + H, n)):
            if (l[j] <= stop) if d == 1 else (h[j] >= stop):
                out, x = j, stop
                break
            if d * (c[j] - tgt) >= 0:
                out, x = j, c[j]
                break
        if out is None:
            out = min(t + H, n - 1)
            x = c[out]
        busy = out
        R[q] = ((x - e) * d - spr[t + 1]) / risk
        bars[q] = out - t
    return R, bars


def ttt(df, O, sig, cap=300):
    """實際碰到進場時 μ* 所需根數（不設停損）。"""
    c, MU = df.close.to_numpy(), O.mu.to_numpy()
    res = []
    for t, d in zip(sig.i, sig.dir):
        j = t + 1
        while j < len(c) and j - t <= cap and d * (c[j] - MU[t]) < 0:
            j += 1
        res.append(j - t if j < len(c) and j - t <= cap else np.nan)
    return np.array(res)


def line(FR, R, half, msk=None):
    msk = np.ones(len(R), bool) if msk is None else msk
    st = [tstat(FR[h].to_numpy()[msk]) for h in (3, 10, 20, 40)]
    return ("".join(f"{m:>+8.3f}({t:>+5.1f})" for m, t, _ in st) + "  " + fmt_R(R[msk], half[msk]))


def main():
    m1 = load("data/XAUUSD_M1_2026.csv")
    lines = []
    P = lines.append
    for tf in TFS:
        df = resample(m1, tf)
        F = features(df)
        c, A = df.close.to_numpy(), F.atr.to_numpy()
        drift = {h: np.nanmean((np.r_[c[h:], [np.nan] * h] - c) / A) for h in HS}
        half_all = np.where(df.index < SPLIT, "H1", "H2")
        P(f"\n{'=' * 120}\n=== {tf}  bars={len(df)}")
        for W in (100, 200):
            O = ou(df, W)
            P(f"\n--- OU 視窗 W={W}：θ 中位數 {O.theta.median():.3f}  半衰期中位數 {O.hl.median():.0f} 根  "
              f"t<-2 佔 {np.mean(O.t < -2):.0%}  t<-2.86 佔 {np.mean(O.t < -2.86):.0%}  σ_eq/ATR 中位數 {(O.seq / F.atr).median():.2f}")
            P(f"[A] 彈簧有沒有作用：依 t_b 分組（前瞻 k=3/10/20/40 去漂移 ATR ＋ 交易：碰 μ* / 停損 1.5ATR / 40 根）")
            for thr in (2.0, 2.5):
                for mode in ("E0", "E1"):
                    sig = ou_signals(O, thr, mode)
                    if len(sig) < 20:
                        continue
                    FR = fwd(df, F, sig, drift)
                    R, _ = sim_ou(df, F, O, sig, np.full(len(sig), 40))
                    half = half_all[sig.i]
                    tb = O.t.to_numpy()[sig.i]
                    for lab, m in (("全部", np.ones(len(sig), bool)), ("t<-2.86 有彈簧", tb < -2.86),
                                   ("-2.86~-2", (tb >= -2.86) & (tb < -2)), ("t≥-2 沒彈簧", tb >= -2)):
                        if m.sum() < 5:
                            continue
                        P(f"  thr{thr} {mode} {lab:<14} n={m.sum():4d}" + line(FR, R, half, m))
            # 以 thr2 E0、t<-2 為主，研究彈性限度與持有時間
            sig = ou_signals(O, 2.0, "E0")
            sig = sig[O.t.to_numpy()[sig.i] < -2].reset_index(drop=True)
            if len(sig) < 20:
                continue
            FR = fwd(df, F, sig, drift)
            half = half_all[sig.i]
            R, _ = sim_ou(df, F, O, sig, np.full(len(sig), 40))
            sub = O.iloc[sig.i.to_numpy()].reset_index(drop=True)
            P(f"[B] 彈性限度（thr2 E0 且 t<-2，n={len(sig)}）：A = sqrt(z² + m·v²/θ)")
            for mn, mm in (("1", np.ones(len(sig))), ("σ", F.m_sig.to_numpy()[sig.i])):
                Aamp = np.sqrt(sub.z ** 2 + mm * sub.v ** 2 / sub.theta).to_numpy()
                for lab, m in (("A≤3", Aamp <= 3), ("3<A≤4", (Aamp > 3) & (Aamp <= 4)), ("A>4", Aamp > 4)):
                    if m.sum() >= 5:
                        P(f"  m={mn} {lab:<6} n={m.sum():4d}" + line(FR, R, half, m))
            tt = ttt(df, O, sig)
            P(f"[D] 持有時間：實際碰到 μ* 根數中位數 {np.nanmedian(tt):.0f}（300 根內沒碰到 {np.mean(np.isnan(tt)):.0%}），"
              f"半衰期中位數 {sub.hl.median():.0f}；Spearman(實際, HL) = "
              f"{pd.Series(tt).rank().corr(sub.hl.rank()):+.3f}")
            for lab, hold in (("固定 40 根", np.full(len(sig), 40)),
                              ("1×HL", np.clip(np.ceil(sub.hl.to_numpy()), 1, 300)),
                              ("2×HL", np.clip(np.ceil(2 * sub.hl.to_numpy()), 1, 300))):
                Rh, bars = sim_ou(df, F, O, sig, hold)
                P(f"  持有上限 {lab:<8} 平均持有 {np.nanmean(bars):5.1f} 根  " + fmt_R(Rh, half))

        # [C] 第 35 部分架構下 k 的寫法
        P("\n[C] 第 35 部分架構（x=log、z 不減均值、E0）換 k 的寫法")
        x = F.x_log
        er = F.er
        for kn, k in (("1/ER", 1 / er), ("ln(1/ER)", np.log(1 / er)), ("1/ln(1/ER)", 1 / np.log(1 / er))):
            f = -k.replace([np.inf, -np.inf], np.nan) * x
            sd = f.rolling(100).std()
            z = f / sd
            v = z.diff().ewm(span=3, adjust=False).mean()
            F["zk"], F["zk_v"], F["zk_v1"], F["zk_a"] = z, v, z.diff(), v.diff()
            for thr in ((2.0, 2.5) if tf == "1h" else (2.5, 3.0)):
                sig = signals(F, "zk", thr, "E0")
                if len(sig) < 10:
                    continue
                FR = fwd(df, F, sig, drift)
                R = simulate(df, F, sig)
                P(f"  {kn:<11} thr{thr} ER平均 {er.to_numpy()[sig.i].mean():.3f} |x|平均 {np.abs(x.to_numpy()[sig.i]).mean():.2f} "
                  f"n={len(sig):4d}" + line(FR, R, half_all[sig.i]))
    txt = "\n".join(lines)
    print(txt)
    open("results_part36.txt", "w").write(txt + "\n")


if __name__ == "__main__":
    main()
