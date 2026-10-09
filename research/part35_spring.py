"""第三十五部分：彈簧策略 f = -k·x（k = ln(1/ER)，x = 偏離/ATR），zf 極值 + 速度/加速度轉向進場，
逃逸速度判斷彈簧是否斷掉，偏離面積輔助持有時間 / 方向。

資料：data/XAUUSD_M1_2026.csv，重採樣成 M5 / M15 / H1。
進場模式
  E0 極值：|zf| 首次 ≥ thr 那根收盤就進
  E1 速度：|zf| ≥ thr 後，等到 zf 的速度（EMA3 平滑）轉向（side·v < 0）才進
  E2 速度+加速度：side·Δzf < 0 且 side·a < 0（未平滑速度已轉向且正在加速回頭）
  arm 之後 10 根內沒觸發、或 |zf| 掉回 1 以下就作廢。方向 = sign(zf)（= 彈簧力方向，往均值）。
評估
  1. 前瞻報酬（往訊號方向，ATR 單位），扣掉同週期所有 K 棒的平均漂移（多空對稱）。
  2. 交易：下一根開盤進；停損 1.5 ATR；收盤回到均值出場，或 H 根收盤出場；扣點差；不重疊。
  3. 前後兩半：1/2~5/31 vs 6/1~10/8。
"""
import numpy as np
import pandas as pd
from load import load
from spring import resample, features

TFS = ["5min", "15min", "1h"]
HS = [1, 3, 5, 10, 20, 40]
SPLIT = pd.Timestamp("2026-06-01")
THRS = [2.0, 2.5, 3.0]
SL = 1.5
HOLD = 40


def signals(F, zc, thr, mode, thr_out=1.0, L=10):
    z = F[zc].to_numpy()
    v = F[zc + "_v"].to_numpy()
    v1 = F[zc + "_v1"].to_numpy()
    a = F[zc + "_a"].to_numpy()
    out = []
    armed, side, t0 = False, 0, 0
    for t in range(1, len(z)):
        if np.isnan(z[t]) or np.isnan(a[t]):
            continue
        if not armed:
            if abs(z[t]) >= thr and abs(z[t - 1]) < thr:
                armed, side, t0 = True, int(np.sign(z[t])), t
                if mode == "E0":
                    out.append((t, side))
                    armed = False
            continue
        if side * z[t] < thr_out or t - t0 > L:
            armed = False
            continue
        fire = (side * v[t] < 0) if mode == "E1" else (side * v1[t] < 0 and side * a[t] < 0)
        if fire:
            out.append((t, side))
            armed = False
    return pd.DataFrame(out, columns=["i", "dir"])


def fwd(df, F, sig, drift):
    c = df.close.to_numpy()
    A = F.atr.to_numpy()
    i, d = sig.i.to_numpy(), sig.dir.to_numpy()
    res = {}
    for h in HS:
        r = np.full(len(i), np.nan)
        ok = i + h < len(c)
        r[ok] = (c[i[ok] + h] - c[i[ok]]) / A[i[ok]] * d[ok] - d[ok] * drift[h]
        res[h] = r
    return pd.DataFrame(res, index=sig.index)


def time_to_mean(df, F, sig, cap=200):
    c, mu = df.close.to_numpy(), F.mu.to_numpy()
    out = []
    for i, d in zip(sig.i, sig.dir):
        j = i + 1
        while j < len(c) and j - i <= cap and d * (c[j] - mu[j]) < 0:
            j += 1
        out.append(j - i if j < len(c) and j - i <= cap else np.nan)
    return np.array(out)


def simulate(df, F, sig, hold=HOLD, sl=SL, exit_mean=True):
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    spr = df.spread.to_numpy(float) * 0.01
    A, mu = F.atr.to_numpy(), F.mu.to_numpy()
    n = len(c)
    R = np.full(len(sig), np.nan)
    busy = -1
    hold = np.broadcast_to(np.asarray(hold), (len(sig),))
    for q, (t, d) in enumerate(zip(sig.i, sig.dir)):
        if t + 1 >= n or t + 1 <= busy:
            continue
        e = o[t + 1]
        risk = sl * A[t]
        stop = e - d * risk
        out, x = None, None
        for j in range(t + 1, min(t + 1 + int(hold[q]), n)):
            if (l[j] <= stop) if d == 1 else (h[j] >= stop):
                out, x = j, stop
                break
            if exit_mean and d * (c[j] - mu[j]) >= 0:
                out, x = j, c[j]
                break
        if out is None:
            out = min(t + int(hold[q]), n - 1)
            x = c[out]
        busy = out
        R[q] = ((x - e) * d - spr[t + 1]) / risk
    return R


def fmt_R(r, half):
    out = []
    for hv in ("H1", "H2"):
        rr = r[(half == hv) & ~np.isnan(r)]
        pf = rr[rr > 0].sum() / -rr[rr < 0].sum() if (rr < 0).any() else np.nan
        out.append(f"{hv} n={len(rr):4d} {rr.mean():+.3f}R PF{pf:.2f}")
    rr = r[~np.isnan(r)]
    return f"n={len(rr):4d} avg={rr.mean():+.3f}R win={np.mean(rr > 0):.0%} | " + " | ".join(out)


def tstat(v):
    v = v[~np.isnan(v)]
    return (v.mean(), v.mean() / v.std(ddof=1) * np.sqrt(len(v)), len(v)) if len(v) > 5 else (np.nan, np.nan, len(v))


def main():
    m1 = load("data/XAUUSD_M1_2026.csv")
    lines = []
    P = lines.append
    grid = []
    for tf in TFS:
        df = resample(m1, tf)
        F = features(df)
        c = df.close.to_numpy()
        A = F.atr.to_numpy()
        drift = {h: np.nanmean((np.r_[c[h:], [np.nan] * h] - c) / A) for h in HS}
        half_all = np.where(df.index < SPLIT, "H1", "H2")
        P(f"\n{'=' * 110}\n=== {tf}  bars={len(df)}  漂移(ATR) " +
          " ".join(f"k{h}={drift[h]:+.3f}" for h in HS))
        P(f"ER 中位數 {F.er.median():.2f}；k 中位數 {F.k.median():.2f}；|d| 中位數 {F.d.abs().median():.2f} ATR")
        P("\n[A] 進場模式比較：前瞻報酬（去漂移，ATR，t 值）＋ 交易（停損 1.5ATR、回均值或 40 根出場）")
        P(f"{'x':<4}{'z':<4}{'thr':>4} {'mode':<3}" + "".join(f"{'k=' + str(h):>15}" for h in HS) + "   交易")
        for xn in ("log", "lin"):
            for zn in ("std", "raw"):
                zc = f"z_{xn}_{zn}"
                for thr in THRS:
                    for mode in ("E0", "E1", "E2"):
                        sig = signals(F, zc, thr, mode)
                        if len(sig) < 10:
                            continue
                        FR = fwd(df, F, sig, drift)
                        R = simulate(df, F, sig)
                        half = half_all[sig.i]
                        st = [tstat(FR[h].to_numpy()) for h in HS]
                        P(f"{xn:<4}{zn:<4}{thr:>4.1f} {mode:<3}" +
                          "".join(f"{m:>+8.3f}({t:>+5.1f})" for m, t, _ in st) + "  " + fmt_R(R, half))
                        rr = R[~np.isnan(R)]
                        h1 = R[(half == "H1") & ~np.isnan(R)]
                        h2 = R[(half == "H2") & ~np.isnan(R)]
                        grid.append(dict(tf=tf, x=xn, z=zn, thr=thr, mode=mode, n=len(rr), R=rr.mean(),
                                         R_H1=h1.mean(), R_H2=h2.mean(),
                                         **{f"fwd{h}": st[j][0] for j, h in enumerate(HS)},
                                         **{f"t{h}": st[j][1] for j, h in enumerate(HS)}))

        # ---- 以 log / thr2.5 為基準，研究逃逸、面積、週期 ----
        for zn in ("std", "raw"):
            zc = f"z_log_{zn}"
            for mode in ("E0", "E1"):
                sig = signals(F, zc, 2.5 if tf != "1h" else 2.0, mode)
                if len(sig) < 30:
                    continue
                FR = fwd(df, F, sig, drift)
                R = simulate(df, F, sig)
                half = half_all[sig.i]
                sub = F.iloc[sig.i.to_numpy()].reset_index(drop=True)
                ttm = time_to_mean(df, F, sig)
                P(f"\n[B] 逃逸速度（{zc} {mode}，n={len(sig)}）：依「本段最大逃逸比」分組，fwd 去漂移 ATR / 交易 R")
                for mn in ("1", "vol", "sig"):
                    for kind in ("", "_max"):
                        e = sub[f"esc_log_{mn}{kind}"].to_numpy()
                        for lab, msk in (("esc≤1", e <= 1), ("esc>1", e > 1)):
                            m10, t10, nn = tstat(FR[10].to_numpy()[msk])
                            m40, t40, _ = tstat(FR[40].to_numpy()[msk])
                            rr = R[msk & ~np.isnan(R)]
                            P(f"  m={mn:<3} {('當根' if kind == '' else '本段最大'):<4} {lab:<6} n={nn:4d} "
                              f"k10 {m10:+.3f}({t10:+.1f})  k40 {m40:+.3f}({t40:+.1f})  "
                              f"交易 {rr.mean() if len(rr) else np.nan:+.3f}R (n={len(rr)})  "
                              f"回均值根數中位 {np.nanmedian(ttm[msk]) if msk.any() else np.nan:.0f}")
                P(f"\n[C] 偏離面積 |area|（/價格）三分位（{zc} {mode}）")
                for col in ("area", "area_atr", "seg_len"):
                    v = sub[col].abs().to_numpy()
                    qs = np.nanquantile(v, [1 / 3, 2 / 3])
                    for b, msk in enumerate((v <= qs[0], (v > qs[0]) & (v <= qs[1]), v > qs[1])):
                        st = [tstat(FR[h].to_numpy()[msk]) for h in (5, 10, 20, 40)]
                        rr = R[msk & ~np.isnan(R)]
                        P(f"  {col:<8} Q{b + 1} ({np.nanmin(v[msk]):.4g}~{np.nanmax(v[msk]):.4g}) n={msk.sum():4d} " +
                          " ".join(f"k{h}:{m:+.3f}({t:+.1f})" for h, (m, t, _) in zip((5, 10, 20, 40), st)) +
                          f"  交易 {rr.mean():+.3f}R  回均值根數中位 {np.nanmedian(ttm[msk]):.0f}"
                          f"  未回均值(200根) {np.mean(np.isnan(ttm[msk])):.0%}")
                P(f"\n[D] 持有時間：實際回到均值根數 vs 理論 T/4、|area|、seg_len 的 Spearman 相關（{zc} {mode}）")
                tt = pd.Series(ttm)
                for col in ("T_1", "T_vol", "T_sig", "area", "area_atr", "seg_len", "k", "er"):
                    rho = tt.rank().corr(sub[col].abs().rank())
                    P(f"  {col:<9} rho={rho:+.3f}   中位數 {sub[col].abs().median():.4g}")
                P(f"  實際回均值根數中位數 {np.nanmedian(ttm):.0f}；理論 T_1/4 中位數 {(sub['T_1'] / 4).median():.1f}")
                # 用 T/4 當持有上限 vs 固定 40 根
                for mult in (1, 2, 4):
                    hl = np.clip(np.nan_to_num(sub["T_vol"].to_numpy() / 4 * mult, nan=HOLD), 1, 200)
                    Rt = simulate(df, F, sig, hold=np.ceil(hl).astype(int))
                    P(f"  持有上限 = {mult}×T_vol/4 → {fmt_R(Rt, half)}")
                P("  固定 40 根 → " + fmt_R(R, half))

    txt = "\n".join(lines)
    print(txt)
    open("results_part35.txt", "w").write(txt + "\n")
    pd.DataFrame(grid).to_csv("part35_grid.csv", index=False)


if __name__ == "__main__":
    main()
