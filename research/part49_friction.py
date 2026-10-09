"""第四十九部分：摩擦力模型 f = μ·N，N = m·a·cosθ（和彈簧 f = −kx 不同的觀點）。

對每一根 K 棒，往回擷取長度 L 的一段行情（滾動），在這段上算：
  μ     = 1 / ER                      ER 小 = 走得來回黏 → 摩擦係數大；ER 大 = 滑 → μ 小
  a     = 二次回歸 p = b0 + b1·t + b2·t² 的 2·b2，換成無因次 a·L² / ATR
          （這段曲率造成的位移，ATR 為單位）
  a_rel = a · sign(段末速度)            > 0 = 順著目前方向加速；< 0 = 減速
  tanθ  = |c_t − c_{t−L}| / (ATR·√L)   這段的「斜度」（隨機漫步約 1）
  cosθ  = 1 / √(1 + tan²θ)             越陡 → 0，越平 → 1
  m     = 1，或相對 tick volume（段內均量 / 過去 500 根中位數）
  N     = m·|a|·cosθ
  f     = μ·N
  f_sgn = μ·m·a_rel·cosθ               帶方向：正 = 順勢加速遇到的摩擦，負 = 減速
  slide = tanθ·ER                      斜面條件：下滑力 m·g·sinθ > 摩擦 μ·m·g·cosθ ⇔ tanθ > μ ⇔ slide > 1

所有欄位只用到該根收盤（含）以前的資料。

評估（在每根收盤判斷，前瞻 H = L 根）
  cont   = (c_{t+H} − c_t)/ATR · sign(段的淨方向)   > 0 延續，< 0 反轉
  absmv  = |c_{t+H} − c_t| / ATR                     之後走多遠（黏 → 小）
  er_nx  = 下一段的 ER                               之後走得乾不乾淨
  只取每 H 根一個樣本（前瞻視窗不重疊）算平均與 t。
  分組用「過去 2000 根的滾動百分位」（事前可知），五分位 Q1..Q5。
  控制 ER：在 ER 五分位內再看 f 的 Q5−Q1，檢查 f 是否比單純 ER 多帶資訊。
交易（事前可知的滾動百分位）
  SLIP_FOLLOW ：f ≤ 20 百分位（滑）→ 順著段的方向
  STICK_FADE  ：f ≥ 80 百分位（黏）→ 反向；STICK_FOLLOW：同條件順勢（對照）
  SLIDE>1 / SLIDE<=1：斜面條件成立 / 不成立時順勢；ALL_FOLLOW：不篩選全部順勢（基準）
  下一根開盤進、停損 1.5 ATR、持有 H 根收盤出、扣點差、不重疊。前後兩半 1/2~5/31 vs 6/1~10/8。
"""
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from load import load

TFS = ["5min", "15min"]
LS = [12, 24, 48]
SPLIT = pd.Timestamp("2026-06-01")
PCT_W = 2000
SL = 1.5


def resample(m1, rule):
    return m1.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last",
         "tickvol": "sum", "spread": "max"}).dropna()


def atr(df, n=14):
    h, l, c = df.high.to_numpy(), df.low.to_numpy(), df.close.to_numpy()
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))
    return pd.Series(tr, index=df.index).rolling(n).mean()


def features(df, L):
    c = df.close
    A = atr(df)
    n = len(c)
    # 二次回歸係數（t = 0..L-1），以固定 pinv 對每個視窗做內積
    t = np.arange(L, dtype=float)
    X = np.c_[np.ones(L), t, t * t]
    P = np.linalg.pinv(X)                       # 3 × L
    W = sliding_window_view(c.to_numpy(float), L)  # (n-L+1) × L
    B = W @ P.T
    b1, b2 = np.full(n, np.nan), np.full(n, np.nan)
    b1[L - 1:], b2[L - 1:] = B[:, 1], B[:, 2]
    a = 2 * b2 * L * L / A.to_numpy()
    v_end = b1 + 2 * b2 * (L - 1)
    net = (c - c.shift(L)).to_numpy()
    er = (np.abs(net) / c.diff().abs().rolling(L).sum().to_numpy()).clip(0.02, 1.0)
    tan = np.abs(net) / (A.to_numpy() * np.sqrt(L))
    cos = 1 / np.sqrt(1 + tan * tan)
    mvol = (df.tickvol.rolling(L).mean() / df.tickvol.rolling(500).median()).to_numpy()
    mu = 1 / er
    out = pd.DataFrame(index=df.index)
    out["atr"], out["dir"] = A, np.sign(net)
    out["er"], out["mu"], out["cos"], out["a"] = er, mu, cos, a
    out["a_rel"] = a * np.sign(v_end)
    out["m_vol"] = mvol
    out["f"] = mu * np.abs(a) * cos
    out["f_vol"] = out.f * mvol
    out["f_sgn"] = mu * out.a_rel * cos
    out["muxcos"] = mu * cos
    out["slide"] = tan * er   # 斜面：tanθ > μ 才會滑 ⇔ tanθ·ER > 1
    return out


def roll_pct(s, w=PCT_W):
    # 目前值在過去 w 根（含當根）中的百分位
    return s.rolling(w, min_periods=w // 2).rank(pct=True)


def outcomes(df, F, H):
    c = df.close.to_numpy()
    A = F.atr.to_numpy()
    fut = np.r_[c[H:], np.full(H, np.nan)]
    mv = (fut - c) / A
    o = pd.DataFrame(index=df.index)
    o["cont"] = mv * F.dir.to_numpy()
    o["absmv"] = np.abs(mv)
    o["er_nx"] = F.er.shift(-H)
    return o


def tst(x):
    x = x[~np.isnan(x)]
    if len(x) < 5:
        return np.nan, np.nan, len(x)
    return x.mean(), x.mean() / (x.std(ddof=1) / np.sqrt(len(x))), len(x)


def quint_table(F, O, var, idx, out):
    q = np.ceil(roll_pct(F[var]).iloc[idx] * 5).clip(1, 5)
    Os = O.iloc[idx]
    rows = []
    for k in ("cont", "absmv", "er_nx"):
        vals = []
        for g in range(1, 6):
            m, _, n = tst(Os[k][q == g].to_numpy())
            vals.append(m)
        top = Os[k][q == 5].to_numpy()
        bot = Os[k][q == 1].to_numpy()
        top, bot = top[~np.isnan(top)], bot[~np.isnan(bot)]
        se = np.sqrt(top.var(ddof=1) / len(top) + bot.var(ddof=1) / len(bot))
        d = top.mean() - bot.mean()
        rows.append(f"    {k:5s} " + " ".join(f"{v:+.3f}" for v in vals) + f" | Q5−Q1 {d:+.3f} (t={d / se:+.1f})")
    out.append(f"  [{var}] Q1..Q5")
    out.extend(rows)


def half_split(F, O, var, idx, k):
    q = np.ceil(roll_pct(F[var]).iloc[idx] * 5).clip(1, 5)
    Os = O.iloc[idx]
    res = []
    for hv, msk in (("H1", Os.index < SPLIT), ("H2", Os.index >= SPLIT)):
        top = Os[k][(q == 5) & msk].dropna().to_numpy()
        bot = Os[k][(q == 1) & msk].dropna().to_numpy()
        se = np.sqrt(top.var(ddof=1) / len(top) + bot.var(ddof=1) / len(bot))
        d = top.mean() - bot.mean()
        res.append(f"{hv} {d:+.3f}(t={d / se:+.1f})")
    return " ".join(res)


def er_controlled(F, O, var, idx, k):
    """在 ER 五分位內，var 的 Q5−Q1（var 的五分位在每個 ER 組內重新切）。"""
    Fs, Os = F.iloc[idx], O.iloc[idx]
    qe = np.ceil(roll_pct(F.er).iloc[idx] * 5).clip(1, 5)
    ds, ws = [], []
    for g in range(1, 6):
        m = qe == g
        v = Fs[var][m]
        y = Os[k][m]
        ok = v.notna() & y.notna()
        v, y = v[ok], y[ok]
        if len(v) < 50:
            continue
        r = v.rank(pct=True)
        top, bot = y[r > 0.8].to_numpy(), y[r <= 0.2].to_numpy()
        ds.append((top.mean() - bot.mean(), top.var(ddof=1) / len(top) + bot.var(ddof=1) / len(bot)))
    d = np.mean([x[0] for x in ds])
    se = np.sqrt(np.sum([x[1] for x in ds])) / len(ds)
    return d, d / se


def simulate(df, F, sig_i, sig_d, H):
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    spr = df.spread.to_numpy(float) * 0.01
    A = F.atr.to_numpy()
    n = len(c)
    R, T = [], []
    busy = -1
    for t, d in zip(sig_i, sig_d):
        if t + 1 >= n or t + 1 <= busy or np.isnan(A[t]):
            continue
        e = o[t + 1]
        risk = SL * A[t]
        stop = e - d * risk
        x, out = None, min(t + H, n - 1)
        for j in range(t + 1, out + 1):
            if (l[j] <= stop) if d == 1 else (h[j] >= stop):
                x, out = stop, j
                break
        if x is None:
            x = c[out]
        busy = out
        R.append(((x - e) * d - spr[t + 1]) / risk)
        T.append(df.index[t])
    return np.array(R), pd.DatetimeIndex(T)


def fmt_trades(R, T):
    def one(r):
        if len(r) == 0:
            return "n=0"
        pf = r[r > 0].sum() / -r[r < 0].sum() if (r < 0).any() else np.nan
        m, t, _ = tst(r)
        return f"n={len(r):4d} {m:+.3f}R (t={t:+.1f}) PF{pf:.2f}"
    return (one(R) + " | H1 " + one(R[T < SPLIT]) + " | H2 " + one(R[T >= SPLIT]))


def main():
    m1 = load("data/XAUUSD_M1_2026.csv")
    out = ["第四十九部分：摩擦力 f = μ·m·a·cosθ（μ = 1/ER）", ""]
    summary = []
    for tf in TFS:
        df = resample(m1, tf)
        for L in LS:
            H = L
            F = features(df, L)
            O = outcomes(df, F, H)
            valid = np.where(F.f.notna() & roll_pct(F.f).notna() & O.cont.notna())[0]
            idx = valid[::H]
            out.append(f"===== {tf}  L={L}  H={H}  樣本(不重疊)={len(idx)} =====")
            Fs = F.iloc[idx]
            out.append("  相關（Spearman）: " + ", ".join(
                f"{a}~{b} {Fs[a].rank().corr(Fs[b].rank()):+.2f}"
                for a, b in (("f", "er"), ("f", "cos"), ("f", "a"), ("cos", "er"), ("f", "f_vol"))))
            for var in ("mu", "cos", "a_rel", "muxcos", "f", "f_vol", "f_sgn", "slide"):
                quint_table(F, O, var, idx, out)
            out.append("  前後兩半 Q5−Q1：")
            for var in ("mu", "f", "f_vol", "f_sgn", "slide"):
                out.append(f"    {var:6s} cont  " + half_split(F, O, var, idx, "cont")
                           + " | absmv " + half_split(F, O, var, idx, "absmv")
                           + " | er_nx " + half_split(F, O, var, idx, "er_nx"))
            out.append("  控制 ER 後（ER 五分位內）Q5−Q1：")
            for var in ("f", "f_vol", "f_sgn", "cos", "a_rel", "slide"):
                s = []
                for k in ("cont", "absmv", "er_nx"):
                    d, t = er_controlled(F, O, var, idx, k)
                    s.append(f"{k} {d:+.3f}(t={t:+.1f})")
                out.append(f"    {var:6s} " + " | ".join(s))
            # 交易
            p = roll_pct(F.f).to_numpy()
            dirn = F.dir.to_numpy()
            ok = ~np.isnan(p) & (dirn != 0)
            sl = F.slide.to_numpy()
            for name, msk, sgn in (("SLIP_FOLLOW", ok & (p <= 0.2), 1),
                                   ("STICK_FADE", ok & (p >= 0.8), -1),
                                   ("STICK_FOLLOW", ok & (p >= 0.8), 1),
                                   ("SLIDE>1", ok & (sl > 1), 1),
                                   ("SLIDE<=1", ok & (sl <= 1), 1),
                                   ("ALL_FOLLOW", ok, 1)):
                ii = np.where(msk)[0]
                R, T = simulate(df, F, ii, (sgn * dirn[ii]).astype(int), H)
                line = fmt_trades(R, T)
                out.append(f"  交易 {name:12s} {line}")
                summary.append(f"{tf:6s} L={L:2d} {name:12s} {line}")
            out.append("")
    out.append("===== 交易總表 =====")
    out.extend(summary)
    txt = "\n".join(out)
    print(txt)
    with open("results_part49.txt", "w") as fh:
        fh.write(txt + "\n")


if __name__ == "__main__":
    main()
