"""第二十二部分：K 線「角度」、短長角度差、速度（一次微分）與加速度（二次微分）。
資料：data/XAUUSD_M15_2023_2026.csv（2023/1~2026/10），另合成 H1 再看一次。
一開始就用 4 個年度，逐年列出結果；不挑參數，只看事先列好的幾組假設。

定義（全部只用 t 當根收完以前的資料）：
  slope_N  = 過去 N 根收盤的線性回歸斜率（價格 / 根）
  k_N      = slope_N / ATR96            → 每根走幾個 ATR，與價格水準、波動無關
  角度 θ_N = atan(k_N)（度）            → tan θ = 斜率 = 速度；θ 只是 k 的單調壓縮
  R²_N     = 回歸擬合度（線走得「直」還是「抖」）
  二次擬合：過去 N 根做 y = b0 + b1·t + b2·t²（t=0 為當根），
            速度 v = b1 / ATR（當根切線斜率），加速度 a = 2·b2 / ATR
前瞻：fwd_h = (close[t+h] − close[t]) / ATR96[t]；成本 = 點差（points × 0.01）/ ATR。
事件測試會去重疊（訊號出現後 h 根內不重複計），t 值才不會被灌水。
"""
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as swv
from load import load

H = (4, 8, 16)


def fit_weights(n, deg):
    t = np.arange(-(n - 1), 1, dtype=float)
    X = np.vander(t, deg + 1, increasing=True)
    return np.linalg.pinv(X)          # 每一列 = 對應係數的權重


def rolling_coef(y, n, deg):
    W = fit_weights(n, deg)
    out = np.full((len(y), deg + 1), np.nan)
    out[n - 1:] = swv(y, n) @ W.T
    return out


def r2(y, n):
    win = swv(y, n)
    t = np.arange(n, dtype=float)
    tm = t - t.mean()
    ym = win - win.mean(1, keepdims=True)
    num = (ym @ tm) ** 2
    den = (ym ** 2).sum(1) * (tm ** 2).sum()
    out = np.full(len(y), np.nan)
    out[n - 1:] = np.where(den > 0, num / den, np.nan)
    return out


def build(d, ns=8, nl=64, nq=16):
    c = d.close.to_numpy(float)
    pc = d.close.shift()
    tr = pd.concat([d.high - d.low, (d.high - pc).abs(), (d.low - pc).abs()], axis=1).max(axis=1)
    atr = tr.rolling(96).mean().to_numpy()
    X = pd.DataFrame(index=d.index)
    X["atr"] = atr
    X["cost"] = d.spread.to_numpy() * 0.01 / atr
    for n, tag in ((ns, "s"), (nl, "l")):
        k = rolling_coef(c, n, 1)[:, 1] / atr
        X[f"k{tag}"] = k
        X[f"ang{tag}"] = np.degrees(np.arctan(k))
        X[f"r2{tag}"] = r2(c, n)
    q = rolling_coef(c, nq, 2)
    X["v"] = q[:, 1] / atr
    X["a"] = 2 * q[:, 2] / atr
    X["dang"] = X.angs.diff(4)            # 角度本身的變化（≈ 加速度）
    for h in H:
        X[f"f{h}"] = (d.close.shift(-h) - d.close).to_numpy() / atr
    X["year"] = X.index.year
    return X.dropna(subset=["angs", "angl", "v", "a", "dang"])


def tstat(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return x.mean() / x.std(ddof=1) * np.sqrt(len(x)) if len(x) > 2 else np.nan


def dedup(idx_pos, h):
    keep, last = [], -10 ** 9
    for p in idx_pos:
        if p - last >= h:
            keep.append(p)
            last = p
    return np.array(keep, int)


def event_line(X, mask, sign, h, label):
    """mask 成立的 K 棒，方向 sign 進場，抱 h 根；去重疊後逐年。"""
    pos = dedup(np.flatnonzero(mask.to_numpy()), h)
    E = X.iloc[pos]
    s = sign.iloc[pos].to_numpy()
    g = s * E[f"f{h}"].to_numpy()
    net = g - E.cost.to_numpy()
    ok = ~np.isnan(net)
    yrs = "  ".join(f"{y}:{net[ok & (E.year.to_numpy() == y)].mean():+.3f}({(ok & (E.year.to_numpy() == y)).sum()})"
                    for y in sorted(E.year.unique()))
    print(f"  {label:<38s} h={h:<2d} n={ok.sum():5d} 毛 {np.nanmean(g):+.3f} 淨 {np.nanmean(net):+.3f} "
          f"t={tstat(net):+.2f} | {yrs}")
    return net


def quant_rank(s, win=2000):
    """當下值在過去 win 根中的百分位（只用過去）"""
    return s.rolling(win, min_periods=500).rank(pct=True)


def section_A(X):
    print("\n[A] 短角度十分位 → 之後的原始報酬（ATR，正 = 漲）。看角度大小本身有沒有延續/反轉")
    X = X.copy()
    X["dec"] = X.groupby("year").angs.transform(lambda s: pd.qcut(s, 10, labels=False))
    for h in H:
        t = X.pivot_table(index="dec", columns="year", values=f"f{h}", aggfunc="mean")
        print(f"  h={h}\n" + t.round(3).to_string())
    print("  各年 corr(angs, f8)：", X.groupby("year").apply(lambda g: round(g.angs.corr(g.f8), 4)).to_dict())
    print("  各年 corr(angl, f16)：", X.groupby("year").apply(lambda g: round(g.angl.corr(g.f16), 4)).to_dict())


def section_B(X):
    print("\n[B] 短 × 長角度 3×3 格（以短角度方向計算前瞻報酬，h=8，單位 ATR，未扣成本）")
    X = X.copy()
    X["S"] = pd.cut(quant_rank(X.angs.abs()), [0, .4, .8, 1], labels=["短平", "短中", "短陡"])
    X["L"] = pd.cut(quant_rank(X.angl.abs()), [0, .4, .8, 1], labels=["長平", "長中", "長陡"])
    X["same"] = np.where(np.sign(X.angs) == np.sign(X.angl), "同向", "反向")
    X["g8"] = np.sign(X.angs) * X.f8
    t = X.pivot_table(index=["S", "L"], columns="year", values="g8", aggfunc="mean", observed=True)
    t["n"] = X.groupby(["S", "L"], observed=True).size()
    print(t.round(3).to_string())
    print("  短陡 × 長同向/反向：")
    t2 = X[X.S == "短陡"].pivot_table(index=["L", "same"], columns="year", values="g8", aggfunc="mean", observed=True)
    print(t2.round(3).to_string())


def section_C(X):
    print("\n[C] 事件：短陡（|θs| 過去 2000 根前 20%）且 長平（|θl| 後 40%）→ 順短方向 vs 逆短方向（已扣點差）")
    rs, rl = quant_rank(X.angs.abs()), quant_rank(X.angl.abs())
    base = (rs >= .8) & (rl <= .4)
    sgn = np.sign(X.angs)
    for h in H:
        event_line(X, base, sgn, h, "短陡長平 順短")
        event_line(X, base, -sgn, h, "短陡長平 逆短")
    print("  對照：短陡長陡且同向（大小週期共振）")
    co = (rs >= .8) & (rl >= .8) & (np.sign(X.angs) == np.sign(X.angl))
    for h in H:
        event_line(X, co, sgn, h, "短陡長陡同向 順勢")
    print("  對照：短陡但與長反向（逆大週期的急動）")
    op = (rs >= .8) & (rl >= .6) & (np.sign(X.angs) != np.sign(X.angl))
    for h in H:
        event_line(X, op, sgn, h, "短陡逆長 順短")
        event_line(X, op, -sgn, h, "短陡逆長 逆短(=順長)")
    print("  短陡長平，再依短線 R²（走得直 vs 抖）分")
    for lo, hi, nm in ((0, .6, "R²<0.6 抖"), (.6, .85, "0.6~0.85"), (.85, 1.01, "R²≥0.85 直")):
        m = base & (X.r2s >= lo) & (X.r2s < hi)
        event_line(X, m, sgn, 8, f"短陡長平 {nm} 順短")


def section_D(X):
    print("\n[D] 速度 v 與加速度 a（16 根二次擬合，當根切線）四象限 → 順 v 方向前瞻（h=8，毛，ATR）")
    X = X.copy()
    rv = quant_rank(X.v.abs())
    X["vbin"] = pd.cut(rv, [0, .5, .8, 1], labels=["v小", "v中", "v大"])
    X["acc"] = np.where(np.sign(X.a) == np.sign(X.v), "加速", "減速")
    X["g8"] = np.sign(X.v) * X.f8
    t = X.pivot_table(index=["vbin", "acc"], columns="year", values="g8", aggfunc="mean", observed=True)
    t["n"] = X.groupby(["vbin", "acc"], observed=True).size()
    print(t.round(3).to_string())
    print("  事件（去重疊、扣點差）：")
    ra = quant_rank((X.a * np.sign(X.v)))
    sgn = np.sign(X.v)
    for h in (4, 8):
        event_line(X, (rv >= .8) & (ra >= .8), sgn, h, "v大 + 強加速 順勢")
        event_line(X, (rv >= .8) & (ra <= .2), -sgn, h, "v大 + 強減速 反向（竭盡）")
        event_line(X, (rv <= .3) & (quant_rank(X.a.abs()) >= .9), np.sign(X.a), h, "v≈0 + |a|大 順 a（轉折起動）")
    print("  角度變化 dθ（4 根）：角度轉陡 vs 轉平")
    rd = quant_rank(X.dang * np.sign(X.angs))
    for h in (4, 8):
        event_line(X, (quant_rank(X.angs.abs()) >= .7) & (rd >= .8), np.sign(X.angs), h, "短角度大且持續變陡 順勢")
        event_line(X, (quant_rank(X.angs.abs()) >= .7) & (rd <= .2), -np.sign(X.angs), h, "短角度大但已轉平 反向")


def section_E(X):
    print("\n[E] 組合：短陡長平 + 短線仍在加速 / 已在減速（h=8）")
    rs, rl = quant_rank(X.angs.abs()), quant_rank(X.angl.abs())
    base = (rs >= .8) & (rl <= .4)
    sgn = np.sign(X.angs)
    acc = np.sign(X.a) == sgn
    for h in (4, 8, 16):
        event_line(X, base & acc, sgn, h, "短陡長平 + 加速 順短")
        event_line(X, base & ~acc, sgn, h, "短陡長平 + 減速 順短")
        event_line(X, base & ~acc, -sgn, h, "短陡長平 + 減速 逆短")


def section_F(X):
    print("\n[F] 「短陡長陡同向」的對照：多空分開、無條件基準（只看長角度方向）、門檻鄰近")
    rs, rl = quant_rank(X.angs.abs()), quant_rank(X.angl.abs())
    same = np.sign(X.angs) == np.sign(X.angl)
    sgn = np.sign(X.angs)
    for h in (4, 8, 16):
        co = (rs >= .8) & (rl >= .8) & same
        event_line(X, co & (sgn > 0), sgn, h, "共振 只做多")
        event_line(X, co & (sgn < 0), sgn, h, "共振 只做空")
        event_line(X, rl.notna(), np.sign(X.angl), h, "基準：每根順長角度方向")
        event_line(X, (rl >= .8), np.sign(X.angl), h, "基準：長陡就順長（不看短）")
        event_line(X, rl.notna(), pd.Series(1.0, index=X.index), h, "基準：每根做多")
    print("  門檻鄰近（h=8）：")
    for a in (.6, .7, .8, .9):
        for b in (.6, .7, .8, .9):
            event_line(X, (rs >= a) & (rl >= b) & same, sgn, 8, f"短≥{a:.1f} 長≥{b:.1f} 同向")


def run(d, label, ns, nl, nq):
    X = build(d, ns, nl, nq)
    print(f"\n{'=' * 100}\n{label}：短 N={ns}、長 N={nl}、二次擬合 N={nq}，K 棒數 {len(X)}")
    print("角度分布（度）：", X.angs.abs().quantile([.5, .8, .95]).round(1).to_dict(), "長：",
          X.angl.abs().quantile([.5, .8, .95]).round(1).to_dict())
    section_A(X)
    section_B(X)
    section_C(X)
    section_D(X)
    section_E(X)
    section_F(X)
    return X


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    m15 = load("data/XAUUSD_M15_2023_2026.csv")
    run(m15, "M15", 8, 64, 16)
    run(m15, "M15（參數鄰近：短 12 / 長 96 / 二次 24）", 12, 96, 24)
    h1 = m15.resample("1h", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "tickvol": "sum", "spread": "max"}).dropna()
    run(h1, "H1", 8, 64, 16)
    run(h1, "H1（參數鄰近：短 6 / 長 48 / 二次 12）", 6, 48, 12)
    run(h1, "H1（參數鄰近：短 12 / 長 96 / 二次 24）", 12, 96, 24)
