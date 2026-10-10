"""第五十四部分 b：唯一候選「低熵 × 低波動 → 往 EMA48 回歸」在 2025 年資料上的預先登記驗證。

規則在看 2025 資料之前就凍結（來自 part54_ev.py 的 2026 結果）：
  狀態：H_T（窗口 144）在前幾個月的下三分位，且 V_des 在前幾個月的下三分位（逐月 expanding 估計）
  方向：−sign(Z48)（往 EMA48 回歸）
  持有：H = 12 與 24 根 M5；進場 open_{t+1}；停損 2σ̂；一次一筆；扣點差
  σ̂：naive × 時段因子（不需估計的版本，避免再挑模型）
資料：data/XAUUSD_M5_2025_2026.csv。沒有 M1，所以每根 RV = r²、bipower 用相鄰 M5 報酬
（2026 年同樣用 M5-only 版本重算一次，確認換資料粒度後原始結果還在）。
樣本外 = 2025/7 ~ 2025/12（5、6 月只當暖機與第一段訓練）。
"""
import numpy as np
import pandas as pd
from load import load
from ev_features import entropy_engine, volatility_engine, future_tod_sum, base_directional

OUT = "results_part54_oos.txt"


def bars_from_m5(m5):
    b = m5[["open", "high", "low", "close", "spread"]].copy()
    b["r"] = np.log(b.close).diff().fillna(0.0)
    b["rv"] = b.r ** 2
    b["bp"] = (b.r.abs() * b.r.abs().shift(1)).fillna(0.0)
    return b


def build(b, H):
    idx = b.index
    n = len(b)
    vol, key = volatility_engine(b)
    sig_prev = np.sqrt(pd.Series(b.rv.to_numpy()).ewm(alpha=0.06, adjust=False).mean().shift(1).to_numpy())
    ent = entropy_engine(b.r.to_numpy(), sig_prev, close=b.close.to_numpy())
    sig_bar = np.sqrt(pd.Series(b.rv.to_numpy()).rolling(288).mean().to_numpy())
    dirf = base_directional(b, sig_bar)
    df = pd.concat([b.reset_index(), vol.reset_index(drop=True), ent, dirf], axis=1).set_index("time")
    c = np.log(b.close.to_numpy())
    gap = np.r_[False, np.diff(idx.values).astype("timedelta64[m]").astype(int) > 15]
    g = np.cumsum(gap)
    ok = np.zeros(n, bool)
    ok[:-H] = g[H:] == g[:-H]
    y = np.full(n, np.nan)
    y[:-H] = c[H:] - c[:-H]
    y[~ok] = np.nan
    df["y"] = y
    fut = future_tod_sum(vol.tod.to_numpy(), key, H)
    tod = vol.tod.to_numpy()
    des12 = pd.Series(b.rv.to_numpy() / tod).rolling(12).mean().to_numpy()
    df["sig_h"] = np.sqrt(des12 * fut)
    df["z"] = df.y / df.sig_h
    df["month"] = idx.to_period("M")
    return df.iloc[288 * 2:]


def run(df, test_months, H):
    mon = df.month.to_numpy()
    st = np.zeros(len(df), bool)
    for m in test_months:
        tr = mon < m
        te = mon == m
        eH = np.nanquantile(df.H_T.to_numpy()[tr], 1 / 3)
        eV = np.nanquantile(df.V_des.to_numpy()[tr], 1 / 3)
        st[te] = (df.H_T.to_numpy()[te] <= eH) & (df.V_des.to_numpy()[te] <= eV)
    d = -np.sign(df.Z48.to_numpy())
    # 前瞻（非重疊取樣）
    s = np.zeros(len(df), bool)
    s[::H] = True
    m = st & s & np.isfinite(df.z.to_numpy()) & (d != 0)
    zr = df.z.to_numpy()[m] * d[m]
    # 交易
    o, hi, lo, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    sh = df.sig_h.to_numpy()
    spr = df.spread.to_numpy() * 0.01
    yok = np.isfinite(df.y.to_numpy())
    R, months = [], []
    busy = -1
    for t in np.flatnonzero(st):
        if t <= busy or t + H >= len(df) or not yok[t] or d[t] == 0 or not np.isfinite(sh[t]):
            continue
        e = o[t + 1]
        risk = 2 * sh[t] * c[t]
        stop = e - d[t] * risk
        x, out = c[t + H], t + H
        for j in range(t + 1, t + H + 1):
            if (lo[j] <= stop) if d[t] > 0 else (hi[j] >= stop):
                x, out = stop, j
                break
        busy = out
        R.append(((x - e) * d[t] - spr[t + 1]) / risk)
        months.append(mon[t])
    R = np.array(R)
    allz = df.z.to_numpy()[s & np.isin(mon, test_months) & np.isfinite(df.z.to_numpy())]
    return zr, R, pd.Series(R, index=months), allz


def tstat(v):
    return v.mean() / v.std(ddof=1) * np.sqrt(len(v)) if len(v) > 2 else np.nan


def main():
    out = open(OUT, "w")
    pr = lambda *a: print(*a, file=out)
    pr("第五十四部分 b：「低熵 × 低波動 → 往 EMA48 回歸」預先登記驗證（規則凍結自 2026 結果）")
    m5_25 = load("data/XAUUSD_M5_2025_2026.csv")
    m1_26 = load("data/XAUUSD_M1_2026.csv")
    m5_26 = m1_26.resample("5min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "spread": "mean"}).dropna()
    sets = [
        ("2025 樣本外（M5 資料，7~12 月）", m5_25[m5_25.index < "2026-01-01"],
         [pd.Period(f"2025-{k:02d}", "M") for k in range(7, 13)]),
        ("2026 原樣本（M5-only 重算，3~10 月）", m5_26,
         [pd.Period(f"2026-{k:02d}", "M") for k in range(3, 11)]),
    ]
    for H in (12, 24):
        pr(f"\n=== H = {H} 根（{H * 5} 分鐘）===")
        for name, raw, months in sets:
            df = build(bars_from_m5(raw), H)
            zr, R, Rm, allz = run(df, months, H)
            pr(f"{name}")
            pr(f"  前瞻：n={len(zr)}  歸 z={zr.mean():+.3f}（t {tstat(zr):+.2f}）  全部樣本 |z| 平均 {np.abs(allz).mean():.2f}")
            if len(R) > 2:
                gp, gl = R[R > 0].sum(), -R[R < 0].sum()
                pr(f"  交易：n={len(R)}  avg={R.mean():+.3f}R（t {tstat(R):+.2f}） win={np.mean(R > 0):.0%}  PF={gp / gl:.2f}  總 {R.sum():+.1f}R")
                pr("  逐月 avgR: " + "  ".join(f"{str(k)[-2:]}月 {v:+.3f}({c})" for k, v, c in
                                              zip(Rm.groupby(level=0).mean().index, Rm.groupby(level=0).mean(), Rm.groupby(level=0).size())))
    out.close()
    print(open(OUT).read())


if __name__ == "__main__":
    main()
