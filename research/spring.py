"""彈簧模型 f = -k·x 的特徵（第三十五部分）。

  均值 μ   = SMA(N) of close
  ER       = |c_t - c_{t-N}| / Σ|Δc|（N 根）
  k        = ln(1/ER)                       （ER→1 趨勢：彈簧變軟；ER→0 盤整：彈簧變硬）
  d        = (c - μ) / ATR14                 （以 ATR 為單位的偏離）
  x (log)  = sign(d)·ln(1+|d|)              （「ln(現值−均值)/ATR」的可計算版本，見 README）
  x (lin)  = d
  f        = -k·x
  zf std   = (f - mean_W(f)) / std_W(f)      （標準 z-score）
  zf raw   = f / std_W(f)                    （分子不減均值）
  v, a     = zf 的一階、二階差分（v 取 EMA3 平滑）

  位能 U = ½ k x²；逃逸速度 v_esc = sqrt(2U/m) = |x|·sqrt(k/m)
  實際速度 u = x 往「離開均值」方向的速度（ATR/根，n 根平均）
  逃逸比 esc = u / v_esc（>1 = 動能足以脫離位能井 → 彈簧斷掉）
  m 候選：1、相對 tick volume（n 根量 / 100 根中位數）、相對波動（σ20 / σ100 of Δc）

  面積 area = Σ (c-μ)/c，從上次穿越均值起累加（帶正負號；/價格標準化）
  SHM 週期 T = 2π·sqrt(m/k)（根）→ 回到均值理論上需 T/4

所有欄位只用到該根 K 棒收盤（含）以前的資料。
"""
import numpy as np
import pandas as pd


def resample(m1, rule):
    if rule == "1min":
        return m1[["open", "high", "low", "close", "tickvol", "spread"]].copy()
    return m1.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last",
         "tickvol": "sum", "spread": "max"}).dropna()


def atr(df, n=14):
    h, l, c = df.high.to_numpy(), df.low.to_numpy(), df.close.to_numpy()
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))
    return pd.Series(tr, index=df.index).rolling(n).mean()


def features(df, N=20, W=100, nv=3):
    c = df.close
    A = atr(df)
    mu = c.rolling(N).mean()
    dc = c.diff()
    er = (c - c.shift(N)).abs() / dc.abs().rolling(N).sum()
    er = er.clip(1e-3, 1.0)
    k = np.log(1.0 / er)
    d = (c - mu) / A
    out = pd.DataFrame(index=df.index)
    out["atr"], out["mu"], out["er"], out["k"], out["d"] = A, mu, er, k, d
    xs = {"log": np.sign(d) * np.log1p(d.abs()), "lin": d}
    for xn, x in xs.items():
        f = -k * x
        sd = f.rolling(W).std()
        out[f"x_{xn}"] = x
        out[f"f_{xn}"] = f
        for zn, z in (("std", (f - f.rolling(W).mean()) / sd), ("raw", f / sd)):
            col = f"z_{xn}_{zn}"
            out[col] = z
            v = z.diff().ewm(span=3, adjust=False).mean()
            out[col + "_v"] = v
            out[col + "_v1"] = z.diff()
            out[col + "_a"] = v.diff()
        # 逃逸：u = x 往外的速度
        u = np.sign(x) * (x - x.shift(nv)) / nv
        out[f"u_{xn}"] = u
    tv = df.tickvol.rolling(nv).sum() / nv
    m = {"1": pd.Series(1.0, index=df.index),
         "vol": tv / df.tickvol.rolling(W).median(),
         "sig": dc.rolling(N).std() / dc.rolling(W).std()}
    for mn, mm in m.items():
        out[f"m_{mn}"] = mm
        for xn in xs:
            vesc = out[f"x_{xn}"].abs() * np.sqrt(k / mm)
            out[f"esc_{xn}_{mn}"] = out[f"u_{xn}"] / vesc.replace(0, np.nan)
            # 本段偏離（上次穿越均值以來）的最大逃逸比
        out[f"T_{mn}"] = 2 * np.pi * np.sqrt(mm / k.replace(0, np.nan))
    # 面積：從上次穿越均值起累加 (c-μ)/c
    s = np.sign(c - mu)
    seg = (s != s.shift()).cumsum()
    out["area"] = ((c - mu) / c).groupby(seg).cumsum()
    out["area_atr"] = d.groupby(seg).cumsum()
    out["seg_len"] = s.groupby(seg).cumcount() + 1
    for col in [x for x in out.columns if x.startswith("esc_")]:
        out[col + "_max"] = out[col].clip(lower=0).groupby(seg).cummax()
    return out
