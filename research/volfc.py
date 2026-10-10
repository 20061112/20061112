"""波動預測（第五十五部分驗證過的 har_rel）的可重用版本。

har_rel：log(RV_{t+1..t+h} / 未來時段因子) − log RV_5d  ~  [log RV_1h, 4h, 1d 相對 5d（皆去季節化）, 跳躍]
逐月 walk-forward（測試月只用之前的資料），Duan smearing 修正平均偏誤。
回傳的 sig2 = 未來 h 根報酬的變異數預測（log 報酬平方的單位）；前 warm 個月為 NaN。
"""
import numpy as np
import pandas as pd
from part55_vol_verify import prep, features, rel_mats, ols


def har_rel(path, h, warm=3, start=None):
    b, step = prep(path, start)
    df = features(b, step, h)
    XT = np.column_stack([df.LS, df.LM, df.LD, df.LW5, df.JUMP]).astype(float)
    rvf = df.rvf.to_numpy()
    ly = np.log(np.where(rvf > 0, rvf, np.nan))
    futf = df.futf.to_numpy()
    Xr, yr = rel_mats(XT, ly, futf)
    mon = df.month.to_numpy()
    months = sorted(df.month.unique())[warm:]
    sig2 = np.full(len(df), np.nan)
    fin = np.isfinite(Xr).all(1)
    for m in months:
        tr = (mon < m) & fin & np.isfinite(yr)
        te = (mon == m) & fin
        bt = ols(Xr[tr], yr[tr])
        sm = np.mean(np.exp(yr[tr] - np.column_stack([np.ones(tr.sum()), Xr[tr]]) @ bt))
        sig2[te] = np.exp(np.column_stack([np.ones(te.sum()), Xr[te]]) @ bt + XT[te, 3]) * sm * futf[te]
    hi, lo, c = b.high.to_numpy(), b.low.to_numpy(), b.close.to_numpy()
    pc = np.r_[c[0], c[:-1]]
    tr_ = np.maximum(hi - lo, np.maximum(abs(hi - pc), abs(lo - pc)))
    out = b[["open", "high", "low", "close", "spread"]].copy()
    out["sig2"] = sig2
    out["sig"] = np.sqrt(sig2) * c                     # 美元 / 盎司
    out["atr"] = pd.Series(tr_, index=b.index).rolling(14).mean()
    out["rvpast"] = np.sqrt(df.rvpast.to_numpy()) * c  # 過去 h 根實現波動（美元）
    out["rvf"] = np.sqrt(rvf) * c                      # 未來 h 根實現波動（只供事後分析）
    out["y"] = df.y.to_numpy()
    out["up"] = df.up.to_numpy()
    out["dn"] = df.dn.to_numpy()
    out["futf"] = futf
    out["gapid"] = b.gapid.to_numpy()
    return out, step
