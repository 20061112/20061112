import numpy as np
import pandas as pd


def atr(df, n=14):
    pc = df.close.shift()
    tr = pd.concat([df.high - df.low, (df.high - pc).abs(), (df.low - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def efficiency_ratio(close, n):
    change = (close - close.shift(n)).abs()
    vol = close.diff().abs().rolling(n).sum()
    return change / vol


def add_features(df):
    f = df.copy()
    f["atr"] = atr(f, 14)
    f["atr60"] = atr(f, 60)
    for n in (5, 8, 10, 12, 15, 20):
        f[f"er{n}"] = efficiency_ratio(f.close, n)
        f[f"mv{n}"] = (f.close - f.close.shift(n)) / f["atr60"]  # 帶方向的幅度（ATR 倍數）
    f["tv_rel"] = f.tickvol / f.tickvol.rolling(60).median()
    return f
