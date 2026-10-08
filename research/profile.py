"""每個標記樣本的微觀特徵剖析：以窗口內極值 K 棒為錨點。"""
import numpy as np
import pandas as pd
from load import load
from labels import load_labels
from features import add_features

df = add_features(load("data/XAUUSD_M1_full.csv"))
idx = df.index
rows = []
for a, b, s in load_labels():
    w = df.loc[a - pd.Timedelta(minutes=1):b]
    e_t = w.low.idxmin() if s == 1 else w.high.idxmax()
    e = idx.get_loc(e_t)
    A = df.atr60.iloc[e]
    ext = df.low.iloc[e] if s == 1 else df.high.iloc[e]
    r = {"win": f"{a:%m-%d %H:%M}", "s": "B" if s == 1 else "S", "ext": f"{e_t:%H:%M}", "atr": round(A, 2)}
    # 極值是幾根 K 棒內的最低/最高
    for L in (10, 20, 30, 60):
        seg = df.iloc[e - L:e]
        r[f"x{L}"] = int(ext < seg.low.min()) if s == 1 else int(ext > seg.high.max())
    # 前段走勢：N 根內反向極值到本極值的距離 (ATR 倍數)
    for N in (8, 12, 20):
        seg = df.iloc[e - N:e + 1]
        run = (seg.high.max() - ext) if s == 1 else (ext - seg.low.min())
        r[f"run{N}"] = round(run / A, 1)
        r[f"er{N}"] = round(df[f"er{N}"].iloc[e], 2) if f"er{N}" in df else None
    # 極值後第幾根收盤回到多少 ATR、確認 K 棒
    for k in (1, 2, 3):
        c = df.close.iloc[e + k]
        r[f"ret{k}"] = round(s * (c - ext) / A, 2)
    # 極值 K 棒本身：影線比例
    o, h, l, c = df[["open", "high", "low", "close"]].iloc[e]
    rng = max(h - l, 1e-9)
    r["wick"] = round(((min(o, c) - l) if s == 1 else (h - max(o, c))) / rng, 2)
    r["tv"] = round(df.tv_rel.iloc[e], 2)
    # 之後 15 根的最大有利/不利（ATR 倍數）
    fut = df.iloc[e + 1:e + 16]
    r["mfe15"] = round(((fut.high.max() - ext) if s == 1 else (ext - fut.low.min())) / A, 1)
    r["mae15"] = round(((ext - fut.low.min()) if s == 1 else (fut.high.max() - ext)) / A, 1)
    rows.append(r)
P = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(P.to_string())
print(P.describe().T[["mean", "25%", "50%", "75%", "min"]].round(2).to_string())
