"""第八部分探索：ER_8h 高（FADE 關閉）時，順 8 小時方向的動能是否延續？
以 5 分鐘 K 棒（由 M1 合成，6/29~10/8）與 M15（2/2~10/8）兩份資料各看一次。
前瞻報酬以 ATR 為單位、方向 = 8 小時趨勢方向；只用當下已收完的 K 棒。"""
import numpy as np
import pandas as pd
from load import load
from backtest import resample
from m1_momentum import efficiency


def table(d, er_n, horizons, ema_n=20):
    c = d.close
    pc = c.shift()
    tr = pd.concat([d.high - d.low, (d.high - pc).abs(), (d.low - pc).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    X = pd.DataFrame(index=d.index)
    X["er8"] = efficiency(c, er_n)
    X["dir"] = np.sign(c - c.shift(er_n))
    X["pull"] = X.dir * (c - c.ewm(span=ema_n, adjust=False).mean()) / atr   # 負 = 回檔到均線下方（順勢方向看）
    for hz in horizons:
        X[f"f{hz}"] = X.dir * (c.shift(-hz) - c) / atr
    return X.dropna()


def show(X, horizons, label):
    print(f"\n=== {label} ===")
    X["er_b"] = pd.cut(X.er8, [0, .1, .18, .25, .35, 1])
    print("依 ER_8h 分組：順 8h 方向的前瞻報酬（ATR）")
    print(X.groupby("er_b", observed=True)[[f"f{h}" for h in horizons]].mean().round(3).assign(n=X.groupby("er_b", observed=True).size()).to_string())
    hi = X[X.er8 >= 0.25].copy()
    hi["pull_b"] = pd.cut(hi.pull, [-99, -1, -0.3, 0.3, 1, 99])
    print("ER_8h >= 0.25 時，依回檔深度分組")
    print(hi.groupby("pull_b", observed=True)[[f"f{h}" for h in horizons]].mean().round(3).assign(n=hi.groupby("pull_b", observed=True).size()).to_string())


if __name__ == "__main__":
    m5 = resample(load("data/XAUUSD_M1_full.csv"), "5min")
    show(table(m5, 96, (6, 12, 24)), (6, 12, 24), "M5，ER 96 根 = 8h，前瞻 30/60/120 分")
    m15 = load("data/XAUUSD_M15_full.csv")
    show(table(m15, 32, (2, 4, 8)), (2, 4, 8), "M15，ER 32 根 = 8h，前瞻 30/60/120 分")
