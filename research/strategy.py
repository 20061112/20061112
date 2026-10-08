"""最終候選策略：M15 趨勢段後的晨星/夜星/吞噬，突破型態高/低點進場（M1 逐根模擬進出場）。

參數（只依 6/29~8/31 樣本內決定）：
  型態 star 或 engulf、型態極值為 20 根 M15 最高/低、40 根內反向幅度 3~7 × ATR14
  前段 ER(20) <= 0.35（前段不能是又直又急的單邊趨勢）
  型態後 3 根內突破型態 K 棒高/低點進場；停損 = 型態極值外 0.1 ATR；停利 2R；最長持有 16 根 M15（4 小時）
"""
import numpy as np
import pandas as pd
from backtest import Cfg, resample, find_setups, backtest_m1

TF, TF_MIN = "15min", 15
CFG = Cfg(pattern="both", swing=20, trend_len=40, trend_atr=3.0, entry="break", wait=3, tp=2.0, buf=0.1, hold=16)
TREND_MAX = 7.0
ER_LEN, ER_MAX = 20, 0.35


def filtered_setups(d, cfg=CFG, trend_max=TREND_MAX, er_max=ER_MAX):
    c = d.close
    er = ((c - c.shift(ER_LEN)).abs() / c.diff().abs().rolling(ER_LEN).sum()).to_numpy()
    return [s for s in find_setups(d, cfg) if s[5] <= trend_max and er[s[0]] <= er_max]


def run(m1, cfg=CFG, trend_max=TREND_MAX, er_max=ER_MAX, flip=False):
    d = resample(m1, TF)
    st = filtered_setups(d, cfg, trend_max, er_max)
    if flip:  # 對照組：同一批型態做「順勢」方向（停損放在突破點另一側），檢查反轉方向是否真的比較好
        raise NotImplementedError
    return backtest_m1(d, m1, cfg, TF_MIN, st)
