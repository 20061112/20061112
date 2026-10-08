"""M1 規則組合：研究期 vs 驗證期（門檻全部取自研究期分位數或事先定好的整數）。"""
import numpy as np
import pandas as pd
from select5 import T, DEV, TEST, SPLIT
from backtest import stats

q = DEV.quantile(numeric_only=True, q=[0.2, 0.5, 0.8])
DAYS_D = DEV.setup.dt.normalize().nunique()
DAYS_T = TEST.setup.dt.normalize().nunique()

RULES = {
    "全部型態": lambda D: D.index == D.index,
    "S1 研究期選出：RSI 不極端 (>中位)": lambda D: D.rsi_e > q.loc[0.5, "rsi_e"],
    "S2 M1 ER20 低 (<中位)": lambda D: D.er_m1_20 < q.loc[0.5, "er_m1_20"],
    "S3 最後 5 根速度不快 (<中位)": lambda D: D.vel5_e < q.loc[0.5, "vel5_e"],
    "S1+S2+S3": lambda D: (D.rsi_e > q.loc[0.5, "rsi_e"]) & (D.er_m1_20 < q.loc[0.5, "er_m1_20"]) & (D.vel5_e < q.loc[0.5, "vel5_e"]),
    "V1 RSI 背離 (>0)": lambda D: D.rsi_div > 0,
    "V2 MACD 柱背離 (>0)": lambda D: D.macd_div > 0,
    "V3 速度背離 (>0)": lambda D: D.vel_div > 0,
    "V1+V2": lambda D: (D.rsi_div > 0) & (D.macd_div > 0),
    "V1+V2+V3": lambda D: (D.rsi_div > 0) & (D.macd_div > 0) & (D.vel_div > 0),
    "A1 已減速 (acc3<0)": lambda D: D.acc3_e < 0,
    "A2 背離+已減速": lambda D: (D.rsi_div > 0) & (D.macd_div > 0) & (D.acc3_e < 0),
    "M1 順 M15 方向": lambda D: D.align_15min == 1,
    "M2 M15 ER 低 (<中位)": lambda D: D.er_15min < q.loc[0.5, "er_15min"],
    "M3 H1 ER 低 (<中位)": lambda D: D.er_60min < q.loc[0.5, "er_60min"],
    "M4 M1/M15 ER 比值高 (>中位)": lambda D: D.er_ratio_m1_m15 > q.loc[0.5, "er_ratio_m1_m15"],
    "V1+V2 + 順 M15": lambda D: (D.rsi_div > 0) & (D.macd_div > 0) & (D.align_15min == 1),
}


def fmt(D, days):
    s = stats(D)
    return f"n={s['n']:4d} ({s['n'] / days:4.1f}/日) 期望={s['expR']:+.3f}R t={s['t']:+.2f} PF={s['pf']:.2f}"


if __name__ == "__main__":
    for name, fn in RULES.items():
        print(f"{name:28s} 研究: {fmt(DEV[fn(DEV)], DAYS_D)} | 驗證: {fmt(TEST[fn(TEST)], DAYS_T)}")
