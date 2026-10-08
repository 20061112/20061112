"""第七部分：借用使用者 FADE 系統的三個概念，套到 M1 晨星/夜星/吞噬上測試。
  1. ER_8h 閘門：最後一根已收完的 5 分鐘 K 棒，ER(96) < 0.18 才做（只在非趨勢狀態做反轉）
  2. layer：過去 35 分鐘內，同方向（同為做多或做空）的型態訊號數量（含被擋掉的）→ 反覆延伸後才做
  3. 出場：不設停利，固定持有 35 分鐘；停損 min(8×ATR, 35)（ATR = M1 TR 的 EMA14，延遲 1 分鐘）
進場：型態 K 棒收盤後下一根開盤；成本 = 進場當下點差。研究期 6/29~8/15、驗證期 8/15~10/8。
"""
import numpy as np
import pandas as pd
from load import load
from backtest import Cfg, find_setups, resample, stats
from m1_momentum import efficiency, htf_to_m1

m1 = load("data/XAUUSD_M1_full.csv")
o, h, l, c = (m1[k].to_numpy(float) for k in ("open", "high", "low", "close"))
spr = m1.spread.to_numpy(float) * 0.01
pc = np.r_[c[0], c[:-1]]
tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))
atr = pd.Series(tr).ewm(span=14, adjust=False).mean().shift(1).to_numpy()
m5 = resample(m1, "5min")
er8 = htf_to_m1(m1.index, efficiency(m5.close, 96), 5).to_numpy()
er1 = htf_to_m1(m1.index, efficiency(m5.close, 12), 5).to_numpy()

setups = [x for x in find_setups(m1, Cfg(pattern="both", swing=15, trend_len=20, trend_atr=0.0)) if x[0] > 600]
t_arr = np.array([x[0] for x in setups]); s_arr = np.array([x[1] for x in setups])
# layer：過去 35 分鐘同方向型態數（含自己）
times = m1.index[t_arr].values.astype("datetime64[m]").astype(np.int64)
layer = np.zeros(len(setups), int)
for side in (1, -1):
    idx = np.where(s_arr == side)[0]
    tt = times[idx]
    layer[idx] = np.arange(len(idx)) - np.searchsorted(tt, tt - 35, side="right") + 1


def trade(t, s, hold=35, stop_mult=8.0, stop_cap=35.0):
    ei = t + 1
    if ei + hold >= len(c):
        return None
    ep = o[ei] + (spr[ei] if s == 1 else 0)          # 做多以 ask 進場（bid + 點差）
    sd = min(stop_mult * atr[t], stop_cap)
    stop = ep - s * sd
    for j in range(ei, ei + hold):
        if (l[j] <= stop) if s == 1 else (h[j] + spr[j] >= stop):
            return -sd, sd
    xp = c[ei + hold - 1] + (spr[ei + hold - 1] if s == -1 else 0)   # 做空以 ask 平倉
    return s * (xp - ep), sd


rows = []
for (t, s, kind, ext, A, trend), L in zip(setups, layer):
    r = trade(t, s)
    if r is None:
        continue
    rows.append(dict(time=m1.index[t], side=s, kind=kind, layer=L, er8=er8[t], er1=er1[t], pnl=r[0], stop=r[1],
                     pnl_R=r[0] / r[1]))
T = pd.DataFrame(rows)
T["entry_time"] = T.time
SPLIT = pd.Timestamp("2026-08-15")
days = T.time.dt.normalize().nunique()


def show(name, m):
    a, b = T[m & (T.time < SPLIT)], T[m & (T.time >= SPLIT)]
    f = lambda x: f"n={len(x):5d} 每筆 {x.pnl.mean():+.3f} 點 (t={x.pnl.mean() / x.pnl.std() * np.sqrt(len(x)):+.2f}) 勝率 {(x.pnl > 0).mean():.2f}"
    print(f"{name:34s} 研究: {f(a)} | 驗證: {f(b)}")


if __name__ == "__main__":
    print(f"M1 型態候選 {len(T)}（每日 {len(T) / days:.0f}），停損中位 {T.stop.median():.2f} 點\n")
    show("全部型態（時間出場 35 分）", T.index == T.index)
    show("ER8h < 0.18", T.er8 < 0.18)
    show("ER8h >= 0.18", T.er8 >= 0.18)
    for lo, hi in ((1, 1), (2, 3), (4, 6), (7, 25), (26, 999)):
        show(f"layer {lo}~{hi}", T.layer.between(lo, hi))
    for lo, hi in ((4, 25), (7, 25)):
        show(f"ER8h<0.18 且 layer {lo}~{hi}", (T.er8 < 0.18) & T.layer.between(lo, hi))
    show("ER8h<0.18 且 layer>=7 且 ER1 下降", (T.er8 < 0.18) & (T.layer >= 7))
