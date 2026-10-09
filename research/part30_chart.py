"""第三十部分示意圖：M15 收盤 + 每天開盤前用過去 20 天插針算出的 KDE 價位（9/14 ~ 9/26）。"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from load import load
from td import resample, atr
from kde_levels import find_pins, levels_from_pins

m1 = load("data/XAUUSD_M1_2026.csv")
d15 = resample(m1, "15min")
pins = find_pins(d15, atr(d15))
s, e = pd.Timestamp("2026-09-14"), pd.Timestamp("2026-09-26")
px = d15[(d15.index >= s) & (d15.index < e)]
fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(px))
ax.plot(x, px.close.to_numpy(), lw=0.8, color="k")
days = px.index.normalize()
for dd in days.unique():
    idx = np.flatnonzero(days == dd)
    p = pins[(pins.index >= dd - pd.Timedelta(days=20)) & (pins.index < dd)]
    lv = levels_from_pins(p, bw=0.5)
    for L, st in zip(lv.price, lv.strength):
        ax.hlines(L, idx[0], idx[-1], color="tab:blue", alpha=min(1, 0.2 + 0.2 * st), lw=1.5)
pw = pins[(pins.index >= s) & (pins.index < e)]
pos = px.index.get_indexer(pw.index)
ax.scatter(pos[pw.side == 1], pw.price[pw.side == 1], marker="^", color="tab:green", s=20, label="lower wick")
ax.scatter(pos[pw.side == -1], pw.price[pw.side == -1], marker="v", color="tab:red", s=20, label="upper wick")
ax.set_ylim(px.low.min() - 10, px.high.max() + 10)
ax.set_title("M15 close, KDE wick levels (rolling 20d, recomputed daily; darker = stronger)")
ax.legend()
fig.tight_layout()
fig.savefig("part30_levels.png", dpi=110)
