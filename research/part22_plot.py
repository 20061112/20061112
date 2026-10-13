"""第二十二部分圖表：part22_direction.png（旗面 vs 安慰劑）、part22_equity.png（突破單累計 R）。"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK TC", "Noto Sans CJK JP", "WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
C1, C2, MUTED = "#2a78d6", "#eb6834", "#8a8984"
TFS = ["M5", "M15", "H1"]

fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
for ax, tf in zip(axes, TFS):
    F = pd.read_csv(f"part22_formations_{tf}.csv")
    F["raw"] = F.up_first - F.exp_up
    b = pd.cut(F.exp_up, np.linspace(0, 1, 6))
    g = F.groupby(b, observed=True).agg(flag=("raw", "mean"), placebo=("placebo", "mean"))
    x = np.arange(len(g))
    ax.bar(x - 0.2, g.flag, 0.38, color=C1, label="Flag (after trend leg)")
    ax.bar(x + 0.2, g.placebo, 0.38, color=C2, label="Placebo (any 5-bar box)")
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks(x, [f"{i.left:.1f}-{i.right:.1f}" for i in g.index], fontsize=8)
    ax.set_title(f"{tf}", fontsize=11)
    ax.set_xlabel("price position in box when flag forms (0=bottom, 1=HH)", fontsize=8)
    ax.grid(axis="y", color="#e5e4e0", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("break-with-trend rate − random-walk baseline")
axes[0].legend(frameon=False, fontsize=8)
fig.suptitle("Which side does the consolidation break first? Flag vs placebo box (2023–2026)", fontsize=12)
fig.tight_layout()
fig.savefig("part22_direction.png", dpi=110)

fig, ax = plt.subplots(figsize=(11, 4))
for tf, col in zip(TFS, [C1, C2, "#1baf7a"]):
    E = pd.read_csv(f"part22_events_{tf}.csv", parse_dates=["time"]).sort_values("time")
    ax.plot(E.time, E.R.cumsum(), color=col, lw=2, label=f"{tf} ({len(E)} trades)")
ax.axvline(pd.Timestamp("2025-01-01"), color=MUTED, ls="--", lw=1)
ax.text(pd.Timestamp("2025-01-15"), ax.get_ylim()[1] * 0.9, "validation →", color=MUTED, fontsize=9)
ax.axhline(0, color=MUTED, lw=1)
ax.set_ylabel("cumulative R (box stop, 2R target, after spread)")
ax.set_title("All 'record HH/LL → close breakout' trades after a trend leg", fontsize=12)
ax.grid(axis="y", color="#e5e4e0", lw=0.6)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("part22_equity.png", dpi=110)
