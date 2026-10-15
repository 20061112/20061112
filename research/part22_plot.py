"""第二十二部分圖表：影片例子重現（M15 + M1）與各版本累積 R。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
from part22_cct import Data, run

plt.rcParams["font.family"] = "WenQuanYi Zen Hei"
plt.rcParams["axes.unicode_minus"] = False


def candles(ax, df, width):
    for t, r in df.iterrows():
        up = r.close >= r.open
        ax.vlines(t, r.low, r.high, color="#333", lw=0.8)
        ax.add_patch(Rectangle((t - width / 2, min(r.open, r.close)), width, max(abs(r.close - r.open), 0.05),
                               facecolor="white" if up else "#333", edgecolor="#333", lw=0.8))


d = Data()
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 左：M15，方向棒 6/10 08:00（H4）之後的窗
ax = axes[0]
m = d.m15.loc["2026-06-10 08:00":"2026-06-10 19:00"]
candles(ax, m, pd.Timedelta(minutes=10))
ax.axvspan(pd.Timestamp("2026-06-10 08:00"), pd.Timestamp("2026-06-10 12:00"), color="#fbe3c8", alpha=0.5, label="H4 看跌方向棒（6/10 08:00）")
ax.axhline(4179.52, xmin=0.35, color="tab:blue", lw=1, ls="--", label="被掃的最上面擺動高點 4179.52")
ax.axhline(4168.93, color="gray", lw=0.8, ls=":", label="下一根 H4 開盤 4168.93")
ax.add_patch(Rectangle((pd.Timestamp("2026-06-10 16:45") - pd.Timedelta(minutes=5), 4179.65), pd.Timedelta(minutes=120),
                       4186.29 - 4179.65, color="tab:red", alpha=0.3, label="矩形（弱勢棒上影線）"))
ax.set_title("M15：16:45 掃高點後收 4179.65（只高出 0.13，嚴格規則不算弱勢）")
ax.legend(loc="lower left", fontsize=8)

# 右：M1 進場
ax = axes[1]
m1 = d.m1.loc["2026-06-10 16:30":"2026-06-10 18:10"]
candles(ax, m1, pd.Timedelta(seconds=40))
ax.add_patch(Rectangle((pd.Timestamp("2026-06-10 16:45"), 4179.65), pd.Timedelta(minutes=85), 4186.29 - 4179.65,
                       color="tab:red", alpha=0.3))
ax.axhline(4174.91, color="black", lw=1, label="進場 4174.91（17:00 收盤跌破矩形）")
ax.axhline(4186.29, color="tab:red", lw=1, label="停損 4186.29")
ax.axhline(4129.39, color="tab:green", lw=1, label="4R 目標 4129.39（17:56 達成）")
ax.set_title("M1：矩形進場（寬鬆版 tol=0.1 ATR 才會觸發）")
ax.legend(loc="lower left", fontsize=8)
fig.suptitle("影片例子重現：XAUUSD 2026/6/10（本資料伺服器時間）")
fig.tight_layout()
fig.savefig("part22_example.png", dpi=110)

# 各版本累積 R
fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
vers = [("基準：結構目標", dict()), ("固定 2R", dict(tp_mode=2)), ("固定 4R", dict(tp_mode=4)),
        ("寬鬆 tol0.1 + K2 + 4R", dict(tol=0.1, K=2, tp_mode=4)), ("位置 20 根極值", dict(loc=20))]
for name, kw in vers:
    T = run(d, **kw).sort_values("entry_time")
    axes[0].plot(T.entry_time, T.pnl_R.cumsum(), label=f"{name}（{len(T)} 筆，{T.pnl_R.mean():+.2f}R）")
    for s, ls in ((1, "-"), (-1, "--")):
        if name in ("固定 2R", "固定 4R"):
            S = T[T.side == s]
            axes[1].plot(S.entry_time, S.pnl_R.cumsum(), ls=ls, label=f"{name} {'多' if s > 0 else '空'}單（{len(S)} 筆）")
for ax in axes:
    ax.axhline(0, color="gray", lw=0.8); ax.legend(fontsize=8); ax.set_ylabel("累積 R（已扣點差）")
axes[0].set_title("CCT 矩形設定：各版本累積 R（2023/6 ~ 2026/10）")
axes[1].set_title("多空拆開：多單略正、空單持續虧損（黃金 3 年大多頭）")
fig.tight_layout()
fig.savefig("part22_equity.png", dpi=110)
print("saved")
