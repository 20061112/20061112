"""第三十一部分：L1 核心 + 反向尾巴 的詳細統計（2023/8 ~ 2026/10，門檻滾動前推），以及 M30 兩個延續組合單獨交易的 PF。

L1 + 尾巴 規則：
  IB：broker 02:00~10:30 每 30 分一個起點，長度 60 分；突破有效 3 小時內；M5 收盤 > IB 高 + 0.05 × ATR10 做多（< IB 低 − 0.05 ATR10 做空）
  停損 IB 另一側；當天收盤平倉；沒有停利；星期一也做
  只有一個篩選：反向尾巴（TPO 15 分字母、1 美元格）> 過去所有訊號 tail 的 90% 分位 → 不做（每月月初重算）
  同一根 M5、同方向只開一筆
金額：以 R 計（1R = 停損距離）；帳戶換算 = 1 萬美元、每筆風險 0.5%（1R = 50 美元）、不複利。
"""
import numpy as np
import pandas as pd
from load import load
from part29_dir_filter import L1, walk, pf

out = open("results_part31.txt", "w")


def P(s=""):
    print(s)
    out.write(s + "\n")
    out.flush()


RISK_USD = 50.0      # 1 萬美元 × 0.5%
T = walk(L1, 0.9, 1.0, 0.0).reset_index(drop=True)
T["entry_time"] = pd.to_datetime(T.entry_time)
T["exit_time"] = pd.to_datetime(T.exit_time)
m1 = load("data/XAUUSD_M1_2023_2026.csv")
tm = m1.index.values
H, L = m1.high.to_numpy(), m1.low.to_numpy()

# ---------- 逐分鐘浮動損益（最差價：多單用 low、空單用 high），單位 R ----------
open_arr = np.zeros(len(tm))
cnt = np.zeros(len(tm) + 1)
mae_min = []
for r in T.itertuples():
    a = np.searchsorted(tm, np.datetime64(r.entry_time), side="right")        # 進場那分鐘收盤進場 → 從下一分鐘開始
    b = np.searchsorted(tm, np.datetime64(r.exit_time), side="left")
    if b > a:
        w = np.maximum((r.side * ((L if r.side == 1 else H)[a:b] - r.lvl) - r.spread) / r.risk, -1.0)
        open_arr[a:b] += w
        cnt[a] += 1
        cnt[b] -= 1
        mae_min.append((tm[a + int(np.argmin(w))] - np.datetime64(r.entry_time)) / np.timedelta64(1, "m"))
    else:
        mae_min.append(0.0)
T["mae_min"] = mae_min
T["mae_R"] = (T.mae / T.risk).clip(upper=1)
T["mfe_R"] = T.mfe / T.risk
n_all = np.cumsum(cnt)[:-1]
act = n_all > 0
open_R = pd.Series(open_arr[act], index=tm[act])
n_open = pd.Series(n_all[act], index=tm[act])
real = T.groupby("exit_time").R.sum().sort_index()
ix = open_R.index.union(real.index)
eq_mtm = real.reindex(ix, fill_value=0).cumsum() + open_R.reindex(ix, fill_value=0)
eq_cl = real.cumsum()
dd_mtm, dd_cl = eq_mtm - eq_mtm.cummax(), eq_cl - eq_cl.cummax()

R = T.R
w, l = R[R > 0], R[R <= 0]
days_all = pd.Series(sorted(set((m1.index - pd.Timedelta(hours=1)).normalize())))
days_all = days_all[(days_all >= T.day.min()) & (days_all <= T.day.max()) & (days_all.dt.dayofweek < 5)]
daily = pd.Series(0.0, index=days_all).add(T.groupby("day").R.sum(), fill_value=0)
lose = (R <= 0).astype(int)
streak_l = lose.groupby((lose != lose.shift()).cumsum()).sum().max()
win = (R > 0).astype(int)
streak_w = win.groupby((win != win.shift()).cumsum()).sum().max()
top = T.groupby("day").R.sum().sort_values(ascending=False)
# 回撤持續時間（平倉權益創新高之間最長的天數）
hi = eq_cl.cummax()
newhi = eq_cl.index[eq_cl >= hi]
gaps = pd.Series(newhi).diff().dt.days.max()


def pct(x):
    return f"{100 * x:.1f}%"


P("第三十一部分：L1 核心 + 反向尾巴 — 詳細統計")
P(f"期間 {T.day.min().date()} ~ {T.day.max().date()}（門檻每月月初只用過去資料重算；2023/6~7 是門檻暖身期，不計）")
P(f"目前門檻（用到 2026/10 的全部訊號）：反向尾巴 ≤ {L1['tail'].quantile(0.9):.4f} × ATR10\n")
P("## 整體")
S = {
    "交易筆數": f"{len(T):,}（多 {(T.side == 1).sum():,} / 空 {(T.side == -1).sum():,}）",
    "交易日 / 有單的日子": f"{len(daily)} / {T.day.nunique()}（{T.day.nunique() / len(daily):.0%}）",
    "每個交易日平均筆數": f"{len(T) / len(daily):.2f}（有單日 {len(T) / T.day.nunique():.2f}）",
    "勝率": pct((R > 0).mean()),
    "平均賺 / 平均賠": f"{w.mean():+.2f}R / {l.mean():+.2f}R（賺賠比 {w.mean() / -l.mean():.2f}）",
    "每筆期望": f"{R.mean():+.3f}R（中位數 {R.median():+.2f}R）",
    "PF（以 R 計）": f"{pf(R):.2f}",
    "PF（美元，每筆 1 盎司）": f"{pf(T.pnl):.2f}（2026 金價高，會被拉高）",
    "總損益": f"{R.sum():+.1f}R = 每筆 0.5% 風險、1 萬美元不複利 {R.sum() * RISK_USD:+,.0f} 美元（{R.sum() * 0.5:+.0f}%）",
    "最大回撤（平倉）": f"{dd_cl.min():.1f}R（{dd_cl.min() * 0.5:.1f}%），發生於 {dd_cl.idxmin():%Y-%m-%d}",
    "最大回撤（含浮動，逐分鐘）": f"{dd_mtm.min():.1f}R（{dd_mtm.min() * 0.5:.1f}%），發生於 {dd_mtm.idxmin():%Y-%m-%d %H:%M}",
    "總損益 / 含浮動最大回撤": f"{R.sum() / -dd_mtm.min():.1f}",
    "最長回撤期間（平倉權益沒創新高）": f"{gaps:.0f} 天",
    "日 Sharpe（年化）": f"{daily.mean() / daily.std() * np.sqrt(252):.2f}",
    "正報酬日": pct((daily > 0).mean()),
    "最差單日 / 最好單日": f"{daily.min():+.1f}R / {daily.max():+.1f}R",
    "最長連虧 / 連勝": f"{streak_l} / {streak_w} 筆",
    "獲利集中": f"最好 10 天占 {top.head(10).sum() / R.sum():.0%}；拿掉最好 10 天剩 {top.iloc[10:].sum():+.1f}R、拿掉最好 20 天剩 {top.iloc[20:].sum():+.1f}R",
    "同時持倉 最多 / 平均": f"{int(n_open.max())} / {n_open[n_open > 0].mean():.1f} 筆",
    "帳戶同時浮虧最大": f"{open_R.min():.1f}R（{open_R.min() * 0.5:.1f}%），{open_R.idxmin():%Y-%m-%d %H:%M}，當時 {int(n_open[open_R.idxmin()])} 筆",
    "停損距離": f"中位數 {T.risk.median():.2f} 美元（{(T.risk / T.atr).median():.2f} ATR10）",
    "點差占停損": f"中位數 {(T.spread / T.risk).median():.3f}R",
    "平均持倉": f"{T.hold_h.mean():.1f} 小時（中位數 {T.hold_h.median():.1f}）；停損單 {T[T.exit_reason == '停損'].hold_h.mean():.1f} / 收盤單 {T[T.exit_reason == '收盤'].hold_h.mean():.1f}",
    "出場：停損 / 收盤": f"{(T.exit_reason == '停損').mean():.0%} / {(T.exit_reason == '收盤').mean():.0%}（收盤出場單勝率 {(T[T.exit_reason == '收盤'].R > 0).mean():.0%}）",
    "單筆浮虧（R）中位數 / 贏單中位數": f"{T.mae_R.median():.2f} / {T[T.R > 0].mae_R.median():.2f}",
    "輸單中曾浮盈 ≥ 1R": pct((T[T.R <= 0].mfe_R >= 1).mean()),
    "單筆最大獲利": f"{R.max():+.1f}R",
}
for k, v in S.items():
    P(f"  {k}：{v}")


def grp(key, label):
    g = T.groupby(key)
    tb = pd.DataFrame({"筆數": g.size(), "勝率%": g.R.apply(lambda x: 100 * (x > 0).mean()), "每筆R": g.R.mean(),
                       "總R": g.R.sum(), "PF": g.R.apply(pf), "多PF": g.apply(lambda x: pf(x[x.side == 1].R), include_groups=False),
                       "空PF": g.apply(lambda x: pf(x[x.side == -1].R), include_groups=False)})
    P(f"\n## {label}\n" + tb.round(2).to_string())


grp(T.day.dt.year, "逐年")
grp(T.day.dt.to_period("Q").astype(str), "逐季")
grp(T.side.map({1: "多", -1: "空"}), "方向")
grp(T.day.dt.dayofweek.map(dict(enumerate("一二三四五"))), "星期")
grp(T.h.map(lambda h: f"{int(h):02d}:{int(round(h % 1 * 60)):02d}"), "IB 起點")
grp(T.exit_reason, "出場原因")
grp(pd.cut(T.hold_h, [0, 1, 2, 4, 8, 12, 16, 24], right=False).astype(str), "持倉時間（小時）")
mon = T.groupby(T.day.dt.to_period("M")).R.sum()
P(f"\n## 月份：{len(mon)} 個月中 {(mon > 0).sum()} 個月賺錢；最差月 {mon.min():+.1f}R（{mon.idxmin()}），最好月 {mon.max():+.1f}R（{mon.idxmax()}）")
P("  " + "  ".join(f"{k}:{v:+.0f}" for k, v in mon.items()))

# 帳戶不同風險比例
P("\n## 換算成帳戶（1 萬美元、不複利）")
for r in (0.1, 0.15, 0.25, 0.5):
    P(f"  每筆 {r}%：總報酬 {R.sum() * r:+.0f}%；最大回撤 平倉 {dd_cl.min() * r:.1f}% / 含浮動 {dd_mtm.min() * r:.1f}%；同時最大浮虧 {open_R.min() * r:.1f}%")

T.drop(columns=["ym"]).to_csv("part31_L1tail_trades.csv", index=False, float_format="%.4f")


def charts():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw=dict(height_ratios=[2, 1]))
    em = eq_mtm.resample("1h").last().dropna()
    ax[0].plot(em.index, em.values, lw=1, color="#3b6ea8", label="incl. open positions")
    ax[0].plot(eq_cl.index, eq_cl.values, lw=1.2, color="#c9824a", drawstyle="steps-post", label="closed trades")
    for side, col, lab in ((1, "#2e8b57", "long only"), (-1, "#c0392b", "short only")):
        s = T[T.side == side].sort_values("exit_time")
        ax[0].plot(s.exit_time, s.R.cumsum(), lw=.8, color=col, alpha=.7, label=lab)
    ax[0].axvline(pd.Timestamp("2026-01-01"), color="k", ls="--", lw=.8)
    ax[0].set_ylabel("cumulative R"); ax[0].legend(loc="upper left"); ax[0].grid(alpha=.3)
    ax[0].set_title("L1 core + reverse-tail filter, Aug 2023 – Oct 2026 (1R per trade)")
    dm = dd_mtm.resample("1h").min().dropna()
    ax[1].fill_between(dm.index, dm.values, 0, color="#c0392b", alpha=.5); ax[1].set_ylabel("drawdown incl. open (R)"); ax[1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig("part31_L1tail_equity.png", dpi=110)


charts()

# ---------- M30 兩個延續組合，單獨當策略 ----------
P("\n\n# M30 延續組合單獨交易（第二十八部分四年都成立的兩組）")
P("規則：M30 K 棒收盤時，最近 N 根陽 ≥ k → 收盤價做多；陰 ≥ k → 做空；持有 48 根 M30（約 24 小時）後平倉；")
P("     沒有停損；持倉中不重複進場（不重疊）；扣當根點差。損益以「進場時 ATR20(M30)」為 1 單位，另列美元 PF。")
m30 = m1.resample("30min").agg({"open": "first", "high": "max", "low": "min", "close": "last", "spread": "mean"}).dropna()
d = np.sign(m30.close - m30.open)
atr = (m30.high - m30.low).rolling(20).mean()
c = m30.close.to_numpy()
for N, k in ((25, 15), (30, 18)):
    g, r = (d > 0).rolling(N).sum().to_numpy(), (d < 0).rolling(N).sum().to_numpy()
    rows, i = [], N
    while i < len(m30) - 48:
        s = 1 if g[i] >= k else (-1 if r[i] >= k else 0)
        if s == 0 or not np.isfinite(atr.iloc[i]):
            i += 1
            continue
        pnl = s * (c[i + 48] - c[i]) - m30.spread.iloc[i] * 0.01
        rows.append((m30.index[i], s, pnl, pnl / atr.iloc[i]))
        i += 48
    X = pd.DataFrame(rows, columns=["t", "side", "usd", "A"]).set_index("t")
    P(f"\n## N={N}、≥{k} 根（{k / N:.0%}）")
    P(f"  全期 {len(X)} 筆（多 {(X.side == 1).sum()} / 空 {(X.side == -1).sum()}） PF(美元) {pf(X.usd):.2f}  PF(ATR) {pf(X.A):.2f}  每筆 {X.A.mean():+.2f} ATR  勝率 {pct((X.A > 0).mean())}")
    for y, gy in X.groupby(X.index.year):
        lo, sh = gy[gy.side == 1].A, gy[gy.side == -1].A
        P(f"    {y}: {len(gy):3d} 筆 PF {pf(gy.A):.2f} | 多 {len(lo):3d} 筆 PF {pf(lo):.2f} ({lo.mean():+.2f}) | 空 {len(sh):3d} 筆 PF {pf(sh):.2f} ({sh.mean():+.2f})")
    lo, sh = X[X.side == 1].A, X[X.side == -1].A
    P(f"    多 PF {pf(lo):.2f} / 空 PF {pf(sh):.2f}")
    # 對照：同樣時間點「不看顏色、一律做多 48 根」
    allL = pd.Series((c[48:] - c[:-48]) / atr.to_numpy()[:-48], index=m30.index[:-48]).dropna()
    P(f"    對照：任何時間做多 48 根的平均 {allL.mean():+.2f} ATR（黃金多頭漂移）；做空 {-allL.mean():+.2f}")
out.close()
