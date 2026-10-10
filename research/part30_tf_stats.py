"""第三十部分：OW 在 M1 / M5 / M15 的詳細交易數據，與三個一起跑的風險放大。

期間：2026/1/2~10/8（M1 有資料的日子；排除 4/3~4/14 缺口），三個週期用同一組交易日。
規則同 FINAL_STRATEGY_OW（參數不變，只換確認用的 K 棒）。帳戶：1 萬美元、每筆 0.25% 風險、不複利。
MAE / MFE 用各自週期的 K 棒高低計算（M1 最精細）。
"""
import numpy as np
import pandas as pd
from part27_ow import data

out = open("results_part30.txt", "w")
def P(s=""): print(s); out.write(s + "\n"); out.flush()
SRC = {"M15": ("data/XAUUSD_M15_2023_2026.csv", 15), "M5": ("data/XAUUSD_M5_2025_2026.csv", 5), "M1": ("data/XAUUSD_M1_2026.csv", 1)}
A, B = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-10-08")
T = {}
for k in SRC:
    x = pd.read_csv(f"part29_ow_{k}.csv", parse_dates=["day", "t_in", "exit_time"])
    T[k] = x[(x.day >= A) & (x.day <= B) & ~x.day.between("2026-04-03", "2026-04-14")].sort_values("t_in").reset_index(drop=True)
days = sorted(set(T["M1"].day) | set(T["M5"].day) | set(T["M15"].day))
alldays = sorted(set(pd.read_csv("data/XAUUSD_M1_2026.csv", sep="\t")["<DATE>"].pipe(lambda s: pd.to_datetime(s, format="%Y.%m.%d"))))
alldays = [d for d in alldays if A <= d <= B and not (pd.Timestamp("2026-04-03") <= d <= pd.Timestamp("2026-04-14"))]

# MAE / MFE（以 R 計）
for k, (path, bm) in SRC.items():
    days_k, D, ARR = data(path, bm)
    mae, mfe = [], []
    for r in T[k].itertuples():
        t, O, H, L, C, S, mins = ARR[r.day]
        j = int(np.searchsorted(t, r.exit_time - pd.Timedelta(minutes=bm)))
        seg = slice(r.i, max(j, r.i) + 1)
        if r.side == 1:
            mae.append((r.lvl - L[seg].min()) / r.risk); mfe.append((H[seg].max() - r.lvl) / r.risk)
        else:
            mae.append((H[seg].max() - r.lvl) / r.risk); mfe.append((r.lvl - L[seg].min()) / r.risk)
    T[k]["mae_R"] = np.clip(mae, 0, None); T[k]["mfe_R"] = np.clip(mfe, 0, None)
    T[k]["hold_h"] = (T[k].exit_time - T[k].t_in).dt.total_seconds() / 3600


def block(X, lab):
    r = X.R; w, l = r[r > 0], r[r <= 0]
    usd = X.R * 25                                         # 1 萬美元、0.25% = 每 R 25 美元
    eq = usd.groupby(X.exit_time).sum().sort_index().cumsum() + 10000
    dl = X.groupby("day").R.sum().reindex(alldays, fill_value=0)
    lose = (r <= 0).astype(int)
    ev = sorted([(a, 1) for a in X.t_in] + [(b, -1) for b in X.exit_time], key=lambda z: (z[0], z[1])); cur = mx = 0
    for _, e in ev:
        cur += e; mx = max(mx, cur)
    mon = X.groupby(X.day.dt.month).R.sum()
    P(f"\n[{lab}]")
    P(f"  交易 {len(X)} 筆（多 {(X.side == 1).sum()} / 空 {(X.side == -1).sum()}），每個交易日 {len(X) / len(alldays):.1f} 筆，有交易的日子 {X.day.nunique()}/{len(alldays)}")
    P(f"  勝率 {len(w) / len(r) * 100:.1f}%  平均賺 {w.mean():+.2f}R / 平均賠 {l.mean():+.2f}R  賺賠比 {w.mean() / -l.mean():.2f}  每筆 {r.mean():+.3f}R（中位 {r.median():+.2f}R）  PF {w.sum() / -l.sum():.2f}")
    P(f"  多單 每筆 {X[X.side == 1].R.mean():+.2f}R / 空單 {X[X.side == -1].R.mean():+.2f}R；停損出場 {(X.exit_reason == '停損').mean() * 100:.0f}%，收盤出場 {(X.exit_reason == '收盤').mean() * 100:.0f}%")
    P(f"  停損距離 中位 {X.risk.median():.1f} 美元（範圍 {X.risk.min():.1f}~{X.risk.max():.1f}）；平均持倉 {X.hold_h.mean():.1f} 小時（停損單 {X[X.exit_reason == '停損'].hold_h.mean():.1f}h）")
    P(f"  最大浮虧 MAE 中位 {X.mae_R.median():.2f}R（贏單 {X[X.R > 0].mae_R.median():.2f}R）；最大浮盈 MFE 中位 {X.mfe_R.median():.2f}R；輸單中曾浮盈 ≥1R {(X[X.R <= 0].mfe_R >= 1).mean() * 100:.0f}%；單筆最大 {r.max():+.1f}R")
    P(f"  總 {r.sum():+.0f}R；1 萬美元 × 0.25%/筆：{eq.iloc[-1]:,.0f}（{(eq.iloc[-1] / 1e4 - 1) * 100:+.1f}%），最大回撤 {((eq - eq.cummax()) / eq.cummax()).min() * 100:.1f}%")
    P(f"  最長連虧 {lose.groupby((lose != lose.shift()).cumsum()).sum().max()} 筆；同時持倉最多 {mx} 筆；最差單日 {dl.min():+.1f}R、最好單日 {dl.max():+.1f}R；日 Sharpe {dl.mean() / dl.std() * np.sqrt(250):.2f}")
    P("  逐月 R：" + "  ".join(f"{m}月 {v:+.0f}" for m, v in mon.items()))
    return dl


DL = {k: block(T[k], k) for k in ["M15", "M5", "M1"]}

P("\n" + "=" * 100 + "\n三個一起跑（各自每筆 0.25%）= 風險放大")
C = pd.concat([T[k].assign(tf=k) for k in T]).sort_values("t_in")
C["key"] = C.day.dt.strftime("%Y-%m-%d") + "|" + C.h.round(3).astype(str) + "|" + C.side.astype(str)
P(f"  同一天、同一 IB、同方向，三個週期都有開：{(C.groupby('key').tf.nunique() == 3).sum()} 組；只有一個週期開：{(C.groupby('key').tf.nunique() == 1).sum()} 組")
P(f"  日損益相關：M15~M5 {DL['M15'].corr(DL['M5']):+.2f}  M15~M1 {DL['M15'].corr(DL['M1']):+.2f}  M5~M1 {DL['M5'].corr(DL['M1']):+.2f}")
dl3 = block(C, "三個合計（每個週期各 0.25%）")
P(f"\n  對照：只跑 M1、每筆 0.75%（= 三倍風險）→ 回撤約 {3 * 1:.0f} 倍於單一 0.25%；三個合計的日 Sharpe {dl3.mean() / dl3.std() * np.sqrt(250):.2f}"
  f" vs M1 單獨 {DL['M1'].mean() / DL['M1'].std() * np.sqrt(250):.2f}（Sharpe 沒有變好 = 沒有分散，只是放大）")

for k in T:
    T[k].to_csv(f"part30_trades_{k}.csv", index=False)

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"axes.grid": True, "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(2, 2, figsize=(14, 9))
cols = {"M15": "#3b6ea8", "M5": "#c9824a", "M1": "#2e8b57"}
for k in ["M15", "M5", "M1"]:
    x = T[k].sort_values("exit_time")
    ax[0, 0].plot(x.exit_time, x.R.cumsum(), color=cols[k], label=k, lw=1.3)
x = C.sort_values("exit_time"); ax[0, 0].plot(x.exit_time, x.R.cumsum(), color="#888", ls="--", label="all three combined", lw=1)
ax[0, 0].set_title("cumulative R (2026, same days)"); ax[0, 0].legend(); ax[0, 0].set_ylabel("R")
w = 0.27
for i, k in enumerate(["M15", "M5", "M1"]):
    m = T[k].groupby(T[k].day.dt.month).R.sum()
    ax[0, 1].bar(m.index + (i - 1) * w, m.values, width=w, color=cols[k], label=k)
ax[0, 1].set_title("R per month"); ax[0, 1].legend(); ax[0, 1].set_xlabel("month")
for k in ["M15", "M5", "M1"]:
    ax[1, 0].hist(T[k].R.clip(-1.3, 8), bins=40, histtype="step", color=cols[k], lw=1.5, label=k)
ax[1, 0].set_title("R distribution per trade"); ax[1, 0].legend()
ax[1, 1].scatter(DL["M5"], DL["M1"], s=12, color="#2e8b57", alpha=.6, label=f"M1 vs M5 (corr {DL['M5'].corr(DL['M1']):.2f})")
ax[1, 1].scatter(DL["M5"], DL["M15"], s=12, color="#3b6ea8", alpha=.6, label=f"M15 vs M5 (corr {DL['M5'].corr(DL['M15']):.2f})")
ax[1, 1].set_xlabel("M5 daily R"); ax[1, 1].set_ylabel("other timeframe daily R"); ax[1, 1].legend(); ax[1, 1].set_title("daily R: nearly the same trades")
plt.tight_layout(); plt.savefig("part30_tf.png", dpi=110)
out.close()
