"""M1 急跌買（第二十三部分樣本外最佳）的完整交易明細與統計。

規則（全部在 M1 收盤時計算）：
  ATR60 = 過去 60 根 M1 真實波幅平均
  v     = ewm(收盤價變化, span 20, adjust=False) / ATR60   （休市 > 6 分鐘後第一根的價格變化當成缺值）
  z     = (v − 過去 1380 根 v 的平均) / 過去 1380 根 v 的標準差
  進場：z 由 > −2.5 變成 ≤ −2.5（首次跌破）→ 下一根開盤市價做多；手上已有單就不做
  停損：進場價 − 3 × ATR60（訊號那根的 ATR60），掛單；開盤就跳空穿過時以開盤價成交
  出場：沒碰到停損 → 第 48 根（進場那根算第 1 根）收盤市價平倉
成本：進場那根的點差（spread 點 × 0.01 美元）；停損 / 出場不另計滑價（另做敏感度）。
單位：美元/盎司 = 0.01 手的美元；bp = ÷ 進場價 × 1e4；R = ÷ 停損距離。
輸出：final_m1dip_trades.csv、final_m1dip_stats.txt、final_m1dip_equity.png、final_m1dip_breakdown.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from load import load

THR, STOP_ATR, HOLD, ZW = 2.5, 3.0, 48, 1380
SPLIT = pd.Timestamp("2025-01-01")
ACCT, RISK = 10_000, 0.005
out = []


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    out.append(s)


m = load("data/XAUUSD_M1_2023_2026.csv")
c = m.close
pc = c.shift()
tr = pd.concat([m.high - m.low, (m.high - pc).abs(), (m.low - pc).abs()], axis=1).max(axis=1)
atr = tr.rolling(60).mean()
gap = m.index.to_series().diff() > pd.Timedelta(minutes=6)
v = c.diff().where(~gap).ewm(span=20, adjust=False).mean() / atr
z = (v - v.rolling(ZW).mean()) / v.rolling(ZW).std()

o, h, l, cl, A, Z = m.open.to_numpy(), m.high.to_numpy(), m.low.to_numpy(), c.to_numpy(), atr.to_numpy(), z.to_numpy()
spr = m.spread.to_numpy() * 0.01
ts = m.index
trig = Z <= -THR
pos = np.flatnonzero(trig & ~np.r_[False, trig[:-1]])
rows, busy = [], -1
for t in pos:
    if t <= busy or t + HOLD + 1 >= len(m) or not np.isfinite(A[t]):
        continue
    e, x = t + 1, t + HOLD
    if ts[x] - ts[t] > pd.Timedelta(minutes=HOLD * 1.6 + 30):        # 中間跨休市 → 不做
        continue
    ep = o[e]
    sd = STOP_ATR * A[t]
    sp = ep - sd
    hit = np.flatnonzero(l[e:x + 1] <= sp)
    if len(hit):
        xi = e + hit[0]
        xp = min(o[xi], sp) if xi > e else sp                         # 跳空穿過停損 → 開盤價
        why = "停損"
    else:
        xi, xp, why = x, cl[x], "時間"
    busy = xi
    gross = xp - ep
    pnl = gross - spr[e]
    seg_l, seg_h = l[e:xi + 1].min(), h[e:xi + 1].max()
    rows.append(dict(
        訊號時間=ts[t], z=Z[t], v=v.iloc[t], ATR60=A[t], 進場時間=ts[e], 進場價=ep, 停損價=sp, 停損距離=sd,
        出場時間=ts[xi], 出場價=xp, 出場原因=why, 持倉_分鐘=int((ts[xi] - ts[e]).total_seconds() // 60) + 1,
        點差=spr[e], 毛損益=gross, 損益_美元每盎司=pnl, 損益_bp=pnl / ep * 1e4, 損益_R=pnl / sd,
        最大浮虧=seg_l - ep, 最大浮虧_R=(seg_l - ep) / sd, 最大浮盈=seg_h - ep, 最大浮盈_R=(seg_h - ep) / sd,
        建議手數_1萬美元_0_5風險=round(ACCT * RISK / (sd * 100), 2)))
T = pd.DataFrame(rows)
T["年"] = T.訊號時間.dt.year
T["樣本"] = np.where(T.訊號時間 < SPLIT, "樣本內", "樣本外")
T.round({k: 4 for k in T.columns if T[k].dtype == float}).to_csv("final_m1dip_trades.csv", index=False, encoding="utf-8-sig")


def block(S, label):
    r, b, R = S.損益_美元每盎司, S.損益_bp, S.損益_R
    w, lo = R[R > 0], R[R <= 0]
    eqR = R.cumsum()
    ddR = (eqR - eqR.cummax()).min()
    s = (R <= 0).astype(int)
    maxl = s.groupby((s != s.shift()).cumsum()).sum().max()
    days = S.訊號時間.dt.normalize().nunique()
    p(f"[{label}]")
    p(f"  筆數 {len(S)}，交易日 {days}，每日 {len(S) / days:.2f} 筆；勝率 {(R > 0).mean():.1%}")
    p(f"  平均賺 / 賠 {w.mean():+.3f}R / {lo.mean():+.3f}R（賺賠比 {w.mean() / -lo.mean():.2f}）；PF {w.sum() / -lo.sum():.3f}")
    p(f"  每筆期望 {R.mean():+.4f}R = {b.mean():+.3f}bp = {r.mean():+.3f} 美元/盎司（t {R.mean() / R.std() * np.sqrt(len(R)):+.2f}）；中位數 {R.median():+.3f}R")
    p(f"  總計 {R.sum():+.1f}R / {b.sum():+.0f}bp / {r.sum():+.1f} 美元/盎司；最大回撤 {ddR:.1f}R；最長連虧 {maxl} 筆")


p("=" * 100)
p("M1 急跌買：z_v20(1380) 首次 ≤ −2.5 → 下一根開盤做多；停損 3×ATR60；持有 48 根")
p(f"資料 {m.index[0]:%Y-%m-%d} ~ {m.index[-1]:%Y-%m-%d}；樣本內 < 2025-01-01（挑參數用），樣本外 ≥ 2025-01-01")
p("=" * 100)
block(T, "全期")
for k in ("樣本內", "樣本外"):
    block(T[T.樣本 == k], k)

p("")
p("--- 出場原因")
g = T.groupby("出場原因").agg(筆數=("損益_R", "size"), 比例=("損益_R", lambda s: len(s) / len(T)), 平均R=("損益_R", "mean"),
                           勝率=("損益_R", lambda s: (s > 0).mean()), 平均持倉分=("持倉_分鐘", "mean"))
p(g.round(3).to_string())
p(f"  停損單中跳空以開盤價成交（比停損價差）的筆數：{((T.出場原因 == '停損') & (T.出場價 < T.停損價 - 1e-9)).sum()}")

p("")
p("--- 逐年")
g = T.groupby("年").agg(筆數=("損益_R", "size"), 勝率=("損益_R", lambda s: (s > 0).mean()), 每筆R=("損益_R", "mean"),
                      每筆bp=("損益_bp", "mean"), 每筆美元=("損益_美元每盎司", "mean"), 總R=("損益_R", "sum"),
                      停損距離中位=("停損距離", "median"), 點差中位=("點差", "median"), 建議手數中位=("建議手數_1萬美元_0_5風險", "median"))
p(g.round(3).to_string())

p("")
p("--- 每月（總 R）")
mo = T.groupby(T.訊號時間.dt.to_period("M")).損益_R.sum()
mt = mo.to_frame("R")
mt["年"], mt["月"] = mt.index.year, mt.index.month
p(mt.pivot(index="年", columns="月", values="R").round(1).to_string())
p(f"  正報酬月份 {(mo > 0).sum()} / {len(mo)}；最好月 {mo.max():+.1f}R（{mo.idxmax()}）；最差月 {mo.min():+.1f}R（{mo.idxmin()}）")

p("")
p("--- 每日")
dy = T.groupby(T.訊號時間.dt.normalize()).損益_R.sum()
p(f"  正報酬日 {(dy > 0).mean():.0%}；最好日 {dy.max():+.1f}R；最差日 {dy.min():+.1f}R；日 Sharpe（年化 252）{dy.mean() / dy.std() * np.sqrt(252):.2f}")

p("")
p("--- 進場時段（broker 時間 = 紐約 + 7）")
T["時段"] = pd.cut(T.進場時間.dt.hour, [0, 4, 8, 11, 14, 17, 20, 24], right=False,
                 labels=["00-03", "04-07", "08-10", "11-13", "14-16", "17-19", "20-23"])
g = T.groupby("時段", observed=True).agg(筆數=("損益_R", "size"), 每筆R=("損益_R", "mean"), 勝率=("損益_R", lambda s: (s > 0).mean()),
                                         樣本內R=("損益_R", lambda s: s[T.loc[s.index, "樣本"] == "樣本內"].mean()),
                                         樣本外R=("損益_R", lambda s: s[T.loc[s.index, "樣本"] == "樣本外"].mean()),
                                         點差_R=("點差", lambda s: (s / T.loc[s.index, "停損距離"]).mean()))
p(g.round(3).to_string())

p("")
p("--- 星期")
wd = T.groupby(T.進場時間.dt.dayofweek).agg(筆數=("損益_R", "size"), 每筆R=("損益_R", "mean"), 勝率=("損益_R", lambda s: (s > 0).mean()))
wd.index = ["一", "二", "三", "四", "五", "六", "日"][:len(wd)] if len(wd) <= 7 else wd.index
p(wd.round(3).to_string())

p("")
p("--- 訊號深度（z 越低 = 跌越急）與波動（ATR60 相對過去 1 天平均）")
T["z分組"] = pd.cut(T.z, [-np.inf, -4, -3.5, -3, -2.5], labels=["≤−4", "−4~−3.5", "−3.5~−3", "−3~−2.5"])
p(T.groupby("z分組", observed=True).agg(筆數=("損益_R", "size"), 每筆R=("損益_R", "mean"), 勝率=("損益_R", lambda s: (s > 0).mean())).round(3).to_string())
atr_rel = (atr / atr.rolling(1380).mean()).reindex(T.訊號時間).to_numpy()
T["ATR相對"] = pd.qcut(atr_rel, 4, labels=["低", "中低", "中高", "高"])
p(T.groupby("ATR相對", observed=True).agg(筆數=("損益_R", "size"), 每筆R=("損益_R", "mean"), 勝率=("損益_R", lambda s: (s > 0).mean())).round(3).to_string())

p("")
p("--- 持倉與浮動")
p(f"  持倉分鐘：平均 {T.持倉_分鐘.mean():.1f}，停損單中位 {T[T.出場原因 == '停損'].持倉_分鐘.median():.0f}，時間出場固定 {HOLD}")
p(f"  最大浮虧_R 中位 {T.最大浮虧_R.median():+.2f}；贏單的最大浮虧_R 中位 {T[T.損益_R > 0].最大浮虧_R.median():+.2f}")
p(f"  最大浮盈_R 中位 {T.最大浮盈_R.median():+.2f}；輸單中曾浮盈 ≥ 1R 的比例 {(T[T.損益_R <= 0].最大浮盈_R >= 1).mean():.0%}")
p(f"  停損距離：中位 {T.停損距離.median():.2f} 美元（2023 {T[T.年 == 2023].停損距離.median():.2f} → 2026 {T[T.年 == 2026].停損距離.median():.2f}）")

p("")
p("--- 滑價敏感度（每筆額外成本，單位 = 停損距離的比例 / bp）")
for s in (0.02, 0.05, 0.1):
    R2 = T.損益_R - s
    p(f"  每筆再扣 {s:.2f}R：每筆 {R2.mean():+.4f}R，t {R2.mean() / R2.std() * np.sqrt(len(R2)):+.2f}，樣本外 {R2[T.樣本 == '樣本外'].mean():+.4f}R")
for s in (0.5, 1.0, 2.0):
    b2 = T.損益_bp - s
    p(f"  每筆再扣 {s:.1f}bp：每筆 {b2.mean():+.3f}bp，樣本外 {b2[T.樣本 == '樣本外'].mean():+.3f}bp")
p(f"  損益兩平的額外成本 ≈ {T.損益_R.mean():.3f}R ≈ {T.損益_bp.mean():.2f}bp ≈ {T.損益_美元每盎司.mean():.2f} 美元/盎司（全期平均）")

p("")
p(f"--- 帳戶模擬：{ACCT:,} 美元、每筆固定風險 {RISK:.1%}（不複利）")
eq = ACCT + (T.損益_R * ACCT * RISK).cumsum()
dd = (eq - eq.cummax()) / eq.cummax()
p(f"  總報酬 {eq.iloc[-1] / ACCT - 1:+.1%}；平倉最大回撤 {dd.min():.1%}；逐年報酬 "
  + " ".join(f"{y}:{v:+.1%}" for y, v in (T.損益_R * RISK).groupby(T.年).sum().items()))

open("final_m1dip_stats.txt", "w").write("\n".join(out) + "\n")

# ---------------------------------------------------------------- 圖
fig, ax = plt.subplots(3, 1, figsize=(11, 10), sharex=True, gridspec_kw=dict(height_ratios=[3, 1.3, 1.3]))
eqR = T.損益_R.cumsum()
ax[0].plot(T.訊號時間, eqR, color="tab:green", label="cum R (after spread)")
for s in (0.05, 0.1):
    ax[0].plot(T.訊號時間, (T.損益_R - s).cumsum(), lw=0.9, label=f"extra cost {s}R/trade")
ax[0].axvline(SPLIT, color="k", ls="--", lw=0.8)
ax[0].axhline(0, color="grey", lw=0.8)
ax[0].set_title("M1 dip-buy: z<=-2.5, stop 3xATR60, hold 48 min  (dashed = out-of-sample start)")
ax[0].legend(fontsize=8)
ax[1].fill_between(T.訊號時間, eqR - eqR.cummax(), 0, color="tab:red", alpha=0.5)
ax[1].set_title("drawdown (R)")
ax[2].bar(mo.index.to_timestamp(), mo.values, width=20, color=np.where(mo.values > 0, "tab:green", "tab:red"))
ax[2].set_title("monthly R")
fig.tight_layout()
fig.savefig("final_m1dip_equity.png", dpi=110)

fig, ax = plt.subplots(1, 3, figsize=(13, 4))
hr = T.groupby(T.進場時間.dt.hour).損益_R.mean()
ax[0].bar(hr.index, hr.values, color=np.where(hr.values > 0, "tab:green", "tab:red"))
ax[0].set_title("avg R by entry hour (broker)")
ax[1].hist(T.損益_R.clip(-1.5, 3), bins=60, color="tab:blue")
ax[1].set_title("R distribution (clipped)")
ax[2].scatter(T.最大浮虧_R.clip(-1.5, 0), T.最大浮盈_R.clip(0, 4), s=2, c=np.where(T.損益_R > 0, "tab:green", "tab:red"), alpha=0.4)
ax[2].set_xlabel("MAE (R)")
ax[2].set_ylabel("MFE (R)")
ax[2].set_title("MAE vs MFE")
fig.tight_layout()
fig.savefig("final_m1dip_breakdown.png", dpi=110)
