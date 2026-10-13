"""第二十二部分：M1 速度 v / 加速度 a（EWMA 版）+ z-score 長度 / EWMA z-score 比較 + 事件交易明細。

v{n} = ewm(收盤價變化, span=n) / ATR60；a{n} = ewm(v{n} 的變化, span=n)（以 ATR60 為單位）。
z-score：滾動（平均 / 標準差，W 根）與 EWMA（ewm 平均 / ewm 標準差，span=W）兩種，W = 60~7200 根。
事件交易：z 首次越過門檻 → 下一根開盤進場，持有 60 分鐘後以該根收盤出場；同一訊號冷卻 60 分（不重疊）。
成本：進場那根的點差（spread 點 × 0.01 美元）。
去漂移：未來報酬減「過去 20 個交易日每分鐘平均報酬 × 持有分鐘」（無前視），只用來判斷是不是只吃到金價趨勢。
資料：data/XAUUSD_M1_2023_2026.csv（2023/6 ~ 2026/10）。
輸出：results_part22.txt、part22_trades.csv、part22_equity.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from load import load

HOLD, COOL = 60, 60
WINS = (60, 240, 720, 1380, 2880, 7200)          # 1 小時、4 小時、半天、1 天、2 天、5 天（M1 根數）
out_lines = []


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    out_lines.append(s)


m = load("data/XAUUSD_M1_2023_2026.csv")
c = m.close
pc = c.shift()
tr = pd.concat([m.high - m.low, (m.high - pc).abs(), (m.low - pc).abs()], axis=1).max(axis=1)
atr = tr.rolling(60).mean()
gap = m.index.to_series().diff() > pd.Timedelta(minutes=5)       # 週末 / 休市後第一根
d = c.diff().where(~gap)
F = pd.DataFrame(index=m.index)
F["atr"] = atr
for n in (5, 20, 60):
    F[f"v{n}"] = d.ewm(span=n, adjust=False).mean() / atr
    F[f"a{n}"] = F[f"v{n}"].diff().ewm(span=n, adjust=False).mean()
F["f60"] = (c.shift(-HOLD) - c) / atr
F["rng15"] = (m.high.rolling(15).max().shift(-15) - m.low.rolling(15).min().shift(-15)) / atr
mu1 = (d / atr).rolling(27600, min_periods=5000).mean()          # 過去 20 個交易日的每分鐘漂移
F["x60"] = F.f60 - mu1 * HOLD
F["yr"] = F.index.year


def z_roll(s, W):
    return (s - s.rolling(W).mean()) / s.rolling(W).std()


def z_ewm(s, W):
    return (s - s.ewm(span=W, adjust=False).mean()) / s.ewm(span=W, adjust=False).std()


def z_roll_wins(s, W, q=0.01):
    return z_roll(s.clip(s.rolling(W).quantile(q), s.rolling(W).quantile(1 - q)), W)


# ---------------------------------------------------------------- 事件交易模擬
o_, h_, l_, c_ = (m[k].to_numpy() for k in ("open", "high", "low", "close"))
spr = m.spread.to_numpy() * 0.01
ts = m.index.to_numpy()
atr_ = atr.to_numpy()
x60_ = F.x60.to_numpy()


def trades(z, thr, side, name):
    """z 首次越過門檻（side=+1 做多：z≤−thr 為急跌買、z≥+thr 為急漲追多，由呼叫端給 mask）。"""
    mask = z.to_numpy()
    pos = np.flatnonzero(mask)
    rows, last = [], -10 ** 9
    for t in pos:
        if t - last < COOL or t + HOLD >= len(m) or not np.isfinite(atr_[t]):
            continue
        e, x = t + 1, t + HOLD
        if (ts[x] - ts[t]) > np.timedelta64(HOLD + 15, "m"):      # 中間跨休市就跳過
            continue
        last = t
        ep, xp, A = o_[e], c_[x], atr_[t]
        pnl = side * (xp - ep) - spr[e]
        hi, lo = h_[e:x + 1].max(), l_[e:x + 1].min()
        mae = (lo - ep) if side == 1 else (ep - hi)
        mfe = (hi - ep) if side == 1 else (ep - lo)
        rows.append(dict(訊號=name, 方向="多" if side == 1 else "空", 觸發時間=m.index[t], 進場時間=m.index[e], 進場價=ep,
                         出場時間=m.index[x], 出場價=xp, 點差_美元=spr[e], 點差_ATR=spr[e] / A, ATR60=A, 損益_美元=pnl, 損益_ATR=pnl / A,
                         去漂移_ATR=side * x60_[t] - spr[e] / A, 最大浮虧_ATR=mae / A, 最大浮盈_ATR=mfe / A))
    return pd.DataFrame(rows)


def summary(T, label):
    if len(T) == 0:
        p(f"{label:40s} 無交易")
        return
    r = T.損益_ATR
    w, l = r[r > 0], r[r <= 0]
    pf = w.sum() / -l.sum() if len(l) else np.inf
    t = r.mean() / r.std() * np.sqrt(len(r))
    eq = T.損益_美元.cumsum()
    dd = (eq - eq.cummax()).min()
    streak = (r <= 0).astype(int)
    maxl = streak.groupby((streak != streak.shift()).cumsum()).sum().max()
    days = T.觸發時間.dt.normalize().nunique()
    yr = r.groupby(T.觸發時間.dt.year).mean()
    xd = T.去漂移_ATR.groupby(T.觸發時間.dt.year).mean()
    p(f"{label:40s} n={len(T):5d} 每交易日 {len(T) / days:.2f} 勝率 {len(w) / len(r):.1%} 均賺 {w.mean():+.2f} 均賠 {l.mean():+.2f} "
      f"PF {pf:.2f} 期望 {r.mean():+.3f}ATR (t {t:+.2f}) 去漂移 {T.去漂移_ATR.mean():+.3f} | 總 ${T.損益_美元.sum():+.0f} 最大回撤 ${dd:.0f} 最長連敗 {maxl}")
    p(f"{'':40s} 逐年 ATR: " + " ".join(f"{y}:{v:+.3f}" for y, v in yr.items())
      + " | 去漂移: " + " ".join(f"{y}:{v:+.3f}" for y, v in xd.items()))


# ================================================================ A. z-score 長度 / EWMA
p("=" * 120)
p("第二十二部分 A：z_v20 的 z-score 方式比較（滾動 vs EWMA，W = 60~7200 根）")
p("  IC60 = Spearman(z, 未來 60 分報酬)，各年平均（最差年）；volIC = Spearman(|z|, 未來 15 分振幅/ATR) 各年平均")
p("  事件：門檻 ±2、冷卻 60 分、持有 60 分、扣點差，單位 ATR；跌買 = z≤−2 做多，漲追 = z≥+2 做多（括號內為去漂移）")
p("=" * 120)
Z = {}
for W in WINS:
    Z[f"roll{W}"] = z_roll(F.v20, W)
    Z[f"ewm{W}"] = z_ewm(F.v20, W)
Z["roll1380_wins"] = z_roll_wins(F.v20, 1380)
valid = F.dropna(subset=["f60", "x60", "rng15"]).index.intersection(Z["roll7200"].dropna().index)
V = F.loc[valid]
order = [f"roll{W}" for W in WINS] + [f"ewm{W}" for W in WINS] + ["roll1380_wins"]
rank_f60 = V.groupby("yr").f60.rank()
rank_rng = V.groupby("yr").rng15.rank()
for k in order:
    z = Z[k].loc[valid]
    ic = pd.Series({y: z[g.index].rank().corr(rank_f60[g.index]) for y, g in V.groupby("yr")})
    vic = pd.Series({y: z[g.index].abs().rank().corr(rank_rng[g.index]) for y, g in V.groupby("yr")})
    f2 = (z.abs() > 2).mean()
    mc = (z.abs() > 2).groupby(z.index.to_period("M")).mean()
    zf = Z[k].where(F.index.isin(valid))
    Tb = trades(zf <= -2, 2, 1, "跌買")
    Tc = trades(zf >= 2, 2, 1, "漲追")
    p(f"{k:14s} IC60 {ic.mean():+.4f}({ic.max():+.4f}) volIC {vic.mean():+.3f}  |z|>2 {f2:5.1%} 每月CV {mc.std() / mc.mean():.2f} 峰度 {z.kurt():5.1f}"
      f" | 跌買 n={len(Tb):5d} {Tb.損益_ATR.mean():+.3f}({Tb.去漂移_ATR.mean():+.3f}) 漲追 n={len(Tc):5d} {Tc.損益_ATR.mean():+.3f}({Tc.去漂移_ATR.mean():+.3f})")

# ================================================================ B. 交易明細（事先指定：滾動 1380 + winsorize 1%）
p("")
p("=" * 120)
p("第二十二部分 B：事件交易明細（z = 滾動 1380 根 + winsorize 1%，上一輪建議的設定；門檻 2.0 與 2.5）")
p("  ATR = 觸發當下的 ATR60（M1 每根平均真實波幅，60 分鐘的典型波動約 5~9 ATR）；美元 = 每盎司（= 0.01 手）。2023 金價約 2000、2026 約 4000，美元不能跨年直接比，看 ATR。")
p("=" * 120)
zb = Z["roll1380_wins"]
allT = []
for thr in (2.0, 2.5):
    for name, mask, side in ((f"急跌買 z≤−{thr}", zb <= -thr, 1), (f"急漲追多 z≥+{thr}", zb >= thr, 1),
                             (f"急跌追空 z≤−{thr}", zb <= -thr, -1), (f"急漲做空 z≥+{thr}", zb >= thr, -1)):
        T = trades(mask, thr, side, name)
        summary(T, name)
        allT.append(T)
T = pd.concat(allT, ignore_index=True)
T.round({k: 4 for k in T.columns if T[k].dtype == float}).to_csv("part22_trades.csv", index=False, encoding="utf-8-sig")

for name in ("急跌買 z≤−2.5", "急漲追多 z≥+2.5"):
    S = T[T.訊號 == name].copy()
    p("")
    p(f"--- {name}：依進場時段（broker 時間）")
    S["時段"] = pd.cut(S.進場時間.dt.hour, [0, 8, 14, 18, 24], right=False, labels=["01-07 亞洲", "08-13 倫敦", "14-17 紐約上半", "18-23 紐約下半"])
    g = S.groupby("時段", observed=True).agg(n=("損益_ATR", "size"), 期望_ATR=("損益_ATR", "mean"), 去漂移=("去漂移_ATR", "mean"),
                                               勝率=("損益_ATR", lambda r: (r > 0).mean()), 點差_ATR=("點差_ATR", "mean"))
    p(g.round(3).to_string())
    p(f"--- {name}：每月損益（ATR 加總）為正的月份比例 "
      f"{(S.groupby(S.觸發時間.dt.to_period('M')).損益_ATR.sum() > 0).mean():.0%}；"
      f"最大浮虧中位 {S.最大浮虧_ATR.median():+.2f} ATR，10% 分位 {S.最大浮虧_ATR.quantile(.1):+.2f} ATR")

# ---------------------------------------------------------------- 權益曲線（ATR 累積）
fig, ax = plt.subplots(figsize=(11, 5))
for name in ("急跌買 z≤−2.0", "急跌買 z≤−2.5", "急漲追多 z≥+2.0", "急漲追多 z≥+2.5"):
    S = T[T.訊號 == name]
    ax.plot(S.觸發時間, S.損益_ATR.cumsum(), label=name.replace("急跌買", "dip-buy").replace("急漲追多", "spike-long")
            .replace("z≤−", "z<=-").replace("z≥+", "z>=+"))
ax.axhline(0, color="grey", lw=0.8)
ax.set_title("Part 22 event trades: cumulative P&L (ATR units, after spread)")
ax.set_ylabel("cum ATR")
ax.legend()
fig.tight_layout()
fig.savefig("part22_equity.png", dpi=110)

open("results_part22.txt", "w").write("\n".join(out_lines) + "\n")
