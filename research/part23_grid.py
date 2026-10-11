"""第二十三部分：中性網格，格距 = 日 ATR × 0.05，可留過夜，每週五收盤前清倉。

規則（每週重設）：
  - 錨點 = 本週第一根 M1 開盤價；格距 g = 0.05 × ATR14（本週第一天之前 14 個交易日的日真實波幅平均），整週固定
  - 價格位於錨點上方第 k 條格線 → 持有 k 單位空單；下方第 k 條 → k 單位多單（= 每條線掛反向單、停利一格）
  - 有最大層數 N 時：持倉 = −clip(k, −N, N)，超過 N 層不再加碼，回到 N 層以內才開始平倉
  - 週五最後一根 K 棒收盤全部平倉；1 單位 = 1 盎司（0.01 手），損益單位為美元
成本：資料價格為 bid，買進成交在 bid + 點差，賣出成交在 bid；隔夜利息另列「單位·夜」數，請自行乘上券商 swap。
M1 內部路徑：陽線 O→L→H→C、陰線 O→H→L→C，逐段計算碰到的格線。
對照：
  - 每日清倉版（同規則，每天最後一根平倉、隔天重設錨點）
  - 隨機漫步基準：每週的 M1 收盤變動幅度不變、方向隨機（只用收盤路徑，與「實際，只用收盤」比較）
    注意：以 K 棒端點直線內插時，穿過格線後價格「必定」繼續到端點，零成本下也會偏負；
    因此兩者只能互相比較，不能把基準當成 0。
"""
import numpy as np, pandas as pd
from load import load

out = open("results_part23.txt", "w")
def P(s=""): print(s); out.write(s + "\n"); out.flush()

d = load("data/XAUUSD_M1_2023_2026.csv")
d["day"] = d.index.normalize()
day = d.groupby("day").agg(h=("high", "max"), l=("low", "min"), c=("close", "last"))
tr = np.maximum(day.h, day.c.shift()) - np.minimum(day.l, day.c.shift())
atr = tr.shift(1).rolling(14).mean()                       # 只用前 14 個交易日
d["week"] = d.index.to_period("W-SUN")
rng = np.random.default_rng(0)
MULT = 0.05


def bar_path(o, h, l, c):
    return (o, l, h, c) if c >= o else (o, h, l, c)


def run(px_paths, spread, last_of_day, g, anchor, cap, swap_tally):
    """px_paths: 每根 K 棒的價格路徑；回傳 (損益, 交易單位數, 最大持倉, 最差浮動權益, 多單夜數, 空單夜數)。"""
    k = 0; pos = 0; cash = 0.0; trades = 0; maxpos = 0; worst = 0.0; ln = sn = 0
    for i, path in enumerate(px_paths):
        s = spread[i]
        for x in path[1:]:
            while x >= anchor + (k + 1) * g:            # 往上碰線
                k += 1; new = -max(-cap, min(cap, k))
                q = new - pos
                if q: cash += -q * (anchor + k * g) - (s * q if q > 0 else 0); trades += abs(q); pos = new
            while x <= anchor + (k - 1) * g:            # 往下碰線
                k -= 1; new = -max(-cap, min(cap, k))
                q = new - pos
                if q: cash += -q * (anchor + k * g) - (s * q if q > 0 else 0); trades += abs(q); pos = new
        c = path[-1]
        maxpos = max(maxpos, abs(pos))
        worst = min(worst, cash + pos * c - (s * -pos if pos < 0 else 0))
        if swap_tally and last_of_day[i]:
            if pos > 0: ln += pos
            if pos < 0: sn += -pos
    c = px_paths[-1][-1]
    cash += pos * c - (s * -pos if pos < 0 else 0)          # 收盤平倉：空單買回付點差
    trades += abs(pos)
    return cash, trades, maxpos, worst, ln, sn


def simulate(period_key, cap, mode="ohlc"):
    rows = []
    for key, x in d.groupby(period_key):
        a = atr.get(x.day.iloc[0], np.nan)
        if np.isnan(a) or len(x) < 500:
            continue
        g = MULT * a
        o, h, l, c = x.open.values, x.high.values, x.low.values, x.close.values
        sp = x.spread.values / 100
        lod = np.r_[x.day.values[1:] != x.day.values[:-1], False]
        if mode == "ohlc":
            paths = [bar_path(*v) for v in zip(o, h, l, c)]
        else:
            if mode == "rw":
                st = np.diff(c); c = c[0] + np.r_[0, np.cumsum(st * rng.choice((-1, 1), len(st)))]
            paths = [(p, q) for p, q in zip(np.r_[o[0], c[:-1]], c)]
        mult = np.where(x.index.dayofweek.values == 2, 3, 1)  # 星期三三倍 swap
        pnl, tr_, mp, worst, ln, sn = run(paths, sp, lod * mult, g, o[0], cap, True)
        rows.append(dict(key=str(key), start=x.index[0], g=g, atr=a, pnl=pnl, pnl_atr=pnl / a, trades=tr_,
                         maxpos=mp, worst=worst, worst_atr=worst / a, long_n=ln, short_n=sn,
                         net=c[-1] - o[0], net_atr=(c[-1] - o[0]) / a))
    return pd.DataFrame(rows)


def summary(R, lab):
    R = R.copy(); R["year"] = R.start.dt.year
    t = R.pnl_atr.mean() / R.pnl_atr.std() * np.sqrt(len(R))
    P(f"\n{lab}：{len(R)} 期，每期平均 {R.pnl_atr.mean():+.2f} ATR（t {t:+.2f}），勝率 {np.mean(R.pnl > 0):.0%}，"
      f"最差一期 {R.pnl_atr.min():+.1f} ATR，最差浮動 中位 {R.worst_atr.median():+.2f} / 最差 {R.worst_atr.min():+.1f} ATR，"
      f"最大持倉 中位 {R.maxpos.median():.0f} / 最大 {R.maxpos.max():.0f} 單位")
    y = R.groupby("year").agg(期數=("pnl", "size"), 平均ATR=("pnl_atr", "mean"), 勝率=("pnl", lambda s: np.mean(s > 0)),
                              總損益美元=("pnl", "sum"), 最差ATR=("pnl_atr", "min"), 最差浮動ATR=("worst_atr", "min"),
                              多單夜=("long_n", "sum"), 空單夜=("short_n", "sum"))
    P(y.round(2).to_string())
    return R


P("第二十三部分：中性網格 格距 = 0.05 × ATR14，可留過夜，每週收盤清倉（XAUUSD M1 2023/6 ~ 2026/10）")
P("損益以 ATR 為單位（每單位 1 盎司；以美元計時 2026 年 1 ATR ≈ $100 / 盎司）")

W = {}
for cap in (10 ** 9, 40, 20, 10):
    lab = "不限層數" if cap > 1e6 else f"最多 {cap} 層"
    W[cap] = summary(simulate("week", cap), f"[週] {lab}")
W[10 ** 9].to_csv("part23_weekly.csv", index=False, float_format="%.4f")

P("\n" + "=" * 100 + "\n對照組")
summary(simulate("day", 10 ** 9), "[日] 不限層數、每天收盤清倉")
summary(simulate("week", 10 ** 9, "close"), "[週] 不限層數，只用 M1 收盤路徑（實際）")
summary(simulate("week", 10 ** 9, "rw"), "[週] 不限層數，只用 M1 收盤路徑（隨機漫步基準）")

R = W[10 ** 9]
P("\n" + "=" * 100 + "\n週損益 vs 本週淨漲跌（不限層數）")
b = pd.cut(R.net_atr.abs(), [0, .5, 1, 2, 3, 99])
P(R.groupby(b, observed=True).agg(週數=("pnl", "size"), 平均ATR=("pnl_atr", "mean"), 最差ATR=("pnl_atr", "min"),
                                  勝率=("pnl", lambda s: np.mean(s > 0))).round(2).to_string())
P("\n最差 8 週：")
P(R.nsmallest(8, "pnl_atr")[["key", "atr", "g", "net_atr", "pnl", "pnl_atr", "worst_atr", "maxpos"]].round(2).to_string(index=False))
out.close()
