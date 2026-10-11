"""第二十二部分：以 M1 路徑計算每日 ER（Kaufman 效率比），評估黃金網格的適用性。

ER_day = |當日最後收盤 − 當日開盤| / Σ|逐分鐘收盤變動|（第一根用 |C0 − O0|）
  0 = 來回震盪（網格友善）、1 = 單邊直線（網格最怕）
交易日 = broker 01:00 ~ 23:59（紐約 + 7），只保留 ≥ 1,000 根 M1 的完整日。

另外計算：
  - 隨機漫步基準：N 步隨機漫步 E[ER] ≈ sqrt(2/(πN))/sqrt(2/π) = 1/sqrt(N)·…（用洗牌模擬）
  - 解析度：M1 / M5 / M15 / H1 路徑的 ER（M1 路徑含點差雜訊，格距大時會高估可吃到的來回）
  - 網格尺度：以格距 g 的 renko 磚數 B 計算 ER_g = |淨變動| / (B·g)、
    完成來回 ≈ (B − |淨|/g) / 2、收盤留倉格數 ≈ |淨|/g
  - 可預測性：昨日 ER / 前 5 日 ER 與今日 ER 的相關
"""
import numpy as np, pandas as pd
from load import load

out = open("results_part22.txt", "w")
def P(s=""): print(s); out.write(s + "\n"); out.flush()

d = load("data/XAUUSD_M1_2023_2026.csv")
d["day"] = d.index.normalize()
n = d.groupby("day").size()
d = d[d.day.isin(n[n >= 1000].index)]
rng = np.random.default_rng(0)


def path_er(c, o0):
    steps = np.abs(np.diff(np.r_[o0, c]))
    return abs(c[-1] - o0) / steps.sum(), steps.sum()


def renko(c, g):
    """固定格線（g 的整數倍）上的觸線序列：碰到目前格線上一條記 +1、下一條記 −1（M1 收盤，可一次多條）。"""
    lv, b = round(c[0] / g), []
    for x in c[1:]:
        while x >= (lv + 1) * g: lv += 1; b.append(1)
        while x <= (lv - 1) * g: lv -= 1; b.append(-1)
    return np.array(b)


def grid_stats(c, g):
    b = renko(c, g); B = len(b)
    return B, (b[1:] == b[:-1]).sum(), max(B - 1, 0)


rows = []
for day, x in d.groupby("day"):
    c, o0 = x.close.values, x.open.values[0]
    er1, path = path_er(c, o0)
    r = dict(day=day, open=o0, close=c[-1], hi=x.high.max(), lo=x.low.min(),
             net=c[-1] - o0, path=path, er_m1=er1, spread=x.spread.median() / 100)
    for m in (5, 15, 60):
        cc = c[m - 1::m]
        r[f"er_m{m}"] = path_er(cc, o0)[0]
    # 隨機漫步基準：同一天的 M1 變動幅度不變、方向隨機
    st = np.diff(np.r_[o0, c])
    r["er_rw"] = np.mean([abs((st * rng.choice((-1, 1), len(st))).sum()) / np.abs(st).sum() for _ in range(20)])
    # 時段 ER（broker 時間）：亞洲 01-09、倫敦 09-15、紐約 15-24
    for lab, a, b in (("asia", 1, 9), ("ldn", 9, 15), ("ny", 15, 24)):
        s = x[(x.index.hour >= a) & (x.index.hour < b)]
        r[f"er_{lab}"] = path_er(s.close.values, s.open.values[0])[0] if len(s) > 30 else np.nan
    rows.append(r)

D = pd.DataFrame(rows).set_index("day")
D["range"] = D.hi - D.lo
D["atr10"] = D.range.shift(1).rolling(10).mean()
for g_atr in (0.02, 0.05, 0.10):            # 格距 = ATR10 的 2% / 5% / 10%
    col = f"{int(g_atr*100)}"
    vals = []
    for day, x in d.groupby("day"):
        a = D.at[day, "atr10"]
        if np.isnan(a):
            vals.append((np.nan,) * 7); continue
        g = g_atr * a
        c = x.close.values
        B, cont, pairs = grid_stats(c, g)      # cont = 下一條觸線與上一條同方向（順勢）的次數
        st = np.diff(c)                        # 隨機漫步基準：同樣的 M1 變動幅度、方向隨機
        _, cr, pr = grid_stats(c[0] + np.r_[0, np.cumsum(st * rng.choice((-1, 1), len(st)))], g)
        k = abs(D.at[day, "net"]) / g
        vals.append((abs(D.at[day, "net"]) / (B * g) if B else np.nan, (B - k) / 2, k, cont, pairs, cr, pr))
    D[[f"erg{col}", f"rt{col}", f"inv{col}", f"cont{col}", f"pairs{col}", f"crw{col}", f"prw{col}"]] = np.array(vals)
D.to_csv("part22_daily_er.csv", float_format="%.5f")

q = [.1, .25, .5, .75, .9]
def dist(s): return "  ".join(f"{v:.3f}" for v in s.quantile(q)) + f"  | 平均 {s.mean():.3f}"

P("第二十二部分：每日 ER（M1 路徑）— XAUUSD 2023/6 ~ 2026/10，%d 個完整交易日" % len(D))
P("\n1. 分布（P10 P25 P50 P75 P90 | 平均）")
P(f"  ER（M1 路徑）       {dist(D.er_m1)}")
P(f"  隨機漫步基準      {dist(D.er_rw)}")
for m in (5, 15, 60):
    P(f"  ER（M{m} 路徑）{'':6s}{dist(D[f'er_m{m}'])}")
P(f"  ER 門檻占比：<0.05 {np.mean(D.er_m1<.05):.0%}  <0.10 {np.mean(D.er_m1<.10):.0%}  "
  f"0.10~0.20 {np.mean((D.er_m1>=.1)&(D.er_m1<.2)):.0%}  ≥0.20 {np.mean(D.er_m1>=.2):.0%}  ≥0.30 {np.mean(D.er_m1>=.3):.0%}")

P("\n2. 逐年（中位數）")
D["year"] = D.index.year
g = D.groupby("year")
P(g.agg(天數=("er_m1", "size"), ER_M1=("er_m1", "median"), ER_M1平均=("er_m1", "mean"), ER_M5=("er_m5", "median"),
        ER_M15=("er_m15", "median"), 路徑=("path", "median"), 淨變動=("net", lambda s: s.abs().median()),
        區間=("range", "median"), 路徑_區間=("path", lambda s: (s / D.loc[s.index, "range"]).median())).round(3).to_string())

P("\n3. 星期（中位 ER_M1）")
P(D.groupby(D.index.dayofweek).er_m1.agg(["size", "median", "mean"]).rename(index=dict(enumerate("一二三四五六日"))).round(3).to_string())

P("\n4. 時段 ER（中位）  亞洲 01-09 %.3f   倫敦 09-15 %.3f   紐約 15-24 %.3f" %
  (D.er_asia.median(), D.er_ldn.median(), D.er_ny.median()))

P("\n5. 網格尺度（格距 = ATR10 × 比例；固定格線觸線數，M1 收盤價計）")
for col in ("2", "5", "10"):
    P(f"  格距 {col:>2s}% ATR：ER_g 中位 {D[f'erg{col}'].median():.3f}（P90 {D[f'erg{col}'].quantile(.9):.3f}）"
      f"  完成來回 中位 {D[f'rt{col}'].median():6.1f}  收盤留倉格數 中位 {D[f'inv{col}'].median():4.1f}（P90 {D[f'inv{col}'].quantile(.9):4.1f}）")

P("  順勢機率 P(下一條觸線同方向)：括號內為隨機漫步基準（同幅度、方向隨機）；低於基準 = 均值回歸（網格有利）")
for col in ("2", "5", "10"):
    def pc(m, a="cont", b="pairs"): return D.loc[m, f"{a}{col}"].sum() / D.loc[m, f"{b}{col}"].sum()
    P(f"  格距 {col:>2s}% ATR：" + "  ".join(f"{y} {pc(D.year == y):.3f}({pc(D.year == y, 'crw', 'prw'):.3f})"
                                         for y in sorted(D.year.unique()))
      + f"  | 全部 {pc(D.year > 0):.3f}({pc(D.year > 0, 'crw', 'prw'):.3f}) n={int(D[f'pairs{col}'].sum())}")

P("\n5b. 變異數比 VR(q) = Var(q 分鐘報酬) / (q × Var(1 分鐘報酬))：1 = 隨機漫步，< 1 = 均值回歸")
lr = np.log(d.close).diff()
lr[d.day != d.day.shift()] = np.nan          # 不跨日
for y in sorted(D.year.unique()):
    s1 = lr[lr.index.year == y]
    cells = []
    for qn in (5, 15, 60, 240):
        rq = s1.rolling(qn).sum()
        cells.append(f"VR({qn}) {rq.var() / (qn * s1.var()):.3f}")
    P(f"  {y}  " + "  ".join(cells))

P("\n6. 可預測性（Spearman 相關）")
D["er_prev"] = D.er_m1.shift(1); D["er_5"] = D.er_m1.shift(1).rolling(5).mean()
D["er_asia_now"] = D.er_asia
D["er_rest"] = D.er_ny
for a, b, lab in (("er_prev", "er_m1", "昨日 ER → 今日 ER"), ("er_5", "er_m1", "前 5 日平均 ER → 今日 ER"),
                  ("er_asia", "er_ldn", "亞洲 ER → 倫敦 ER"), ("er_ldn", "er_ny", "倫敦 ER → 紐約 ER")):
    m = D[[a, b]].dropna()
    P(f"  {lab:22s} ρ = {m[a].rank().corr(m[b].rank()):+.3f}（n={len(m)}）")
P("  昨日 ER 五分位 → 今日 ER 中位：")
qq = pd.qcut(D.er_prev, 5)
P("   " + "  ".join(f"{c.left:.2f}~{c.right:.2f}: {v:.3f}" for c, v in D.groupby(qq, observed=True).er_m1.median().items()))
out.close()
