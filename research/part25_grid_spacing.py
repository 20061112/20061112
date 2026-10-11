"""第二十五部分：網格格距掃描 — 格距 = m × ATR14，m = 0.05 ~ 1.0；可留過夜、週五清倉。

同第二十三部分的雙向中性網格（另測只做多、只做空、浮虧 20 ATR 本週停止）。
理論：庫存虧損 ≈ (週淨漲跌)² / (2g)、來回利潤 ≈ g × 來回次數；隨機漫步下兩者期望相抵，
格距越大 → 交易次數越少（點差越少）、庫存虧損越小，但每週可收的來回利潤也越少。
損益以 ATR 為單位；不同格距的「每單位風險」不同，所以比較時看 t 值（與部位大小無關）與 平均 / 標準差。
"""
import numpy as np, pandas as pd
from load import load

out = open("results_part25.txt", "w")
def P(s=""): print(s); out.write(s + "\n"); out.flush()

d = load("data/XAUUSD_M1_2023_2026.csv")
d["day"] = d.index.normalize()
day = d.groupby("day").agg(h=("high", "max"), l=("low", "min"), c=("close", "last"))
tr = np.maximum(day.h, day.c.shift()) - np.minimum(day.l, day.c.shift())
atr = tr.shift(1).rolling(14).mean()
d["week"] = d.index.to_period("W-SUN")
CUT = 2025

W = []
for key, x in d.groupby("week"):
    s = x.day.iloc[0]; a = atr.get(s, np.nan)
    if np.isnan(a) or len(x) < 500 or (day.index < s).sum() < 60:   # 與第二十四部分同樣的 162 週
        continue
    o, h, l, c = x.open.values, x.high.values, x.low.values, x.close.values
    W.append(dict(key=str(key), year=x.index[0].year, atr=a, net_atr=(c[-1] - o[0]) / a,
                  paths=[(p, q, r, t) if t >= p else (p, r, q, t) for p, q, r, t in zip(o, l, h, c)],
                  spread=x.spread.values / 100))
W = pd.DataFrame(W)
IS = (W.year < CUT).values


def run(paths, spread, g, anchor, side="both", stop=None):
    """回傳 (週損益 美元/盎司, 交易單位數, 最大持倉)。"""
    k = 0; pos = 0; cash = 0.0; n = 0; mp = 0
    def target(k):
        if side == "both": return -k
        if side == "long": return max(0, -k)
        return -max(0, k)
    for i, path in enumerate(paths):
        s = spread[i]
        for x in path[1:]:
            while x >= anchor + (k + 1) * g or x <= anchor + (k - 1) * g:
                k += 1 if x >= anchor + (k + 1) * g else -1
                q = target(k) - pos
                if q: cash += -q * (anchor + k * g) - (s * q if q > 0 else 0); pos += q; n += abs(q)
            mp = max(mp, abs(pos))
            if stop is not None and cash + pos * x - (s * -pos if pos < 0 else 0) <= -stop:
                return cash + pos * x - (s * -pos if pos < 0 else 0), n + abs(pos), mp
    c = paths[-1][-1]
    return cash + pos * c - (s * -pos if pos < 0 else 0), n + abs(pos), mp


def sweep(side="both", stop_atr=None, zero_cost=False):
    rows = []
    for m in MS:
        pnl = np.zeros(len(W)); n = np.zeros(len(W)); mp = np.zeros(len(W))
        for i, r in enumerate(W.itertuples()):
            sp = r.spread * 0 if zero_cost else r.spread
            p, n[i], mp[i] = run(r.paths, sp, m * r.atr, r.paths[0][0], side, None if stop_atr is None else stop_atr * r.atr)
            pnl[i] = p / r.atr
        rows.append((m, pnl, n, mp))
    return rows


def t(p): return p.mean() / p.std() * np.sqrt(len(p)) if p.std() > 0 else 0


def table(rows, lab):
    P(f"\n{lab}")
    P(f"  {'格距':>6s} | {'樣本內 23-24 平均':>16s} {'t':>6s} | {'樣本外 25-26 平均':>16s} {'t':>6s} | {'全部 t':>6s} {'平均/標準差':>8s} {'最差週':>7s} {'P5':>6s} "
      f"{'交易/週':>7s} {'最大持倉':>6s} | 逐年平均")
    for m, pnl, n, mp in rows:
        yr = "  ".join(f"{y}:{pnl[W.year == y].mean():+6.2f}" for y in sorted(W.year.unique()))
        P(f"  {m:5.2f}A | {pnl[IS].mean():+16.2f} {t(pnl[IS]):+6.2f} | {pnl[~IS].mean():+16.2f} {t(pnl[~IS]):+6.2f} | "
          f"{t(pnl):+6.2f} {pnl.mean() / pnl.std():+8.3f} {pnl.min():+7.1f} {np.quantile(pnl, .05):+6.1f} "
          f"{np.median(n):7.0f} {np.median(mp):6.0f}   | {yr}")


MS = (0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0)
P(f"第二十五部分：格距掃描（{len(W)} 週；樣本內 2023~2024 {IS.sum()} 週、樣本外 2025~2026 {(~IS).sum()} 週）")
P("損益單位：ATR × 盎司 / 週（每格 1 盎司 = 0.01 手）；未計 swap。格距越大每單位風險越小，請看 t 與 平均/標準差")
base = sweep()
table(base, "雙向中性網格")
table(sweep(zero_cost=True), "雙向中性網格（零點差，看毛利）")
table(sweep("long"), "只做多（含金價上漲的 beta）")
table(sweep("short"), "只做空")
table(sweep(stop_atr=20), "雙向 + 浮虧 20 ATR 本週停止")

P("\n雙向網格：週損益 vs 本週 |淨漲跌|（各格距的平均）")
b = pd.cut(W.net_atr.abs(), [0, .5, 1, 2, 3, 99])
P("  " + f"{'格距':>6s} " + " ".join(f"{str(c):>12s}" for c in b.cat.categories))
for m, pnl, _, _ in base:
    g = pd.Series(pnl).groupby(b.values, observed=False).mean()
    P(f"  {m:5.2f}A " + " ".join(f"{v:+12.2f}" for v in g.values))
P("  週數   " + " ".join(f"{v:12d}" for v in b.value_counts(sort=False).values))
pd.DataFrame({f"m{m}": pnl for m, pnl, _, _ in base}).assign(key=W.key, year=W.year, net_atr=W.net_atr).to_csv(
    "part25_weekly.csv", index=False, float_format="%.4f")
out.close()
