"""第二十四部分：第二十三部分網格（格距 0.05 × ATR14、可留過夜、週五清倉）加上過濾。

A. 預測力：週初可得的特徵 vs 本週 |淨漲跌|（決定網格損益）與 本週方向
B. 方向偏向 → 單邊網格（多：錨點下方每格加一單位多單、停利一格；錨點上方不持倉）
C. 較大週期 ER → 預期趨勢週不開網格
D. 當週浮動損益停損：權益 ≤ −X ATR 就全平，(a) 本週停止 或 (b) 以當下價格為新錨點重開
事先登記：門檻只用 2023~2024（樣本內）決定，2025~2026 為樣本外；特徵只用週初以前的資料。
"""
import numpy as np, pandas as pd
from load import load

out = open("results_part24.txt", "w")
def P(s=""): print(s); out.write(s + "\n"); out.flush()

d = load("data/XAUUSD_M1_2023_2026.csv")
d["day"] = d.index.normalize()
day = d.groupby("day").agg(o=("open", "first"), h=("high", "max"), l=("low", "min"), c=("close", "last"))
tr = np.maximum(day.h, day.c.shift()) - np.minimum(day.l, day.c.shift())
atr = tr.shift(1).rolling(14).mean()
d["week"] = d.index.to_period("W-SUN")
MULT, CUT = 0.05, 2025


def er(c):
    c = np.asarray(c); p = np.abs(np.diff(c)).sum()
    return abs(c[-1] - c[0]) / p if p > 0 else np.nan


# ---------- 週資料：路徑 + 週初特徵 ----------
h1 = d.close.resample("1h").last().dropna()
h4 = d.close.resample("4h").last().dropna()
W = []
for key, x in d.groupby("week"):
    s = x.day.iloc[0]; a = atr.get(s, np.nan)
    prev = day[day.index < s]
    if np.isnan(a) or len(x) < 500 or len(prev) < 60:
        continue
    o, h, l, c = x.open.values, x.high.values, x.low.values, x.close.values
    pc = prev.c
    f = dict(key=str(key), start=x.index[0], year=x.index[0].year, atr=a, g=MULT * a,
             net_atr=(c[-1] - o[0]) / a,
             trend20=np.sign(pc.iloc[-1] - pc.iloc[-20:].mean()),
             trend50=np.sign(pc.iloc[-1] - pc.iloc[-50:].mean()),
             ret20=(pc.iloc[-1] - pc.iloc[-21]) / a,
             ret5=(pc.iloc[-1] - pc.iloc[-6]) / a,
             er_d10=er(pc.iloc[-11:]), er_d20=er(pc.iloc[-21:]),
             er_h1_5d=er(h1[(h1.index >= prev.index[-5]) & (h1.index < s)]),
             er_h4_20d=er(h4[(h4.index >= prev.index[-20]) & (h4.index < s)]),
             atr_ratio=tr[tr.index < s].iloc[-5:].mean() / tr[tr.index < s].iloc[-20:].mean())
    f["paths"] = [(p, q, r, t) if t >= p else (p, r, q, t) for p, q, r, t in zip(o, l, h, c)]
    f["spread"] = x.spread.values / 100
    W.append(f)
W = pd.DataFrame(W)
W["prev_absnet"] = W.net_atr.abs().shift(1)
W["prev_net"] = W.net_atr.shift(1)
IS = W.year < CUT


def run(paths, spread, g, anchor, cap=10 ** 9, side="both", stop=None, reset=False):
    """回傳週損益（美元/盎司）。stop = 浮動權益停損（美元，正數）。"""
    k = 0; pos = 0; cash = 0.0
    def target(k):
        if side == "both": return -max(-cap, min(cap, k))
        if side == "long": return max(0, min(cap, -k))
        return -max(0, min(cap, k))
    for i, path in enumerate(paths):
        s = spread[i]
        for x in path[1:]:
            while x >= anchor + (k + 1) * g or x <= anchor + (k - 1) * g:
                k += 1 if x >= anchor + (k + 1) * g else -1
                q = target(k) - pos
                if q: cash += -q * (anchor + k * g) - (s * q if q > 0 else 0); pos += q
            if stop is not None and cash + pos * x - (s * -pos if pos < 0 else 0) <= -stop:
                cash += pos * x - (s * -pos if pos < 0 else 0); pos = 0
                if not reset:
                    return cash
                anchor, k = x, 0
    c = paths[-1][-1]
    return cash + pos * c - (s * -pos if pos < 0 else 0)


def evaluate(mask=None, side=None, **kw):
    """side 可以是字串或每週的方向陣列（+1 多 / −1 空 / 0 雙向）；mask = 本週要不要開。"""
    res = np.zeros(len(W))
    for i, r in enumerate(W.itertuples()):
        if mask is not None and not mask[i]:
            continue
        sd = side if isinstance(side, str) or side is None else {1: "long", -1: "short", 0: "both"}[int(side[i])]
        res[i] = run(r.paths, r.spread, r.g, r.paths[0][0], side=sd or "both", **kw) / r.atr
    return res


def line(lab, pnl, mask=None):
    m = np.ones(len(W), bool) if mask is None else np.asarray(mask)
    cells = []
    for sel, nm in ((IS, "樣本內23-24"), (~IS, "樣本外25-26")):
        p = pnl[sel]; n = (m & sel).sum()
        t = p.mean() / p.std() * np.sqrt(len(p)) if p.std() > 0 else 0
        cells.append(f"{nm} 每週 {p.mean():+6.2f}（t {t:+5.2f}，開 {n:3d} 週，最差 {p.min():+7.1f}）")
    yr = "  ".join(f"{y}:{pnl[W.year == y].mean():+6.1f}" for y in sorted(W.year.unique()))
    P(f"  {lab:34s} " + " | ".join(cells) + f" | {yr}")


def rk(a, b):
    m = ~(np.isnan(a) | np.isnan(b)); return pd.Series(a[m]).rank().corr(pd.Series(b[m]).rank())


P(f"第二十四部分：網格 + 過濾（{len(W)} 週，樣本內 2023~2024 {IS.sum()} 週、樣本外 2025~2026 {(~IS).sum()} 週）")
P("損益單位：ATR × 盎司 / 週（0.01 手 / 格）；未計 swap")

# ---------- A. 預測力 ----------
P("\nA. 週初特徵的預測力（等級相關）")
P("   |本週淨漲跌| 越大 → 網格越虧；要能預測它才能避開趨勢週")
absnet = W.net_atr.abs().values
for f in ("er_d10", "er_d20", "er_h1_5d", "er_h4_20d", "prev_absnet", "atr_ratio"):
    v = W[f].values
    P(f"   {f:12s} vs |淨漲跌|：樣本內 {rk(v[IS], absnet[IS]):+.3f}  樣本外 {rk(v[~IS], absnet[~IS]):+.3f}")
P("   方向：特徵符號與本週方向一致的比例（50% = 無預測力）")
net = W.net_atr.values
for f in ("trend20", "trend50", "ret20", "ret5", "prev_net"):
    sg = np.sign(W[f].values)
    P(f"   {f:12s} 樣本內 {np.mean(sg[IS] == np.sign(net[IS])):.0%}  樣本外 {np.mean(sg[~IS] == np.sign(net[~IS])):.0%}"
      f"   （本週上漲比例：樣本內 {np.mean(net[IS] > 0):.0%}、樣本外 {np.mean(net[~IS] > 0):.0%}）")

# ---------- B / C / D ----------
P("\n" + "=" * 150)
P("基準")
base = evaluate(); line("雙向網格（第二十三部分）", base)
line("永遠只做多（金價上漲的 beta 基準）", evaluate(side="long"))
line("永遠只做空", evaluate(side="short"))

P("\nB. 方向偏向 → 順勢單邊網格")
for f in ("trend20", "trend50", "ret20", "ret5", "prev_net"):
    line(f"順 {f} 單邊", evaluate(side=np.sign(np.nan_to_num(W[f].values))))
for f in ("trend20", "trend50"):
    line(f"逆 {f} 單邊", evaluate(side=-np.sign(np.nan_to_num(W[f].values))))

P("\nC. 較大週期 ER 過濾（樣本內 67% 分位以上 = 趨勢狀態 → 不開；另測反向：只在高 ER 開）")
for f in ("er_d10", "er_d20", "er_h1_5d", "er_h4_20d", "prev_absnet", "atr_ratio"):
    thr = np.nanquantile(W.loc[IS, f], 0.67)
    lo = (W[f] < thr).values
    line(f"{f} < {thr:.3f} 才開", evaluate(mask=lo), lo)
    line(f"{f} ≥ {thr:.3f} 才開（反向檢查）", evaluate(mask=~lo), ~lo)

P("\nD. 當週浮動損益停損")
for X in (5, 10, 20, 40):
    stop_pnl = np.zeros(len(W)); reset_pnl = np.zeros(len(W))
    for i, r in enumerate(W.itertuples()):
        stop_pnl[i] = run(r.paths, r.spread, r.g, r.paths[0][0], stop=X * r.atr) / r.atr
        reset_pnl[i] = run(r.paths, r.spread, r.g, r.paths[0][0], stop=X * r.atr, reset=True) / r.atr
    line(f"浮虧 {X} ATR 全平、本週停止", stop_pnl)
    line(f"浮虧 {X} ATR 全平、新錨點重開", reset_pnl)

P("\nE. 組合（事先登記一組，不再調整）：順 trend50 單邊 + 浮虧 20 ATR 本週停止")
comb = np.zeros(len(W))
for i, r in enumerate(W.itertuples()):
    sd = "long" if r.trend50 > 0 else "short"
    comb[i] = run(r.paths, r.spread, r.g, r.paths[0][0], side=sd, stop=20 * r.atr) / r.atr
line("trend50 單邊 + 停損 20", comb)
W.drop(columns=["paths", "spread"]).assign(base=base, combo=comb).to_csv("part24_weekly.csv", index=False, float_format="%.4f")
out.close()
