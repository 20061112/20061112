"""第二十六部分：有偏向的網格（非中性）。可留過夜、週五清倉；格距 g = m × ATR14。

四個基本元件（持倉只由「目前在錨點上下第 k 條線」決定，彼此不重疊，所以任何組合的損益與成本都可直接相加）：
  L  逢低買：錨點下方第 k 條線持有 k 單位多單（停利一格）       pos = max(0, −k)
  S  逢高賣：錨點上方第 k 條線持有 k 單位空單                   pos = −max(0, k)
  PU 順勢加多：錨點上方第 k 條線持有 k 單位多單（突破加碼）     pos = max(0, k)
  PD 順勢加空：錨點下方第 k 條線持有 k 單位空單                 pos = −max(0, −k)
組合：中性 = L + S；反網格（順勢網格）= PU + PD；
      偏多網格 = L + w·S（w = 0.5：上方只賣一半）；依趨勢方向：多頭用 L / PU、空頭用 S / PD。
方向指標（週初可得）：trend50 = sign(前一日收盤 − 50 日均線)，trend20 同理；事先登記，不調參。

模型偏誤基準：M1 內部路徑用 O→L→H→C 直線內插，順勢單（PU/PD）在格線上以「剛好格線價」成交，
在真實市場（停損單、跳價）是偏樂觀的。因此另建「隨機方向」基準：每根 M1 的 (H−O, L−O, C−O) 與跳空
以 50% 機率整根上下鏡像，K 棒內部結構完全相同，但 K 棒之間沒有方向性（= 真實的零期望值），
用同一個模型跑。實際 − 基準 = 扣除模型偏誤後的優勢。
"""
import numpy as np, pandas as pd
from load import load

out = open("results_part26.txt", "w")
def P(s=""): print(s); out.write(s + "\n"); out.flush()

d = load("data/XAUUSD_M1_2023_2026.csv")
d["day"] = d.index.normalize()
day = d.groupby("day").agg(h=("high", "max"), l=("low", "min"), c=("close", "last"))
tr = np.maximum(day.h, day.c.shift()) - np.minimum(day.l, day.c.shift())
atr = tr.shift(1).rolling(14).mean()
d["week"] = d.index.to_period("W-SUN")
CUT, SEEDS, MS = 2025, 20, (0.1, 0.2, 0.3, 0.5)
COMP = ("L", "S", "PU", "PD")

W = []
for key, x in d.groupby("week"):
    s = x.day.iloc[0]; a = atr.get(s, np.nan)
    pc = day.c[day.index < s]
    if np.isnan(a) or len(x) < 500 or len(pc) < 60:
        continue
    W.append(dict(key=str(key), year=x.index[0].year, atr=a,
                  trend50=1 if pc.iloc[-1] > pc.iloc[-50:].mean() else -1,
                  trend20=1 if pc.iloc[-1] > pc.iloc[-20:].mean() else -1,
                  ohlc=x[["open", "high", "low", "close"]].values, spread=x.spread.values / 100))
W = pd.DataFrame(W)
IS = (W.year < CUT).values


def mirrored(ohlc, rng):
    o, h, l, c = ohlc.T
    gap = np.r_[0, o[1:] - c[:-1]]
    sg = rng.choice((-1, 1), len(o))
    ho, lo, co = np.where(sg > 0, h - o, -(l - o)), np.where(sg > 0, l - o, -(h - o)), sg * (c - o)
    gap = gap * rng.choice((-1, 1), len(o))
    o2 = np.empty(len(o)); o2[0] = o[0]
    for i in range(1, len(o)):
        o2[i] = o2[i - 1] + co[i - 1] + gap[i]
    return np.c_[o2, o2 + ho, o2 + lo, o2 + co]


TGT = dict(L=lambda k: max(0, -k), S=lambda k: -max(0, k), PU=lambda k: max(0, k), PD=lambda k: -max(0, -k))


def run(ohlc, spread, g):
    """一次走完一週，回傳四個元件的損益（美元/盎司）與多/空單位·夜。"""
    anchor = ohlc[0, 0]; k = 0
    pos = dict.fromkeys(COMP, 0); cash = dict.fromkeys(COMP, 0.0)
    for (o, h, l, c), s in zip(ohlc, spread):
        for x in ((o, l, h, c) if c >= o else (o, h, l, c))[1:]:
            while x >= anchor + (k + 1) * g or x <= anchor + (k - 1) * g:
                k += 1 if x >= anchor + (k + 1) * g else -1
                px = anchor + k * g
                for nm in COMP:
                    q = TGT[nm](k) - pos[nm]
                    if q:
                        cash[nm] += -q * px - (s * q if q > 0 else 0); pos[nm] += q
    c = ohlc[-1, 3]; s = spread[-1]
    return {nm: cash[nm] + pos[nm] * c - (s * -pos[nm] if pos[nm] < 0 else 0) for nm in COMP}


rng = np.random.default_rng(0)
R = {}
for m in MS:
    real = pd.DataFrame([run(r.ohlc, r.spread, m * r.atr) for r in W.itertuples()]).div(W.atr, axis=0)
    base = sum(pd.DataFrame([run(mirrored(r.ohlc, rng), r.spread, m * r.atr) for r in W.itertuples()]).div(W.atr, axis=0)
               for _ in range(SEEDS)) / SEEDS
    R[m] = (real, base)
    print("done", m, flush=True)


def t(p): return p.mean() / p.std() * np.sqrt(len(p)) if p.std() > 0 else 0


def strategies(df):
    up50, up20 = (W.trend50 > 0).values, (W.trend20 > 0).values
    return {
        "中性 L+S（參考）": df.L + df.S,
        "只做多 L（逢低買）": df.L,
        "只做空 S": df.S,
        "偏多 L + 0.5·S": df.L + 0.5 * df.S,
        "反網格 PU+PD（順勢加碼）": df.PU + df.PD,
        "只順勢加多 PU": df.PU,
        "trend50 單邊逢低買/逢高賣": np.where(up50, df.L, df.S),
        "trend50 偏向 1 : 0.5": np.where(up50, df.L + 0.5 * df.S, df.S + 0.5 * df.L),
        "trend50 順勢加碼 PU/PD": np.where(up50, df.PU, df.PD),
        "trend20 單邊逢低買/逢高賣": np.where(up20, df.L, df.S),
        "trend20 順勢加碼 PU/PD": np.where(up20, df.PU, df.PD),
    }


P(f"第二十六部分：有偏向的網格（{len(W)} 週；樣本內 2023~2024 {IS.sum()} 週、樣本外 2025~2026 {(~IS).sum()} 週）")
P("單位：ATR × 盎司 / 週；未計 swap。「基準」= 同模型跑在隨機方向（鏡像）K 棒上，理論上應為 0（減點差）；")
P(f"「優勢」= 實際 − 基準平均（基準為 {SEEDS} 次鏡像的平均），括號為實際損益本身的 t。")
for m in MS:
    real, base = R[m]
    sr, sb = strategies(real), strategies(base)
    P(f"\n格距 {m} × ATR")
    P(f"  {'策略':28s} | {'實際 樣本內':>10s} {'樣本外':>8s} {'t':>6s} {'最差週':>7s} | {'基準':>7s} | {'優勢 樣本內':>14s} {'樣本外':>14s} | 逐年（實際）")
    for nm in sr:
        a, b = np.asarray(sr[nm]), np.asarray(sb[nm])
        yr = " ".join(f"{y % 100}:{a[(W.year == y).values].mean():+5.2f}" for y in sorted(W.year.unique()))
        P(f"  {nm:28s} | {a[IS].mean():+10.2f} {a[~IS].mean():+8.2f} {t(a):+6.2f} {a.min():+7.1f} | {b.mean():+7.2f} | "
          f"{a[IS].mean() - b[IS].mean():+7.2f}({t(a[IS]):+5.2f}) {a[~IS].mean() - b[~IS].mean():+7.2f}({t(a[~IS]):+5.2f}) | {yr}")

P("\n方向指標本身：trend50 / trend20 與本週方向一致的比例")
net = np.array([(r.ohlc[-1, 3] - r.ohlc[0, 0]) for r in W.itertuples()])
for f in ("trend50", "trend20"):
    P(f"  {f}：樣本內 {np.mean(np.sign(net[IS]) == W[f].values[IS]):.0%}  樣本外 {np.mean(np.sign(net[~IS]) == W[f].values[~IS]):.0%}")
real, _ = R[0.2]
pd.DataFrame(strategies(real)).assign(key=W.key, year=W.year, trend50=W.trend50).to_csv(
    "part26_weekly_g0.2.csv", index=False, float_format="%.4f")
out.close()
