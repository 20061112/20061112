"""第二十八部分：IB 定方向的順勢偏向網格（雛形）。日內、不留過夜。

規則（事先登記，不調參）：
  1. IB = 某時段開始的 60 根 M1 最高 / 最低；起點測三個：01:00（亞洲開盤）、10:00（倫敦）、16:30（紐約，broker = 紐約 + 7）。
  2. IB 結束後，第一根 M1 收盤 > IB 高 + 0.05 × ATR10 → 今天做多；< IB 低 − 0.05 × ATR10 → 今天做空（只取第一個突破，一天一個方向）。
     下一根 M1 開盤 = 錨點 E。
  3. 格距 g = m × ATR10（日真實波幅 10 日平均，只用昨天以前），m = 0.05 / 0.10。
  4. 停損 = IB 另一側（多單 = IB 低、空單 = IB 高），碰到全平、今天停止；否則 23:50 全平。
     另跑「不停損、只到收盤平倉、最多 10 層」作對照。

元件（持倉只由「目前在錨點順方向第 k 條線」決定，損益可直接相加；做空時全部鏡像）：
  BASE 底倉：突破時進 1 單位，抱到停損 / 收盤（= 不加網格的 IB 突破核心）
  L    順勢逢回買：錨點逆方向第 j 條線持有 j 單位（停利一格）—— 只做順 IB 方向的網格
  PU   順勢加碼：錨點順方向第 k 條線持有 k 單位（突破加碼）
組合：L、BASE+L、PU、BASE+PU、BASE。

成交模型同第二十六部分：M1 內部路徑 陽線 O→L→H→C、陰線 O→H→L→C，格線價成交（順勢加碼單偏樂觀），買進付點差。
模型偏誤基準：每根 M1 以 50% 機率上下鏡像（K 棒之間沒有方向性），同一模型跑 SEEDS 次取平均。
單位：ATR × 盎司 / 天（每日損益 ÷ ATR10），另列以 R（= |E − 停損|）計的每筆平均。
樣本內 2023~2024、樣本外 2025~2026。
"""
import numpy as np, pandas as pd
from multiprocessing import Pool
from load import load

SEEDS, CUT, BUF, IB_MIN = 10, 2025, 0.05, 60
STARTS = {"01:00 亞洲": 60, "10:00 倫敦": 600, "16:30 紐約": 990}
MS = (0.05, 0.10)
FLAT = 23 * 60 + 50
COMP = ("BASE", "L", "PU")

d = load("data/XAUUSD_M1_2023_2026.csv")
d["day"] = d.index.normalize()
day = d.groupby("day").agg(h=("high", "max"), l=("low", "min"), c=("close", "last"))
tr = np.maximum(day.h, day.c.shift()) - np.minimum(day.l, day.c.shift())
ATR = tr.shift(1).rolling(10).mean()
DAYS = []
for k, x in d.groupby("day"):
    a = ATR.get(k, np.nan)
    if np.isfinite(a) and len(x) > 1000:
        DAYS.append((k, a, ((x.index - k).total_seconds().values // 60).astype(int),
                     x[["open", "high", "low", "close"]].values, x.spread.values / 100))


def mirrored(ohlc, rng):
    o, h, l, c = ohlc.T
    gap = np.r_[0, o[1:] - c[:-1]] * rng.choice((-1, 1), len(o))
    sg = rng.choice((-1, 1), len(o))
    ho, lo, co = np.where(sg > 0, h - o, -(l - o)), np.where(sg > 0, l - o, -(h - o)), sg * (c - o)
    o2 = o[0] + np.r_[0, np.cumsum(co[:-1] + gap[1:])]
    return np.c_[o2, o2 + ho, o2 + lo, o2 + co]


def target(nm, kd, cap):
    """kd = 順方向的線數（> 0 = 往有利方向走了 kd 格）。回傳順方向的持倉單位數。"""
    if nm == "BASE": return 1
    if nm == "L": return min(max(0, -kd), cap)
    return min(max(0, kd), cap)


def run_day(mins, ohlc, spread, atr, s0, m, stop_on, cap):
    """回傳 dict：side、R、各元件損益（美元/盎司）；沒有突破回傳 None。"""
    ib = (mins >= s0) & (mins < s0 + IB_MIN)
    if ib.sum() < IB_MIN * 0.8: return None
    ibh, ibl = ohlc[ib, 1].max(), ohlc[ib, 2].min()
    after = np.where((mins >= s0 + IB_MIN) & (mins < FLAT))[0]
    c = ohlc[after, 3]
    up, dn = np.where(c > ibh + BUF * atr)[0], np.where(c < ibl - BUF * atr)[0]
    if not len(up) and not len(dn): return None
    side, i = (1, after[up[0]]) if len(up) and (not len(dn) or up[0] < dn[0]) else (-1, after[dn[0]])
    i += 1
    if i >= len(mins) or mins[i] >= FLAT: return None
    E = ohlc[i, 0]; stop = ibl if side == 1 else ibh; R = abs(E - stop); g = m * atr
    pos = {nm: 0 for nm in COMP}; cash = {nm: 0.0 for nm in COMP}; k = 0

    def trade(px, s, tgt):
        for nm in COMP:
            q = side * tgt(nm) - pos[nm]
            if q:
                cash[nm] += -q * px - (s * q if q > 0 else 0); pos[nm] += q

    trade(E, spread[i], lambda nm: target(nm, 0, cap))
    last = i; stopped = False
    for j in range(i, len(mins)):
        if mins[j] >= FLAT: break
        last = j
        o, h, l, cl = ohlc[j]; s = spread[j]
        for x in ((o, l, h, cl) if cl >= o else (o, h, l, cl))[(1 if j == i else 0):]:
            if stop_on and side * (x - stop) <= 0:
                px = x if j > i and x == o else stop   # 跳空穿過停損 → 以開盤價成交
                trade(px, s, lambda nm: 0); stopped = True; break
            while x >= E + (k + 1) * g or x <= E + (k - 1) * g:
                k += 1 if x >= E + (k + 1) * g else -1
                kd = side * k
                trade(E + k * g, s, lambda nm: target(nm, kd, cap))
        if stopped: break
    if not stopped:
        trade(ohlc[last, 3], spread[last], lambda nm: 0)
    return dict(side=side, R=R, **cash)


def job(args):
    s0, m, stop_on, seed = args
    rng = np.random.default_rng(seed) if seed is not None else None
    cap = 10 ** 6 if stop_on else 10
    rows = []
    for k, a, mins, ohlc, sp in DAYS:
        oh = mirrored(ohlc, rng) if rng is not None else ohlc
        r = run_day(mins, oh, sp, a, s0, m, stop_on, cap)
        if r: rows.append(dict(day=k, atr=a, **r))
    return args, pd.DataFrame(rows)


def t(p): p = np.asarray(p); return p.mean() / p.std() * np.sqrt(len(p)) if p.std() > 0 else 0


def combos(df):
    return {"BASE（只底倉 = 核心）": df.BASE, "L（只順勢逢回買網格）": df.L, "BASE + L": df.BASE + df.L,
            "PU（只順勢加碼）": df.PU, "BASE + PU": df.BASE + df.PU}


if __name__ == "__main__":
    jobs = [(s0, m, st, sd) for s0 in STARTS.values() for m in MS for st in (True, False)
            for sd in [None] + list(range(SEEDS))]
    with Pool(4) as pool:
        res = dict(pool.map(job, jobs))
    out = open("results_part28.txt", "w")
    def P(s=""): print(s); out.write(s + "\n")
    P("第二十八部分：IB 定方向的順勢偏向網格（日內、IB 另一側停損、23:50 全平）")
    P("單位：每日損益 ÷ ATR10（ATR × 盎司 / 交易日，只算有突破的日子）；括號 = t。基準 = 同模型跑在鏡像 K 棒上（理論 0 − 點差）。")
    P(f"樣本內 2023~2024、樣本外 2025~2026；/R = 每筆以 R（E 到停損）計的平均；基準為 {SEEDS} 次平均。")
    keep = []
    for nm0, s0 in STARTS.items():
        for st in (True, False):
            for m in MS:
                real = res[(s0, m, st, None)]
                base = pd.concat([res[(s0, m, st, sd)] for sd in range(SEEDS)])
                IS = (real.day.dt.year < CUT).values
                rr, bb = combos(real), combos(base)
                P(f"\nIB {nm0}｜格距 {m} ATR｜{'IB 另一側停損' if st else '不停損、最多 10 層、收盤平'}｜"
                  f"{len(real)} 天（內 {IS.sum()} / 外 {(~IS).sum()}），R 中位 {np.median(real.R / real.atr):.2f} ATR")
                P(f"  {'組合':22s} | {'樣本內':>13s} {'樣本外':>13s} | {'全部/R':>7s} {'最差日':>7s} | {'基準':>6s} | 逐年")
                for nm in rr:
                    a = rr[nm] / real.atr; b = bb[nm] / base.atr
                    yr = " ".join(f"{y % 100}:{a[(real.day.dt.year == y).values].mean():+.3f}" for y in range(2023, 2027))
                    P(f"  {nm:22s} | {a[IS].mean():+.3f}({t(a[IS]):+5.2f}) {a[~IS].mean():+.3f}({t(a[~IS]):+5.2f}) | "
                      f"{(rr[nm] / real.R).mean():+7.3f} {a.min():+7.2f} | {b.mean():+6.3f} | {yr}")
                    keep.append(dict(ib=nm0, m=m, stop=st, combo=nm, IS=a[IS].mean(), t_IS=t(a[IS]), OOS=a[~IS].mean(),
                                     t_OOS=t(a[~IS]), perR=(rr[nm] / real.R).mean(), worst=a.min(), base=b.mean()))
    pd.DataFrame(keep).to_csv("part28_summary.csv", index=False, float_format="%.4f")
    out.close()
