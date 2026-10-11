"""第二十二部分（續）：把「短陡長陡同向」共振搬到 M1 / M5 / M15 / H1，掃短、長窗口與持有長度。
規則同 part22_angle.py：|斜率/ATR96| 在過去一段時間的百分位，短 ≥ 0.8、長 ≥ 0.8、同向 → 順勢；
第 t 根收盤成立，用 close[t] 進、close[t+h] 出，扣當根點差，去重疊。
百分位視窗統一成約 83 天（= H1 的 2000 根），不同時框才是同一把尺。
資料：M1 2023/6~2026/10（M5 由 M1 合成）；M15、H1 2023/1~2026/10。
"""
import numpy as np
import pandas as pd
from load import load
from backtest import resample

RANK_DAYS = 83


def slope(y, n):
    """過去 n 根（含當根）線性回歸斜率，用累積和 O(N)"""
    y = np.asarray(y, float)
    t = np.arange(len(y), dtype=float)
    cy, cty = np.concatenate([[0], np.cumsum(y)]), np.concatenate([[0], np.cumsum(t * y)])
    sy = cy[n:] - cy[:-n]
    sty = cty[n:] - cty[:-n]
    t0 = np.arange(len(y) - n + 1, dtype=float)          # 視窗起點
    # 以視窗內相對時間 u = t - t0 計算
    suy = sty - t0 * sy
    ub = (n - 1) / 2
    var = n * (n * n - 1) / 12
    out = np.full(len(y), np.nan)
    out[n - 1:] = (suy - ub * sy) / var
    return out


def prep(d, bars_per_day):
    c = d.close
    pc = c.shift()
    tr = pd.concat([d.high - d.low, (d.high - pc).abs(), (d.low - pc).abs()], axis=1).max(axis=1)
    atr = tr.rolling(96).mean()
    return dict(c=c.to_numpy(float), atr=atr.to_numpy(), cost=(d.spread * 0.01 / atr).to_numpy(),
                year=d.index.year.to_numpy(), win=int(RANK_DAYS * bars_per_day), cache={})


def rank_abs(P, n):
    if n not in P["cache"]:
        k = pd.Series(np.abs(slope(P["c"], n) / P["atr"]))
        P["cache"][n] = (k.rolling(P["win"], min_periods=P["win"] // 4).rank(pct=True).to_numpy(),
                         np.sign(slope(P["c"], n)))
    return P["cache"][n]


def test(P, ns, nl, h, a=.8, b=.8):
    rs, ss = rank_abs(P, ns)
    rl, sl = rank_abs(P, nl)
    m = (rs >= a) & (rl >= b) & (ss == sl) & (ss != 0)
    pos, last = [], -10 ** 9
    for p in np.flatnonzero(m):
        if p - last >= h and p + h < len(P["c"]):
            pos.append(p)
            last = p
    pos = np.array(pos, int)
    if len(pos) < 20:
        return None
    s = ss[pos]
    g = s * (P["c"][pos + h] - P["c"][pos]) / P["atr"][pos]
    net = g - P["cost"][pos]
    yr = P["year"][pos]
    ys = {y: net[yr == y].mean() for y in np.unique(yr)}
    # 基準：同時框同持有長度，每 h 根順長角度方向
    bp = np.arange(P["win"], len(P["c"]) - h, h)
    bp = bp[~np.isnan(rl[bp])]
    bnet = sl[bp] * (P["c"][bp + h] - P["c"][bp]) / P["atr"][bp] - P["cost"][bp]
    return dict(n=len(net), gross=g.mean(), net=net.mean(), t=net.mean() / net.std(ddof=1) * np.sqrt(len(net)),
                win=(net > 0).mean(), cost=P["cost"][pos].mean(), base=np.nanmean(bnet),
                pos_years=sum(v > 0 for v in ys.values()), n_years=len(ys),
                years=" ".join(f"{y % 100}:{v:+.2f}" for y, v in ys.items()))


def grid(P, label, shorts, mults, holds_rel, minutes):
    print(f"\n{'=' * 110}\n{label}（1 根 = {minutes} 分鐘；百分位視窗 {P['win']} 根）")
    print(f"{'短':>5s} {'長':>5s} {'持有':>5s} {'短(時)':>6s} {'長(時)':>6s} {'持(時)':>6s} {'筆數':>6s} {'毛':>7s} {'成本':>6s} {'淨':>7s} {'t':>6s} {'勝率':>5s} {'基準':>7s} {'正年':>5s}  逐年淨")
    rows = []
    for ns in shorts:
        for mlt in mults:
            nl = ns * mlt
            for hr in holds_rel:
                h = max(1, int(ns * hr))
                r = test(P, ns, nl, h)
                if r is None:
                    continue
                r.update(tf=label, ns=ns, nl=nl, h=h, hs=ns * minutes / 60, hl=nl * minutes / 60, hh=h * minutes / 60)
                rows.append(r)
                print(f"{ns:5d} {nl:5d} {h:5d} {r['hs']:6.1f} {r['hl']:6.1f} {r['hh']:6.1f} {r['n']:6d} {r['gross']:+7.3f} {r['cost']:6.3f} "
                      f"{r['net']:+7.3f} {r['t']:+6.2f} {r['win']:5.2f} {r['base']:+7.3f} {r['pos_years']:2d}/{r['n_years']}  {r['years']}")
    return rows


if __name__ == "__main__":
    m1 = load("data/XAUUSD_M1_2023_2026.csv")
    m15 = load("data/XAUUSD_M15_2023_2026.csv")
    h1 = resample(m15, "1h")
    m5 = resample(m1, "5min")
    mults, holds = (4, 8, 16), (0.5, 1, 2)
    R = []
    R += grid(prep(h1, 22), "H1", (4, 8, 16), mults, holds, 60)
    R += grid(prep(m15, 88), "M15", (8, 16, 32, 64), mults, holds, 15)
    R += grid(prep(m5, 264), "M5", (12, 24, 48, 96, 192), mults, holds, 5)
    R += grid(prep(m1, 1320), "M1", (15, 30, 60, 120, 240, 480), mults, holds, 1)
    R = pd.DataFrame(R)
    R.to_csv("part22_mtf_grid.csv", index=False)
    print(f"\n{'=' * 110}\n各時框彙總（全部組合）")
    s = R.groupby("tf", sort=False).agg(組數=("net", "size"), 淨為正=("net", lambda x: (x > 0).mean()),
                                       淨中位=("net", "median"), t中位=("t", "median"), t大於2=("t", lambda x: (x > 2).sum()),
                                       四年全正=("pos_years", lambda x: (x == R.loc[x.index, "n_years"]).sum()),
                                       成本中位=("cost", "median"), 超額中位=("net", lambda x: (x - R.loc[x.index, "base"]).median()))
    print(s.round(3).to_string())
    print("\n等時間換算（短 8h / 長 32~128h / 持有 4~16h）在各時框：")
    eq = R[(R.hs == 8)]
    print(eq[["tf", "ns", "nl", "h", "n", "net", "t", "base", "pos_years", "n_years", "years"]].round(3).to_string(index=False))
