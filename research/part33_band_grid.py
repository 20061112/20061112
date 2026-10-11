"""第三十三部分（rev=-1 為反向版，見 part34_band_trend.py）：布林 / 肯特納通道的回歸網格 —— 每離中線 1 個單位（σ 或 ATR）加一層，往中線回歸平倉。

通道（只用「上一根已收完」的 K 棒，沒有未來函數）：
  BB：中線 = SMA20(收盤)，單位 u = 20 根收盤標準差
  KC：中線 = EMA20(收盤)，單位 u = ATR20（該週期的真實波幅）
  週期：M5 / M15 / H1（由 M1 重採樣）
網格：價格 ≤ 中線 − k·u（k = 1, 2, …, K）→ 第 k 層買 1 單位；上方鏡像做空。
出場兩種：
  inward 往內一層：第 k 層在價格回到 中線 − (k−1)·u 時賣出（第 1 層 = 回到中線）
  mean   回中線：所有層在價格回到中線時一起平
停損兩種：
  K = 4，碰到 5 單位全平該邊，該邊鎖住直到價格回到中線
  K = 10，不停損
可留過夜，週五最後一根全平（swap 未計）。M1 路徑 陽線 O→L→H→C、陰線 O→H→L→C；K 棒內碰線以線價成交，
K 棒開盤時（通道剛更新或跳空）已越線則以開盤價成交；買進付點差。
基準：整段 M1 隨機上下鏡像後重算通道、同模型跑（無方向性），SEEDS 次平均。
單位：每週損益 ÷ ATR10（日線，週初以前）；樣本內 2023~2024、樣本外 2025~2026。
"""
import numpy as np, pandas as pd
from multiprocessing import Pool
from load import load
from part28_ib_grid import mirrored, t

SEEDS, CUT = 6, 2025
d = load("data/XAUUSD_M1_2023_2026.csv")
OHLC0 = d[["open", "high", "low", "close"]].values; SP = d.spread.values / 100; IDX = d.index
dd = d.resample("1D").agg({"high": "max", "low": "min", "close": "last"}).dropna()
tr = np.maximum(dd.high, dd.close.shift()) - np.minimum(dd.low, dd.close.shift())
ATRD = tr.shift(1).rolling(10).mean()
WEEK = IDX.to_period("W-SUN")
WSTART = np.r_[0, np.where(WEEK[1:] != WEEK[:-1])[0] + 1]
WEND = np.r_[WSTART[1:] - 1, len(IDX) - 1]


def bands(ohlc, tf, kind):
    df = pd.DataFrame(ohlc, index=IDX, columns=["o", "h", "l", "c"])
    b = df.resample(tf, label="left", closed="left").agg({"o": "first", "h": "max", "l": "min", "c": "last"}).dropna()
    if kind == "BB":
        mid = b.c.rolling(20).mean(); u = b.c.rolling(20).std(ddof=0)
    else:
        trr = np.maximum(b.h, b.c.shift()) - np.minimum(b.l, b.c.shift())
        mid = b.c.ewm(span=20, adjust=False).mean(); u = trr.rolling(20).mean()
    # 每根 M1 用「已收完的上一根」：把值往後移到下一根 K 棒的起點
    end = b.index + pd.Timedelta(tf)
    s = pd.DataFrame({"mid": mid.values, "u": u.values}, index=end)
    s = s[~s.index.duplicated()].reindex(IDX, method="ffill")
    return s.mid.values, s.u.values


def sim(ohlc, mid, u, K, exit_kind, rev=1):
    cash = 0.0; nopen = 0; out = []
    for w0, w1 in zip(WSTART, WEND):
        a = ATRD.asof(IDX[w0].normalize())
        if not np.isfinite(a) or w1 - w0 < 3000: continue
        cw = 0.0; n = {1: 0, -1: 0}; lock = {1: False, -1: False}; trades = 0
        for j in range(w0, w1 + 1):
            m, uu = mid[j], u[j]
            if not (np.isfinite(m) and uu > 0): continue
            o, h, l, c = ohlc[j]; s = SP[j]
            for q, x in enumerate((o, l, h, c) if c >= o else (o, h, l, c)):
                for sd in (1, -1):                      # 1 = 下方做多、−1 = 上方做空
                    dep = sd * (m - x) / uu             # 往不利方向離中線幾個單位
                    if lock[sd]:
                        if dep <= 0: lock[sd] = False
                        else: continue
                    # 停損
                    if K < 10 and dep >= K + 1:
                        px = x if q == 0 else m - sd * (K + 1) * uu
                        if n[sd]:
                            cw += sd * rev * n[sd] * px - (s * n[sd] if sd * rev == -1 else 0); n[sd] = 0
                        lock[sd] = True; continue
                    # 出場
                    if exit_kind == "mean":
                        if n[sd] and dep <= 0:
                            px = x if q == 0 else m
                            cw += sd * rev * n[sd] * px - (s * n[sd] if sd * rev == -1 else 0); n[sd] = 0
                    else:
                        while n[sd] and dep <= n[sd] - 1:
                            px = x if q == 0 else m - sd * (n[sd] - 1) * uu
                            cw += sd * rev * px - (s if sd * rev == -1 else 0); n[sd] -= 1
                    # 進場
                    while n[sd] < K and dep >= n[sd] + 1:
                        px = x if q == 0 else m - sd * (n[sd] + 1) * uu
                        cw += -sd * rev * px - (s if sd * rev == 1 else 0); n[sd] += 1; trades += 1
        c = ohlc[w1, 3]; s = SP[w1]
        for sd in (1, -1):
            if n[sd]: cw += sd * rev * n[sd] * c - (s * n[sd] if sd * rev == -1 else 0)
        out.append(dict(week=IDX[w0], year=IDX[w0].year, atr=a, pnl=cw, trades=trades))
    return pd.DataFrame(out)


def job(args):
    tf, kind, K, ex, seed = args
    oh = OHLC0 if seed is None else mirrored(OHLC0, np.random.default_rng(seed))
    mid, u = bands(oh, tf, kind)
    return args, sim(oh, mid, u, K, ex)


if __name__ == "__main__":
    cfg = [(tf, kind, K, ex) for tf in ("5min", "15min", "1h") for kind in ("BB", "KC") for K in (4, 10) for ex in ("inward", "mean")]
    jobs = [c + (sd,) for c in cfg for sd in [None] + list(range(SEEDS))]
    with Pool(4) as pool:
        res = dict(pool.map(job, jobs, chunksize=1))
    out = open("results_part33.txt", "w")
    def P(s=""): print(s); out.write(s + "\n")
    P("第三十三部分：布林 / 肯特納通道回歸網格（每 1 個 σ / ATR 一層，可留過夜、週五全平）")
    P(f"單位：每週損益 ÷ 日 ATR10；括號 = t；基準 = 鏡像 M1 重算通道同模型 {SEEDS} 次平均；樣本內 2023~24 / 樣本外 2025~26。")
    P(f"\n  {'週期':5s} {'通道':3s} {'層/停損':9s} {'出場':6s} | {'樣本內':>14s} {'樣本外':>14s} | {'勝率':>4s} {'最差週':>7s} {'每日筆':>6s} | {'基準':>6s} | 逐年")
    keep = []
    for c in cfg:
        tf, kind, K, ex = c
        r = res[c + (None,)]; b = pd.concat([res[c + (sd,)] for sd in range(SEEDS)])
        a = r.pnl / r.atr; ab = b.pnl / b.atr; IS = (r.year < CUT).values
        yr = " ".join(f"{y % 100}:{a[(r.year == y).values].mean():+.2f}" for y in range(2023, 2027))
        P(f"  {tf:5s} {kind:3s} {('K4 停損5' if K == 4 else 'K10 不停'):9s} {('往內' if ex == 'inward' else '回中線'):6s} | "
          f"{a[IS].mean():+6.2f}({t(a[IS]):+5.2f}) {a[~IS].mean():+6.2f}({t(a[~IS]):+5.2f}) | {(a > 0).mean():4.0%} {a.min():+7.1f} "
          f"{r.trades.mean() / 5:6.1f} | {ab.mean():+6.2f} | {yr}")
        keep.append(dict(tf=tf, kind=kind, K=K, exit=ex, IS=a[IS].mean(), t_IS=t(a[IS]), OOS=a[~IS].mean(), t_OOS=t(a[~IS]),
                         win=(a > 0).mean(), worst=a.min(), per_day=r.trades.mean() / 5, base=ab.mean(),
                         edge=a.mean() - ab.mean()))
    S = pd.DataFrame(keep); S.to_csv("part33_summary.csv", index=False, float_format="%.4f")
    P("\n實際 − 基準（扣掉點差 / 模型偏誤後，價格本身的均值回歸貢獻），每週 ATR：")
    for r in S.itertuples():
        P(f"  {r.tf:5s} {r.kind} K{r.K:<2d} {r.exit:6s}  {r.edge:+.2f}")
    out.close()
