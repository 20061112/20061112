"""第三十部分：H1 六合一 —— (1) 回到日線 SMA20 之上停止做空；(2) 其他出場方法。
輸出：python3 part30_exits.py > results_part30_exits.txt

進場與第二十八部分完全相同（六個背離版本、D+E 盤面過濾、突破訊號棒進場、停損回檔低點、各版本 1/6 R、各版本一次一筆）。

(1) SMA20 過濾（日線 SMA20，只用已收完的資料）：
    F0  不加
    F1  前一日收盤 > 日線 SMA20 → 不做空
    F2  F1 + 對稱：前一日收盤 < SMA20 → 不做多
    F3  訊號當下 H1 收盤 > 前一日的日線 SMA20 → 不做空（反應較快）
    F4  F3 + 對稱
(2) 出場（初始停損都一樣）：
    X0  2R 停利 / 32 根（原本）
    X1  1R / 32        X2  1.5R / 32      X3  3R / 32
    X4  不設停利，32 根時間出場
    X5  2R / 64 根     X6  2R / 16 根
    X7  浮盈 1R 後停損移到進場價（保本），2R / 32
    X8  吊燈追蹤停損：最高點 − 3 ATR（不低於初始停損），不設停利，最長 96 根
    X9  吊燈追蹤 2 ATR，最長 96 根
    X10 H1 收盤跌破 EMA20 出場（下一根開盤），最長 96 根
    X11 一半 1R 停利，另一半保本 + 吊燈 3 ATR 追蹤，最長 96 根
    同一根同時碰到停損與停利 → 停損；追蹤停損用「上一根收完後」更新的值。
"""
import numpy as np
import pandas as pd
from part22_divergence import get_frames, indicators, detect, nonoverlap, SPLIT, BUF, MIN_R
from part24_common import daily_regime, both_spaces, tstat

SIX = ["push", "obv", "speed", "rsi", "cci", "stoch"]
W = 1 / 6
WAIT_C = 4


# ---------------------------------------------------------------- 出場（Y 空間做多）
def ex_fixed(Y, ei, ep, stop, tp=2.0, hold=32, be=None):
    o, h, l, c, spr = Y["o"], Y["h"], Y["l"], Y["c"], Y["spr"]
    n = len(c)
    R = ep - stop
    tgt = ep + tp * R if tp else np.inf
    cur = stop
    for j in range(ei, min(ei + hold + 1, n)):
        if l[j] <= cur:
            xp = cur if j == ei else min(cur, o[j])
            return j, (xp - ep - spr[ei]) / R, "stop" if cur < ep else "be"
        if j > ei and h[j] >= tgt:
            return j, (tgt - ep - spr[ei]) / R, "tp"
        if be is not None and h[j] >= ep + be * R:
            cur = max(cur, ep)
    j = min(ei + hold, n - 1)
    return j, (c[j] - ep - spr[ei]) / R, "time"


def ex_trail(Y, ei, ep, stop, k=3.0, hold=96):
    o, h, l, c, atr, spr = Y["o"], Y["h"], Y["l"], Y["c"], Y["atr"], Y["spr"]
    n = len(c)
    R = ep - stop
    cur, hh = stop, ep
    for j in range(ei, min(ei + hold + 1, n)):
        if l[j] <= cur:
            xp = cur if j == ei else min(cur, o[j])
            return j, (xp - ep - spr[ei]) / R, "trail"
        hh = max(hh, h[j])
        cur = max(cur, hh - k * atr[j])
    j = min(ei + hold, n - 1)
    return j, (c[j] - ep - spr[ei]) / R, "time"


def ex_ema(Y, ei, ep, stop, hold=96):
    o, h, l, c, spr, ema = Y["o"], Y["h"], Y["l"], Y["c"], Y["spr"], Y["ema20"]
    n = len(c)
    R = ep - stop
    for j in range(ei, min(ei + hold + 1, n)):
        if l[j] <= stop:
            xp = stop if j == ei else min(stop, o[j])
            return j, (xp - ep - spr[ei]) / R, "stop"
        if c[j] < ema[j] and j + 1 < n:
            return j + 1, (o[j + 1] - ep - spr[ei]) / R, "ema"
    j = min(ei + hold, n - 1)
    return j, (c[j] - ep - spr[ei]) / R, "time"


def ex_partial(Y, ei, ep, stop, hold=96):
    o, h, l, c, atr, spr = Y["o"], Y["h"], Y["l"], Y["c"], Y["atr"], Y["spr"]
    n = len(c)
    R = ep - stop
    t1 = ep + R
    half_done, r1 = False, 0.0
    cur, hh = stop, ep
    for j in range(ei, min(ei + hold + 1, n)):
        if l[j] <= cur:
            xp = cur if j == ei else min(cur, o[j])
            r2 = (xp - ep) / R
            tot = (0.5 * r1 + 0.5 * r2) if half_done else r2
            return j, tot - spr[ei] / R, "partial"
        if not half_done and j > ei and h[j] >= t1:
            half_done, r1 = True, 1.0
            cur = max(cur, ep)
        hh = max(hh, h[j])
        if half_done:
            cur = max(cur, hh - 3 * atr[j])
    j = min(ei + hold, n - 1)
    r2 = (c[j] - ep) / R
    return j, ((0.5 * r1 + 0.5 * r2) if half_done else r2) - spr[ei] / R, "time"


EXITS = {
    "X0 2R/32(原本)": lambda Y, e, p, s: ex_fixed(Y, e, p, s, 2.0, 32),
    "X1 1R/32": lambda Y, e, p, s: ex_fixed(Y, e, p, s, 1.0, 32),
    "X2 1.5R/32": lambda Y, e, p, s: ex_fixed(Y, e, p, s, 1.5, 32),
    "X3 3R/32": lambda Y, e, p, s: ex_fixed(Y, e, p, s, 3.0, 32),
    "X4 無停利/32": lambda Y, e, p, s: ex_fixed(Y, e, p, s, None, 32),
    "X5 2R/64": lambda Y, e, p, s: ex_fixed(Y, e, p, s, 2.0, 64),
    "X6 2R/16": lambda Y, e, p, s: ex_fixed(Y, e, p, s, 2.0, 16),
    "X7 保本1R+2R/32": lambda Y, e, p, s: ex_fixed(Y, e, p, s, 2.0, 32, be=1.0),
    "X8 吊燈3ATR": lambda Y, e, p, s: ex_trail(Y, e, p, s, 3.0),
    "X9 吊燈2ATR": lambda Y, e, p, s: ex_trail(Y, e, p, s, 2.0),
    "X10 跌破EMA20": ex_ema,
    "X11 半倉1R+追蹤": ex_partial,
}


# ---------------------------------------------------------------- 進場（與第二十三 / 二十八部分 C1S2 相同）
def entries(df):
    X = indicators(df)
    idx = df.index
    reg = daily_regime(df)
    ema200 = np.sign(df.close - df.close.ewm(span=200, adjust=False).mean()).shift(1).to_numpy()
    d = df.close.resample("1D").last().dropna()
    sma20 = d.rolling(20).mean()
    prev_close = d.shift(1).reindex(idx.normalize()).to_numpy()
    prev_sma = sma20.shift(1).reindex(idx.normalize()).to_numpy()
    h1c = df.close.to_numpy()
    rows = []
    for Y, s_short in both_spaces(X):
        Y = dict(Y)
        Y["ema20"] = pd.Series(Y["c"]).ewm(span=20, adjust=False).mean().to_numpy()
        o, h, l, atr = Y["o"], Y["h"], Y["l"], Y["atr"]
        n = len(h)
        side = -s_short
        for e in detect(Y):
            if e["cond"] not in SIX:
                continue
            t, P1 = e["t"], e["P1"]
            A = atr[t]
            s2 = P1 + int(np.argmin(l[P1:t + 1]))
            base_stop = l[s2] - BUF * A
            ei = None
            for j in range(t + 1, min(t + 1 + WAIT_C, n - 1)):
                if h[j] > h[t]:
                    ei, ep = j, max(h[t], o[j])
                    break
                if l[j] <= base_stop:
                    break
            if ei is None:
                continue
            stop = min(base_stop, ep - MIN_R * A)
            k = ei - 1
            if not (side == reg[k] and side == ema200[k]):
                continue
            rows.append(dict(cond=e["cond"], side=side, ei=ei, ep=ep, stop=stop, Y=Y,
                             above_d=prev_close[k] > prev_sma[k], above_h=h1c[k] > prev_sma[k],
                             valid_sma=np.isfinite(prev_sma[k])))
    return idx, rows


FILTERS = {
    "F0 不加": lambda r: True,
    "F1 日收>SMA20不空": lambda r: not (r["side"] == -1 and r["above_d"]),
    "F2 F1+對稱": lambda r: not (r["side"] == -1 and r["above_d"]) and not (r["side"] == 1 and not r["above_d"]),
    "F3 H1收>SMA20不空": lambda r: not (r["side"] == -1 and r["above_h"]),
    "F4 F3+對稱": lambda r: not (r["side"] == -1 and r["above_h"]) and not (r["side"] == 1 and not r["above_h"]),
}


def run(idx, rows, fkey, xkey):
    f, ex = FILTERS[fkey], EXITS[xkey]
    out = []
    for r in rows:
        if not r["valid_sma"] or not f(r):
            continue
        xi, pr, why = ex(r["Y"], r["ei"], r["ep"], r["stop"])
        out.append(dict(cond=r["cond"], side=r["side"], entry_time=idx[r["ei"]], exit_time=idx[xi], pnl_R=pr, exit=why))
    T = pd.DataFrame(out)
    return pd.concat([nonoverlap(T[T.cond == k]) for k in SIX]).sort_values("entry_time")


def summarize(T):
    c = T.pnl_R * W
    days = pd.date_range("2023-04-01", "2026-10-09", freq="B")
    daily = c.groupby(T.exit_time.dt.normalize()).sum().reindex(days, fill_value=0)
    eq = daily.cumsum()
    a, b = T[T.entry_time < SPLIT].pnl_R, T[T.entry_time >= SPLIT].pnl_R
    yr = (T.pnl_R * W).groupby(T.entry_time.dt.year).sum()
    sh = T[T.side == -1]
    jul = T[(T.exit_time >= "2026-07-01") & (T.exit_time < "2026-08-01")]
    hold = (T.exit_time - T.entry_time).dt.total_seconds().mean() / 3600
    return dict(n=len(T), tot=c.sum(), dd=(eq.cummax() - eq).max(), sharpe=daily.mean() / daily.std() * np.sqrt(252),
                ra=a.mean(), ta=tstat(a), rb=b.mean(), tb=tstat(b), y=yr, short_n=len(sh), short_R=(sh.pnl_R * W).sum(),
                jul=(jul.pnl_R * W).sum(), hold=hold, win=(T.pnl_R > 0).mean())


def line(name, s):
    yrs = " ".join(f"{s['y'].get(y, 0):+5.1f}" for y in (2023, 2024, 2025, 2026))
    return (f"  {name:18s} {s['n']:5d} {s['win']:5.0%} {s['ra']:+.3f}({s['ta']:+.1f}) {s['rb']:+.3f}({s['tb']:+.1f}) "
            f"{s['tot']:+6.1f} {s['dd']:5.1f} {s['tot'] / s['dd']:5.1f} {s['sharpe']:5.2f}  {yrs}  "
            f"{s['short_n']:4d} {s['short_R']:+5.1f}  {s['jul']:+5.2f}  {s['hold']:5.1f}")


HEAD = (f"  {'':18s} {'筆數':>5s} {'勝率':>5s} {'研究期每筆':>12s} {'驗證期每筆':>12s} {'組合R':>6s} {'回撤':>5s} {'總/撤':>5s} "
        f"{'Sharpe':>5s}  {'2023':>5s} {'2024':>5s} {'2025':>5s} {'2026':>5s}  {'空筆':>4s} {'空R':>5s}  {'26/7':>5s}  {'持有h':>5s}")


def main():
    idx, rows = entries(get_frames()["H1"])
    print("H1 六合一（每版本 1/6 R），期間 2023/4 ~ 2026/10。組合R / 回撤 / 逐年 / 空R / 26/7 單位都是組合 R；每筆為單筆 R。")
    print("\n## 1. SMA20 過濾（出場 = 原本 2R / 32）")
    print(HEAD)
    for fk in FILTERS:
        print(line(fk, summarize(run(idx, rows, fk, "X0 2R/32(原本)"))))

    for fk in ("F0 不加", "F1 日收>SMA20不空", "F3 H1收>SMA20不空"):
        print(f"\n## 2. 出場方法（過濾 = {fk}）")
        print(HEAD)
        res = []
        for xk in EXITS:
            s = summarize(run(idx, rows, fk, xk))
            res.append((xk, s))
            print(line(xk, s))
        best = max(res, key=lambda z: z[1]["ra"])
        print(f"  → 只用研究期挑每筆最好的：{best[0]}，驗證期每筆 {best[1]['rb']:+.3f}R（原本 X0 {res[0][1]['rb']:+.3f}R）")


if __name__ == "__main__":
    main()
