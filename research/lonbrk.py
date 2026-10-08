"""
XAUUSD  LONBRK  —  倫敦開盤突破亞洲區間（與 FADE 互補的日級模組）參考實作
=====================================================================================
設計理由（來自 FADE 與本研究）：
  - FADE 在倫敦開盤 (broker 10-12) 排除交易，在倫敦紐約重疊 (15-19) 獲利 → 該時段「延伸會回歸」。
  - 本研究：同樣的突破規則，只有在倫敦開盤時段有延續性（+0.2R）；搬到 16-19 反而 −0.15R。
  → LONBRK 只在 FADE 不做的時段進場，做「順勢延續」，與 FADE 的「均值回歸」互補。

規則（broker 時間 = 紐約 + 7；交易日 = broker 01:00~23:59）
  1. 亞洲區間 = 當日 broker 01:00~09:59 的最高 / 最低（中間價或 bid 皆可，需與實盤一致）
  2. broker 10:00~12:59 第一次突破區間高點 → 做多；第一次跌破低點 → 做空（一天最多一筆）
     進場價 = 突破價（stop 單）；若跳空開在區間外，則以該根開盤價
  3. 停損 = 區間另一側（伺服器端停損）
  4. 出場 = broker 19:00 市價平倉（不跨日）；沒有停利、沒有移動停損
  5. 星期五同樣規則（19:00 前已平倉，不受 FADE 星期五 20:00 截止影響）
  6. 倉位：固定風險 → units = RISK_PER_TRADE / R（R = 進場價到停損的距離，點）

用法：python3 lonbrk.py [M15 或 M1 CSV]   → 印出摘要、輸出 lonbrk_trades.csv 與 lonbrk_daily.csv
"""
import sys
import numpy as np
import pandas as pd


class CFG:
    DAY_START_H = 1
    ASIA = (1, 10)            # [01:00, 10:00)
    WINDOW = (10, 13)         # [10:00, 13:00)
    EXIT_H = 19
    RISK_PER_TRADE = 100.0    # 每筆風險（點 × 單位），依帳戶調整
    MIN_BARS_ASIA = 8


def load_bars(path):
    df = pd.read_csv(path, sep="\t")
    df.columns = [c.strip("<>").lower() for c in df.columns]
    df.index = pd.to_datetime(df["date"] + " " + df["time"], format="%Y.%m.%d %H:%M:%S")
    df["sp"] = df["spread"] * 0.01
    return df[["open", "high", "low", "close", "sp"]]


def trades(bars):
    day = (bars.index - pd.Timedelta(hours=CFG.DAY_START_H)).normalize()
    out = []
    for k, g in bars.groupby(day):
        hr = g.index.hour
        asia = g[(hr >= CFG.ASIA[0]) & (hr < CFG.ASIA[1])]
        win = g[(hr >= CFG.WINDOW[0]) & (hr < CFG.WINDOW[1])]
        if len(asia) < CFG.MIN_BARS_ASIA or len(win) == 0:
            continue
        hi, lo = asia.high.max(), asia.low.min()
        for t, b in win.iterrows():
            s = 1 if b.high > hi else -1 if b.low < lo else 0
            if s == 0:
                continue
            ep = max(hi, b.open) if s == 1 else min(lo, b.open)
            stop = lo if s == 1 else hi
            R = abs(ep - stop)
            path = g[(g.index >= t) & (g.index.hour < CFG.EXIT_H)]
            xp, xt, why = path.close.iloc[-1], path.index[-1], "time"
            for j, (tt, bb) in enumerate(path.iterrows()):
                if (bb.low <= stop) if s == 1 else (bb.high >= stop):
                    xp = stop if j == 0 else (min(stop, bb.open) if s == 1 else max(stop, bb.open))
                    xt, why = tt, "stop"
                    break
            pts = s * (xp - ep) - b.sp
            units = CFG.RISK_PER_TRADE / R
            out.append(dict(day=k, entry_time=t, exit_time=xt, side=s, asia_hi=hi, asia_lo=lo, entry=ep, stop=stop,
                            R=R, exit=why, pnl_pts=pts, pnl_R=pts / R, units=units, net=pts * units))
            break
    return pd.DataFrame(out)


def summary(T):
    r = T.pnl_R
    eq = T.net.cumsum()
    m = T.groupby(T.day.dt.to_period("M")).net.sum()
    w = T.groupby(T.day.dt.to_period("W")).net.sum()
    return dict(trades=len(T), win=round((r > 0).mean(), 3), exp_R=round(r.mean(), 3),
                t=round(r.mean() / r.std() * np.sqrt(len(r)), 2), pf=round(T.net[T.net > 0].sum() / -T.net[T.net < 0].sum(), 2),
                total=round(T.net.sum(), 1), mdd=round((eq.cummax() - eq).max(), 1),
                months_pos=f"{(m > 0).sum()}/{len(m)}", weeks_pos=f"{(w > 0).sum()}/{len(w)}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/XAUUSD_M15_full.csv"
    T = trades(load_bars(path))
    print(summary(T))
    half = T.day < pd.Timestamp("2026-06-01")
    print("前半", summary(T[half]))
    print("後半", summary(T[~half]))
    T.to_csv("lonbrk_trades.csv", index=False)
    T.groupby(T.day.dt.date).net.sum().rename("lonbrk_net").to_csv("lonbrk_daily.csv")
