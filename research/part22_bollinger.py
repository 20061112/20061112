"""第二十二部分：布林帶 2σ / 3σ 觸碰反彈單。

規則（多單；空單鏡像）：
  布林帶：中軌 = SMA(n)，σ = 收盤 n 根標準差；用「上一根已收盤」的帶值，不偷看
  進場：K 棒最低價碰到 中軌 − k·σ（k = 2 或 3）→ 掛限價單在該價成交（開盤就跳空在下方則以開盤價成交）
  停利：價格回到目標帶 tp（以 σ 表示，-2 = 下軌 2σ、-1、0 = 中軌、+1、+2 = 上軌 2σ），帶值逐根更新
  停損：進場價再往外 sl 個 σ（固定價位，進場時決定）；sl = None 表示不設停損，只靠時間平倉
  時間平倉：持有超過 hold 根以收盤價平倉
  成本：每筆扣進場當下點差（MT5 K 棒為 bid 價，spread 單位 0.01 美元）
  保守假設：進場那根只檢查停損、不算停利；同一根同時碰到停損與停利一律算停損
  同一時間只持有一筆

資料：data/XAUUSD_M1_2023_2026.csv（2023/6 ~ 2026/10），重採樣為 M5 / M15 / H1。
樣本內 2023/6~2024/12 挑參數，樣本外 2025/1~2026/10 驗證；另列逐年結果。
"""
import itertools
import sys

import numpy as np
import pandas as pd

from backtest import resample
from load import load

IS_END = pd.Timestamp("2025-01-01")


def bands(df, n):
    mid = df.close.rolling(n).mean()
    sd = df.close.rolling(n).std(ddof=0)
    # 用上一根已收盤的帶值
    return mid.shift(1).to_numpy(), sd.shift(1).to_numpy()


def simulate(df, mid, sd, k, tp, sl, hold):
    o, h, l, c = (df[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    spr = df.spread.to_numpy(float) * 0.01
    idx = df.index
    n = len(df)
    out = []
    t = 0
    while t < n - 1:
        m, s = mid[t], sd[t]
        if not s > 0:
            t += 1
            continue
        side = 0
        if l[t] <= m - k * s:
            side, px = 1, min(o[t], m - k * s)
        elif h[t] >= m + k * s:
            side, px = -1, max(o[t], m + k * s)
        if side == 0:
            t += 1
            continue
        s0 = s
        stop = px - side * sl * s0 if sl is not None else None
        exit_px, why, j = None, None, t
        # 進場那根只檢查停損
        if stop is not None and (l[t] <= stop if side == 1 else h[t] >= stop):
            exit_px, why = stop, "sl"
        else:
            for j in range(t + 1, min(t + 1 + hold, n)):
                if stop is not None:
                    if side == 1 and l[j] <= stop:
                        exit_px, why = min(o[j], stop), "sl"
                        break
                    if side == -1 and h[j] >= stop:
                        exit_px, why = max(o[j], stop), "sl"
                        break
                target = mid[j] + side * tp * sd[j]  # 多單目標 = 中軌 + tp·σ；空單鏡像
                if side == 1 and h[j] >= target:
                    exit_px, why = max(o[j], target), "tp"
                    break
                if side == -1 and l[j] <= target:
                    exit_px, why = min(o[j], target), "tp"
                    break
            else:
                exit_px, why = c[j], "time"
        pnl = side * (exit_px - px) - spr[t]
        out.append((idx[t], side, px, exit_px, why, j - t, pnl, pnl / s0))
        t = j + 1
    return pd.DataFrame(out, columns=["time", "side", "entry", "exit", "why", "bars", "pnl", "pnl_sd"])


def stats(tr):
    if len(tr) == 0:
        return dict(n=0, win=np.nan, avg=np.nan, avg_sd=np.nan, t=np.nan, pf=np.nan, total=0.0, mdd=0.0)
    p = tr.pnl.to_numpy()
    eq = p.cumsum()
    mdd = (np.maximum.accumulate(np.r_[0, eq]) - np.r_[0, eq]).max()
    g, b = p[p > 0].sum(), -p[p < 0].sum()
    q = tr.pnl_sd.to_numpy()
    return dict(n=len(p), win=(p > 0).mean(), avg=p.mean(), avg_sd=q.mean(),
                t=q.mean() / (q.std(ddof=1) / np.sqrt(len(q))) if len(q) > 1 else np.nan,
                pf=g / b if b > 0 else np.inf, total=p.sum(), mdd=mdd)


def main():
    m1 = load("data/XAUUSD_M1_2023_2026.csv")
    tfs = {"M5": "5min", "M15": "15min", "H1": "1h"}
    holds = {"M5": 96, "M15": 64, "H1": 48}  # 約 8 小時 / 16 小時 / 2 天
    rows = []
    for tf, rule in tfs.items():
        df = resample(m1, rule)
        for n in (20,):
            mid, sd = bands(df, n)
            for k, tp, sl in itertools.product((2, 3), (-2, -1, 0, 1, 2), (None, 1, 2)):
                if tp <= -k:
                    continue
                tr = simulate(df, mid, sd, k, tp, sl, holds[tf])
                ins, oos = tr[tr.time < IS_END], tr[tr.time >= IS_END]
                r = dict(tf=tf, n=n, k=k, tp=tp, sl=sl if sl is not None else "無")
                for tag, part in (("IS", ins), ("OOS", oos)):
                    for key, v in stats(part).items():
                        r[f"{tag}_{key}"] = v
                for y, part in tr.groupby(tr.time.dt.year):
                    r[f"y{y}"] = part.pnl_sd.mean()
                rows.append(r)
                print(tf, n, k, tp, sl, len(tr), file=sys.stderr)
    res = pd.DataFrame(rows)
    res.to_csv("part22_bollinger_scan.csv", index=False)

    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 200)
    fmt = res[["tf", "k", "tp", "sl", "IS_n", "IS_win", "IS_avg_sd", "IS_t", "IS_pf",
               "OOS_n", "OOS_win", "OOS_avg", "OOS_avg_sd", "OOS_t", "OOS_pf",
               "y2023", "y2024", "y2025", "y2026"]].round(3)
    print(fmt.to_string(index=False))


if __name__ == "__main__":
    main()
