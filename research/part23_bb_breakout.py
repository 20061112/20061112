"""第二十三部分：布林帶 2σ / 3σ 觸碰「順勢」突破單（第二十二部分反彈單的反向假設）。

規則（多單；空單鏡像）：
  布林帶：BB(20)，用「上一根已收盤」的帶值（M15 / H1）
  進場：價格碰到 中軌 + k·σ（k = 2 或 3）→ 停損買單在該價成交（跳空則以開盤價），
        成交時點用 M1 逐根找，不靠 K 棒內順序假設
  停損：進場價往回 sl 個 σ（固定價位，進場時決定），sl = 1 / 2，或 "mid" = 進場時的中軌
  出場：
    R2 / R3     固定停利 2R / 3R
    trail_mid   該週期 K 棒收盤跌回中軌下方 → 以該收盤平倉
    trail_1     收盤跌回 +1σ 下方 → 平倉
    time        只靠時間平倉
  所有版本都有停損，並在 hold 根後以收盤平倉
  停損、停利用 M1 逐根檢查；同一根 M1 同時碰到停損與停利算停損；成交那根 M1 只檢查停損
  成本：扣進場當下點差（MT5 為 bid 價）

基準：黃金 2023~2026 漲了一倍多，多單會吃到趨勢 drift。
  因此另做「隨機進場」基準：在隨機的 K 棒以開盤價進場、方向與停損/出場規則相同，
  用來判斷訊號本身是否比單純做多 / 做空更好。

資料：data/XAUUSD_M1_2023_2026.csv（2023/6 ~ 2026/10）。
樣本內 2023/6~2024/12，樣本外 2025/1~2026/10。
"""
import itertools
import sys

import numpy as np
import pandas as pd

from backtest import resample
from load import load

IS_END = pd.Timestamp("2025-01-01")
HOLD = {"M15": 64, "H1": 48}
RULE = {"M15": "15min", "H1": "1h"}


class Data:
    def __init__(self, m1, tf, n=20):
        df = resample(m1, RULE[tf])
        self.df = df
        mid = df.close.rolling(n).mean()
        sd = df.close.rolling(n).std(ddof=0)
        self.mid_now, self.sd_now = mid.to_numpy(), sd.to_numpy()        # 收盤時的帶值（出場判斷用）
        self.mid, self.sd = mid.shift(1).to_numpy(), sd.shift(1).to_numpy()  # 上一根的帶值（進場用）
        self.o, self.h, self.l, self.c = (df[x].to_numpy(float) for x in ("open", "high", "low", "close"))
        self.mo, self.mh, self.ml = (m1[x].to_numpy(float) for x in ("open", "high", "low"))
        self.mspr = m1.spread.to_numpy(float) * 0.01
        # 每根 TF K 棒對應的 M1 索引範圍 [start[t], start[t+1])
        self.start = np.searchsorted(m1.index.values, df.index.values)
        self.end = np.r_[self.start[1:], len(m1)]
        self.time = df.index


def run_trade(d, t, side, px, a, s0, m0, sl, ex, hold):
    """從 M1 索引 a（成交那根）開始模擬一筆單，回傳 (出場價, 原因, 出場的 TF 索引)。"""
    stop = m0 if sl == "mid" else px - side * sl * s0
    risk = side * (px - stop)
    if not risk > 0:
        return None
    tp = px + side * float(ex[1:]) * risk if ex in ("R2", "R3") else None
    # 成交那根 M1 只檢查停損
    if (d.ml[a] <= stop) if side == 1 else (d.mh[a] >= stop):
        return stop, "sl", t, risk
    last = min(t + hold, len(d.c) - 1)
    for j in range(t, last + 1):
        b0 = a + 1 if j == t else d.start[j]
        b1 = d.end[j]
        if b1 > b0:
            lo, hi, op = d.ml[b0:b1], d.mh[b0:b1], d.mo[b0:b1]
            hit_sl = (lo <= stop) if side == 1 else (hi >= stop)
            i_sl = np.argmax(hit_sl) if hit_sl.any() else None
            i_tp = None
            if tp is not None:
                hit_tp = (hi >= tp) if side == 1 else (lo <= tp)
                i_tp = np.argmax(hit_tp) if hit_tp.any() else None
            if i_sl is not None and (i_tp is None or i_sl <= i_tp):
                p = min(op[i_sl], stop) if side == 1 else max(op[i_sl], stop)
                return p, "sl", j, risk
            if i_tp is not None:
                p = max(op[i_tp], tp) if side == 1 else min(op[i_tp], tp)
                return p, "tp", j, risk
        # 收盤出場條件（以該 TF K 棒收盤判斷）
        if ex in ("trail_mid", "trail_1"):
            lvl = d.mid_now[j] + side * (1.0 if ex == "trail_1" else 0.0) * d.sd_now[j]
            if (d.c[j] < lvl) if side == 1 else (d.c[j] > lvl):
                return d.c[j], "trail", j, risk
    return d.c[last], "time", last, risk


def simulate(d, k, sl, ex, hold):
    out = []
    t, n = 21, len(d.c)
    while t < n - 1:
        m, s = d.mid[t], d.sd[t]
        if not s > 0:
            t += 1
            continue
        up, dn = m + k * s, m - k * s
        side = 1 if d.h[t] >= up else (-1 if d.l[t] <= dn else 0)
        if d.h[t] >= up and d.l[t] <= dn:
            side = 0  # 同一根上下都碰到，順序不明，跳過
        if side == 0:
            t += 1
            continue
        lvl = up if side == 1 else dn
        b0, b1 = d.start[t], d.end[t]
        hit = (d.mh[b0:b1] >= lvl) if side == 1 else (d.ml[b0:b1] <= lvl)
        if not hit.any():
            t += 1
            continue
        a = b0 + int(np.argmax(hit))
        px = max(d.mo[a], lvl) if side == 1 else min(d.mo[a], lvl)
        r = run_trade(d, t, side, px, a, s, m, sl, ex, hold)
        if r is None:
            t += 1
            continue
        exit_px, why, j, risk = r
        pnl = side * (exit_px - px) - d.mspr[a]
        out.append((d.time[t], side, px, exit_px, why, j - t, pnl, pnl / s, pnl / risk))
        t = j + 1
    return pd.DataFrame(out, columns=["time", "side", "entry", "exit", "why", "bars", "pnl", "pnl_sd", "pnl_r"])


def random_baseline(d, tr, sl, ex, hold, reps=5, seed=0):
    """隨機 K 棒開盤進場，方向分布與訊號相同，停損距離用該時點的 σ，同樣規則出場。"""
    rng = np.random.default_rng(seed)
    n = len(d.c)
    res = []
    valid = np.where(d.sd > 0)[0]
    valid = valid[(valid > 21) & (valid < n - 1)]
    for _ in range(reps):
        for side in tr.side:
            t = int(rng.choice(valid))
            a = d.start[t]
            px = d.mo[a]
            r = run_trade(d, t, side, px, a, d.sd[t], d.mid[t] if sl != "mid" else px - side * 3 * d.sd[t],
                          sl, ex, hold)
            if r is None:
                continue
            exit_px, why, j, risk = r
            pnl = side * (exit_px - px) - d.mspr[a]
            res.append((d.time[t], side, pnl / d.sd[t]))
    return pd.DataFrame(res, columns=["time", "side", "pnl_sd"])


def tstat(x):
    x = np.asarray(x)
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan


def summarize(tr):
    r = {}
    for tag, part in (("IS", tr[tr.time < IS_END]), ("OOS", tr[tr.time >= IS_END])):
        p = part.pnl.to_numpy()
        g, b = p[p > 0].sum(), -p[p < 0].sum()
        r[f"{tag}_n"] = len(p)
        r[f"{tag}_win"] = (p > 0).mean() if len(p) else np.nan
        r[f"{tag}_sd"] = part.pnl_sd.mean()
        r[f"{tag}_t"] = tstat(part.pnl_sd)
        r[f"{tag}_pf"] = g / b if b > 0 else np.nan
    for sd_, name in ((1, "long"), (-1, "short")):
        r[f"{name}_sd"] = tr[tr.side == sd_].pnl_sd.mean()
    for y, part in tr.groupby(tr.time.dt.year):
        r[f"y{y}"] = part.pnl_sd.mean()
    return r


def main():
    m1 = load("data/XAUUSD_M1_2023_2026.csv")
    rows = []
    for tf in ("M15", "H1"):
        d = Data(m1, tf)
        for k, sl, ex in itertools.product((2, 3), (1, 2, "mid"), ("R2", "R3", "trail_mid", "trail_1", "time")):
            if sl == "mid" and k == 2:
                continue  # k=2 時 mid 即 2σ 停損，重複
            tr = simulate(d, k, sl, ex, HOLD[tf])
            r = dict(tf=tf, k=k, sl=sl, ex=ex, **summarize(tr))
            base = random_baseline(d, tr, sl, ex, HOLD[tf])
            r["base_long"] = base[base.side == 1].pnl_sd.mean()
            r["base_short"] = base[base.side == -1].pnl_sd.mean()
            # 訊號減去隨機基準（多空分開扣掉，再依訊號的多空比例加權）
            w = (tr.side == 1).mean()
            r["edge"] = (r["long_sd"] - r["base_long"]) * w + (r["short_sd"] - r["base_short"]) * (1 - w)
            rows.append(r)
            print(tf, k, sl, ex, len(tr), round(r["OOS_sd"], 3), round(r["edge"], 3), file=sys.stderr)
    res = pd.DataFrame(rows)
    res.to_csv("part23_bb_breakout_scan.csv", index=False)
    pd.set_option("display.width", 260, "display.max_columns", 40, "display.max_rows", 200)
    cols = ["tf", "k", "sl", "ex", "IS_n", "IS_win", "IS_sd", "IS_t", "IS_pf", "OOS_n", "OOS_win", "OOS_sd",
            "OOS_t", "OOS_pf", "long_sd", "short_sd", "base_long", "base_short", "edge",
            "y2023", "y2024", "y2025", "y2026"]
    print(res[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
