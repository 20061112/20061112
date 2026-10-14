"""第二十三部分：哪一種「均值」比較適合當均值回歸的中線？

候選：SMA、EMA、HMA、KAMA、Kalman（局部線性趨勢）、滾動中位數、滾動 Huber M 估計。
（局部水準 Kalman 的穩態解 = EMA，所以不另列；這裡用「水準 + 斜率」版本。）

週期：M5（由 M1 合成，2023/6 ~ 2026/10）、M15（2023/1 ~ 2026/10）。
長度 n：M5 = 12 / 48 / 144 根（1h / 4h / 12h），M15 = 16 / 48 / 96 根（4h / 12h / 24h）。

「品質」分兩類：
1. 描述性（濾波器本身）：平滑度 = std(Δ中線) / std(Δ收盤)，越小越平滑；
   落後 = Δ中線 對 過去 Δ收盤 的交叉相關重心（根）。這只說明它長什麼樣，不代表能賺錢。
2. 均值回歸的預測力（真正的問題）：
   a. 斜率 b：未來 h 根報酬 / ATR = a + b · (收盤 − 中線) / ATR，h = n/2 與 n。b < 0 才是回歸。
      t 以週叢集；另外把每週 K 棒順序打亂 4 次，給出虛無分佈，z = (b − 打亂平均) / 打亂標準差。
   b. 事件 1:1：偏離第一次越過 ±2σ（σ = 偏離的滾動標準差，長度 4n）→ 反向進，目標碰中線、停損等距、最長持有 2n 根。
   c. 穩定性：2023-24 與 2025-26 分開看 b 的方向。
"""
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as swv
from load import load

rng = np.random.default_rng(23)
OUT = []


def P(s=""):
    print(s)
    OUT.append(s)


# ---------------------------------------------------------------- 濾波器
def sma(x, n):
    return pd.Series(x).rolling(n).mean().values


def ema(x, n):
    return pd.Series(x).ewm(span=n, adjust=False).mean().values


def wma(x, n):
    w = np.arange(1, n + 1, dtype=float)
    out = np.full(len(x), np.nan)
    m = np.isfinite(x)
    xi = np.where(m, x, 0.0)
    v = swv(xi, n) @ w / w.sum()
    ok = swv(m, n).all(1)
    out[n - 1:] = np.where(ok, v, np.nan)
    return out


def hma(x, n):
    return wma(2 * wma(x, n // 2) - wma(x, n), int(round(np.sqrt(n))))


def kama(x, n, fast=2):
    """ER 長度 n，最快 = EMA(2)，最慢 = EMA(n)（震盪時退化成 EMA(n)）。"""
    slow = n
    ch = np.abs(x - np.r_[np.full(n, np.nan), x[:-n]])
    vol = pd.Series(np.abs(np.diff(x, prepend=x[0]))).rolling(n).sum().values
    er = np.where(vol > 0, ch / vol, 0)
    fs, ss = 2 / (fast + 1), 2 / (slow + 1)
    sc = (er * (fs - ss) + ss) ** 2
    out = np.full(len(x), np.nan)
    out[n] = x[n]
    for i in range(n + 1, len(x)):
        out[i] = out[i - 1] + sc[i] * (x[i] - out[i - 1])
    return out


def kalman_llt(x, n):
    """局部線性趨勢 Kalman：狀態 = [水準, 斜率]。Q 依長度 n 設定，量測雜訊 R 以滾動報酬變異估計。"""
    a = 2 / (n + 1)
    ql = a * a / (1 - a)
    qs = ql / n ** 2
    r = pd.Series(np.diff(x, prepend=x[0])).rolling(4 * n, min_periods=20).var().bfill().values
    lvl, slp = x[0], 0.0
    P11, P12, P22 = 1.0, 0.0, 1.0
    out = np.empty(len(x))
    for i in range(len(x)):
        R = r[i] if r[i] > 0 else 1e-6
        # 預測
        lvl, slp = lvl + slp, slp
        P11, P12, P22 = P11 + 2 * P12 + P22 + ql * R, P12 + P22, P22 + qs * R
        # 更新
        S = P11 + R
        k1, k2 = P11 / S, P12 / S
        e = x[i] - lvl
        lvl += k1 * e; slp += k2 * e
        P11, P12, P22 = (1 - k1) * P11, (1 - k1) * P12, P22 - k2 * P12
        out[i] = lvl
    return out


def rmedian(x, n):
    return pd.Series(x).rolling(n).median().values


def rhuber(x, n, c=1.345, iters=8):
    out = np.full(len(x), np.nan)
    W = swv(x, n)
    mu = np.median(W, 1)
    s = 1.4826 * np.median(np.abs(W - mu[:, None]), 1)
    s = np.where(s > 0, s, 1e-9)
    for _ in range(iters):
        u = np.abs(W - mu[:, None]) / (c * s[:, None])
        w = 1.0 / np.maximum(u, 1.0)
        mu = (w * W).sum(1) / w.sum(1)
    out[n - 1:] = mu
    return out


FILTERS = {"SMA": sma, "EMA": ema, "HMA": hma, "KAMA": kama, "Kalman": kalman_llt, "中位數": rmedian, "Huber": rhuber}


# ---------------------------------------------------------------- 資料
def m5_from_m1():
    d = load("data/XAUUSD_M1_2023_2026.csv")
    r = d.resample("5min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "tickvol": "sum", "spread": "mean"})
    return r.dropna(subset=["close"])


def prep(d):
    d = d.copy()
    pc = d.close.shift()
    tr = np.maximum(d.high - d.low, np.maximum((d.high - pc).abs(), (d.low - pc).abs()))
    d["atr"] = tr.rolling(96).mean()
    d["week"] = d.index.to_period("W").astype(str)
    return d.iloc[100:]


def shuffle_weeks(d):
    out = []
    for wk, g in d.groupby("week", sort=True):
        c = g.close.values
        pc = np.r_[g.open.values[0], c[:-1]]
        rel = np.c_[g.open.values - pc, g.high.values - pc, g.low.values - pc, c - pc]
        idx = rng.permutation(len(g))
        rel = rel[idx]
        nc = pc[0] + np.cumsum(rel[:, 3])
        npc = np.r_[pc[0], nc[:-1]]
        out.append(pd.DataFrame({"open": npc + rel[:, 0], "high": npc + rel[:, 1], "low": npc + rel[:, 2], "close": nc,
                                 "spread": g.spread.values[idx], "atr": g.atr.values[idx], "week": wk}, index=g.index))
    s = pd.concat(out)
    pc = s.close.shift()
    tr = np.maximum(s.high - s.low, np.maximum((s.high - pc).abs(), (s.low - pc).abs()))
    s["atr"] = tr.rolling(96, min_periods=1).mean()
    return s


# ---------------------------------------------------------------- 指標
def slope(x, y, grp):
    m = np.isfinite(x) & np.isfinite(y)
    x, y, grp = x[m], y[m], grp[m]
    xc = x - x.mean()
    sxx = (xc ** 2).sum()
    b = (xc * (y - y.mean())).sum() / sxx
    e = (y - y.mean()) - b * xc
    sc = pd.Series(xc * e).groupby(grp).sum().values
    return b, b / (np.sqrt((sc ** 2).sum()) / sxx)


def slopes(d, f, h):
    c, atr = d.close.values, d.atr.values
    dev = (c - f) / atr
    fwd = (np.r_[c[h:], np.full(h, np.nan)] - c) / atr
    grp = d.week.values
    yr = d.index.year.values
    full = slope(dev, fwd, grp)
    is_ = slope(dev[yr <= 2024], fwd[yr <= 2024], grp[yr <= 2024])
    oos = slope(dev[yr >= 2025], fwd[yr >= 2025], grp[yr >= 2025])
    return full, is_, oos


def describe(c, f, n):
    dc, df = np.diff(c), np.diff(f)
    m = np.isfinite(df)
    dc, df = dc[m], df[m]
    smooth = df.std() / dc.std()
    K = min(3 * n, 300)
    cors = np.array([np.corrcoef(df[k:], dc[:len(dc) - k])[0, 1] for k in range(K)])
    cp = np.clip(cors, 0, None)
    lag = (np.arange(K) * cp).sum() / cp.sum()
    return smooth, lag


def events(d, f, n, T=2.0):
    c, h, l = d.close.values, d.high.values, d.low.values
    spr = d.spread.values * 0.01
    dev = c - f
    sd = pd.Series(dev).rolling(4 * n).std().values
    z = dev / sd
    zp = np.r_[np.nan, z[:-1]]
    ev = np.where((np.abs(z) >= T) & (np.abs(zp) < T) & (np.sign(z) == np.sign(zp)))[0]
    R, Rn, hit = [], [], []
    H = 2 * n
    for i in ev:
        if i + 1 >= len(c):
            continue
        side = -np.sign(dev[i]); e = c[i]; risk = abs(dev[i])
        stop = e - side * risk
        k = np.arange(i + 1, min(i + 1 + H, len(c)))
        tgt = f[k - 1]
        if side > 0:
            hs, ht = l[k] <= stop, h[k] >= tgt
        else:
            hs, ht = h[k] >= stop, l[k] <= tgt
        fs = np.argmax(hs) if hs.any() else 10 ** 9
        ft = np.argmax(ht) if ht.any() else 10 ** 9
        if fs == ft == 10 ** 9:
            pnl, w = (c[k[-1]] - e) * side, np.nan
        elif fs <= ft:
            pnl, w = -risk, 0.0
        else:
            pnl, w = (tgt[ft] - e) * side, 1.0
        R.append(pnl / risk); Rn.append((pnl - spr[i]) / risk); hit.append(w)
    R = np.array(R)
    return len(R), np.nanmean(hit), R.mean(), R.mean() / (R.std() / np.sqrt(len(R))), np.mean(Rn)


# ---------------------------------------------------------------- main
def run(tf, d, lengths, nshuf=4):
    sh = [shuffle_weeks(d) for _ in range(nshuf)]
    P("=" * 130)
    P(f"{tf}：{d.index[0]} ~ {d.index[-1]}，{len(d):,} 根")
    for n in lengths:
        P("-" * 130)
        P(f"  長度 n = {n}（h = {n//2} / {n} 根）")
        P(f"  {'濾波器':<8}{'平滑':>6}{'落後':>6} | {'b(h=n/2)':>9}{'t':>6}{'z虛無':>7} | {'b(h=n)':>9}{'t':>6}{'z虛無':>7}"
          f" | {'23-24 b':>8}{'25-26 b':>9} | 事件±2σ: {'n':>6}{'先到中線':>8}{'平均R':>8}{'t':>6}{'扣點差':>8}")
        for name, fn in FILTERS.items():
            f = fn(d.close.values, n)
            sm, lag = describe(d.close.values, f, n)
            row = f"  {name:<8}{sm:6.3f}{lag:6.1f} |"
            sf = [fn(s.close.values, n) for s in sh]
            for h in (n // 2, n):
                (b, t), _, _ = slopes(d, f, h)
                nb = np.array([slopes(s, g, h)[0][0] for s, g in zip(sh, sf)])
                row += f" {b:+9.4f}{t:+6.1f}{(b - nb.mean()) / nb.std():+7.1f} |"
            _, (bi, ti), (bo, to) = slopes(d, f, n)
            row += f" {bi:+8.4f}{bo:+9.4f} |"
            ne, hr, mr, tr, mrn = events(d, f, n)
            row += f"          {ne:6d}{hr:8.1%}{mr:+8.3f}{tr:+6.1f}{mrn:+8.3f}"
            P(row)


def main():
    P(__doc__.strip().splitlines()[0])
    P("欄位：平滑 = std(Δ中線)/std(Δ收盤)（小 = 平滑）；落後 = 交叉相關重心（根）；b < 0 = 回歸；"
      "z虛無 = 相對於週內打亂 4 次的 z 值（< −2 才算比隨機更回歸）")
    m15 = prep(load("data/XAUUSD_M15_2023_2026.csv"))
    run("M15", m15, [16, 48, 96])
    m5 = prep(m5_from_m1())
    run("M5", m5, [12, 48, 144])
    with open("results_part23.txt", "w") as fh:
        fh.write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
