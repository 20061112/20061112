"""第二十二部分：大漲/大跌途中的「小根 K 線」— 是規律還是巧合？

使用者的觀察：
  (A) 漲很多的途中出現一根小 K 線之後，會繼續往原方向走。
  (B) 之後若回撤到那根小 K 線，那裡是很強的支撐（大跌時則是壓力）。

資料：data/XAUUSD_M1_2023_2026.csv（2023/6/1 ~ 2026/10/8），M5 / M15 / H1 由 M1 重新取樣。
一開始就用多年資料（第十部分的教訓：只用 2026 研究的策略到其他年份全部失效）。

定義（多頭；空頭鏡像），全部只用到小 K 線收盤前已知的資料：
  ATR       = ATR(20)，取到前一根（不含小 K 線本身）
  大漲背景  = 前 N=10 根內最低點到前一根收盤 >= K × ATR，且前一根收盤距 10 根最高點 <= 1 ATR（仍在高檔、走勢進行中）
  小 K 線   = 本根全長 (high-low) <= 0.5 ATR
  對照組    = 同樣大漲背景、但本根不是小 K 線（一般大小的 K 線）

檢驗方法（用 M1 逐分鐘路徑判定先碰到哪邊，同一分鐘兩邊都碰到算不利的一邊）：
  (A) 從小 K 線收盤價出發，上下各 X ATR 的對稱關卡，先碰上方的機率。
      隨機漫步（沒有規律）時剛好 50%。
  (B) 小 K 線中點當作價位 L。之後價格必須先離開 L 至少 1 ATR（真的走了一段，期間沒先跌破 L−X），
      再回到 L 才算一次「回撤測試」（等待最多 W 根）。觸價那根 M1 只採計不利方向。
      從碰到 L 那一刻起，先反彈 +X ATR 還是先跌破 −X ATR？沒有支撐時 50%。
      對照價位：一般 K 線中點、以及小 K 線中點往上/下平移 0.5 ATR 的「假價位」。
"""
import numpy as np
import pandas as pd
from load import load

M1 = load("data/XAUUSD_M1_2023_2026.csv")
T1 = M1.index.values
H1A, L1A = M1.high.values, M1.low.values
SP1 = M1.spread.values * 0.01
TF_MIN = {"M5": 5, "M15": 15, "H1": 60}


def resample(rule):
    return M1.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "spread": "max"}).dropna()


def atr(df, n=20):
    pc = df.close.shift()
    tr = pd.concat([df.high - df.low, (df.high - pc).abs(), (df.low - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean().shift()          # 不含本根


def events(df, N=10, K=4.0, small=0.5):
    a = atr(df)
    lo_n = df.low.shift().rolling(N).min()
    hi_n = df.high.shift().rolling(N).max()
    pc = df.close.shift()
    up = ((pc - lo_n) >= K * a) & ((hi_n - pc) <= a)
    dn = ((hi_n - pc) >= K * a) & ((pc - lo_n) <= a)
    rng = df.high - df.low
    E = pd.DataFrame({"atr": a, "close": df.close, "mid": (df.high + df.low) / 2,
                      "dir": np.where(up, 1, np.where(dn, -1, 0)),
                      "small": rng <= small * a, "spread": df.spread * 0.01}, index=df.index)
    E = E[(E.dir != 0) & a.notna()]
    return E


def first_passage(start, up, dn, maxlen, skip_first=None):
    """從 M1 索引 start 開始，先碰 up 回 +1、先碰 dn 回 -1、同分鐘都碰算 -1、逾時 0。
    skip_first='up'/'dn'：第一根 M1 不採計該側（觸價那根的另一端可能發生在觸價之前）。"""
    h, l = H1A[start:start + maxlen].copy(), L1A[start:start + maxlen].copy()
    if skip_first == "up" and len(h):
        h[0] = -np.inf
    if skip_first == "dn" and len(l):
        l[0] = np.inf
    iu = np.argmax(h >= up) if (h >= up).any() else 10 ** 9
    idn = np.argmax(l <= dn) if (l <= dn).any() else 10 ** 9
    if iu == idn == 10 ** 9:
        return 0
    return 1 if iu < idn else -1


def continuation(E, tf, X, maxlen=5 * 1440):
    """(A) 從收盤價出發的對稱關卡，順勢方向先碰到 → +1。"""
    out = []
    end_times = E.index.values + np.timedelta64(TF_MIN[tf], "m")
    starts = np.searchsorted(T1, end_times)
    for s, c, a, d in zip(starts, E.close.values, E.atr.values, E.dir.values):
        r = first_passage(s, c + X * a, c - X * a, maxlen)
        out.append(r * d)
    return np.array(out)


def level_test(E, tf, X, level_shift=0.0, away=1.0, W=100, maxlen=5 * 1440):
    """(B) 回撤到價位 L 時是否有支撐/壓力。回傳每筆：+1 守住、-1 跌破、nan 沒有發生回撤測試。"""
    out = []
    end_times = E.index.values + np.timedelta64(TF_MIN[tf], "m")
    starts = np.searchsorted(T1, end_times)
    wmin = W * TF_MIN[tf]
    for s, m, a, d in zip(starts, E.mid.values, E.atr.values, E.dir.values):
        L = m + d * level_shift * a
        h, l = H1A[s:s + wmin], L1A[s:s + wmin]
        if d == 1:
            # 先離開 L 到 L+away 之前不能先跌破 L
            hit_away = np.argmax(h >= L + away * a) if (h >= L + away * a).any() else None
            broke = np.argmax(l <= L - X * a) if (l <= L - X * a).any() else None
            if hit_away is None or (broke is not None and broke <= hit_away):
                out.append(np.nan); continue
            back = l[hit_away + 1:] <= L
            if not back.any():
                out.append(np.nan); continue
            t = s + hit_away + 1 + np.argmax(back)
            out.append(first_passage(t, L + X * a, L - X * a, maxlen, skip_first="up"))
        else:
            hit_away = np.argmax(l <= L - away * a) if (l <= L - away * a).any() else None
            broke = np.argmax(h >= L + X * a) if (h >= L + X * a).any() else None
            if hit_away is None or (broke is not None and broke <= hit_away):
                out.append(np.nan); continue
            back = h[hit_away + 1:] >= L
            if not back.any():
                out.append(np.nan); continue
            t = s + hit_away + 1 + np.argmax(back)
            out.append(-first_passage(t, L + X * a, L - X * a, maxlen, skip_first="dn"))
    return np.array(out, dtype=float)


def breakout_trades(E, df, tf, tp=2.0, wait=3, buf=0.1, maxlen=1440):
    """策略化：小 K 線之後 wait 根內突破小 K 高點（空單為低點）進場，停損 = 小 K 另一端外 buf ATR，停利 tp R。
    進出場用 M1 逐根；成交那根同時碰到停損算停損；逾時以最後一根 M1 收盤（這裡簡化為 0R 以外的實際價差）平倉。
    每筆扣進場當下點差。回傳 R 序列（含成本）。"""
    C1 = M1.close.values
    R, when = [], []
    idx = df.index.get_indexer(E.index)
    for (t, row), i in zip(E.iterrows(), idx):
        d, a = row.dir, row.atr
        hi, lo = df.high.iat[i], df.low.iat[i]
        entry = hi if d == 1 else lo
        stop = lo - buf * a if d == 1 else hi + buf * a
        risk = abs(entry - stop)
        s = np.searchsorted(T1, (t + pd.Timedelta(minutes=TF_MIN[tf])).to_datetime64())
        e = np.searchsorted(T1, (t + pd.Timedelta(minutes=TF_MIN[tf] * (wait + 1))).to_datetime64())
        h, l = H1A[s:e], L1A[s:e]
        trig = np.argmax(h >= entry) if d == 1 and (h >= entry).any() else (
            np.argmax(l <= entry) if d == -1 and (l <= entry).any() else None)
        if trig is None:
            continue
        cancel = (l[:trig] <= stop).any() if d == 1 else (h[:trig] >= stop).any()
        if cancel:
            continue
        f = s + trig
        tgt = entry + d * tp * risk
        up, dn = (tgt, stop) if d == 1 else (stop, tgt)
        r = first_passage(f, up, dn, maxlen)
        if r == 0:
            px = C1[min(f + maxlen, len(C1)) - 1]
            res = d * (px - entry) / risk
        else:
            res = tp if r * d == 1 else -1.0
        R.append(res - SP1[f] / risk)
        when.append(t)
    return pd.Series(R, index=pd.DatetimeIndex(when))


def summ(r):
    r = r[~np.isnan(r)]
    r = r[r != 0]
    n = len(r)
    if n == 0:
        return "n=0", np.nan, 0
    p = (r > 0).mean()
    se = np.sqrt(0.25 / n)
    return f"{p * 100:5.1f}%  (n={n:5d}, z={(p - .5) / se:+5.1f})", p, n


def fmt_years(E, r):
    s = pd.Series(r, index=E.index)
    parts = []
    for y, g in s.groupby(s.index.year):
        g = g.dropna(); g = g[g != 0]
        if len(g):
            parts.append(f"{y}:{(g > 0).mean() * 100:4.0f}%({len(g)})")
    return " ".join(parts)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    print("資料", M1.index[0], "~", M1.index[-1], f"{len(M1):,} 根 M1")
    hindsight = {}
    for tf, rule in [("M5", "5min"), ("M15", "15min"), ("H1", "60min")]:
        df = resample(rule)
        print(f"\n{'=' * 100}\n### {tf}（{len(df):,} 根）")
        for K in (3.0, 4.0, 6.0):
            E = events(df, K=K)
            S, C = E[E.small], E[~E.small]
            print(f"\n-- 大漲/大跌背景 K={K} ATR：小 K 線 {len(S):,} 次，一般 K 線（對照）{len(C):,} 次")
            print("  (A) 小K之後是否續行：從收盤價先碰順勢 +X ATR 的機率（隨機 = 50%）")
            for X in (0.5, 1.0, 2.0):
                rs, rc = continuation(S, tf, X), continuation(C, tf, X)
                print(f"     X={X}: 小K {summ(rs)[0]}   一般K {summ(rc)[0]}")
                if K == 4.0 and X == 1.0:
                    print(f"           小K 逐年：{fmt_years(S, rs)}")
                    print(f"           一般K逐年：{fmt_years(C, rc)}")
            print("  (B) 回撤到小K中點時，先反彈 +X 還是先穿越 −X（守住的機率，隨機 = 50%）")
            for X in (0.5, 1.0):
                rs = level_test(S, tf, X)
                rc = level_test(C, tf, X)
                ru = level_test(S, tf, X, level_shift=+0.5)
                rd = level_test(S, tf, X, level_shift=-0.5)
                print(f"     X={X}: 小K中點 {summ(rs)[0]}  一般K中點 {summ(rc)[0]}")
                print(f"            假價位 +0.5ATR {summ(ru)[0]}  假價位 -0.5ATR {summ(rd)[0]}")
                if K == 4.0 and X == 0.5:
                    print(f"           小K 逐年：{fmt_years(S, rs)}")
                    # 扣成本後：在 L 掛限價、停損/停利各 X ATR 的期望值（以 ATR 計）
                    sp = (S.spread / S.atr).values
                    m = ~np.isnan(rs) & (rs != 0)
                    ev = np.mean(rs[m] * X - sp[m])
                    print(f"           掛單在 L、停損停利各 {X} ATR：扣點差後每筆 {ev:+.3f} ATR（點差中位 {np.median(sp[m]):.3f} ATR）")
            if K == 4.0:
                # 事後偏誤示範：只看「事後看起來真的大漲一段」的小 K（之後 20 根內再走 >= 4 ATR）
                fut = []
                idx = df.index.get_indexer(S.index)
                for i, d, a, c in zip(idx, S.dir.values, S.atr.values, S.close.values):
                    seg = df.iloc[i + 1:i + 21]
                    fut.append(((seg.high.max() - c) if d == 1 else (c - seg.low.min())) / a if len(seg) else np.nan)
                fut = np.array(fut)
                rs = continuation(S, tf, 1.0)
                hs = fut >= 4
                hindsight[tf] = (summ(rs[hs])[0], summ(rs[~hs])[0], hs.mean())
    print(f"\n{'=' * 100}\n### 策略化：小 K 之後 3 根內突破小 K 高/低點進場，停損小 K 另一端外 0.1 ATR，停利 2R（扣點差）")
    for tf, rule in [("M5", "5min"), ("M15", "15min"), ("H1", "60min")]:
        df = resample(rule)
        for K in (3.0, 4.0):
            E = events(df, K=K)
            for name, sub in [("小K", E[E.small]), ("一般K", E[~E.small])]:
                R = breakout_trades(sub, df, tf)
                dirs = sub.dir.reindex(R.index).values
                gp, gl = R[R > 0].sum(), -R[R < 0].sum()
                yr = " ".join(f"{y}:{g.mean():+.2f}" for y, g in R.groupby(R.index.year))
                print(f"  {tf} K={K} {name:4s}: n={len(R):5d} 期望 {R.mean():+.3f}R t={R.mean() / R.std() * np.sqrt(len(R)):+.2f} "
                      f"勝率 {(R > 0).mean():.2f} PF {gp / gl:.2f} | 多 {R[dirs == 1].mean():+.2f} 空 {R[dirs == -1].mean():+.2f} | 逐年 {yr}")

    print(f"\n{'=' * 100}\n### 事後偏誤示範（K=4，X=1）：只挑「之後真的又大漲一段」的圖來看")
    for tf, (a, b, f) in hindsight.items():
        print(f"  {tf}: 事後挑選的（佔 {f * 100:.0f}%）續行 {a}   其餘 {b}")
