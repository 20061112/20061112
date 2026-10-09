"""第二十五部分：ER 回檔順勢策略 — 績效指標（PF、P/MDD、Sharpe）、出場方式、延伸條件。

進場（同第二十四部分的順勢版）：d·ER長 ≥ a、d·ER中 ≥ 0 且比 S 根前小、d·ER短 ≤ 0 → 下一根開盤順 d 進場
初始停損：前 M 根反向極值外 0.1 ATR14（每筆 1R = 進場價到停損的距離）
主要設定：
  A  M15 ER40/20/5  a 0.3  冷卻 5
  B  M30 ER20/10/3  a 0.3  冷卻 3
  C  M15 ER20/10/3  a 0.4  冷卻 10
出場（全部共用同一個進場與初始停損，同一根同時碰停損/目標算停損，跳空以開盤價成交，每筆扣點差）
  tpX      : X R 停利，最長 2L 根
  time N   : 不設停利，N 根後收盤平倉
  chand k  : 吊燈停損，最高收盤（做多）− k × ATR14，只往有利方向移動；最長 4L 根
  dc N     : 停損移到最近 N 根最低點（做多），只往有利方向移動；最長 4L 根
  erM / erL: 中 ER（或長 ER）轉為反向（d·ER < 0）時收盤平倉；最長 4L 根
  be+tp3   : 到 1R 後停損移到進場價，tp 3R
  half     : 1R 先平一半，剩下用吊燈 3ATR
指標：PF；總 R；最大回撤（以出場時間排序的累積 R）；P/MDD；Sharpe = 每日 R 合計（含沒交易的日子）平均 / 標準差 × √252
"""
import numpy as np
import pandas as pd
from part23_ma_cycle import load_m1, resample, SPLIT
from part24_er_cascade import ser, atr

CFGS = {"A M15 40/20/5": ("15min", 40, 20, 5, 0.3, 5),
        "B M30 20/10/3": ("30min", 20, 10, 3, 0.3, 3),
        "C M15 20/10/3": ("15min", 20, 10, 3, 0.4, 10)}


def prep(m1, tf):
    df = resample(m1, tf)
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    return dict(df=df, idx=df.index, o=o, h=h, l=l, c=c, spr=df.spread.to_numpy(float) * 0.01,
                A=atr(h, l, c), E={})


def er(B, n):
    if n not in B["E"]:
        B["E"][n] = ser(B["c"], n)
    return B["E"][n]


def entries(B, L, M, S, a, cool, extra=None):
    eL, eM, eS = er(B, L), er(B, M), er(B, S)
    d = np.sign(eL)
    eMp = np.r_[np.full(S, np.nan), eM[:-S]]
    with np.errstate(invalid="ignore"):
        cond = (d * eL >= a) & (d * eM >= 0) & (d * eM < d * eMp) & (d * eS <= 0)
    if extra is not None:
        cond &= extra(d)
    tt, last = [], -10 ** 9
    for t in np.flatnonzero(cond):
        if t - last > cool:
            tt.append(t)
        last = t
    t = np.array(tt, int)
    t = t[(t > 60) & (t < len(eL) - 2)]
    return t, d[t].astype(int)


def trade(B, k, s, L, M, exit_, buf=0.1):
    """回傳 (R, 出場 bar index)。"""
    o, h, l, c, A = B["o"], B["h"], B["l"], B["c"], B["A"]
    n = len(c)
    e = k + 1
    ext = l[k - M + 1:k + 1].min() if s == 1 else h[k - M + 1:k + 1].max()
    stop = ext - s * buf * A[k]
    ep = o[e]
    R = s * (ep - stop)
    if not R > 0:
        return np.nan, e
    kind, par = exit_
    hold = {"tp": 2 * L, "time": par if kind == "time" else 0}.get(kind, 4 * L)
    end = min(e + hold, n)
    tgt = ep + s * par * R if kind == "tp" else (ep + s * 3 * R if kind == "be" else None)
    best = ep
    half_done = False
    realized = 0.0
    for j in range(e, end):
        # 1. 停損（跳空以開盤價）
        hit = l[j] <= stop if s == 1 else h[j] >= stop
        if hit:
            xp = stop if j == e else (min(stop, o[j]) if s == 1 else max(stop, o[j]))
            pnl = s * (xp - ep)
            return ((realized + (0.5 if half_done else 1.0) * pnl) - B["spr"][e]) / R, j
        # 2. 半倉 1R
        if kind == "half" and not half_done and ((h[j] >= ep + R) if s == 1 else (l[j] <= ep - R)):
            half_done, realized = True, 0.5 * R
        # 3. 停利
        if tgt is not None and ((h[j] >= tgt) if s == 1 else (l[j] <= tgt)):
            return (s * (tgt - ep) - B["spr"][e]) / R, j
        # 4. 收盤型出場
        if kind in ("erM", "erL"):
            v = er(B, M if kind == "erM" else L)[j]
            if s * v < 0:
                return (s * (c[j] - ep) - B["spr"][e]) / R, j
        # 5. 移動停損（收盤後更新，下一根生效）
        if kind == "be" and ((h[j] >= ep + R) if s == 1 else (l[j] <= ep - R)):
            stop = max(stop, ep) if s == 1 else min(stop, ep)
        if kind in ("chand", "half"):
            kk = par if kind == "chand" else 3
            best = max(best, c[j]) if s == 1 else min(best, c[j])
            ns = best - s * kk * A[j]
            stop = max(stop, ns) if s == 1 else min(stop, ns)
        if kind == "dc":
            ns = l[max(e, j - par + 1):j + 1].min() if s == 1 else h[max(e, j - par + 1):j + 1].max()
            if j - e + 1 >= par:
                stop = max(stop, ns) if s == 1 else min(stop, ns)
    j = end - 1
    pnl = s * (c[j] - ep)
    return ((realized + (0.5 if half_done else 1.0) * pnl) - B["spr"][e]) / R, j


def metrics(T, days):
    v = T.R.to_numpy()
    if len(v) < 5:
        return {}
    g, b = v[v > 0].sum(), -v[v < 0].sum()
    eq = np.cumsum(T.sort_values("xtime").R.to_numpy())
    mdd = -(eq - np.maximum.accumulate(np.r_[0, eq])[1:]).min()
    dr = T.groupby(T.xtime.dt.date).R.sum().reindex(days, fill_value=0)
    h = T.time < SPLIT
    r = dict(n=len(v), 每天=len(v) / len(days), avgR=v.mean(), win=(v > 0).mean(), PF=g / b, 總R=v.sum(), MDD=mdd,
             P_MDD=v.sum() / mdd, Sharpe=dr.mean() / dr.std() * np.sqrt(252), 前半=T.R[h].mean(),
             後半=T.R[~h].mean(), 平均持有=T.bars.mean())
    return {k: (x if k == "n" else round(float(x), 3 if k in ("avgR", "前半", "後半") else 2)) for k, x in r.items()}


def run(B, t, side, L, M, exit_):
    rows = []
    for k, s in zip(t, side):
        R, j = trade(B, k, s, L, M, exit_)
        if not np.isnan(R):
            rows.append(dict(time=B["idx"][k], xtime=B["idx"][j], side=s, R=R, bars=j - k))
    return pd.DataFrame(rows)


EXITS = [("tp", 1), ("tp", 1.5), ("tp", 2), ("tp", 3), ("tp", 4), ("time", None), ("chand", 2), ("chand", 3),
         ("chand", 4), ("dc", None), ("erM", 0), ("erL", 0), ("be", 0), ("half", 0)]


def exits_for(L, M):
    out = []
    for kind, p in EXITS:
        if kind == "time":
            out += [("time", L // 2), ("time", L), ("time", 2 * L)]
        elif kind == "dc":
            out += [("dc", M // 2), ("dc", M)]
        else:
            out.append((kind, p))
    return out


def main():
    m1 = load_m1()
    days = sorted(set(m1.index.date))
    cache = {}
    out = []
    P = out.append
    P(f"交易日 {len(days)}；R = 進場到初始停損距離；Sharpe 以每日 R 合計（含 0 交易日）年化 √252")
    best = {}
    for name, (tf, L, M, S, a, cool) in CFGS.items():
        B = cache.setdefault(tf, prep(m1, tf))
        t, side = entries(B, L, M, S, a, cool)
        rows = []
        for ex in exits_for(L, M):
            T = run(B, t, side, L, M, ex)
            rows.append(dict(出場=f"{ex[0]} {ex[1]}", **metrics(T, days)))
            if ex in (("tp", 2), ("chand", 3)):
                best[(name, ex)] = T
        P(f"\n=== {name}  a={a} 冷卻={cool}：出場比較 ===")
        P(pd.DataFrame(rows).to_string(index=False))

    # 合併（tp2 與 chand3 各一版）
    for ex in (("tp", 2), ("chand", 3)):
        T = pd.concat([best[(n, ex)].assign(leg=n) for n in CFGS])
        P(f"\n=== A+B+C 合併（{ex[0]} {ex[1]}）===")
        P(str(metrics(T, days)))
        mo = T.groupby(T.xtime.dt.to_period("M")).R.sum().round(1)
        P("逐月 R：" + "  ".join(f"{k.month}月 {v:+.1f}" for k, v in mo.items()))

    pd.concat([best[(n, ("chand", 3))].assign(leg=n) for n in CFGS]).to_csv("part25_trades_chand3.csv", index=False)
    open("results_part25_exits.txt", "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
