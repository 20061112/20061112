"""第二十四部分 (4)：順勢回檔版的穩健性與筆數 → results_part24_robust.txt

1. 主要設定在不同長 ER 門檻 a 下的：筆數、每天筆數、平均 R、t 值、PF、最大回撤（R）、月份勝率
2. 對照「在趨勢中但沒回檔」：d·ER長 ≥ a 且 d·ER短 > 0.3（還在衝），同樣順勢進場、同樣停損
   → 回檔條件到底有沒有加分，還是只是「今年順勢就賺」
3. 增加筆數的方法
   a. 冷卻期縮短（同一段趨勢允許多次回檔進場）：M 根 → S 根
   b. 多組合併（M15 ER40/20/5 + M30 ER20/10/3 + M15 ER20/10/3 a≥0.4）：每天筆數、日損益 t 值、組合間相關
   c. 大週期判斷 + 小週期進場：M30 已收完 K 棒的 ER20 ≥ a 決定方向，M5 的 ER3 ≤ 0 當回檔進場，停損用 M5 前 10 根極值
"""
import numpy as np
import pandas as pd
from part23_ma_cycle import load_m1, resample, st, SPLIT
from part24_er_cascade import ser, atr
from part24_continuation import sim

m1 = load_m1()
NDAYS = len(np.unique(m1.index.date))
out = []
P = out.append
P(f"交易日數：{NDAYS}（2026/1/2 ~ 10/8，4/7~4/14 缺資料）")

cache = {}


def bars(tf):
    if tf not in cache:
        df = resample(m1, tf)
        o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        cache[tf] = dict(df=df, o=o, h=h, l=l, c=c, spr=df.spread.to_numpy(float) * 0.01, A=atr(h, l, c), E={})
    return cache[tf]


def er(B, n):
    if n not in B["E"]:
        B["E"][n] = ser(B["c"], n)
    return B["E"][n]


def gen(tf, L, M, S, a, cc=0.0, mode="pull", cool=None):
    """回傳 (t, side)。mode=pull：長 ER ≥ a、中 ER 同向且減速、短 ER ≤ cc；mode=run：短 ER > 0.3（沒回檔）。"""
    B = bars(tf)
    eL, eM, eS = er(B, L), er(B, M), er(B, S)
    d = np.sign(eL)
    eMp = np.r_[np.full(S, np.nan), eM[:-S]]
    with np.errstate(invalid="ignore"):
        cond = d * eL >= a
        if mode == "pull":
            cond &= (d * eM >= 0) & (d * eM < d * eMp) & (d * eS <= cc)
        else:
            cond &= (d * eM >= 0) & (d * eS > 0.3)
    cool = M if cool is None else cool
    out_t, last = [], -10 ** 9
    for t in np.flatnonzero(cond):
        if t - last > cool:
            out_t.append(t)
        last = t
    t = np.array(out_t, int)
    t = t[(t > 30) & (t < len(eL) - 2)]
    return t, d[t].astype(int)


def run(tf, L, M, S, t, side, tp=2):
    B = bars(tf)
    R = sim(B["o"], B["h"], B["l"], B["c"], B["spr"], B["A"], t, side, M, 2 * L, tp)
    return pd.DataFrame(dict(time=B["df"].index[t], side=side, R=R)).dropna()


def summ(T):
    v = T.R.to_numpy()
    n, mu, t, win, pf = st(v)
    eq = np.cumsum(v)
    dd = (eq - np.maximum.accumulate(np.r_[0, eq])[1:]).min() if len(v) else np.nan
    m = T.groupby(T.time.dt.to_period("M")).R.sum()
    h = T.time < SPLIT
    return dict(n=n, 每天=round(n / NDAYS, 2), avgR=round(float(mu), 3), t=round(float(t), 2), win=round(float(win), 2),
                pf=round(float(pf), 2), 總R=round(float(v.sum()), 1), 最大回撤R=round(float(dd), 1), 月份正=f"{(m > 0).sum()}/{len(m)}",
                前半=round(float(T.R[h].mean()), 3), 後半=round(float(T.R[~h].mean()), 3))


MAIN = [("30min", 20, 10, 3), ("15min", 40, 20, 5), ("15min", 20, 10, 3), ("5min", 20, 10, 3)]

P("\n=== 1 & 2. 回檔進場 vs 趨勢中沒回檔就進場（同方向、同停損、tp2）===")
rows = []
for tf, L, M, S in MAIN:
    for a in (0.3, 0.4, 0.5):
        for mode in ("pull", "run"):
            t, side = gen(tf, L, M, S, a, mode=mode)
            rows.append(dict(設定=f"{tf} {L}/{M}/{S}", a=a, 進場=("回檔" if mode == "pull" else "沒回檔"),
                             **summ(run(tf, L, M, S, t, side))))
P(pd.DataFrame(rows).to_string(index=False))

P("\n=== 3a. 冷卻期縮短（同段趨勢可多次進場）a=0.3 ===")
rows = []
for tf, L, M, S in MAIN[:3]:
    for cool in (M, S, 1):
        t, side = gen(tf, L, M, S, 0.3, cool=cool)
        rows.append(dict(設定=f"{tf} {L}/{M}/{S}", 冷卻=cool, **summ(run(tf, L, M, S, t, side))))
P(pd.DataFrame(rows).to_string(index=False))

P("\n=== 3b. 多組合併（每組各自獨立下單，每筆 1R）===")
legs = {"M30 20/10/3 a.3": ("30min", 20, 10, 3, 0.3), "M15 40/20/5 a.3": ("15min", 40, 20, 5, 0.3),
        "M15 20/10/3 a.4": ("15min", 20, 10, 3, 0.4), "M30 40/20/5 a.3": ("30min", 40, 20, 5, 0.3),
        "M30 14/7/2 a.4": ("30min", 14, 7, 2, 0.4)}
T_all, daily, rows_b = [], {}, []
for name, (tf, L, M, S, a) in legs.items():
    t, side = gen(tf, L, M, S, a)
    T = run(tf, L, M, S, t, side).assign(leg=name)
    T_all.append(T)
    daily[name] = T.groupby(T.time.dt.date).R.sum()
    rows_b.append(dict(組合=name, **summ(T)))
T = pd.concat(T_all).sort_values("time")
rows_b.append(dict(組合="合併", **summ(T)))
P(pd.DataFrame(rows_b).to_string(index=False))
D = pd.DataFrame(daily).fillna(0)
P("日損益相關係數")
P(D.corr().round(2).to_string())
dr = T.groupby(T.time.dt.date).R.sum()
dr = dr.reindex(sorted(set(m1.index.date)), fill_value=0)
P(f"合併日損益：平均 {dr.mean():+.3f}R/天、日 t 值 {dr.mean() / dr.std() * np.sqrt(len(dr)):.2f}、"
  f"虧損日 {(dr < 0).mean():.0%}、最差一天 {dr.min():.1f}R")

P("\n=== 3c. M30 判斷趨勢 + M5 回檔進場 ===")
B30, B5 = bars("30min"), bars("5min")
rows = []
for L30 in (20, 40):
    e30 = pd.Series(er(B30, L30), index=B30["df"].index)
    # 只用已收完的 M30：M30 K 棒標籤是開始時間，收完 = 標籤 + 30 分 → 對 M5 往後移一根 M30
    e30_done = e30.shift(1).reindex(B5["df"].index, method="ffill").to_numpy()
    for a in (0.3, 0.4, 0.5):
        for S5, M5 in ((3, 10), (5, 20)):
            eS = er(B5, S5)
            d = np.sign(e30_done)
            with np.errstate(invalid="ignore"):
                cond = (d * e30_done >= a) & (d * eS <= 0)
            tt, last = [], -10 ** 9
            for t in np.flatnonzero(cond):
                if t - last > M5:
                    tt.append(t)
                last = t
            tt = np.array(tt, int)
            tt = tt[(tt > 30) & (tt < len(eS) - 2)]
            side = d[tt].astype(int)
            for tp in (2, 3):
                R = sim(B5["o"], B5["h"], B5["l"], B5["c"], B5["spr"], B5["A"], tt, side, M5, 4 * M5, tp)
                Tm = pd.DataFrame(dict(time=B5["df"].index[tt], side=side, R=R)).dropna()
                rows.append(dict(M30_ER=L30, a=a, M5短ER=S5, 停損回看=M5, tp=tp, **summ(Tm)))
P(pd.DataFrame(rows).to_string(index=False))

open("results_part24_robust.txt", "w").write("\n".join(out) + "\n")
print("\n".join(out))
