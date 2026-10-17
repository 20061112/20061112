"""第二十七部分：(1) covered call 參數區域 + 逐日評價；(2) 每天進場 vs 每週進場的穩健性。

(1) QQQ 全期貨 L=3，2002/3 起：持 1 倍時賣 call，delta × 天數 × VRP 門檻 × 偏斜 共 120 組；
    空方 call 逐日用 BS（IV = VXN − 偏斜）評價，成交扣權利金 5%；開槓桿前買回。報告整片區域，不挑最佳。
(2) 每天進場：QQQ 1999 起前後兩半（1999~2012 / 2013~2026）、逐年相對 QQQ 最差落後；SPY / DIA / SOXX 2006 起（沒參與發現）。
"""
import itertools
import math
from statistics import NormalDist
import numpy as np
import pandas as pd
from part22_core import load, signals, run, metrics, year_starts

out = open("results_part27.txt", "w")


def P(s=""):
    print(s); out.write(s + "\n"); out.flush()


def cagr(x):
    return (1 + x).prod() ** (252 / len(x)) - 1


def daily_regime(S):
    up, z = S.up.values, S.z.values
    on, r = False, np.zeros(len(S), bool)
    for i in range(len(S)):
        ok = up[i] and z[i] == z[i]
        if on and (not ok or z[i] >= 0.5):
            on = False
        if not on and ok and z[i] < 0.5:
            on = True
        r[i] = on
    return r


# =============================================================== (1) covered call
DF = load()
S = signals(DF)
START, SPLIT = pd.Timestamp("2002-03-01"), pd.Timestamp("2014-01-01")
I0 = S.index.searchsorted(START)
YS = year_starts(S, 2003, 2016)
WIN = list(range(2003, 2017))
KW = dict(eng="allfut", L=3.0)
vxn = pd.read_csv("data/VXNCLS.csv", parse_dates=["observation_date"]).set_index("observation_date").VXNCLS
VXN = vxn.reindex(S.index.union(vxn.index)).ffill(limit=3).reindex(S.index)
RV = S.atrp * np.sqrt(252) * 100 / 1.25                 # ATR 換算的年化實現波動（點）
VRP = (VXN - RV).values
Nc = NormalDist().cdf


def bs_call(Sp, K, T, sig, r):
    if T <= 1e-9 or sig <= 0:
        return max(Sp - K, 0.0)
    d1 = (math.log(Sp / K) + (r + sig * sig / 2) * T) / (sig * math.sqrt(T))
    return Sp * Nc(d1) - K * math.exp(-r * T) * Nc(d1 - sig * math.sqrt(T))


def cc_returns(reg, delta, days, skew, thr, cost=0.05):
    """回傳期貨部位的日報酬（含空方 call 逐日評價）與賣出次數。現金與評價都以「每 1 倍名目」計，再除以當天實際槓桿。"""
    Pc, rf, vv = S.close.values, S.rf_ann.values, VXN.values
    fr = S.ret.values.copy()
    q = -NormalDist().inv_cdf(delta)
    pos, n = None, 0                                     # (K, 到期 index, 前一天評價)
    lev = lambda i: 3.0 if reg[i - 1] else 1.0
    for i in range(1, len(S)):
        if pos is not None:
            K, ei, prev = pos
            iv = max(vv[i] - skew, 5.0) / 100 if vv[i] == vv[i] else None
            if i >= ei:
                val = max(Pc[i] - K, 0.0)
                fr[i] -= (val - prev) / Pc[i - 1] / lev(i); pos = None
            else:
                val = bs_call(Pc[i], K, (ei - i) / 252, iv if iv else 0.2, rf[i])
                fr[i] -= (val - prev) / Pc[i - 1] / lev(i)
                if reg[i]:                               # 今天收盤開槓桿 → 買回（多付 5%）
                    fr[i] -= cost * val / Pc[i - 1] / lev(i); pos = None
                else:
                    pos = (K, ei, val)
        if pos is None and not reg[i] and i >= I0 and vv[i] == vv[i] and (thr is None or VRP[i] >= thr):
            iv, T = max(vv[i] - skew, 5.0) / 100, days / 252
            K = Pc[i] * math.exp(q * iv * math.sqrt(T) + (rf[i] + iv * iv / 2) * T)
            c = bs_call(Pc[i], K, T, iv, rf[i])
            fr[i] -= cost * c / Pc[i - 1] / lev(i)         # 賣出當下：只損失價差，負債以評價計
            pos, n = (K, i + days, c), n + 1
    return fr, n


def ev(**extra):
    kw = {**KW, **extra}
    o = run(S, i0=I0, **kw)
    m = metrics(o)
    tw = pd.Series(o["twr"], index=S.index[I0:])
    m["h1"], m["h2"] = cagr(tw[tw.index < SPLIT]), cagr(tw[tw.index >= SPLIT])
    m["w"] = np.array([metrics(run(S, i0=YS[y], i1=YS[y + 10], **kw))["tot"] for y in WIN])
    return m


B = ev()
P("第二十七部分 (1)：covered call 參數區域（QQQ 全期貨 L=3，2002/3 起；空方 call 逐日評價）")
P(f"  基準 IRR {B['irr'] * 100:.1f}%、最慘相對本金 {B['worst'] * 100:.0f}%；前半 2002~2013、後半 2014~2026\n")
rows = []
reg0 = S.on.values
for delta, days, thr, skew in itertools.product((0.10, 0.15, 0.20, 0.30), (21, 31, 42), (None, 0, 2, 4, 6), (2.0, 4.0)):
    fr, n = cc_returns(reg0, delta, days, skew, thr)
    m = ev(fut_ret=fr)
    rows.append(dict(delta=delta, days=days, thr="不限" if thr is None else thr, skew=skew, n=n, irr=m["irr"] - B["irr"],
                     h1=m["h1"] - B["h1"], h2=m["h2"] - B["h2"], win=int((m["w"] > B["w"]).sum()), worst=m["worst"]))
R = pd.DataFrame(rows)
R["both"] = (R.h1 > 0.002) & (R.h2 > 0.002)
R.to_csv("part27_cc_grid.csv", index=False)
P(f"  120 組：IRR 差中位 {R.irr.median() * 100:+.2f}%，兩半都 > +0.2% 的比例 {R.both.mean() * 100:.0f}%，最慘相對本金 {R.worst.min() * 100:.0f}% ~ {R.worst.max() * 100:.0f}%")
for col in ("delta", "days", "thr", "skew"):
    g = R.groupby(col, sort=False).agg(irr=("irr", "median"), h1=("h1", "median"), h2=("h2", "median"), both=("both", "mean"), win=("win", "median"), n=("n", "median"))
    P(f"\n  依 {col}（各組中位數）")
    for k, r in g.iterrows():
        P(f"    {str(k):>5s}：IRR {r.irr * 100:+.2f}%｜前半 {r.h1 * 100:+.2f}% 後半 {r.h2 * 100:+.2f}%｜兩半同向+ {r.both * 100:3.0f}%｜勝段中位 {r.win:.0f}/14｜賣出 {r.n:.0f} 次")
g = R.groupby(["thr", "skew"], sort=False).agg(irr=("irr", "median"), both=("both", "mean"))
P("\n  VRP 門檻 × 偏斜（IRR 差中位 / 兩半同向+ 比例）")
for (t, s_), r in g.iterrows():
    P(f"    門檻 {str(t):>3s}、偏斜 {s_:.0f}：{r.irr * 100:+.2f}% / {r.both * 100:3.0f}%")
R["score"] = R[["h1", "h2"]].min(axis=1)
P("\n  兩半中較差那半最好的 5 組（只供參考，挑最佳本身是過擬合）")
for _, r in R.sort_values("score", ascending=False).head(5).iterrows():
    P(f"    delta {r.delta:.2f}、{r.days} 天、門檻 {r.thr}、偏斜 {r['skew']:.0f}：IRR {r.irr * 100:+.2f}%｜{r.h1 * 100:+.2f}% / {r.h2 * 100:+.2f}%｜{r.win}/14｜{r.n} 次")
d = R[(R.delta == 0.20) & (R.days == 31)]
P("\n  原報告那組（0.2 delta、約 45 天 = 31 交易日）：")
for _, r in d.iterrows():
    P(f"    門檻 {str(r.thr):>3s}、偏斜 {r['skew']:.0f}：IRR {r.irr * 100:+.2f}%｜{r.h1 * 100:+.2f}% / {r.h2 * 100:+.2f}%｜{r.win}/14｜最慘 {r.worst * 100:.0f}%")

# =============================================================== (2) 每天進場
P("\n\n第二十七部分 (2)：每天進場 vs 每週進場")
YS99 = year_starts(S)
for eng in ("allfut", "etf"):
    kw = dict(eng=eng, L=3.0) if eng == "allfut" else dict(eng="etf")
    res = {}
    for lab, reg in (("每週", None), ("每天", daily_regime(S))):
        k2 = dict(kw, regime_arr=reg) if reg is not None else kw
        o = run(S, **k2); m = metrics(o)
        tw = pd.Series(o["twr"], index=S.index)
        yr = (1 + tw).groupby(S.index.year).prod() - 1
        qy = (1 + S.ret).groupby(S.index.year).prod() - 1
        w = np.array([metrics(run(S, i0=YS99[y], i1=YS99[y + 10], **k2))["tot"] for y in range(2000, 2017)])
        res[lab] = dict(m=m, h1=cagr(tw[tw.index < "2013-01-01"]), h2=cagr(tw[tw.index >= "2013-01-01"]),
                        lag=(yr - qy).min(), lagy=(yr - qy).idxmin(), w=w, mdd=m["mdd"])
    a, b = res["每週"], res["每天"]
    P(f"  QQQ 1999 起（{'全期貨 L=3' if eng == 'allfut' else 'ETF 切換'}）")
    for lab, r in res.items():
        P(f"    {lab}：IRR {r['m']['irr'] * 100:.1f}%｜前半 {r['h1'] * 100:.1f}% 後半 {r['h2'] * 100:.1f}%｜最慘相對本金 {r['m']['worst'] * 100:.0f}%｜"
          f"最大回撤 {r['mdd'] * 100:.0f}%｜相對 QQQ 最差落後 {r['lag'] * 100:.0f}%（{r['lagy']}）｜換倉 {r['m']['sw_per_yr']:.1f}/年")
    P(f"    每天 − 每週：前半 {(b['h1'] - a['h1']) * 100:+.1f}%、後半 {(b['h2'] - a['h2']) * 100:+.1f}%｜10 年段 {(b['w'] > a['w']).sum()}/17 勝")

P("\n  其他指數（2006/3 起，全期貨 L=3；SOXX 用 ETF 3 倍；這些指數沒參與「每天進場」的發現）")
for name, fut, ratio in (("SPY", "MES", 10.0), ("DIA", "MYM", 100.0), ("SOXX", None, None)):
    Sx = signals(load(f"data/{name.lower()}_us_d.csv"))
    i0 = Sx.index.searchsorted(pd.Timestamp("2006-03-01"))
    YSx = year_starts(Sx, 2007, 2016)
    kw = dict(eng="etf") if fut is None else dict(eng="allfut", L=3.0, fut_name=fut, ratio=ratio)
    res = {}
    for lab, reg in (("每週", None), ("每天", daily_regime(Sx))):
        k2 = dict(kw, regime_arr=reg) if reg is not None else kw
        o = run(Sx, i0=i0, **k2); m = metrics(o)
        tw = pd.Series(o["twr"], index=Sx.index[i0:])
        w = np.array([metrics(run(Sx, i0=YSx[y], i1=YSx[y + 10], **k2))["tot"] for y in range(2007, 2017)])
        res[lab] = dict(m=m, h1=cagr(tw[tw.index < "2016-01-01"]), h2=cagr(tw[tw.index >= "2016-01-01"]), w=w)
    a, b = res["每週"], res["每天"]
    P(f"    {name:5s} IRR {a['m']['irr'] * 100:.1f}% → {b['m']['irr'] * 100:.1f}%｜前半 {(b['h1'] - a['h1']) * 100:+.1f}% 後半 {(b['h2'] - a['h2']) * 100:+.1f}%｜"
      f"最慘 {a['m']['worst'] * 100:.0f}% → {b['m']['worst'] * 100:.0f}%｜10 年段 {(b['w'] > a['w']).sum()}/10 勝｜換倉 {a['m']['sw_per_yr']:.1f} → {b['m']['sw_per_yr']:.1f}")
out.close()
