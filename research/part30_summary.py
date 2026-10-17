"""第三十部分：總結 —— 純定投 vs 原策略 vs 改善後各版本（同一套回測一次跑完，不把各項改善相加）。

改善後版本（逐步疊加）：
  1. 原策略：QQQ ↔ TQQQ（合成），每週進場、每天出場
  2. 期貨 3 倍（全期貨，每週重設）
  3. + 盤中停損：持槓桿時 QQQ 盤中跌破前收 −4%，額外 2 倍出場
  4. + covered call：持 1 倍時、VXN 高於實現波動才賣 0.2 delta、約 45 天 call（偏斜 4 點保守假設；VXN 2001/2 起才有）
  5. + 每天都可進場（取捨：報酬多、下檔多一點）
1999/3 起每週 1,000（布林加碼時 2,000）；10 年滾動 2000~2016 起 17 段。
"""
import math
from statistics import NormalDist
import numpy as np
import pandas as pd
from part22_core import load, signals, run, metrics, year_starts

out = open("results_part30.txt", "w")


def P(s=""):
    print(s); out.write(s + "\n"); out.flush()


DF = load()
S = signals(DF)
YS = year_starts(S)
WIN = list(range(2000, 2017))
I10 = S.index.searchsorted(pd.Timestamp("2010-02-11"))
vxn = pd.read_csv("data/VXNCLS.csv", parse_dates=["observation_date"]).set_index("observation_date").VXNCLS
VXN = vxn.reindex(S.index.union(vxn.index)).ffill(limit=3).reindex(S.index).values
VRP = VXN - (S.atrp * np.sqrt(252) * 100 / 1.25).values
Nc = NormalDist().cdf


def regime(daily=False, stop=None):
    """回傳 (reg, fr)：fr 為期貨部位日報酬（含盤中停損的出場價）。"""
    pc = DF.close.shift(1).values
    o_r, l_r = DF.open.values / pc - 1, DF.low.values / pc - 1
    up, z, ws, R = S.up.values, S.z.values, S.ws.values, S.ret.values
    reg, fr, on = np.zeros(len(S), bool), R.copy(), False
    for i in range(len(S)):
        ok = up[i] and z[i] == z[i]
        if on and stop is not None and l_r[i] <= -stop:
            fr[i] = (R[i] + 2 * (min(o_r[i], -stop) - 0.001)) / 3; on = False
        elif on and (not ok or z[i] >= 0.5):
            on = False
        elif not on and ok and z[i] < 0.5 and (ws[i] or daily):
            on = True
        reg[i] = on
    return reg, fr


def bs_call(Sp, K, T, sig, r):
    if T <= 1e-9:
        return max(Sp - K, 0.0)
    d1 = (math.log(Sp / K) + (r + sig * sig / 2) * T) / (sig * math.sqrt(T))
    return Sp * Nc(d1) - K * math.exp(-r * T) * Nc(d1 - sig * math.sqrt(T))


def add_cc(reg, fr, delta=0.2, days=31, skew=4.0, thr=0.0, cost=0.05):
    """持 1 倍時賣 call（逐日 BS 評價），現金以每 1 倍名目計、除以當天實際槓桿。"""
    fr = fr.copy()
    Pc, rf = S.close.values, S.rf_ann.values
    q = -NormalDist().inv_cdf(delta)
    pos = None
    lev = lambda i: 3.0 if reg[i - 1] else 1.0
    for i in range(1, len(S)):
        if pos is not None:
            K, ei, prev = pos
            if i >= ei:
                fr[i] -= (max(Pc[i] - K, 0.0) - prev) / Pc[i - 1] / lev(i); pos = None
            else:
                iv = max((VXN[i] if VXN[i] == VXN[i] else VXN[i - 1]) - skew, 5.0) / 100
                val = bs_call(Pc[i], K, (ei - i) / 252, iv, rf[i])
                fr[i] -= (val - prev) / Pc[i - 1] / lev(i)
                if reg[i]:
                    fr[i] -= cost * val / Pc[i - 1] / lev(i); pos = None
                else:
                    pos = (K, ei, val)
        if pos is None and not reg[i] and VXN[i] == VXN[i] and VRP[i] >= thr:
            iv, T = max(VXN[i] - skew, 5.0) / 100, days / 252
            K = Pc[i] * math.exp(q * iv * math.sqrt(T) + (rf[i] + iv * iv / 2) * T)
            c = bs_call(Pc[i], K, T, iv, rf[i])
            fr[i] -= cost * c / Pc[i - 1] / lev(i)
            pos = (K, i + days, c)
    return fr


reg_w, fr_w = regime()
reg_ws, fr_ws = regime(stop=0.04)
reg_ds, fr_ds = regime(daily=True, stop=0.04)
VERS = [
    ("純週定投 QQQ", dict(eng="dca")),
    ("原策略 QQQ↔TQQQ", dict(eng="etf")),
    ("期貨 3 倍", dict(eng="allfut", L=3.0)),
    ("+ 盤中停損 −4%", dict(eng="allfut", L=3.0, regime_arr=reg_ws, fut_ret=fr_ws)),
    ("+ covered call", dict(eng="allfut", L=3.0, regime_arr=reg_ws, fut_ret=add_cc(reg_ws, fr_ws))),
    ("+ 每天可進場", dict(eng="allfut", L=3.0, regime_arr=reg_ds, fut_ret=add_cc(reg_ds, fr_ds))),
]
rows, yrs = [], {}
for lab, kw in VERS:
    o = run(S, **kw); m = metrics(o)
    m10 = metrics(run(S, i0=I10, **kw))
    w = np.array([metrics(run(S, i0=YS[y], i1=YS[y + 10], **kw))["tot"] for y in WIN])
    tw = pd.Series(o["twr"], index=S.index)
    yr = (1 + tw).groupby(S.index.year).prod() - 1
    yrs[lab] = yr
    rows.append(dict(lab=lab, irr=m["irr"], irr10=m10["irr"], mult=o["V"].iloc[-1] / o["inv"], avg10=w.mean(), min10=w.min(),
                     worst=m["worst"], mdd=m["mdd"], worst_yr=yr.min(), wy=yr.idxmin(), lev=m["avg_lev"], sw=m["sw_per_yr"], w=w))
D = rows[0]
P("第三十部分：總結（1999/3 ~ 2026/10，每週 1,000 定投；10 年滾動 17 段）\n")
P(f"  {'版本':22s}{'IRR':>7s}{'2010起':>7s}{'期末/投入':>9s}{'10年均':>8s}{'10年最差':>8s}{'最慘/本金':>8s}{'最大回撤':>8s}{'最差年':>12s}{'均槓桿':>6s}{'換倉':>5s}{'勝定投段':>7s}")
for r in rows:
    P(f"  {r['lab']:22s}{r['irr'] * 100:6.1f}%{r['irr10'] * 100:6.1f}%{r['mult']:8.1f}倍{r['avg10'] * 100:7.0f}%{r['min10'] * 100:7.0f}%"
      f"{r['worst'] * 100:7.0f}%{r['mdd'] * 100:7.0f}%{r['worst_yr'] * 100:6.0f}%({r['wy']}){r['lev']:6.2f}{r['sw']:5.1f}"
      f"{(r['w'] > D['w']).sum():4d}/17")
P(f"\n  每投入 1 元、27.6 年後：純定投 {rows[0]['mult']:.1f} 元｜原策略 {rows[1]['mult']:.1f} 元｜改善後（第 5 版）{rows[4]['mult']:.1f} 元")
Y = pd.DataFrame(yrs)
P("\n  逐年報酬（時間加權）")
P("  年份" + "".join(f"{c:>16s}" for c in Y.columns))
for y, r in Y.iterrows():
    P(f"  {y}" + "".join(f"{v * 100:>15.0f}%" for v in r.values))
for c in Y.columns[1:]:
    d = Y[c] - Y[Y.columns[0]]
    P(f"  {c:22s}勝定投 {(d > 0).sum()}/{len(d)} 年｜落後最多 {d.min() * 100:+.0f}%（{d.idxmin()}）｜領先最多 {d.max() * 100:+.0f}%（{d.idxmax()}）")
out.close()
