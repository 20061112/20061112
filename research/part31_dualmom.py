"""第三十一部分：防大回撤 —— 雙動能（跌破趨勢且 12 個月報酬 < T-bill → 1 倍部位改持短債）。

claude/festive-lovelace-1975n0 已發現：雙動能是唯一不犧牲歷史報酬就降低回撤的做法（QQQ），但區塊蒙地卡羅驗不了它。
這裡補三種驗證：
  A. 疊在改善後版本上（期貨 3 倍 + 盤中停損 −4% + covered call）
  B. 搬到 SPY / DIA / SOXX（沒參與發現）
  C. 參數區域：MA 150/200/250 × 動能回看 126/189/252/315 天，每個指數 12 組
另報：逐次空頭的一次投入報酬、躲在短債時錯過的反彈。
三種狀態：3 倍（原規則）、1 倍、短債（0 倍，期貨帳戶的錢只賺 T-bill）；狀態每天依前一天收盤判斷。
"""
import itertools
import math
from statistics import NormalDist
import numpy as np
import pandas as pd
from part22_core import load, signals, run, metrics, year_starts

out = open("results_part31.txt", "w")


def P(s=""):
    print(s); out.write(s + "\n"); out.flush()


def cagr(x):
    return (1 + x).prod() ** (252 / len(x)) - 1


Nc = NormalDist().cdf


def bs_call(Sp, K, T, sig, r):
    if T <= 1e-9:
        return max(Sp - K, 0.0)
    d1 = (math.log(Sp / K) + (r + sig * sig / 2) * T) / (sig * math.sqrt(T))
    return Sp * Nc(d1) - K * math.exp(-r * T) * Nc(d1 - sig * math.sqrt(T))


class Book:
    def __init__(self, name, fut, ratio, start=None):
        self.name = name
        self.DF = load(f"data/{name.lower()}_us_d.csv")
        self.S = signals(self.DF)
        self.i0 = 0 if start is None else self.S.index.searchsorted(start)
        y0 = self.S.index[self.i0].year + 1
        self.win = [y for y in range(y0, 2017)]
        self.YS = year_starts(self.S, y0, 2016)
        self.kw = dict(eng="allfut", rebal="daily", fut_name=fut, ratio=ratio)
        self.split = pd.Timestamp("2013-01-01") if start is None else pd.Timestamp("2016-01-01")

    def states(self, stop=0.04, dm=None):
        """回傳 state（3 / 1 / 0）與期貨日報酬 fr。dm = (MA 天數, 動能回看天數) 或 None。"""
        S, DF = self.S, self.DF
        c = S.close
        pc = DF.close.shift(1).values
        o_r, l_r = DF.open.values / pc - 1, DF.low.values / pc - 1
        up, z, ws, R = S.up.values, S.z.values, S.ws.values, S.ret.values
        if dm is not None:
            ma, lb = dm
            below = (c < c.rolling(ma).mean()).shift(1).fillna(False).values
            mom = (c / c.shift(lb) - 1).shift(1).values
            hurdle = (S.rf_ann * lb / 252).values
            cash = below & (mom < hurdle)
        st, fr, lev = np.ones(len(S), int), R.copy(), False
        for i in range(len(S)):
            ok = up[i] and z[i] == z[i]
            if lev and stop is not None and l_r[i] <= -stop:
                fr[i] = (R[i] + 2 * (min(o_r[i], -stop) - 0.001)) / 3; lev = False
            elif lev and (not ok or z[i] >= 0.5):
                lev = False
            elif not lev and ok and z[i] < 0.5 and ws[i]:
                lev = True
            st[i] = 3 if lev else (0 if dm is not None and cash[i] else 1)
        return st, fr

    def add_cc(self, st, fr, vxn, delta=0.2, days=31, skew=4.0, cost=0.05):
        S = self.S
        fr = fr.copy()
        Pc, rf = S.close.values, S.rf_ann.values
        VX = vxn.reindex(S.index.union(vxn.index)).ffill(limit=3).reindex(S.index).values
        VRP = VX - (S.atrp * np.sqrt(252) * 100 / 1.25).values
        q = -NormalDist().inv_cdf(delta)
        pos = None
        for i in range(1, len(S)):
            lv = float(st[i - 1]) or 1.0
            if pos is not None:
                K, ei, prev = pos
                if i >= ei:
                    fr[i] -= (max(Pc[i] - K, 0.0) - prev) / Pc[i - 1] / lv; pos = None
                else:
                    iv = max((VX[i] if VX[i] == VX[i] else VX[i - 1]) - skew, 5.0) / 100
                    val = bs_call(Pc[i], K, (ei - i) / 252, iv, rf[i])
                    fr[i] -= (val - prev) / Pc[i - 1] / lv
                    if st[i] != 1:
                        fr[i] -= cost * val / Pc[i - 1] / lv; pos = None
                    else:
                        pos = (K, ei, val)
            if pos is None and st[i] == 1 and VX[i] == VX[i] and VRP[i] >= 0:
                iv, T = max(VX[i] - skew, 5.0) / 100, days / 252
                K = Pc[i] * math.exp(q * iv * math.sqrt(T) + (rf[i] + iv * iv / 2) * T)
                c = bs_call(Pc[i], K, T, iv, rf[i])
                fr[i] -= cost * c / Pc[i - 1] / lv
                pos = (K, i + days, c)
        # 短債狀態時名目為 0，期貨報酬不影響淨值；選擇權現金只在 1 倍狀態發生
        return fr

    def ev(self, st, fr):
        S = self.S
        # 全期貨引擎在「關」時固定 1 倍，所以一律標成「開」，由 L_arr（3 / 1 / 0）決定曝險
        kw = dict(self.kw, L=3.0, regime_arr=np.ones(len(S), bool), L_arr=st.astype(float), fut_ret=fr)
        o = run(S, i0=self.i0, **kw)
        m = metrics(o)
        tw = pd.Series(o["twr"], index=S.index[self.i0:])
        m["h1"], m["h2"] = cagr(tw[tw.index < self.split]), cagr(tw[tw.index >= self.split])
        m["worst_yr"] = ((1 + tw).groupby(tw.index.year).prod() - 1).min()
        V = o["V"]
        m["acct_dd"] = (V / V.cummax() - 1).min()
        m["cash"] = (st[self.i0:] == 0).mean()
        m["sw"] = (np.diff(st[self.i0:]) != 0).sum() / ((S.index[-1] - S.index[self.i0]).days / 365.25)
        m["w"] = np.array([metrics(run(S, i0=self.YS[y], i1=self.YS[y + 10], **kw))["worst"] for y in self.win])
        m["wt"] = np.array([metrics(run(S, i0=self.YS[y], i1=self.YS[y + 10], **kw))["tot"] for y in self.win])
        m["tw"] = tw
        return m


def row(lab, m, b=None):
    s = (f"  {lab:30s}{m['irr'] * 100:6.1f}%{m['worst'] * 100:6.0f}%{m['mdd'] * 100:6.0f}%{m['acct_dd'] * 100:7.0f}%{m['worst_yr'] * 100:6.0f}%"
         f"{np.median(m['w']) * 100:7.0f}%{m['w'].min() * 100:6.0f}%{m['cash'] * 100:5.0f}%{m['sw']:5.1f}")
    if b is not None:
        s += (f"{(m['h1'] - b['h1']) * 100:+7.1f}%{(m['h2'] - b['h2']) * 100:+7.1f}%"
              f"{(m['wt'] > b['wt']).sum():3d}/{len(m['wt'])}")
    P(s)


HDR = (f"  {'版本':30s}{'IRR':>7s}{'最慘':>6s}{'回撤':>6s}{'帳面回落':>7s}{'最差年':>6s}{'10年最慘中位':>7s}{'最差':>6s}{'短債':>5s}{'換倉':>5s}"
       f"{'前半差':>8s}{'後半差':>7s}{'勝段':>5s}")
P("第三十一部分：防大回撤 —— 雙動能（MA 之下且動能 < T-bill → 1 倍部位改持短債）\n")
P("  最慘 = 最慘相對本金；回撤 = 時間加權最大回撤；帳面回落 = 帳戶市值從高點最多跌多少（含持續投入）")
P("  10 年最慘中位 / 最差 = 各 10 年定投段的「最慘相對本金」；短債 = 持短債的時間比例\n")

vxn = pd.read_csv("data/VXNCLS.csv", parse_dates=["observation_date"]).set_index("observation_date").VXNCLS

# ---------------------------------------------------------------- A
Q = Book("QQQ", "MNQ", 41.0)
P("A. QQQ 1999 起（全期貨、每天重設）")
P(HDR)
st0, fr0 = Q.states(stop=None)
b0 = Q.ev(st0, fr0)
row("期貨 3 倍（無停損）", b0)
st1, fr1 = Q.states()
fr1c = Q.add_cc(st1, fr1, vxn)
b1 = Q.ev(st1, fr1c)
row("改善版（停損 + covered call）", b1, b0)
st2, fr2 = Q.states(dm=(200, 252))
m2 = Q.ev(st2, Q.add_cc(st2, fr2, vxn))
row("改善版 + 雙動能 MA200/12 月", m2, b1)

P("\n  逐次空頭（QQQ 跌 > 20% 的段落，一次投入的報酬）")
c = Q.S.close
dd = c / c.cummax() - 1
eps, in_dd, start = [], False, None
for d, v in dd.items():
    if not in_dd and v < 0:
        in_dd, start = True, d
    if in_dd and v == 0:
        seg = dd[start:d]
        if seg.min() < -0.2:
            eps.append((c[:start].idxmax() if start != c.index[0] else start, seg.idxmin()))
        in_dd = False
seg = dd[start:]
if in_dd and seg.min() < -0.2:
    eps.append((start, seg.idxmin()))
for a, b in eps:
    a = c[:a].index[-1]
    f = lambda tw: (1 + tw[(tw.index > a) & (tw.index <= b)]).prod() - 1
    P(f"    {a.date()} ~ {b.date()}：QQQ {c[b] / c[a] - 1:+.0%}｜改善版 {f(b1['tw']):+.0%}｜+雙動能 {f(m2['tw']):+.0%}")
yr1 = (1 + b1["tw"]).groupby(b1["tw"].index.year).prod() - 1
yr2 = (1 + m2["tw"]).groupby(m2["tw"].index.year).prod() - 1
diff = (yr2 - yr1)
P("  雙動能 − 改善版，各年差異（|差| ≥ 3%）：" + "、".join(f"{y} {d * 100:+.0f}%" for y, d in diff.items() if abs(d) >= 0.03))

# ---------------------------------------------------------------- B + C
P("\nB/C. 四個指數（QQQ 1999 起；SPY / DIA / SOXX 2006/3 起，SOXX 假設有期貨），停損 −4%、無 covered call")
for name, fut, ratio, start in (("QQQ", "MNQ", 41.0, None), ("SPY", "MES", 10.0, pd.Timestamp("2006-03-01")),
                                ("DIA", "MYM", 100.0, pd.Timestamp("2006-03-01")), ("SOXX", "MNQ", 41.0, pd.Timestamp("2006-03-01"))):
    Bk = Book(name, fut, ratio, start)
    stb, frb = Bk.states()
    base = Bk.ev(stb, frb)
    P(f"\n  {name}（前後半切在 {Bk.split.year}）")
    P(HDR)
    row("不加雙動能", base)
    res = []
    for ma, lb in itertools.product((150, 200, 250), (126, 189, 252, 315)):
        st, fr = Bk.states(dm=(ma, lb))
        m = Bk.ev(st, fr)
        res.append((ma, lb, m))
        if (ma, lb) == (200, 252):
            row("+ 雙動能 MA200 / 12 月", m, base)
    irr_d = np.array([m["irr"] - base["irr"] for *_, m in res])
    wst_d = np.array([m["worst"] - base["worst"] for *_, m in res])
    mdd_d = np.array([m["mdd"] - base["mdd"] for *_, m in res])
    both = np.array([(m["h1"] - base["h1"] > -0.005) and (m["h2"] - base["h2"] > -0.005) for *_, m in res])
    P(f"  12 組參數：IRR 差中位 {np.median(irr_d) * 100:+.1f}%（{irr_d.min() * 100:+.1f} ~ {irr_d.max() * 100:+.1f}）｜"
      f"最慘改善中位 {np.median(wst_d) * 100:+.0f} 點（改善的組數 {(wst_d > 0.01).sum()}/12）｜"
      f"回撤改善中位 {np.median(mdd_d) * 100:+.0f} 點｜兩半報酬都沒少超過 0.5% 的組數 {both.sum()}/12")
out.close()
