"""第三十二部分：空頭時用期貨做空（淨空 −0.5 / −1 倍），對照持短債（0 倍）與原本 1 倍。

原報告測過「跌破均線改持 PSQ / QID」每年少 14% / 23%；期貨做空沒有反向 ETF 的管理費與每日重設耗損，所以重測。
空頭定義（全用前一天收盤）：
  MA      ：在 MA200 之下
  MA↓     ：在 MA200 之下且 MA200 比 20 天前低
  DM      ：在 MA200 之下且 12 個月報酬 < T-bill（第三十一部分的雙動能）
  DM+亂   ：DM 且 ATR z ≥ 0.5（空頭中的劇烈段）
3 倍規則與 −4% 盤中停損不變；只改「原本 1 倍、且符合空頭定義」時的部位。全期貨、每天重設。
QQQ 1999 起；SPY / DIA / SOXX 2006/3 起（SOXX 假設有期貨）。另做參數區域：MA 150/200/250 × 動能 189/252 天（DM）。
"""
import itertools
import numpy as np
import pandas as pd
from part22_core import load, signals, run, metrics, year_starts

out = open("results_part32.txt", "w")


def P(s=""):
    print(s); out.write(s + "\n"); out.flush()


def cagr(x):
    return (1 + x).prod() ** (252 / len(x)) - 1


class Book:
    def __init__(self, name, fut, ratio, start=None):
        self.DF = load(f"data/{name.lower()}_us_d.csv")
        self.S = S = signals(self.DF)
        self.i0 = 0 if start is None else S.index.searchsorted(start)
        y0 = S.index[self.i0].year + 1
        self.win = list(range(y0, 2017))
        self.YS = year_starts(S, y0, 2016)
        self.kw = dict(eng="allfut", rebal="daily", fut_name=fut, ratio=ratio, L=3.0, regime_arr=np.ones(len(S), bool))
        self.split = pd.Timestamp("2013-01-01") if start is None else pd.Timestamp("2016-01-01")

    def bear(self, kind, ma=200, lb=252):
        S, c = self.S, self.S.close
        m = c.rolling(ma).mean()
        below = c < m
        if kind == "MA":
            b = below
        elif kind == "MA↓":
            b = below & (m < m.shift(20))
        else:
            b = below & ((c / c.shift(lb) - 1) < S.rf_ann * lb / 252)
        b = b.shift(1).fillna(False).values
        if kind == "DM+亂":                                  # S.z 已經是前一天的值
            b = b & (np.nan_to_num(S.z.values, nan=0.0) >= 0.5)
        return b

    def levels(self, bear_arr, bear_lev, stop=0.04):
        S, DF = self.S, self.DF
        pc = DF.close.shift(1).values
        o_r, l_r = DF.open.values / pc - 1, DF.low.values / pc - 1
        up, z, ws, R = S.up.values, S.z.values, S.ws.values, S.ret.values
        L, fr, lev = np.ones(len(S)), R.copy(), False
        for i in range(len(S)):
            ok = up[i] and z[i] == z[i]
            if lev and stop is not None and l_r[i] <= -stop:
                fr[i] = (R[i] + 2 * (min(o_r[i], -stop) - 0.001)) / 3; lev = False
            elif lev and (not ok or z[i] >= 0.5):
                lev = False
            elif not lev and ok and z[i] < 0.5 and ws[i]:
                lev = True
            L[i] = 3.0 if lev else (bear_lev if bear_arr is not None and bear_arr[i] else 1.0)
        return L, fr

    def ev(self, L, fr, windows=True):
        S = self.S
        kw = dict(self.kw, L_arr=L, fut_ret=fr)
        o = run(S, i0=self.i0, **kw)
        m = metrics(o)
        tw = pd.Series(o["twr"], index=S.index[self.i0:])
        m["h1"], m["h2"] = cagr(tw[tw.index < self.split]), cagr(tw[tw.index >= self.split])
        m["worst_yr"] = ((1 + tw).groupby(tw.index.year).prod() - 1).min()
        V = o["V"]
        m["acct_dd"] = (V / V.cummax() - 1).min()
        m["bear_t"] = (L[self.i0:] < 1).mean()
        m["tw"] = tw
        if windows:
            m["w"] = np.array([metrics(run(S, i0=self.YS[y], i1=self.YS[y + 10], **kw))["worst"] for y in self.win])
            m["wt"] = np.array([metrics(run(S, i0=self.YS[y], i1=self.YS[y + 10], **kw))["tot"] for y in self.win])
        return m


def row(lab, m, b=None):
    s = (f"  {lab:24s}{m['irr'] * 100:6.1f}%{m['worst'] * 100:6.0f}%{m['mdd'] * 100:6.0f}%{m['acct_dd'] * 100:7.0f}%{m['worst_yr'] * 100:6.0f}%"
         f"{m['w'].min() * 100:7.0f}%{m['bear_t'] * 100:5.0f}%")
    if b is not None:
        s += f"{(m['h1'] - b['h1']) * 100:+7.1f}%{(m['h2'] - b['h2']) * 100:+7.1f}%{(m['wt'] > b['wt']).sum():3d}/{len(m['wt'])}"
    P(s)


HDR = (f"  {'版本':24s}{'IRR':>7s}{'最慘':>6s}{'回撤':>6s}{'帳面回落':>7s}{'最差年':>6s}{'10年段最慘':>7s}{'空頭時':>5s}"
       f"{'前半差':>8s}{'後半差':>7s}{'勝段':>5s}")
P("第三十二部分：空頭時用期貨做空\n")
P("  最慘 = 最慘相對本金；回撤 = 時間加權最大回撤；帳面回落 = 帳戶市值從高點最多跌多少；空頭時 = 不是 1 倍也不是 3 倍的時間\n")

BOOKS = [("QQQ", "MNQ", 41.0, None), ("SPY", "MES", 10.0, pd.Timestamp("2006-03-01")),
         ("DIA", "MYM", 100.0, pd.Timestamp("2006-03-01")), ("SOXX", "MNQ", 41.0, pd.Timestamp("2006-03-01"))]
summary = []
for name, fut, ratio, start in BOOKS:
    Bk = Book(name, fut, ratio, start)
    L0, fr0 = Bk.levels(None, 1.0)
    base = Bk.ev(L0, fr0)
    P(f"{name}（前後半切在 {Bk.split.year}）")
    P(HDR)
    row("原本（空頭仍 1 倍）", base)
    for kind in ("MA", "MA↓", "DM", "DM+亂"):
        ba = Bk.bear(kind)
        for bl in (0.0, -0.5, -1.0):
            L, fr = Bk.levels(ba, bl)
            m = Bk.ev(L, fr)
            row(f"{kind:5s} → {bl:+.1f} 倍", m, base)
            summary.append(dict(idx=name, kind=kind, bl=bl, d_irr=m["irr"] - base["irr"], d_worst=m["worst"] - base["worst"],
                                d_mdd=m["mdd"] - base["mdd"], d_h1=m["h1"] - base["h1"], d_h2=m["h2"] - base["h2"]))
    if name == "QQQ":
        tw0 = base["tw"]
        P("\n  QQQ 逐年（原本 / DM→0 / DM→−0.5 / DM→−1），只列有差異 ≥ 5% 的年份")
        ys = {}
        for bl in (0.0, -0.5, -1.0):
            L, fr = Bk.levels(Bk.bear("DM"), bl)
            tw = Bk.ev(L, fr, windows=False)["tw"]
            ys[bl] = (1 + tw).groupby(tw.index.year).prod() - 1
        y0 = (1 + tw0).groupby(tw0.index.year).prod() - 1
        for y in y0.index:
            if max(abs(ys[b][y] - y0[y]) for b in ys) >= 0.05:
                P(f"    {y}：{y0[y] * 100:+5.0f}% / {ys[0.0][y] * 100:+5.0f}% / {ys[-0.5][y] * 100:+5.0f}% / {ys[-1.0][y] * 100:+5.0f}%")
        r = Bk.S.ret
        best = r.sort_values().index[-20:]
        ba = pd.Series(Bk.bear("DM"), index=Bk.S.index)
        P(f"  最好的 20 天裡，DM 空頭訊號亮著的有 {ba[best].sum()} 天（做空會在這些天虧，−1 倍時平均每天 {-r[best][ba[best]].mean() * 100:.1f}%）")
    P("")

    # 參數區域（DM）
    res = []
    for ma, lb, bl in itertools.product((150, 200, 250), (189, 252), (0.0, -0.5, -1.0)):
        L, fr = Bk.levels(Bk.bear("DM", ma, lb), bl)
        m = Bk.ev(L, fr, windows=False)
        res.append((bl, m["irr"] - base["irr"], m["worst"] - base["worst"], m["mdd"] - base["mdd"]))
    R = pd.DataFrame(res, columns=["bl", "irr", "worst", "mdd"])
    P(f"  DM 參數區域（MA 150/200/250 × 動能 189/252，各 6 組）：")
    for bl, g in R.groupby("bl", sort=False):
        P(f"    {bl:+.1f} 倍：IRR 差中位 {g.irr.median() * 100:+.1f}%（{g.irr.min() * 100:+.1f} ~ {g.irr.max() * 100:+.1f}）｜"
          f"最慘改善中位 {g.worst.median() * 100:+.0f} 點｜回撤改善中位 {g.mdd.median() * 100:+.0f} 點")
    P("")

T = pd.DataFrame(summary)
P("總結：四個指數平均（相對「空頭仍 1 倍」）")
for (kind, bl), g in T.groupby(["kind", "bl"], sort=False):
    P(f"  {kind:5s} → {bl:+.1f} 倍：IRR {g.d_irr.mean() * 100:+.1f}%｜最慘 {g.d_worst.mean() * 100:+.0f} 點｜回撤 {g.d_mdd.mean() * 100:+.0f} 點｜"
      f"前半 {g.d_h1.mean() * 100:+.1f}% 後半 {g.d_h2.mean() * 100:+.1f}%｜IRR 變好的指數 {(g.d_irr > 0).sum()}/4")
out.close()
