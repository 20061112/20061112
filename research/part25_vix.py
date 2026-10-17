"""第二十五部分：VXN / VIX 對 MA200 + ATR z 槓桿開關策略的用處（全期貨 L=3、每週重設）。

claude/festive-lovelace-1975n0 已測過 VXN 當「開關訊號」（取代 ATR z、AND、VRP、5 日急升、(ATR+VXN)/2、波動目標），全部輸原策略。
這裡只測沒測過的用法：
  A. 重現：VXN z 取代 ATR z（確認與舊分支一致）
  B. 快速出場：前一天 VXN/VIX 單日急升或超過水準 → 當天收盤先退回 1 倍（進場規則不變）
  C. 恐慌補倉改用 VXN / VIX 水準（取代 ATR z > 2）
  D. 盤中停損距離依隱含波動縮放：k × VXN / √252（取代固定 −4%）
  E. SPY / DIA 用 VIX（舊分支只測過 QQQ + VXN）
QQQ 從 2002/3/1 起定投（VXN 2001/2 起，需暖機），10 年滾動 2003~2016 起 14 段；前半 2002~2013、後半 2014~2026。
"""
import numpy as np
import pandas as pd
from part22_core import load, signals, run, metrics, year_starts

START = pd.Timestamp("2002-03-01")
SPLIT = pd.Timestamp("2014-01-01")
out = open("results_part25.txt", "w")


def P(s=""):
    print(s); out.write(s + "\n"); out.flush()


def cagr(x):
    return (1 + x).prod() ** (252 / len(x)) - 1


vxn = pd.read_csv("data/VXNCLS.csv", parse_dates=["observation_date"]).set_index("observation_date").VXNCLS
vix = pd.read_csv("data/VIX_History.csv")
vix.index = pd.to_datetime(vix.DATE, format="%m/%d/%Y")
vix = vix[["HIGH", "CLOSE"]].astype(float)


def align(s, idx):
    return s.reindex(idx.union(s.index)).ffill(limit=3).reindex(idx)


def iv_z(iv, win=252):
    l = np.log(iv)
    return (l - l.rolling(win).mean()) / l.rolling(win).std()


def regime(S, z=None, exit_now=None, z_on=0.5):
    """原規則；z 可換成別的波動 z；exit_now[i]（只用到 i−1 的資料）為真時當天收盤退出槓桿。"""
    up, ws = S.up.values, S.ws.values
    z = S.z.values if z is None else z
    on, res = False, np.zeros(len(S), bool)
    for i in range(len(S)):
        ok = up[i] and z[i] == z[i]
        if on and (not ok or z[i] >= z_on or (exit_now is not None and exit_now[i])):
            on = False
        if ws[i] and not on and ok and z[i] < z_on and not (exit_now is not None and exit_now[i]):
            on = True
        res[i] = on
    return res


def stop_regime(S, DF, Xarr):
    """持有槓桿時盤中跌破前收 −X[i] → 額外 2 倍在 −X（跳空則開盤價）出場，再扣 0.1%。"""
    pc = DF.close.shift(1).values
    o_r, l_r = DF.open.values / pc - 1, DF.low.values / pc - 1
    up, zz, ws, R = S.up.values, S.z.values, S.ws.values, S.ret.values
    reg, fr, cur, n = np.zeros(len(S), bool), R.copy(), False, 0
    for i in range(len(S)):
        ok = up[i] and zz[i] == zz[i]
        X = Xarr[i]
        if cur and X == X and l_r[i] <= -X:
            fr[i] = (R[i] + 2 * (min(o_r[i], -X) - 0.001)) / 3; cur = False; n += 1
        elif cur and (not ok or zz[i] >= 0.5):
            cur = False
        elif ws[i] and not cur and ok and zz[i] < 0.5:
            cur = True
        reg[i] = cur
    return reg, fr, n


class Book:
    def __init__(self, name, fut, ratio, start):
        self.DF = load(f"data/{name.lower()}_us_d.csv")
        self.S = signals(self.DF)
        self.i0 = self.S.index.searchsorted(start)
        y0 = start.year + 1
        self.YS = year_starts(self.S, y0, 2016)
        self.win = list(range(y0, 2017))
        self.kw = dict(eng="allfut", L=3.0, fut_name=fut, ratio=ratio)
        self.base = None
        self.base = self.ev()

    def ev(self, **extra):
        kw = {**self.kw, **extra}
        o = run(self.S, i0=self.i0, **kw)
        m = metrics(o)
        tw = pd.Series(o["twr"], index=self.S.index[self.i0:])
        m["h1"], m["h2"] = cagr(tw[tw.index < SPLIT]), cagr(tw[tw.index >= SPLIT])
        m["w"] = np.array([metrics(run(self.S, i0=self.YS[y], i1=self.YS[y + 10], **kw))["tot"] for y in self.win])
        yr = (1 + tw).groupby(tw.index.year).prod() - 1
        m["worst_yr"] = yr.min()
        m["sw"] = o["switches"] / ((self.S.index[-1] - self.S.index[self.i0]).days / 365.25)
        return m

    def show(self, lab, m):
        b = self.base or m
        d1, d2 = m["h1"] - b["h1"], m["h2"] - b["h2"]
        tag = "同向+" if d1 > 0.002 and d2 > 0.002 else ("同向−" if d1 < -0.002 and d2 < -0.002 else "  —  ")
        P(f"  {lab:34s}{m['irr'] * 100:6.1f}%{m['worst'] * 100:6.0f}%{m['worst_yr'] * 100:6.0f}%{m['sw']:6.1f}"
          f"{d1 * 100:+7.1f}%{d2 * 100:+7.1f}%  {tag} {(m['w'] > b['w']).sum():2d}/{len(m['w'])}")


HDR = f"  {'變體':34s}{'IRR':>7s}{'最慘':>6s}{'最差年':>6s}{'換倉':>6s}{'前半差':>8s}{'後半差':>8s}  兩半   勝基準段"

# ---------------------------------------------------------------- QQQ
Q = Book("QQQ", "MNQ", 41.0, START)
S, DF = Q.S, Q.DF
VXN = align(vxn, S.index)
VIXc, VIXh = align(vix.CLOSE, S.index), align(vix.HIGH, S.index)
P("第二十五部分：VXN / VIX 的用處（QQQ 全期貨 L=3 每週重設，2002/3 起定投，10 年滾動 14 段）")
P(f"  前半 = 2002~2013、後半 = 2014~2026（年化，相對基準）；同向+ = 兩半都改善 > 0.2%\n")
P(HDR)
Q.show("基準（ATR z）", Q.base)

P("\nA. 重現舊分支：VXN z 取代 ATR z / 兩者任一平靜 / 兩者都平靜")
vz = iv_z(VXN).shift(1).values
Q.show("VXN z < 0.5 取代 ATR z", Q.ev(regime_arr=regime(S, z=vz)))
Q.show("min(ATR z, VXN z) < 0.5", Q.ev(regime_arr=regime(S, z=np.fmin(S.z.values, vz))))
Q.show("max(ATR z, VXN z) < 0.5", Q.ev(regime_arr=regime(S, z=np.fmax(S.z.values, vz))))

P("\nB. 快速出場：前一天收盤 VXN / VIX 觸發 → 今天收盤退回 1 倍（下週一照原規則再進場）")
chg = (VXN / VXN.shift(1) - 1)
for x in (0.15, 0.20, 0.30):
    e = (chg >= x).shift(1).fillna(False).values
    Q.show(f"VXN 單日漲 ≥ {x * 100:.0f}%（{int((chg >= x).sum())} 天）", Q.ev(regime_arr=regime(S, exit_now=e)))
for lv in (25, 30, 35):
    e = (VXN >= lv).shift(1).fillna(False).values
    Q.show(f"VXN ≥ {lv}", Q.ev(regime_arr=regime(S, exit_now=e)))
vspike = VIXh / VIXc.shift(1) - 1
for x in (0.20, 0.30, 0.50):
    e = (vspike >= x).shift(1).fillna(False).values
    Q.show(f"VIX 盤中高點 ≥ 前收 +{x * 100:.0f}%（{int((vspike >= x).sum())} 天）", Q.ev(regime_arr=regime(S, exit_now=e)))
gap = VXN - (S.atrp * np.sqrt(252) * 100 / 1.25)   # 隱含 − 以 ATR 換算的實現波動（ATR ≈ 1.25 × 日標準差）
for g in (8, 12):
    e = (gap >= g).shift(1).fillna(False).values
    Q.show(f"VXN 比實現波動高 ≥ {g} 點", Q.ev(regime_arr=regime(S, exit_now=e)))

P("\nC. 恐慌補倉（20% 現金全部買 1 倍）改用隱含波動水準")
for lab, arr in [("VXN ≥ 40", (VXN >= 40)), ("VXN ≥ 50", (VXN >= 50)), ("VIX ≥ 35", (VIXc >= 35)), ("VIX ≥ 45", (VIXc >= 45)),
                 ("VXN z > 2", pd.Series(vz > 2, index=S.index))]:
    a = arr.shift(1).fillna(False).values if lab.startswith("VXN ≥") or lab.startswith("VIX") else arr.values
    Q.show(f"{lab}（原 ATR z > 2）", Q.ev(panic_arr=a))

P("\nD. 盤中停損：固定 −4% vs 依前一天 VXN 縮放（停損距離 = k × VXN / √252）")
reg, fr, n = stop_regime(S, DF, np.full(len(S), 0.04))
Q.show(f"固定 −4%（觸發 {n} 次）", Q.ev(regime_arr=reg, fut_ret=fr))
for k in (2.0, 2.5, 3.0):
    X = (k * VXN.shift(1) / 100 / np.sqrt(252)).values
    reg, fr, n = stop_regime(S, DF, X)
    Q.show(f"k = {k}（中位 {np.nanmedian(X[S.on.values]) * 100:.1f}%，觸發 {n} 次）", Q.ev(regime_arr=reg, fut_ret=fr))

# ---------------------------------------------------------------- SPY / DIA
P("\nE. SPY / DIA 用 VIX（2006/3 起，10 年滾動 10 段；前後半切點同上）")
for name, fut, ratio in (("SPY", "MES", 10.0), ("DIA", "MYM", 100.0)):
    B = Book(name, fut, ratio, pd.Timestamp("2006-03-01"))
    Sx = B.S
    V = align(vix.CLOSE, Sx.index)
    Vh = align(vix.HIGH, Sx.index)
    vzx = iv_z(V).shift(1).values
    P(f"  {name}" + HDR[6:])
    B.show("基準（ATR z）", B.base)
    B.show("VIX z < 0.5 取代 ATR z", B.ev(regime_arr=regime(Sx, z=vzx)))
    B.show("min(ATR z, VIX z) < 0.5", B.ev(regime_arr=regime(Sx, z=np.fmin(Sx.z.values, vzx))))
    B.show("max(ATR z, VIX z) < 0.5", B.ev(regime_arr=regime(Sx, z=np.fmax(Sx.z.values, vzx))))
    c2 = V / V.shift(1) - 1
    B.show("VIX 單日漲 ≥ 20% → 快速出場", B.ev(regime_arr=regime(Sx, exit_now=(c2 >= 0.2).shift(1).fillna(False).values)))
    B.show("VIX ≥ 25 → 快速出場", B.ev(regime_arr=regime(Sx, exit_now=(V >= 25).shift(1).fillna(False).values)))
    B.show("恐慌補倉改 VIX ≥ 35", B.ev(panic_arr=(V >= 35).shift(1).fillna(False).values))

# ---------------------------------------------------------------- 目前
P("\nF. 目前（2026/10/16）")
last = vix.index[-1]
P(f"  VIX 10/15 收 {vix.CLOSE.iloc[-2]:.2f} → 10/16 盤中高 {vix.HIGH.iloc[-1]:.2f}（+{(vix.HIGH.iloc[-1] / vix.CLOSE.iloc[-2] - 1) * 100:.0f}%）、收 {vix.CLOSE.iloc[-1]:.2f}"
  f"（+{(vix.CLOSE.iloc[-1] / vix.CLOSE.iloc[-2] - 1) * 100:.0f}%）；VXN 10/15 {vxn.dropna().iloc[-1]:.2f}")
vzq = iv_z(VXN)
P(f"  VXN z（10/15）{vzq.dropna().iloc[-1]:+.2f}；VIX z（10/16）{iv_z(vix.CLOSE).iloc[-1]:+.2f}；QQQ ATR z（10/16 收盤算）見 part22_live.json")
P(f"  VXN − 實現波動：{gap.dropna().iloc[-1]:.1f} 點")
out.close()
