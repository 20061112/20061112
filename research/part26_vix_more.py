"""第二十六部分：VXN / VIX 的更多用法（QQQ 全期貨 L=3 每週重設，2002/3 起定投）。

A. 快速進場：平日也能進場（原本只有週一），用 VXN 確認平靜；對照組 = 平日進場但不看 VXN
B. 恐慌退潮再進場：出場後 VIX 從 10 日高點回落 ≥ 20~30% 且 VXN z 低 → 不等 ATR z 回落就進場
C. 定投金額：VIX 高時加倍（取代或加在布林加碼上）
D. 恐慌補倉時機：VXN ≥ 40 之後等「VXN 跌破 5 日均」（高峰已過）才買，對照原本 ATR z > 2
E. 槓桿大小依 VXN 絕對水準（< 20 才 3 倍，否則 2 倍）
F. 價格 + VIX 雙確認的緊急出場（單日跌 ≥ X% 且 VIX 漲 ≥ Y%）
G. 科技股特有恐慌：VXN / VIX 比值 z 高時出場或不進場
H. 依隱含波動決定是否賣 covered call（持 1 倍時賣 0.2 delta、約 45 天，Black-Scholes 定價、權利金扣 5% 價差）
出場類規則另做安慰劑：同樣天數的隨機出場日 100 次，看規則排在第幾百分位。
"""
import math
import numpy as np
import pandas as pd
from part22_core import load, signals, run, metrics, year_starts

START = pd.Timestamp("2002-03-01")
SPLIT = pd.Timestamp("2014-01-01")
out = open("results_part26.txt", "w")


def P(s=""):
    print(s); out.write(s + "\n"); out.flush()


def cagr(x):
    return (1 + x).prod() ** (252 / len(x)) - 1


DF = load()
S = signals(DF)
I0 = S.index.searchsorted(START)
YS = year_starts(S, 2003, 2016)
WIN = list(range(2003, 2017))
KW = dict(eng="allfut", L=3.0)


def align(s):
    return s.reindex(S.index.union(s.index)).ffill(limit=3).reindex(S.index)


vxn_raw = pd.read_csv("data/VXNCLS.csv", parse_dates=["observation_date"]).set_index("observation_date").VXNCLS
vix_raw = pd.read_csv("data/VIX_History.csv")
vix_raw.index = pd.to_datetime(vix_raw.DATE, format="%m/%d/%Y")
VXN, VIX = align(vxn_raw), align(vix_raw.CLOSE.astype(float))


def zs(x, win=252):
    l = np.log(x)
    return (l - l.rolling(win).mean()) / l.rolling(win).std()


VXNZ = zs(VXN)
lag = lambda s: s.shift(1).fillna(False).astype(bool).values if s.dtype == bool else s.shift(1).values


def ev(full=True, **extra):
    kw = {**KW, **extra}
    o = run(S, i0=I0, **kw)
    m = metrics(o)
    tw = pd.Series(o["twr"], index=S.index[I0:])
    m["h1"], m["h2"] = cagr(tw[tw.index < SPLIT]), cagr(tw[tw.index >= SPLIT])
    m["worst_yr"] = ((1 + tw).groupby(tw.index.year).prod() - 1).min()
    m["sw"] = o["switches"] / ((S.index[-1] - S.index[I0]).days / 365.25)
    if full:
        m["w"] = np.array([metrics(run(S, i0=YS[y], i1=YS[y + 10], **kw))["tot"] for y in WIN])
    return m


BASE = ev()
NTEST = [0]


def show(lab, m, placebo=None):
    NTEST[0] += 1
    d1, d2 = m["h1"] - BASE["h1"], m["h2"] - BASE["h2"]
    tag = "同向+" if d1 > 0.002 and d2 > 0.002 else ("同向−" if d1 < -0.002 and d2 < -0.002 else "  —  ")
    s = (f"  {lab:40s}{m['irr'] * 100:6.1f}%{m['worst'] * 100:6.0f}%{m['worst_yr'] * 100:6.0f}%{m['sw']:6.1f}"
         f"{d1 * 100:+7.1f}%{d2 * 100:+7.1f}%  {tag} {(m['w'] > BASE['w']).sum():2d}/14")
    if placebo is not None:
        s += f"  安慰劑百分位 {placebo:3.0f}%"
    P(s)


def regime(z=None, exit_now=None, entry_day=None, iv_entry=None, iv_exit=None):
    """原規則 + 可選：exit_now 額外出場；entry_day 平日也可進場的日子；
    iv_entry 不看 ATR z 的進場（之後在 ATR z 回到 < 0.5 前，只用 iv_exit 判斷出場）。所有輸入只用到 i−1。"""
    up, ws = S.up.values, S.ws.values
    z = S.z.values if z is None else z
    on, via_iv, res = False, False, np.zeros(len(S), bool)
    for i in range(len(S)):
        ok = up[i] and z[i] == z[i]
        calm = ok and z[i] < 0.5
        ex = exit_now is not None and exit_now[i]
        if on:
            if via_iv and calm:
                via_iv = False
            if via_iv:
                if not up[i] or iv_exit[i] or ex:
                    on = via_iv = False
            elif not calm or ex:
                on = False
        if not on and not ex:
            if calm and (ws[i] or (entry_day is not None and entry_day[i])):
                on = True
            elif iv_entry is not None and up[i] and iv_entry[i]:
                on = via_iv = True
        res[i] = on
    return res


def placebo(exit_now, n=100, seed=0):
    rng = np.random.default_rng(seed)
    k = int(exit_now[I0:].sum())
    act = ev(full=False, regime_arr=regime(exit_now=exit_now))["irr"]
    vals = []
    for _ in range(n):
        e = np.zeros(len(S), bool)
        e[I0 + rng.choice(len(S) - I0, k, replace=False)] = True
        vals.append(ev(full=False, regime_arr=regime(exit_now=e))["irr"])
    return (np.array(vals) < act).mean() * 100


HDR = f"  {'變體':40s}{'IRR':>7s}{'最慘':>6s}{'最差年':>6s}{'換倉':>6s}{'前半差':>8s}{'後半差':>8s}  兩半   勝基準段"
P("第二十六部分：VXN / VIX 的更多用法（QQQ 全期貨 L=3 每週重設，2002/3 起定投，10 年滾動 14 段）")
P("  前半 2002~2013、後半 2014~2026（年化，相對基準）；同向+ = 兩半都改善 > 0.2%\n")
P(HDR)
show("基準", BASE)

P("\nA. 快速進場：平日也能進場（原本只在每週第一個交易日）")
show("對照：每天都可進場（不看 VXN）", ev(regime_arr=regime(entry_day=np.ones(len(S), bool))))
for t in (0.0, -0.5):
    show(f"平日進場需前一天 VXN z < {t}", ev(regime_arr=regime(entry_day=lag(VXNZ < t))))
for lv in (18, 20):
    show(f"平日進場需前一天 VXN < {lv}", ev(regime_arr=regime(entry_day=lag(VXN < lv))))

P("\nB. 恐慌退潮再進場：VIX 從 10 日高點回落 ≥ X 且 VXN z < t → 不等 ATR z（之後 VXN z ≥ 0.5 就出）")
peak = VIX.rolling(10).max()
for x, t in ((0.2, 0.5), (0.3, 0.5), (0.3, 0.0)):
    ie = lag((VIX <= (1 - x) * peak) & (VXNZ < t))
    show(f"回落 {x * 100:.0f}%、VXN z < {t}", ev(regime_arr=regime(iv_entry=ie, iv_exit=lag(VXNZ >= 0.5))))

P("\nC. 定投金額：VIX 高時加倍")
bzv = S.bz.values <= -0.6
for lv in (25, 30):
    v = lag(VIX >= lv)
    show(f"VIX ≥ {lv} 加倍（取代布林）", ev(boost_arr=v))
    show(f"VIX ≥ {lv} 或布林 加倍", ev(boost_arr=v | bzv))
show("VXN z > 1 加倍（取代布林）", ev(boost_arr=lag(VXNZ > 1)))

P("\nD. 恐慌補倉時機")
hi40 = (VXN.rolling(10).max() >= 40) & (VXN < VXN.rolling(5).mean())
show("VXN 近 10 日 ≥ 40 且跌破 5 日均", ev(panic_arr=lag(hi40)))
hi35 = (VIX.rolling(10).max() >= 35) & (VIX < VIX.rolling(5).mean())
show("VIX 近 10 日 ≥ 35 且跌破 5 日均", ev(panic_arr=lag(hi35)))
show("ATR z > 2 且 VXN 跌破 5 日均", ev(panic_arr=(S.z.values > 2) & lag(VXN < VXN.rolling(5).mean())))

P("\nE. 槓桿大小依 VXN 絕對水準（每週重設時套用）")
vx = VXN.shift(1).values
for lv in (18, 20, 25):
    show(f"VXN < {lv} 才 3 倍，否則 2 倍", ev(L_arr=np.where(vx < lv, 3.0, 2.0)))
show("VXN < 1 年中位 才 3 倍，否則 2 倍", ev(L_arr=np.where(vx < VXN.rolling(252).median().shift(1).values, 3.0, 2.0)))

P("\nF. 價格 + VIX 雙確認的緊急出場（前一天 QQQ 跌 ≥ X 且 VIX 漲 ≥ Y → 今天收盤退回 1 倍）")
vchg = VIX / VIX.shift(1) - 1
for x, y in ((0.015, 0.15), (0.02, 0.20), (0.025, 0.25), (0.03, 0.30)):
    e = lag((S.ret <= -x) & (vchg >= y))
    show(f"跌 ≥ {x * 100:.1f}% 且 VIX ≥ +{y * 100:.0f}%（{int(e[I0:].sum())} 天）", ev(regime_arr=regime(exit_now=e)), placebo(e))
P("  對照：只看價格")
for x in (0.02, 0.03):
    e = lag(S.ret <= -x)
    show(f"只看 QQQ 跌 ≥ {x * 100:.0f}%（{int(e[I0:].sum())} 天）", ev(regime_arr=regime(exit_now=e)), placebo(e))

P("\nG. 科技股特有恐慌：VXN / VIX 比值的 z（252 日）")
rz = zs(VXN / VIX)
for t in (1.5, 2.0):
    e = lag(rz >= t)
    show(f"比值 z ≥ {t} 出場且不進場（{int(e[I0:].sum())} 天）", ev(regime_arr=regime(exit_now=e)), placebo(e))

# ---------------------------------------------------------------- H. covered call
P("\nH. 持 1 倍時賣 0.2 delta、45 天 call（BS 定價，IV = VXN − 偏斜，權利金扣 5%；開槓桿前買回；到期才結算，不逐日評價）")
Nc = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_call(Sp, K, T, sig, r):
    if T <= 0 or sig <= 0:
        return max(Sp - K, 0.0)
    d1 = (math.log(Sp / K) + (r + sig * sig / 2) * T) / (sig * math.sqrt(T))
    return Sp * Nc(d1) - K * math.exp(-r * T) * Nc(d1 - sig * math.sqrt(T))


def cc_returns(reg, skew, cond, days=31):
    Pc, rf, vv = S.close.values, S.rf_ann.values, VXN.values
    fr = S.ret.values.copy()
    pos, n = None, 0
    def add(i, x):                                       # x = 每 1 倍名目的現金；當天的期貨名目 = 前一天收盤後的槓桿
        fr[i] += x / (3.0 if reg[i - 1] else 1.0)
    for i in range(1, len(S)):
        if pos is not None:
            K, ei = pos
            if reg[i]:                                   # 今天收盤要開槓桿 → 先買回
                iv = (vv[i] - skew) / 100
                add(i, -bs_call(Pc[i], K, (ei - i) / 252, iv, rf[i]) * 1.05 / Pc[i - 1]); pos = None
            elif i >= ei:
                add(i, -max(Pc[i] - K, 0.0) / Pc[i - 1]); pos = None
        if pos is None and not reg[i] and cond[i] and vv[i] == vv[i] and i >= I0:
            iv, T = (vv[i] - skew) / 100, days / 252
            K = Pc[i] * math.exp(0.8416 * iv * math.sqrt(T) + (rf[i] + iv * iv / 2) * T)
            add(i, bs_call(Pc[i], K, T, iv, rf[i]) * 0.95 / Pc[i - 1])
            pos, n = (K, i + days), n + 1
    return fr, n


reg0 = S.on.values
rv = S.atrp * np.sqrt(252) * 100 / 1.25
for skew in (2.0, 4.0):
    for lab, cond in [("持 1 倍時都賣", np.ones(len(S), bool)), ("VXN − 實現 ≥ 4 點才賣", (VXN - rv >= 4).values),
                      ("VXN z > 0 才賣", (VXNZ > 0).values), ("VXN ≥ 20 才賣", (VXN >= 20).values)]:
        fr, n = cc_returns(reg0, skew, cond)
        show(f"偏斜 {skew:.0f}：{lab}（{n} 次）", ev(fut_ret=fr))

# ---------------------------------------------------------------- I. 副產品：平日進場（1999 起全樣本）
P("\nI. 副產品（與 VIX 無關）：A 節對照組顯示「每天都可進場」比較好，用 1999 起全樣本確認")
S99 = signals(DF)
YS99 = year_starts(S99)
daily = regime(entry_day=np.ones(len(S), bool))
for lab, kw in [("每週進場（原規則）", dict(eng="allfut", L=3.0)), ("每天都可進場", dict(eng="allfut", L=3.0, regime_arr=daily)),
                ("ETF 每週進場", dict(eng="etf")), ("ETF 每天都可進場", dict(eng="etf", regime_arr=daily))]:
    o = run(S99, **kw); m = metrics(o)
    w = [metrics(run(S99, i0=YS99[y], i1=YS99[y + 10], **kw))["tot"] for y in range(2000, 2017)]
    tw = pd.Series(o["twr"], index=S99.index); yr = (1 + tw).groupby(S99.index.year).prod() - 1
    P(f"  {lab:18s}IRR {m['irr'] * 100:5.1f}%｜最慘相對本金 {m['worst'] * 100:4.0f}%｜換倉 {m['sw_per_yr']:.1f}/年｜10 年平均 {np.mean(w) * 100:4.0f}%、"
      f"最差 {np.min(w) * 100:3.0f}%｜2005 {yr[2005] * 100:+.0f}% 2011 {yr[2011] * 100:+.0f}% 2015 {yr[2015] * 100:+.0f}%")

P(f"\n本部分共比較 {NTEST[0] - 1} 個變體；隨機情況下約 1/4 會兩半同向，應以「同向+」且安慰劑百分位高者為準。")
out.close()
