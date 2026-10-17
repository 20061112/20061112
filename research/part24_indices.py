"""第二十四部分：同一套策略（MA200 + ATR z 開關 3 倍）搬到 SPY、DIA、SOXX，比較 ETF 與期貨執行。

資料：data/{qqq,spy,dia,soxx}_us_d.csv（SPY/DIA/SOXX 取自 claude/festive-lovelace-1975n0 分支，2005/2 起）。
四個指數一律 2006/3/1 開始定投（SPY/DIA/SOXX 要 1 年暖機），10 年滾動 2007 ~ 2016 起共 10 段。
期貨：QQQ → MNQ/NQ、SPY → MES/ES、DIA → MYM/YM；SOXX 沒有流動性足夠的期貨，只能用槓桿 ETF（SOXL / USD）。
價格報酬、未計股息（DIA 約 2%/年、SPY 約 1.3%/年，純定投與策略都少算）。
"""
import itertools
import numpy as np
import pandas as pd
from part22_core import load, signals, run, metrics, year_starts, xirr, CONTRACTS

START = pd.Timestamp("2006-03-01")
SPLIT = pd.Timestamp("2016-01-01")
WIN = list(range(2007, 2017))
IDX = {"QQQ": dict(fut="MNQ", big="NQ", ratio=41.0), "SPY": dict(fut="MES", big="ES", ratio=10.0),
       "DIA": dict(fut="MYM", big="YM", ratio=100.0), "SOXX": dict(fut=None, big=None, ratio=None)}
out = open("results_part24.txt", "w")


def P(s=""):
    print(s); out.write(s + "\n"); out.flush()


def cagr(x):
    return (1 + x).prod() ** (252 / len(x)) - 1


def stop_regime(S, DF, X):
    """持有槓桿時盤中跌破前收 −X% → 額外 2 倍在 −X%（跳空則開盤價）出場，再扣 0.1%。"""
    pc = DF.close.shift(1).values
    o_r, l_r = DF.open.values / pc - 1, DF.low.values / pc - 1
    up, zz, ws, R = S.up.values, S.z.values, S.ws.values, S.ret.values
    reg, fr, cur, n = np.zeros(len(S), bool), R.copy(), False, 0
    for i in range(len(S)):
        ok = up[i] and zz[i] == zz[i]
        if cur and l_r[i] <= -X:
            fr[i] = (R[i] + 2 * (min(o_r[i], -X) - 0.001)) / 3; cur = False; n += 1
        elif cur and (not ok or zz[i] >= 0.5):
            cur = False
        elif ws[i] and not cur and ok and zz[i] < 0.5:
            cur = True
        reg[i] = cur
    return reg, fr, n


DATA = {}
for name, spec in IDX.items():
    DF = load(f"data/{name.lower()}_us_d.csv")
    S = signals(DF)
    DATA[name] = (DF, S, S.index.searchsorted(START), year_starts(S, 2007, 2016),
                  S.index.searchsorted(pd.Timestamp("2010-02-11")))


def ev(name, **kw):
    DF, S, i0, YS, i10 = DATA[name]
    spec = IDX[name]
    if spec["fut"]:
        kw.setdefault("fut_name", spec["fut"]); kw.setdefault("ratio", spec["ratio"])
    o = run(S, i0=i0, **kw)
    m = metrics(o)
    m["irr10"] = metrics(run(S, i0=i10, **kw))["irr"]
    w = [metrics(run(S, i0=YS[y], i1=YS[y + 10], **kw))["tot"] for y in WIN]
    m["avg10"], m["min10"], m["w"] = np.mean(w), np.min(w), np.array(w)
    tw = pd.Series(o["twr"], index=S.index[i0:])
    m["h1"], m["h2"] = cagr(tw[tw.index < SPLIT]), cagr(tw[tw.index >= SPLIT])
    m["yr"] = (1 + tw).groupby(tw.index.year).prod() - 1
    return m


P("第二十四部分：MA200 + ATR z 槓桿開關 —— QQQ / SPY / DIA / SOXX，ETF 與期貨執行（2006/3 起定投，每週 1,000）")
P("  未計股息；10 年滾動 2007~2016 起 10 段；前半 = 2006~2015、後半 = 2016~2026（時間加權年化）\n")

# ---------------------------------------------------------------- A. 總表
P("A. 總表")
P(f"  {'':6s}{'執行':22s}{'IRR06':>7s}{'IRR10':>7s}{'10年均':>8s}{'10年最差':>8s}{'最慘/本金':>8s}{'均槓桿':>7s}{'前半':>7s}{'後半':>7s}")
RES = {}
for name in IDX:
    DF, S, i0, *_ = DATA[name]
    vol = S.ret[S.index >= START].std() * np.sqrt(252)
    rows = [("純定投", dict(eng="dca")), ("ETF 切換 3 倍", dict(eng="etf")),
            ("全期貨 L=3 每週", dict(eng="allfut", L=3.0)), ("現股+期貨/借款 L=3", dict(eng="fut", L=3.0)),
            ("全期貨 L=2 每週", dict(eng="allfut", L=2.0))]
    for lab, kw in rows:
        if IDX[name]["fut"] is None and kw["eng"] in ("allfut", "fut"):
            lab = "（假設有期貨）" + lab
        m = ev(name, **kw)
        RES[(name, lab)] = m
        P(f"  {name:6s}{lab:22s}{m['irr'] * 100:6.1f}%{m['irr10'] * 100:6.1f}%{m['avg10'] * 100:7.0f}%{m['min10'] * 100:7.0f}%"
          f"{m['worst'] * 100:7.0f}%{m['avg_lev']:7.2f}{m['h1'] * 100:6.1f}%{m['h2'] * 100:6.1f}%")
    P(f"  {'':6s}年化波動 {vol * 100:.0f}%（3 倍每年耗損約 {3 * vol ** 2 * 100:.0f}%），開槓桿時間 {S.on[S.index >= START].mean() * 100:.0f}%，"
      f"每年換倉 {RES[(name, 'ETF 切換 3 倍')]['sw_per_yr']:.1f} 次\n")

# ---------------------------------------------------------------- B. 時機是否有用
P("B. 時機是否有用（全期貨 L=3）：固定同平均槓桿、隨機打亂開槓桿時段 300 次")
rng = np.random.default_rng(24)
for name in IDX:
    DF, S, i0, *_ = DATA[name]
    act = RES[(name, [k for n_, k in RES if n_ == name and "全期貨 L=3" in k][0])]
    kwf = dict(fut_name=IDX[name]["fut"] or "MNQ", ratio=IDX[name]["ratio"] or 41.0)
    const = metrics(run(S, eng="allfut", L=act["avg_lev"], rebal="daily", regime_arr=np.ones(len(S), bool), i0=i0, **kwf))
    on = S.on.values.astype(int)
    e = np.flatnonzero(np.diff(np.r_[0, on, 0]))
    on_runs, off_runs = list(e[1::2] - e[::2]), [r for r in np.diff(np.r_[0, e])[0::2] if r > 0]
    ri, rw = [], []
    for _ in range(300):
        a, b = rng.permutation(on_runs), rng.permutation(off_runs)
        seq = []
        for k in range(max(len(a), len(b))):
            if k < len(b): seq += [False] * b[k]
            if k < len(a): seq += [True] * a[k]
        reg = np.array((seq + [False] * len(S))[:len(S)])
        m = metrics(run(S, eng="allfut", L=3.0, regime_arr=reg, i0=i0, **kwf))
        ri.append(m["irr"]); rw.append(m["worst"])
    ri, rw = np.array(ri), np.array(rw)
    P(f"  {name:5s} 策略 IRR {act['irr'] * 100:5.1f}%、最慘 {act['worst'] * 100:4.0f}%｜固定 {act['avg_lev']:.2f} 倍 {const['irr'] * 100:5.1f}%、"
      f"{const['worst'] * 100:4.0f}%｜隨機時段 IRR 中位 {np.median(ri) * 100:5.1f}%，策略勝過 {(ri < act['irr']).mean() * 100:3.0f}%；"
      f"最慘勝過 {(rw < act['worst']).mean() * 100:3.0f}%")

# ---------------------------------------------------------------- C. 凍結參數
P("\nC. 凍結參數樣本外：MA 150/200/250 × z 窗 126/252/504 × 門檻 0/0.5/1，只用 2006~2015 挑，2016~2026 檢驗（全期貨 L=3）")
for name in IDX:
    DF, S0, i0, *_ = DATA[name]
    kwf = dict(fut_name=IDX[name]["fut"] or "MNQ", ratio=IDX[name]["ratio"] or 41.0)
    res = []
    for ma, zw, zo in itertools.product((150, 200, 250), (126, 252, 504), (0.0, 0.5, 1.0)):
        S2 = signals(DF, ma=ma, zwin=zw, z_on=zo)
        j0 = S2.index.searchsorted(START)
        tw = pd.Series(run(S2, eng="allfut", L=3.0, i0=j0, **kwf)["twr"], index=S2.index[j0:])
        res.append(dict(ma=ma, zw=zw, zo=zo, h1=cagr(tw[tw.index < SPLIT]), h2=cagr(tw[tw.index >= SPLIT])))
    R = pd.DataFrame(res)
    R["r2"] = R.h2.rank(ascending=False)
    best = R.loc[R.h1.idxmax()]
    base = R[(R.ma == 200) & (R.zw == 252) & (R.zo == 0.5)].iloc[0]
    q2 = cagr(S0.ret[S0.index >= SPLIT])
    P(f"  {name:5s} 前半最佳 MA{int(best.ma)}/{int(best.zw)}/{best.zo} → 後半 {best.h2 * 100:5.1f}%（第 {int(best.r2):2d}/27）｜"
      f"報告參數後半 {base.h2 * 100:5.1f}%（第 {int(base.r2):2d}）｜27 組後半中位 {R.h2.median() * 100:5.1f}%、最差 {R.h2.min() * 100:5.1f}%｜"
      f"指數本身 {q2 * 100:5.1f}%，勝過的組數 {(R.h2 > q2).sum():2d}/27")

# ---------------------------------------------------------------- D. 盤中停損
P("\nD. 盤中停損（全期貨 L=3；QQQ 正規盤價格，夜盤觸發未計）")
for name in ("QQQ", "SPY", "DIA"):
    DF, S, i0, *_ = DATA[name]
    base = RES[(name, "全期貨 L=3 每週")]
    for X in (0.04, 0.05):
        reg, fr, n = stop_regime(S, DF, X)
        m = ev(name, eng="allfut", L=3.0, regime_arr=reg, fut_ret=fr)
        P(f"  {name:5s} −{X * 100:.0f}%：觸發 {n:2d} 次｜IRR {base['irr'] * 100:.1f}% → {m['irr'] * 100:.1f}%｜前半 {(m['h1'] - base['h1']) * 100:+.1f}%、"
          f"後半 {(m['h2'] - base['h2']) * 100:+.1f}%｜10 年段勝基準 {(m['w'] > base['w']).sum()}/10")

# ---------------------------------------------------------------- E. 合約顆粒度
P("\nE. 合約顆粒度（以今天合約大小起算，L=3 全期貨；10 段 10 年的開槓桿時實際槓桿 10–90%）")
for name in ("QQQ", "SPY", "DIA"):
    DF, S, i0, YS, _ = DATA[name]
    spec = IDX[name]
    for con in (spec["fut"], spec["big"]):
        mult = CONTRACTS[con][0]
        notional = mult * spec["ratio"] * DF.close.iloc[-1]
        cells = []
        for lump, wk in ((10_000, 250), (30_000, 500), (60_000, 1_000), (150_000, 2_000), (500_000, 5_000)):
            lv = []
            for y in WIN:
                o = run(S, eng="allfut", L=3.0, base=wk, lump=lump, i0=YS[y], i1=YS[y + 10], contract=con,
                        start_notional=notional)
                lv.append(o["lev"][S.on.values[YS[y]:YS[y + 10]]])
            lv = np.concatenate(lv)
            cells.append(f"{lump // 1000:>3d}k {np.percentile(lv, 10):.2f}~{np.percentile(lv, 90):.2f}")
        P(f"  {name:5s}{con:4s}一口約 {notional:>9,.0f} 美元｜" + "｜".join(cells))

# ---------------------------------------------------------------- F. 逐年
P("\nF. 逐年報酬（時間加權）：純定投 / 策略（QQQ、SPY、DIA 用全期貨 L=3；SOXX 用 ETF 3 倍）")
cols = {}
for name in IDX:
    k = "ETF 切換 3 倍" if IDX[name]["fut"] is None else "全期貨 L=3 每週"
    cols[f"{name} 定投"] = RES[(name, "純定投")]["yr"]
    cols[f"{name} 策略"] = RES[(name, k)]["yr"]
Y = pd.DataFrame(cols)
P("  年份" + "".join(f"{c:>11s}" for c in Y.columns))
for y, r in Y.iterrows():
    P(f"  {y}" + "".join(f"{v * 100:>10.0f}%" for v in r.values))
for name in IDX:
    d = Y[f"{name} 策略"] - Y[f"{name} 定投"]
    P(f"  {name:5s} 策略勝過定投 {(d > 0).sum()}/{len(d)} 年；落後最多 {d.min() * 100:+.0f}%（{d.idxmin()}）")

# ---------------------------------------------------------------- G. 組合
P("\nG. 組合（每週金額平均分給各指數，各自獨立跑策略；QQQ/SPY/DIA 全期貨 L=3、SOXX ETF 3 倍）")
tws = {}
for name in IDX:
    DF, S, i0, *_ = DATA[name]
    kw = dict(eng="etf") if IDX[name]["fut"] is None else dict(eng="allfut", L=3.0, fut_name=IDX[name]["fut"], ratio=IDX[name]["ratio"])
    o = run(S, i0=i0, **kw)
    tws[name] = o
idx = sorted(set.intersection(*[set(o["V"].index) for o in tws.values()]))
for combo in (("QQQ",), ("QQQ", "SPY"), ("QQQ", "DIA"), ("QQQ", "SOXX"), ("QQQ", "SPY", "DIA"), ("QQQ", "SPY", "DIA", "SOXX")):
    V = sum(tws[c]["V"].reindex(idx).ffill() / len(combo) for c in combo)
    flows = tws[combo[0]]["flows"]
    cum = pd.Series(0.0, index=idx)
    for d, a in flows:
        if d in cum.index:
            cum[d] += a
    cum = cum.cumsum()
    rel = (V / cum - 1)[cum > 0]
    P(f"  {' + '.join(combo):22s} IRR {xirr(flows, idx[-1], V.iloc[-1]) * 100:5.1f}%｜最慘相對本金 {rel.min() * 100:4.0f}%")
P("  （各指數的定投金額與日期相同，所以組合 = 各自市值相加 / 檔數；組合只取四檔共同日期，截至 10/15，所以 QQQ 單獨與 A 節略有差異）")
out.close()
