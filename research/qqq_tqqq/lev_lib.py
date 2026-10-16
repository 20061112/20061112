"""把 QQQ 槓桿切換策略套到任意指數 ETF 的共用工具（日線 OHLC，Stooq 格式）。

build(ticker)        讀 data/<ticker>_us_d.csv，加日報酬、T-bill、合成 2 / 3 / −3 倍（管理費 0.95%、融資 T-bill + 0.45%）
calibrate(d, ...)    合成 vs 真實槓桿 ETF
signals / timing     原策略訊號（MA200 & ATR z < 0.5）、每週進場 / 每天出場
dca_only             純定投（每週 1 單位）
evaluate             全期 / 2010 起 / 10 年滾動 IRR、最慘相對本金
mc                   3 個月區塊拼貼蒙地卡羅
"""
import numpy as np
import pandas as pd
import engine as E
from run_qld import run3


def build(ticker):
    d = E._read(f"{ticker}_us_d.csv")[["open", "high", "low", "close"]].copy()
    d["r"] = d.close.pct_change().fillna(0.0)
    rf_y = np.array([E.TBILL[y] for y in d.index.year]) / 100
    d["rf"] = rf_y / 252
    for k in (2, 3, -3):
        fin = (rf_y + E.SPREAD) * (k - 1) if k > 0 else -rf_y * (1 - k)
        rk = k * d.r - E.FEE / 252 - fin / 252
        rk.iloc[0] = 0.0
        d[f"x{k}"] = (1 + rk).cumprod()
    d["bill"] = (1 + d.rf).cumprod()
    d["base"] = d.close
    p = d.index.to_period("W").to_series(index=d.index)
    d["week_start"] = p.ne(p.shift(1)).values
    return d


def calibrate(d, real_ticker, k):
    real = E._read(f"{real_ticker}_us_d.csv").close.reindex(d.index)
    m = real.notna() & (real > 0)
    rr = real[m].pct_change().dropna(); rs = d[f"x{k}"][m].pct_change().dropna()
    keep = rr.abs() < 0.6
    rr, rs = rr[keep], rs[keep]
    yrs = (rr.index[-1] - rr.index[0]).days / 365.25
    cg = lambda x: (1 + x).prod() ** (1 / yrs) - 1
    return dict(start=rr.index[0].date(), real=cg(rr), syn=cg(rs), corr=np.corrcoef(rr, rs)[0, 1],
                te=(rr - rs).std() * np.sqrt(252))


def real_series(d, real_ticker, k):
    """真實槓桿 ETF 接在合成價格之後（比例銜接），讓全期都能用、上市後是真實價格。"""
    real = E._read(f"{real_ticker}_us_d.csv").close.reindex(d.index)
    first = real[real.notna() & (real > 0)].index[0]
    syn = d[f"x{k}"]
    return pd.Series(np.where(d.index < first, syn, real / real[first] * syn[first]), index=d.index).ffill()


def signals(d, ma=200, zthr=0.5):
    z = E.atr_z(d)
    above = d.close > d.close.rolling(ma).mean()
    return above, z, above & (z < zthr)


def timing(d, L):
    t = L.shift(1).fillna(1).astype(int).values
    ws = d.week_start.values; o = np.ones(len(t), int); cur = int(t[0])
    for i in range(len(t)):
        if t[i] < cur or ws[i]:
            cur = t[i]
        o[i] = cur
    return o


def codes(d, ok, alt=None):
    """3 = 槓桿、2 = 替代資產（alt 為 True 時）、1 = 底倉。"""
    L = np.where(ok, 3, 1)
    if alt is not None:
        L = np.where(ok, 3, np.where(alt, 2, 1))
    return timing(d, pd.Series(L, index=d.index))


def dca_only(d, start=None, end=None):
    idx = d.index; i0 = 0 if start is None else idx.searchsorted(pd.Timestamp(start))
    i1 = len(idx) if end is None else idx.searchsorted(pd.Timestamp(end))
    px = d.close.values; ws = d.week_start.values; units = 0.0; inv = 0.0; flows = []; vals = []; invs = []
    for k, i in enumerate(range(i0, i1)):
        if ws[i] or k == 0:
            units += 1 / px[i]; inv += 1; flows.append((idx[i], -1.0))
        vals.append(units * px[i]); invs.append(inv)
    flows.append((idx[i1 - 1], vals[-1]))
    return dict(value=np.array(vals), invested=np.array(invs), flows=flows, switches=0, years=(idx[i1 - 1] - idx[i0]).days / 365.25)


def strat(d, lev, cols):
    return lambda a, b: run3(d, lev, a, b, cols=cols)


def mix(runners, weights):
    """多個策略按比例同時定投，合併現金流。"""
    def f(a, b):
        rs = [r(a, b) for r in runners]
        fl = []
        for r, w in zip(rs, weights):
            fl += [(dt, c * w) for dt, c in r["flows"][:-1]]
        fl.sort(key=lambda x: x[0])
        v = sum(w * r["value"] for r, w in zip(rs, weights)); inv = sum(w * r["invested"] for r, w in zip(rs, weights))
        fl.append((rs[0]["flows"][-1][0], v[-1]))
        return dict(value=v, invested=inv, flows=fl, switches=sum(r["switches"] for r in rs), years=rs[0]["years"])
    return f


def evaluate(label, runner, starts, P, recent="2010-03-12"):
    f = E.stats(runner(None, None)); s10 = E.stats(runner(recent, None))
    roll = pd.DataFrame([E.stats(runner(f"{y}-01-01", f"{y + 10}-01-01")) for y in starts])
    P(f"{label:40s} 全期 IRR {f['irr']:6.1%}｜2010 起 {s10['irr']:6.1%}｜10 年滾動 IRR 中位 {roll.irr.median():6.1%} 最差 {roll.irr.min():6.1%}｜"
      f"最慘相對本金 {f['worst']:5.0%}（10 年段最差 {roll.worst.min():4.0%}）｜換倉 {f['sw_per_year']:.1f}/年")
    return dict(label=label, irr=f["irr"], irr10=s10["irr"], roll_med=roll.irr.median(), roll_min=roll.irr.min(),
                worst=f["worst"], roll_worst=roll.worst.min(), sw=f["sw_per_year"])


def mc(d, n_paths=300, seed=5, B=63, warm=300, years=10):
    rng = np.random.default_rng(seed)
    lr = np.log(d.close).diff().fillna(0).values; ho = np.log(d.high / d.close).values; lo = np.log(d.low / d.close).values
    oo = np.log(d.open / d.close).values; rfv = d.rf.values
    N = warm + 252 * years; idx = pd.bdate_range("2000-01-03", periods=N)
    res = []
    for _ in range(n_paths):
        pos = []
        while len(pos) < N:
            st = rng.integers(1, len(lr) - B); pos.extend(range(st, st + B))
        pos = np.array(pos[:N]); cl = 100 * np.exp(np.cumsum(lr[pos]))
        p = pd.DataFrame(dict(close=cl, high=cl * np.exp(ho[pos]), low=cl * np.exp(lo[pos]), open=cl * np.exp(oo[pos])), index=idx)
        p["r"] = p.close.pct_change().fillna(0); p["rf"] = rfv[pos]
        for k in (2, 3):
            p[f"x{k}"] = (1 + k * p.r - E.FEE / 252 - (p.rf * 252 + E.SPREAD) * (k - 1) / 252).cumprod()
        p["bill"] = (1 + p.rf).cumprod(); p["base"] = p.close
        w = p.index.to_period("W").to_series(index=p.index); p["week_start"] = w.ne(w.shift(1)).values
        _, _, ok = signals(p); s0 = p.index[warm]
        lev = codes(p, ok)
        row = dict(dca=E.stats(dca_only(p, s0))["irr"])
        for k in (2, 3):
            st = E.stats(run3(p, lev, s0, cols=("base", "bill", f"x{k}"))); row[f"s{k}"] = st["irr"]; row[f"s{k}w"] = st["worst"]
        res.append(row)
    return pd.DataFrame(res)
