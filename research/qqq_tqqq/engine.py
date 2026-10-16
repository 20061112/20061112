"""QQQ↔TQQQ 槓桿切換定投：資料、合成 TQQQ、回測引擎、績效指標。

時點慣例（與原報告相同）：所有訊號用「前一天收盤」判斷（陣列先 shift(1)），交易在「當天收盤」成交。
引擎吃一條「目標 TQQQ 權重」序列 w_t ∈ [0, 1]（曝險 = 1 + 2w），所以二元切換（w ∈ {0,1}）與連續槓桿都能跑。
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

# 3 個月 T-bill 年平均（約略值，%），與原報告相同的處理方式
TBILL = {1999: 4.64, 2000: 5.82, 2001: 3.39, 2002: 1.60, 2003: 1.01, 2004: 1.37, 2005: 3.15,
         2006: 4.73, 2007: 4.36, 2008: 1.37, 2009: 0.15, 2010: 0.14, 2011: 0.05, 2012: 0.09,
         2013: 0.06, 2014: 0.03, 2015: 0.05, 2016: 0.32, 2017: 0.93, 2018: 1.94, 2019: 2.06,
         2020: 0.37, 2021: 0.04, 2022: 2.02, 2023: 5.07, 2024: 4.97, 2025: 4.15, 2026: 3.80}
FEE = 0.0095          # TQQQ 管理費
SPREAD = 0.0045       # 融資利差
COST = 0.0005         # 每次換倉成本（以換倉金額計）


def _read(name):
    d = pd.read_csv(os.path.join(DATA, name), parse_dates=["Date"]).set_index("Date").sort_index()
    d.columns = [c.lower() for c in d.columns]
    return d


def load():
    """回傳 DataFrame（QQQ 交易日）：open/high/low/close/volume、r（QQQ 日報酬）、rf（日利率）、
    tq_syn（合成 TQQQ 指數）、tq_real（真實 TQQQ，2010 前 NaN）、vxn（前值填補）。"""
    q = _read("qqq_us_d.csv")
    q["r"] = q.close.pct_change().fillna(0.0)
    rf_y = np.array([TBILL[y] for y in q.index.year]) / 100
    q["rf"] = rf_y / 252
    r3 = 3 * q.r - FEE / 252 - 2 * (rf_y + SPREAD) / 252
    r3.iloc[0] = 0.0
    q["tq_syn"] = (1 + r3).cumprod()
    tq = _read("tqqq_us_d.csv").close
    q["tq_real"] = tq.reindex(q.index)
    v = pd.read_csv(os.path.join(DATA, "VXNCLS.csv"), parse_dates=["observation_date"])
    v = v.set_index("observation_date").VXNCLS
    v = pd.to_numeric(v, errors="coerce")
    q["vxn"] = v.reindex(q.index).ffill()
    q["week_start"] = q.index.to_period("W").to_series(index=q.index).ne(
        q.index.to_period("W").to_series(index=q.index).shift(1)).values
    return q


# ---------------------------------------------------------------- 基準指標
def atr_z(q, n=20, look=252):
    pc = q.close.shift(1)
    tr = np.maximum(q.high - q.low, np.maximum((q.high - pc).abs(), (q.low - pc).abs()))
    atrp = np.log(tr.rolling(n).mean() / q.close)
    return (atrp - atrp.rolling(look).mean()) / atrp.rolling(look).std()


def boll_z(q, n=100):
    return (q.close - q.close.rolling(n).mean()) / q.close.rolling(n).std()


def base_signals(q, ma=200, zthr=0.5):
    """原報告規則：entry = QQQ > MA 且 ATR z < zthr；exit = 反之。皆未 shift。"""
    z = atr_z(q)
    above = q.close > q.close.rolling(ma).mean()
    ok = above & (z < zthr)
    return ok, z


def weights_from_rules(q, entry_ok, exit_bad, entry_mask=None):
    """把「每週檢查進場、每天檢查出場」轉成二元權重。entry_ok / exit_bad 為當天收盤的布林（尚未 shift）。"""
    e = entry_ok.shift(1).fillna(False).values.astype(bool)
    x = exit_bad.shift(1).fillna(True).values.astype(bool)
    ws = q.week_start.values if entry_mask is None else np.asarray(entry_mask, bool)
    w = np.zeros(len(q))
    cur = 0.0
    for i in range(len(q)):
        if cur > 0 and x[i]:
            cur = 0.0
        elif cur == 0 and ws[i] and e[i]:
            cur = 1.0
        w[i] = cur
    return w


# ---------------------------------------------------------------- 引擎
def run(q, w, start=None, end=None, tq_col="tq_syn", dca_only=False, boost=True, panic=True,
        rebalance_band=0.1, z=None, bz=None, cash_frac=0.2, boost_thr=-0.6, boost_mult=2.0,
        panic_thr=2.0, panic_asset="QQQ", dca_mask=None, off=None, cost=COST):
    """w：目標 TQQQ 權重（已是「當天收盤要持有」的值，即已用前一天訊號）。
    其他參數（預設 = 原報告）：cash_frac 每筆定投存成現金的比例；boost_thr / boost_mult 布林加碼門檻與倍數；
    panic_thr / panic_asset 恐慌補倉門檻與買進標的（"QQQ" 或 "TQQQ"）；dca_mask 哪些日子定投（預設每週第一個交易日）；
    off 布林陣列（已對齊持有日）：為 True 時 1 倍部位改持 T-bill；cost 換倉成本。
    回傳 dict：dates、value（每日市值）、invested（累計投入）、flows（現金流，用於 IRR）、switches。"""
    idx = q.index
    i0 = 0 if start is None else idx.searchsorted(pd.Timestamp(start))
    i1 = len(idx) if end is None else idx.searchsorted(pd.Timestamp(end))
    pq = q.close.values; pt = q[tq_col].values; rf = q.rf.values
    ws = q.week_start.values if dca_mask is None else np.asarray(dca_mask, bool)
    pb = np.cumprod(1 + rf)
    offv = np.zeros(len(q), bool) if off is None else np.asarray(off, bool)
    if z is None:
        z = atr_z(q)
    if bz is None:
        bz = boll_z(q)
    zp = z.shift(1).values; bzp = bz.shift(1).values
    nq = nt = nb = 0.0; cash = 0.0; inv = 0.0
    vals = np.zeros(i1 - i0); invs = np.zeros(i1 - i0); flows = []
    sw = 0; first = True
    for k, i in enumerate(range(i0, i1)):
        cash *= 1 + rf[i]
        if ws[i] or first:
            first = False
            amt = 1.0
            if not dca_only and boost and np.isfinite(bzp[i]) and bzp[i] <= boost_thr:
                amt = boost_mult
            inv += amt; flows.append((idx[i], -amt))
            if dca_only:
                nq += amt / pq[i]
            else:
                put = (1 - cash_frac) * amt; cash += cash_frac * amt
                wt = w[i] if np.isfinite(pt[i]) else 0.0
                if offv[i]:
                    nb += put * (1 - wt) / pb[i]
                else:
                    nq += put * (1 - wt) / pq[i]
                if wt > 0:
                    nt += put * wt / pt[i]
                if panic and np.isfinite(zp[i]) and zp[i] > panic_thr and cash > 0:
                    if panic_asset == "TQQQ" and np.isfinite(pt[i]):
                        nt += cash / pt[i]
                    else:
                        nq += cash / pq[i]
                    cash = 0.0
        if not dca_only:
            vq = nq * pq[i]; vt = nt * pt[i] if nt > 0 else 0.0; vb = nb * pb[i]
            tot = vq + vt + vb
            wt = w[i] if np.isfinite(pt[i]) else 0.0
            cw = vt / tot if tot > 0 else 0.0
            need = (wt in (0.0, 1.0) and cw != wt and abs(cw - wt) > 1e-9) or abs(cw - wt) > rebalance_band
            wrong_base = (offv[i] and vq > 1e-12) or (not offv[i] and vb > 1e-12)
            if tot > 0 and (need or wrong_base):
                trade = abs(wt - cw) * tot + (vq if offv[i] else vb)
                tot -= trade * cost
                base = tot * (1 - wt)
                nq, nb = (0.0, base / pb[i]) if offv[i] else (base / pq[i], 0.0)
                nt = tot * wt / pt[i] if wt > 0 else 0.0
                sw += 1
        vals[k] = nq * pq[i] + (nt * pt[i] if nt > 0 else 0.0) + nb * pb[i] + cash
        invs[k] = inv
    flows.append((idx[i1 - 1], vals[-1]))
    return dict(dates=idx[i0:i1], value=vals, invested=invs, flows=flows, switches=sw,
                years=(idx[i1 - 1] - idx[i0]).days / 365.25)


def xirr(flows):
    t0 = flows[0][0]
    ts = np.array([(d - t0).days / 365.25 for d, _ in flows]); cf = np.array([c for _, c in flows])
    f = lambda r: np.sum(cf / (1 + r) ** ts)
    lo, hi = -0.99, 5.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return mid


def stats(res):
    v, inv = res["value"], res["invested"]
    return dict(irr=xirr(res["flows"]), gain=v[-1] / inv[-1] - 1, worst=np.min(v / inv - 1),
                sw_per_year=res["switches"] / max(res["years"], 1e-9))


def rolling_windows(q, w, years=10, starts=range(2000, 2017), **kw):
    out = []
    for y in starts:
        s = pd.Timestamp(f"{y}-01-01"); e = pd.Timestamp(f"{y + years}-01-01")
        if e > q.index[-1] + pd.Timedelta(days=5):
            break
        out.append(stats(run(q, w, s, e, **kw)) | dict(start=y))
    return pd.DataFrame(out)


def calendar_returns(q, w, **kw):
    """策略整體部位（不含定投現金流的影響很難去除）→ 改用「一次投入、不定投」的日報酬計算逐年報酬。"""
    rq = q.r.values; r3 = q.tq_syn.pct_change().fillna(0).values
    wt = np.nan_to_num(w)
    wp = np.r_[0.0, wt[:-1]]                       # 當天報酬由前一天收盤的持倉賺
    chg = np.r_[0, np.abs(np.diff(wt))]
    rs = (1 - wp) * rq + wp * r3 - chg * COST
    s = pd.Series(rs, index=q.index)
    return (1 + s).groupby(s.index.year).prod() - 1, s
