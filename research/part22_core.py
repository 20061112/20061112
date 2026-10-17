"""第二十二部分（核心）：QQQ↔3 倍槓桿 定投策略 —— 訊號重建 + 四種執行引擎。

訊號（全部用「前一天收盤」為止的資料，在當天收盤成交）：
  每週第一個交易日：投入 base（布林 z100 ≤ −0.6 時加倍），80% 進場、20% 存現金（賺 T-bill）
  每週：持 1 倍、QQQ > MA200、ATR z < 0.5 → 開槓桿
  每天：QQQ < MA200 或 ATR z ≥ 0.5 → 關槓桿
  每週：ATR z > 2 → 存下的現金全部買 1 倍（QQQ）
  ATR z = (ln ATR20% − 252 日平均) / 252 日標準差

引擎
  dca    純週定投 QQQ（基準）
  etf    QQQ ↔ 合成 3 倍 ETF（3×日報酬 − 0.95% − 2×(T-bill+0.45%)），換倉 0.05%
  fut    QQQ 現股 + Nasdaq-100 期貨疊加：開槓桿時期貨名目 = L × 部位淨值 − QQQ 市值
  allfut 全期貨：部位淨值全放 T-bill 當保證金，期貨名目 = 1 × 淨值（平時）/ L × 淨值（開槓桿）
期貨每日結算進「期貨帳戶現金」；融資成本 = T-bill + basis（期貨隱含）；每季轉倉扣成本。
"""
import numpy as np
import pandas as pd

# 3 個月 T-bill 年平均（%，約略值；FRED DTB3 年均四捨五入）
TBILL = {1999: 4.64, 2000: 5.82, 2001: 3.39, 2002: 1.60, 2003: 1.01, 2004: 1.37, 2005: 3.15, 2006: 4.73,
         2007: 4.36, 2008: 1.37, 2009: 0.15, 2010: 0.14, 2011: 0.05, 2012: 0.09, 2013: 0.06, 2014: 0.03,
         2015: 0.05, 2016: 0.32, 2017: 0.93, 2018: 1.94, 2019: 2.06, 2020: 0.37, 2021: 0.05, 2022: 2.02,
         2023: 5.07, 2024: 4.97, 2025: 4.10, 2026: 3.70}
NDX_PER_QQQ = 41.0                       # NDX ≈ 41 × QQQ（近年比值）
# 合約：乘數（美元/點）、單邊成本（手續費 + 0.25 點滑價）
CONTRACTS = {"MNQ": (2.0, 0.62 + 0.25 * 2.0), "NQ": (20.0, 2.25 + 0.25 * 20.0),
             "MES": (5.0, 0.62 + 0.25 * 5.0), "ES": (50.0, 2.25 + 0.25 * 50.0),
             "MYM": (0.5, 0.62 + 1.0 * 0.5), "YM": (5.0, 2.25 + 1.0 * 5.0)}
# 指數 / ETF 比值：NDX ≈ 41 × QQQ、SPX ≈ 10 × SPY、DJIA ≈ 100 × DIA
TODAY_MNQ = 2.0 * NDX_PER_QQQ * 709.80   # 2026/10/16 一口 MNQ 名目 ≈ 58,000 美元


def load(path="data/qqq_us_d.csv"):
    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
    return df.rename(columns=str.lower)


def signals(df, ma=200, zwin=252, z_on=0.5, z_panic=2.0, bwin=100, bthr=-0.6, atrn=20):
    c, h, l = df.close, df.high, df.low
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atrp = tr.rolling(atrn).mean() / c
    la = np.log(atrp)
    z = (la - la.rolling(zwin).mean()) / la.rolling(zwin).std()
    m200 = c.rolling(ma).mean()
    bz = (c - c.rolling(bwin).mean()) / c.rolling(bwin).std()
    S = pd.DataFrame(index=df.index)
    S["close"] = c
    S["ret"] = c.pct_change().fillna(0.0)
    S["up"] = (c > m200).shift(1).fillna(False).astype(bool) & m200.shift(1).notna()
    S["z"] = z.shift(1)
    S["bz"] = bz.shift(1)
    S["atrp"] = atrp.bfill()                                   # 當天收盤可得，用於保證金估算
    wk = S.index.to_period("W-SUN")
    S["ws"] = np.r_[True, wk[1:] != wk[:-1]]
    rf = np.array([TBILL[y] for y in S.index.year]) / 100
    S["rf_ann"] = rf
    S["rf"] = rf / 252
    # 保證金率估算：max(5%, 3.5 × ATR20%)（CME SPAN 大致跟著波動走）
    S["mrate"] = np.maximum(0.05, 3.5 * S.atrp)
    # 轉倉日：3、6、9、12 月第三個星期五前 8 天之後的第一個交易日
    roll = np.zeros(len(S), bool)
    for y in range(S.index.year.min(), S.index.year.max() + 1):
        for m in (3, 6, 9, 12):
            d = pd.Timestamp(y, m, 1)
            third_fri = d + pd.Timedelta(days=(4 - d.dayofweek) % 7 + 14)
            k = S.index.searchsorted(third_fri - pd.Timedelta(days=8))
            if k < len(S):
                roll[k] = True
    S["roll"] = roll
    S["on"] = regime(S, z_on)
    S.attrs.update(z_panic=z_panic, bthr=bthr)
    return S


def regime(S, z_on=0.5):
    """on[i]：第 i 天收盤後是否持有槓桿（進場每週檢查、出場每天檢查）。"""
    up, z, ws = S.up.values, S.z.values, S.ws.values
    on, out = False, np.zeros(len(S), bool)
    for i in range(len(S)):
        ok = up[i] and z[i] == z[i]
        if on and (not ok or z[i] >= z_on):
            on = False
        if ws[i] and not on and ok and z[i] < z_on:
            on = True
        out[i] = on
    return out


def run(S, eng="fut", i0=0, i1=None, L=3.0, rebal="weekly", band=0.25, fund="borrow", basis=0.0025,
        spread=0.01, buffer=1.5, contract=None, start_notional=None, base=1000.0, lump=0.0,
        regime_arr=None, switch_cost=0.0005, boll=True, panic=True, cash_frac=0.2, shock=None,
        L_arr=None, fut_ret=None, fut_name="MNQ", ratio=NDX_PER_QQQ, panic_arr=None, boost_arr=None):
    """回傳 dict：V（每日淨值）、flows、lev（每日總曝險/部位淨值）、twr（每日時間加權報酬）及成本統計。

    contract=None 為可分割（理想）合約；"MNQ"/"NQ" 為整數口數。
    start_notional：設定後，一口合約在 i0 的名目 = start_notional，之後隨 QQQ 價格變動（模擬「從今天的合約大小開始」）。
    fund：borrow（期貨帳戶可為負，借款利率 T-bill + spread）/ sell（維持現金 ≥ buffer × 保證金，不足時賣 QQQ）。
    shock：(日期索引, 跌幅) —— 在該日額外加一個單日跌幅（壓力測試）。
    L_arr：每天開槓桿時的目標倍數（覆蓋 L）；fut_ret：期貨部位當天的報酬（覆蓋 QQQ 日報酬，用於盤中停損）。
    panic_arr：每天是否觸發恐慌補倉（覆蓋 ATR z > 2）；boost_arr：當週是否加倍投入（覆蓋布林 z ≤ −0.6）。
    """
    i1 = len(S) if i1 is None else i1
    P, R, rf, rfa = S.close.values, S.ret.values.copy(), S.rf.values, S.rf_ann.values
    ws, z, bz, mr, roll = S.ws.values, S.z.values, S.bz.values, S.mrate.values, S.roll.values
    reg = S.on.values if regime_arr is None else regime_arr
    RF = R if fut_ret is None else fut_ret
    zp, bthr = S.attrs["z_panic"], S.attrs["bthr"]
    if shock is not None:
        R[shock[0]] = (1 + R[shock[0]]) * (1 - shock[1]) - 1
        P = P.copy(); P[shock[0]:] = P[shock[0]:] * (1 - shock[1])
    mult, side = CONTRACTS[contract or fut_name]
    if start_notional is not None:
        cn = start_notional * P / P[i0]
    else:
        cn = mult * ratio * P
    integer = contract is not None
    overlay, allf = eng == "fut", eng == "allfut"

    q = acct = res = n = u = 0.0
    T = 1.0
    inv = 0.0
    flows, V, lev, twr = [], np.zeros(i1 - i0), np.zeros(i1 - i0), np.zeros(i1 - i0)
    cost_fut = cost_roll = cost_qqq = interest_paid = 0.0
    min_acct = 0.0
    switches = 0
    prev_on, Vprev = False, 0.0
    for k, i in enumerate(range(i0, i1)):
        p = P[i]
        if k:
            r, d = R[i], rf[i]
            res += res * d
            if acct >= 0:
                acct += acct * d
            else:
                ip = -acct * (d + spread / 252); acct -= ip; interest_paid += ip
            if n:
                acct += n * cn[i - 1] * (RF[i] - d - basis / 252)
            T *= 1 + 3 * r - (0.0095 + 2 * (rfa[i] + 0.0045)) / 252
        E = q * p + acct + u * T + res
        twr[k] = E / Vprev - 1 if Vprev > 0 else 0.0
        on = bool(reg[i]) if eng != "dca" else False

        # 1) 換倉（etf）
        if eng == "etf" and on != prev_on:
            switches += 1
            if on:
                u = q * p * (1 - switch_cost) / T; q = 0.0
            else:
                q = u * T * (1 - switch_cost) / p; u = 0.0
        # 2) 定投與恐慌補倉
        if ws[i]:
            dbl = (bz[i] <= bthr) if boost_arr is None else boost_arr[i]
            amt = base * (2 if (boll and eng != "dca" and dbl) else 1) + (lump if k == 0 else 0.0)
            inv += amt; flows.append((S.index[i], amt))
            if eng == "dca":
                q += amt / p
            else:
                put = (1 - cash_frac) * amt; res += cash_frac * amt
                if panic and (z[i] > zp if panic_arr is None else panic_arr[i]):
                    put += res; res = 0.0
                if eng == "etf":
                    if on:
                        u += put / T
                    else:
                        q += put / p
                elif overlay:
                    q += put / p
                else:
                    acct += put
        # 3) 期貨部位
        if overlay or allf:
            if on != prev_on:
                switches += 1
            base_lev = (L if L_arr is None else L_arr[i]) if on else (1.0 if allf else 0.0)
            per = side                                 # 每口單邊成本（可分割合約時 n 以 MNQ 口為單位）
            if overlay and not on:
                if n:
                    c_ = abs(n) * per; cost_fut += c_; acct -= c_
                    n = 0.0
                if acct:                               # 出場：期貨帳戶結清回 QQQ
                    c_ = abs(acct) * switch_cost; cost_qqq += c_
                    q += (acct - c_) / p; acct = 0.0
            else:
                spot = q * p
                Einv = spot + acct
                cur = (spot + n * cn[i]) / Einv if Einv > 0 else 0.0
                need = (on != prev_on or rebal == "daily" or (rebal == "weekly" and ws[i])
                        or (rebal == "band" and abs(cur - base_lev) > band)
                        or (allf and n == 0))
                if overlay and fund == "sell":
                    N = max(base_lev * Einv - spot, 0.0)
                    b = buffer * mr[i]
                    x = (b * N - acct) / (1 - b) if need else b * abs(n) * cn[i] - acct
                    if x > 0:                         # 賣 QQQ 補保證金
                        c_ = x * switch_cost; cost_qqq += c_
                        q -= x / p; acct += x - c_; spot = q * p
                    elif need and ws[i] and acct > 2 * b * N:   # 多餘現金掃回 QQQ（期貨名目同步減少）
                        y = (acct - b * N) / (1 - b)
                        q += y / p; acct -= y; spot = q * p
                if need:
                    Nt = max(base_lev * (spot + acct) - spot, 0.0)
                    nt = Nt / cn[i]
                    if integer:
                        nt = float(np.floor(nt + 0.5))
                    c_ = abs(nt - n) * per
                    cost_fut += c_; acct -= c_
                    n = nt
            if roll[i] and n:                          # 價差單：兩邊手續費 + 1 tick
                c_ = abs(n) * per * (2 * side - 0.25 * mult) / side
                cost_roll += c_; acct -= c_
        Vt = q * p + acct + u * T + res
        if (overlay or allf) and Vt > 0:
            min_acct = min(min_acct, acct / Vt)
        pos = Vt - res
        lev[k] = ((q * p + n * cn[i] + 3 * u * T) / pos) if pos > 0 else 0.0
        V[k] = Vt; Vprev = Vt
        prev_on = on
    return dict(V=pd.Series(V, index=S.index[i0:i1]), flows=flows, lev=lev, twr=twr, inv=inv,
                cost_fut=cost_fut, cost_roll=cost_roll, cost_qqq=cost_qqq, interest=interest_paid,
                min_acct=min_acct, switches=switches)


def xirr(flows, end_date, end_value):
    t0 = flows[0][0]
    ts = np.array([(d - t0).days / 365.25 for d, _ in flows] + [(end_date - t0).days / 365.25])
    cf = np.array([-a for _, a in flows] + [end_value])
    f = lambda r: (cf / (1 + r) ** ts).sum()
    lo, hi = -0.99, 5.0
    if f(lo) * f(hi) > 0:
        return np.nan
    for _ in range(200):
        mid = (lo + hi) / 2
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def metrics(out):
    V, flows = out["V"], out["flows"]
    cum = pd.Series(0.0, index=V.index)
    for d, a in flows:
        cum[d] += a
    cum = cum.cumsum()
    rel = (V / cum - 1)[cum > 0]
    tw = (1 + pd.Series(out["twr"], index=V.index)).cumprod()
    yrs = (V.index[-1] - V.index[0]).days / 365.25
    return dict(irr=xirr(flows, V.index[-1], V.iloc[-1]), tot=V.iloc[-1] / out["inv"] - 1, worst=rel.min(),
                mdd=(tw / tw.cummax() - 1).min(), avg_lev=float(np.mean(out["lev"])),
                sw_per_yr=out["switches"] / max(yrs, 1e-9), min_acct=out["min_acct"],
                cost_pct=(out["cost_fut"] + out["cost_roll"] + out["cost_qqq"] + out["interest"]) / V.iloc[-1])


def year_starts(S, first=2000, last=2016):
    return {y: S.index.searchsorted(pd.Timestamp(y, 1, 1)) for y in range(first, last + 2 + 9)}
