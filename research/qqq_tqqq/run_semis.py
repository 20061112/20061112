"""同一套策略搬到半導體（SOXX）。

底倉 SOXX；平靜上升（SOXX > MA200 且 ATR z < 0.5）時換成槓桿版。其他規則與原策略相同
（每週進場、每天出場、80% 定投 + 20% 現金、布林加碼、ATR z > 2 現金補倉）。
槓桿版用 SOXX 日報酬合成（2 倍 / 3 倍，管理費 0.95%、融資 T-bill + 0.45%），因為 USD 追蹤的是 Dow Jones 美國半導體指數、
不是 SOXX 的指數；先用真實 USD（2007 起）、SOXS（2010 起）校準。
比較：純定投 SOXX、純定投 QQQ、QQQ 原策略（同期間）、SOXX 策略 2 倍 / 3 倍、跌破 MA200 改持 SOXS、雙動能、用 QQQ 的訊號操作 SOXX、
QQQ 策略與 SOXX 策略各半。
輸出 results_semis.txt、results_semis.csv。
"""
import numpy as np
import pandas as pd
import engine as E
from run_qld import run3

out = []
P = lambda s: (print(s, flush=True), out.append(s))


def build(ticker):
    d = E._read(f"{ticker}_us_d.csv")[["open", "high", "low", "close"]].copy()
    d["r"] = d.close.pct_change().fillna(0.0)
    rf_y = np.array([E.TBILL[y] for y in d.index.year]) / 100
    d["rf"] = rf_y / 252
    for k in (2, 3, -3):
        lev = k - 1 if k > 0 else k - 1       # 融資倍數：2 倍借 1、3 倍借 2；−3 倍放空 3 倍、現金收益 = 1 + 3 倍
        fin = (rf_y + E.SPREAD) * (k - 1) if k > 0 else -(rf_y) * (1 - k)
        rk = k * d.r - E.FEE / 252 - fin / 252
        rk.iloc[0] = 0.0
        d[f"x{k}"] = (1 + rk).cumprod()
    d["bill"] = (1 + d.rf).cumprod()
    p = d.index.to_period("W").to_series(index=d.index)
    d["week_start"] = p.ne(p.shift(1)).values
    return d


s = build("soxx")
qq = build("qqq").reindex(s.index)
P(f"資料：SOXX {s.index[0].date()} ~ {s.index[-1].date()}")

# ---------------- 校準
for name, col, k in [("USD（2 倍，Dow Jones 半導體指數）", "usd", 2), ("SOXS（−3 倍）", "soxs", -3)]:
    real = E._read(f"{col}_us_d.csv").close.reindex(s.index)
    m = real.notna() & (real > 0)
    rr = real[m].pct_change().dropna(); rs = s[f"x{k}"][m].pct_change().dropna()
    rr, rs = rr[rr.abs() < 0.6], rs.reindex(rr[rr.abs() < 0.6].index)
    yrs = (rr.index[-1] - rr.index[0]).days / 365.25
    cg = lambda x: (1 + x).prod() ** (1 / yrs) - 1
    P(f"校準 {name}：{rr.index[0].date()} 起 真實年化 {cg(rr):+.1%}、合成 {cg(rs):+.1%}、日報酬相關 {np.corrcoef(rr, rs)[0, 1]:.3f}、"
      f"追蹤誤差 {(rr - rs).std() * np.sqrt(252):.1%}")
vol_s = s.r.std() * np.sqrt(252); vol_q = qq.r.std() * np.sqrt(252)
P(f"年化波動：SOXX {vol_s:.0%}、QQQ（同期） {vol_q:.0%}；3 倍每年波動耗損約 3σ²：SOXX {3 * vol_s ** 2:.0%}、QQQ {3 * vol_q ** 2:.0%}")


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


def dca_only(d, start=None, end=None):
    """純定投（每週 1 單位、不加碼、無現金）。"""
    idx = d.index; i0 = 0 if start is None else idx.searchsorted(pd.Timestamp(start))
    i1 = len(idx) if end is None else idx.searchsorted(pd.Timestamp(end))
    px = d.close.values; ws = d.week_start.values; units = 0.0; inv = 0.0; flows = []; vals = []; invs = []
    for k, i in enumerate(range(i0, i1)):
        if ws[i] or k == 0:
            units += 1 / px[i]; inv += 1; flows.append((idx[i], -1.0))
        vals.append(units * px[i]); invs.append(inv)
    flows.append((idx[i1 - 1], vals[-1]))
    return dict(value=np.array(vals), invested=np.array(invs), flows=flows, switches=0, years=(idx[i1 - 1] - idx[i0]).days / 365.25)


STARTS = list(range(2006, 2017))


def evaluate(label, runner):
    f = E.stats(runner(None, None))
    s10 = E.stats(runner("2010-03-12", None))
    roll = pd.DataFrame([E.stats(runner(f"{y}-01-01", f"{y + 10}-01-01")) | dict(start=y) for y in STARTS])
    res = runner(None, None)
    row = dict(label=label, irr=f["irr"], irr10=s10["irr"], worst=f["worst"], roll_med=roll.irr.median(), roll_min=roll.irr.min(),
               roll_worst=roll.worst.min(), g10_mean=roll.gain.mean(), sw=f["sw_per_year"])
    P(f"{label:40s} 2005 起 IRR {f['irr']:6.1%}｜2010 起 {s10['irr']:6.1%}｜10 年滾動 IRR 中位 {roll.irr.median():6.1%} 最差 {roll.irr.min():6.1%}｜"
      f"最慘相對本金 {f['worst']:5.0%}（10 年段最差 {roll.worst.min():4.0%}）｜換倉 {f['sw_per_year']:.1f}/年")
    return row, roll


rows = []
ab, zb, okb = signals(s)
qa, qz, qok = signals(qq)
codes = lambda ok: timing(s, pd.Series(np.where(ok, 3, 1), index=s.index))
s["soxx"] = s.close


def strat(d, lev, cols):
    return lambda a, b: run3(d, lev, a, b, cols=cols)


P("\n同期間（2005/2 起）比較")
r, _ = evaluate("純定投 SOXX", lambda a, b: dca_only(s, a, b)); rows.append(r)
qq2 = qq.copy(); qq2["qqq"] = qq2.close
r, _ = evaluate("純定投 QQQ", lambda a, b: dca_only(qq2, a, b)); rows.append(r)
r, _ = evaluate("QQQ 原策略（QQQ ↔ 3 倍）", strat(qq2, timing(qq2, pd.Series(np.where(qok, 3, 1), index=qq2.index)), ("qqq", "bill", "x3"))); rows.append(r)
for k in (2, 3):
    r, _ = evaluate(f"SOXX 策略（SOXX ↔ {k} 倍）", strat(s, codes(okb), ("soxx", "bill", f"x{k}"))); rows.append(r)
for zt in (0.0,):
    ok0 = ab & (zb < zt)
    for k in (2, 3):
        r, _ = evaluate(f"SOXX 策略 {k} 倍，ATR z < {zt}", strat(s, codes(ok0), ("soxx", "bill", f"x{k}"))); rows.append(r)
# 用 QQQ 的訊號
for k in (2, 3):
    r, _ = evaluate(f"SOXX ↔ {k} 倍，用 QQQ 的訊號", strat(s, codes(qok.fillna(False)), ("soxx", "bill", f"x{k}"))); rows.append(r)
# 跌破 MA200 改持 SOXS
lev_s = timing(s, pd.Series(np.where(okb, 3, np.where(~ab, 2, 1)), index=s.index))
r, _ = evaluate("SOXX 策略 2 倍 + 跌破 MA200 持 SOXS", strat(s, lev_s, ("soxx", "x-3", "x2"))); rows.append(r)
# 雙動能：跌破 MA200 且 12 月報酬 < T-bill → T-bill
dm = (~ab) & (s.close / s.close.shift(252) - 1 < s.rf.rolling(252).sum())
lev_dm = timing(s, pd.Series(np.where(okb, 3, np.where(dm, 2, 1)), index=s.index))
for k in (2, 3):
    r, _ = evaluate(f"SOXX 策略 {k} 倍 + 雙動能", strat(s, lev_dm, ("soxx", "bill", f"x{k}"))); rows.append(r)

# 各半：QQQ 原策略 + SOXX 2 倍策略
def half(a, b):
    r1 = run3(qq2, timing(qq2, pd.Series(np.where(qok, 3, 1), index=qq2.index)), a, b, cols=("qqq", "bill", "x3"))
    r2 = run3(s, codes(okb), a, b, cols=("soxx", "bill", "x2"))
    fl = [(d, c * 0.5) for d, c in r1["flows"][:-1]] + [(d, c * 0.5) for d, c in r2["flows"][:-1]]
    fl.sort(key=lambda x: x[0])
    end = r1["flows"][-1][0]
    v = 0.5 * (r1["value"] + r2["value"]); inv = 0.5 * (r1["invested"] + r2["invested"])
    fl.append((end, v[-1]))
    return dict(value=v, invested=inv, flows=fl, switches=r1["switches"] + r2["switches"], years=r1["years"])
r, _ = evaluate("一半 QQQ 原策略 + 一半 SOXX 2 倍策略", half); rows.append(r)

pd.DataFrame(rows).to_csv("results_semis.csv", index=False, float_format="%.4f")

# ---------------- 蒙地卡羅
import run_ideas_mc as MC  # noqa: E402（只借用 path 的做法）
rng = np.random.default_rng(5)
lr = np.log(s.close).diff().fillna(0).values; ho = np.log(s.high / s.close).values; lo = np.log(s.low / s.close).values
oo = np.log(s.open / s.close).values; rfv = s.rf.values
B = 63; WARM = 300; N = WARM + 2520; idx = pd.bdate_range("2000-01-03", periods=N)


def path():
    pos = []
    while len(pos) < N:
        st = rng.integers(1, len(lr) - B); pos.extend(range(st, st + B))
    pos = np.array(pos[:N]); cl = 100 * np.exp(np.cumsum(lr[pos]))
    d = pd.DataFrame(dict(close=cl, high=cl * np.exp(ho[pos]), low=cl * np.exp(lo[pos]), open=cl * np.exp(oo[pos])), index=idx)
    d["r"] = d.close.pct_change().fillna(0); d["rf"] = rfv[pos]
    for k in (2, 3):
        d[f"x{k}"] = (1 + k * d.r - E.FEE / 252 - (d.rf * 252 + E.SPREAD) * (k - 1) / 252).cumprod()
    d["bill"] = (1 + d.rf).cumprod(); d["soxx"] = d.close
    p = d.index.to_period("W").to_series(index=d.index); d["week_start"] = p.ne(p.shift(1)).values
    return d


res = []
for k in range(300):
    d = path(); a_, z_, ok_ = signals(d); s0 = d.index[WARM]
    lev = timing(d, pd.Series(np.where(ok_, 3, 1), index=d.index))
    row = dict(dca=E.stats(dca_only(d, s0))["irr"])
    for kk in (2, 3):
        st = E.stats(run3(d, lev, s0, cols=("soxx", "bill", f"x{kk}"))); row[f"s{kk}"] = st["irr"]; row[f"s{kk}w"] = st["worst"]
    res.append(row)
M = pd.DataFrame(res)
P("\n區塊拼貼蒙地卡羅（SOXX 歷史、3 個月區塊、300 條 10 年路徑）：年化 IRR")
P(f"  純定投 SOXX：中位 {M.dca.median():.1%}、5% {M.dca.quantile(.05):.1%}、95% {M.dca.quantile(.95):.1%}")
for kk in (2, 3):
    d_ = M[f"s{kk}"] - M.dca
    P(f"  SOXX 策略 {kk} 倍：中位 {M[f's{kk}'].median():.1%}、5% {M[f's{kk}'].quantile(.05):.1%}、95% {M[f's{kk}'].quantile(.95):.1%}｜"
      f"勝過純定投 {(d_ > 0).mean():.0%}、超額中位 {d_.median():+.1%}｜最慘相對本金 中位 {M[f's{kk}w'].median():.0%}、5% {M[f's{kk}w'].quantile(.05):.0%}")
d_ = M.s2 - M.s3
P(f"  2 倍 − 3 倍：中位 {d_.median():+.1%}、2 倍較好的比例 {(d_ > 0).mean():.0%}")
open("results_semis.txt", "w").write("\n".join(out) + "\n")
