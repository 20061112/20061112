"""第二十二部分：預先排定的數據發布（NFP / CPI / FOMC，對照：初領失業金）前後的行為，
以及 ER、IB、TPO、Pivot、Entropy 在發布當下有沒有規律。

資料：data/XAUUSD_M1_2023_2026.csv（2023/6/1 ~ 2026/10/8，broker = 美東 + 7）。日曆：part22_calendar.py。
所有幅度都除以 ATR10（前 10 個交易日真實波幅平均，不含當天），2023 年金價 1,900 和 2026 年 4,000+ 才能放在一起比。
發布當下的因子一律只用發布前一分鐘（含）以前的 M1。
p(k) = 發布後第 k 分鐘那根 K 棒的收盤；p(0) = 發布前一根（t−1）的收盤；p(−60) = t−61 的收盤。
對照組：同一時刻、同星期幾、當天沒有 NFP/CPI/FOMC/初領的交易日（其他沒列入的數據仍可能在對照組裡 → 對照偏保守）。
"""
import math
import numpy as np
import pandas as pd
from load import load
from tpo import tpo_profile, poc_va
from part22_calendar import calendar

out = open("results_part22.txt", "w")
def P(s=""): print(s); out.write(s + "\n"); out.flush()

m1 = load("data/XAUUSD_M1_2023_2026.csv")
m1["day"] = (m1.index - pd.Timedelta(hours=1)).normalize()
m1["period"] = ((m1.index.hour - 1) * 60 + m1.index.minute) // 30
T = m1.index.values
C = m1.close.values; H = m1.high.values; L = m1.low.values; O = m1.open.values; SP = m1.spread.values * 0.01

# ---------- 日線：ATR10、前日 H/L/C、Pivot ----------
D = m1.groupby("day").agg(o=("open", "first"), h=("high", "max"), l=("low", "min"), c=("close", "last"), n=("close", "size"))
D = D[D.n >= 600]
tr = pd.concat([D.h - D.l, (D.h - D.c.shift()).abs(), (D.l - D.c.shift()).abs()], axis=1).max(axis=1)
D["atr"] = tr.rolling(10).mean().shift(1)
D["ph"], D["pl"], D["pc"], D["po"] = D.h.shift(), D.l.shift(), D.c.shift(), D.o.shift()
D["P"] = (D.ph + D.pl + D.pc) / 3
D["R1"], D["S1"] = 2 * D.P - D.pl, 2 * D.P - D.ph
D["R2"], D["S2"] = D.P + (D.ph - D.pl), D.P - (D.ph - D.pl)

# 前日 TPO（30 分字母）POC / VAH / VAL
prev_va = {}
days = list(D.index)
for i, (day, g) in enumerate(m1.groupby("day")):
    if day not in D.index:
        continue
    pr, ct, _ = tpo_profile(g)
    prev_va[day] = poc_va(pr, ct)
D["ppoc"] = [prev_va.get(days[i - 1], (np.nan,) * 3)[0] if i else np.nan for i in range(len(days))]
D["pvah"] = [prev_va.get(days[i - 1], (np.nan,) * 3)[1] if i else np.nan for i in range(len(days))]
D["pval"] = [prev_va.get(days[i - 1], (np.nan,) * 3)[2] if i else np.nan for i in range(len(days))]


def pos(t):
    return np.searchsorted(T, np.datetime64(t))


def er(c):
    p = np.abs(np.diff(c)).sum()
    return (c[-1] - c[0]) / p if p > 0 else 0.0


def sign_entropy(c):
    r = np.diff(c); r = r[r != 0]
    if len(r) < 10:
        return np.nan
    p = (r > 0).mean()
    return 0.0 if p in (0, 1) else -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def perm_entropy(c, m=3):
    """排列熵（order 3），0 = 完全規律、1 = 完全隨機。"""
    x = np.lib.stride_tricks.sliding_window_view(c, m)
    pat = np.argsort(x, axis=1) @ (m ** np.arange(m))
    _, cnt = np.unique(pat, return_counts=True)
    p = cnt / cnt.sum()
    return -(p * np.log(p)).sum() / np.log(math.factorial(m))


def features(t, horizon_end=None):
    """t = 發布分鐘（broker）。回傳因子與結果，或 None。"""
    i = pos(t)
    if i >= len(T) or T[i] != np.datetime64(t) or i < 300:
        return None
    day = m1.day.iloc[i]
    if day not in D.index or np.isnan(D.at[day, "atr"]):
        return None
    d = D.loc[day]; atr = d.atr
    p0 = C[i - 1]
    f = dict(t=t, day=day, atr=atr, year=t.year)

    # ---- 結果（之後的路徑）----
    end_i = np.searchsorted(T, np.datetime64(day + pd.Timedelta(hours=23, minutes=58)))
    def pk(k):
        j = i + k - 1
        return C[j] if j < end_i and j < len(C) else np.nan
    for k in (1, 5, 15, 30, 60, 120, 240):
        f[f"r{k}"] = (pk(k) - p0) / atr
    f["r_eod"] = (C[end_i - 1] - p0) / atr
    for k in (15, 60):                       # 發布前的漂移
        f[f"pre{k}"] = (p0 - C[i - 1 - k]) / atr
    for a, b in ((0, 30), (0, 60), (0, 120)):
        w = slice(i + a, min(i + b, end_i))
        f[f"rng{a}_{b}"] = (H[w].max() - L[w].min()) / atr
    w = slice(i - 60, i)
    f["rng_pre60"] = (H[w].max() - L[w].min()) / atr
    f["absret_pre60"] = np.abs(np.diff(C[i - 61:i])).mean() / atr
    f["sp_pre"] = np.median(SP[i - 30:i]); f["sp0"] = SP[i]; f["sp_max5"] = SP[i:i + 5].max()

    # ---- 發布前的因子 ----
    for n in (30, 60, 240):
        f[f"er{n}"] = er(C[i - n - 1:i])
    day_i = np.searchsorted(T, np.datetime64(day + pd.Timedelta(hours=1)))
    f["er_day"] = er(C[day_i:i]) if i - day_i > 30 else np.nan
    f["day_move"] = (p0 - O[day_i]) / atr
    f["ent60"] = sign_entropy(C[i - 61:i])
    f["pent60"] = perm_entropy(C[i - 61:i])
    f["pent120"] = perm_entropy(C[i - 121:i])

    # IB：當天 LDN 10:00 起 60 分鐘、以及「發布前 60 分鐘」當成新聞 IB
    li = np.searchsorted(T, np.datetime64(day + pd.Timedelta(hours=10)))
    lj = np.searchsorted(T, np.datetime64(day + pd.Timedelta(hours=11)))
    if lj < i:
        ibh, ibl = H[li:lj].max(), L[li:lj].min()
        f["ldn_ib"] = (ibh - ibl) / atr
        f["ldn_ib_pos"] = (p0 - (ibh + ibl) / 2) / ((ibh - ibl) / 2)          # IB 單位
    # 發展中 TPO（30 分字母，當天 01:00 到發布前）
    g = m1.iloc[day_i:i]
    pr, ct, _ = tpo_profile(g)
    poc, vah, val = poc_va(pr, ct)
    f["tpo_pos"] = (p0 - (poc + 0.5)) / atr
    f["tpo_va_pos"] = (p0 - (vah + val + 1) / 2) / max((vah + 1 - val) / 2, 1e-9)   # VA 單位：±1 = VA 邊緣
    f["prev_va_pos"] = (p0 - (d.pvah + d.pval + 1) / 2) / max((d.pvah + 1 - d.pval) / 2, 1e-9)
    # Pivot：距離最近的樞紐點、位於哪一區
    piv = {k: d[k] for k in ("S2", "S1", "P", "R1", "R2")}
    f["piv_pos"] = (p0 - d.P) / (d.R1 - d.P if p0 >= d.P else d.P - d.S1)        # P=0, R1=+1, S1=−1
    near = min(piv, key=lambda k: abs(p0 - piv[k]))
    f["piv_near"], f["piv_dist"] = near, abs(p0 - piv[near]) / atr
    # 之後當天有沒有碰到 R1 / S1 / P
    post = slice(i, end_i)
    f["hit_R1"] = H[post].max() >= d.R1; f["hit_S1"] = L[post].min() <= d.S1
    f["hit_P"] = (L[post].min() <= d.P) & (H[post].max() >= d.P)
    f["hit_ppoc"] = (L[post].min() <= d.ppoc + 1) & (H[post].max() >= d.ppoc)
    f["dist_ppoc"] = abs(p0 - d.ppoc) / atr
    return f


def build(times, label):
    rows = [features(pd.Timestamp(t)) for t in times]
    X = pd.DataFrame([r for r in rows if r is not None])
    X["set"] = label
    return X


# ---------- 事件與對照 ----------
CAL = calendar()
major_days = set(CAL[CAL.event != "CLAIMS"].date) | set(CAL[CAL.event == "CLAIMS"].date)
EV = {e: build(CAL[CAL.event == e].t, e) for e in ("NFP", "CPI", "FOMC", "CLAIMS")}
all_days = pd.Series(sorted(set(m1.day)))


def control(hh, mm, weekdays):
    ts = [d + pd.Timedelta(hours=hh, minutes=mm) for d in all_days
          if d.weekday() in weekdays and d not in major_days]
    return build(ts, "CTRL")


CTL1530 = control(15, 30, (0, 1, 2, 4))        # 週一二三五 15:30（週四有初領）
CTL2100 = control(21, 0, (2,))                 # 週三 21:00（FOMC 對照）
EV["NFP+CPI"] = pd.concat([EV["NFP"], EV["CPI"]], ignore_index=True)


def tstat(x):
    x = pd.Series(x).dropna()
    return x.mean() / x.std() * np.sqrt(len(x)) if len(x) > 2 and x.std() > 0 else np.nan


def era(X):
    return np.where(X.t < pd.Timestamp("2025-01-01"), "23-24", "25-26")


# ======================================================================
P("=" * 110)
P("第二十二部分：預先排定的數據發布 — 前後行為與 ER / IB / TPO / Pivot / Entropy")
P(f"資料 2023/6/1 ~ 2026/10/8；事件數：" + "、".join(f"{k} {len(v)}" for k, v in EV.items() if k != "NFP+CPI")
  + f"；對照 15:30 {len(CTL1530)} 天、週三 21:00 {len(CTL2100)} 天")
P("單位：ATR10（前 10 日平均真實波幅）。2023 ATR ≈ 25 美元、2026 ≈ 100 美元。")

# ---------- 1. 波動輪廓 ----------
P("\n## 1. 波動：發布前縮、發布後放大多久？（每分鐘 |變動| ÷ ATR，事件 ÷ 對照 的倍數）")
def minute_profile(X, lo=-120, hi=240):
    M = []
    for t, atr in zip(X.t, X.atr):
        i = pos(t)
        if i + hi >= len(C):
            continue
        seg = np.abs(np.diff(C[i + lo - 1:i + hi])) / atr
        if len(seg) == hi - lo:
            M.append(seg)
    return np.nanmean(np.array(M), axis=0)

offs = np.arange(-120, 240)
bands = [(-120, -60), (-60, -15), (-15, -1), (-1, 0), (0, 1), (1, 5), (5, 15), (15, 30), (30, 60), (60, 120), (120, 240)]
for e, ctl in (("NFP", CTL1530), ("CPI", CTL1530), ("FOMC", CTL2100), ("CLAIMS", CTL1530)):
    pe, pc = minute_profile(EV[e]), minute_profile(ctl)
    cells = []
    for a, b in bands:
        m = (offs >= a) & (offs < b)
        cells.append(f"[{a:+d},{b:+d}) {pe[m].mean() / pc[m].mean():4.1f}x")
    P(f"  {e:6s} " + " ".join(cells))
P("  （[0,+1) = 發布那一分鐘；FOMC 的 +30 是記者會）")

P("\n  發布前 60 分鐘的區間（÷ATR）與發布後 30 / 120 分鐘區間：中位數")
for e, ctl in (("NFP", CTL1530), ("CPI", CTL1530), ("FOMC", CTL2100), ("CLAIMS", CTL1530)):
    X = EV[e]
    P(f"  {e:6s} 前60 {X.rng_pre60.median():.3f}（對照 {ctl.rng_pre60.median():.3f}）  "
      f"後30 {X.rng0_30.median():.3f}（對照 {ctl.rng0_30.median():.3f}）  後120 {X.rng0_120.median():.3f}（對照 {ctl.rng0_120.median():.3f}）")
P("\n  點差（美元）：發布前 30 分中位 / 發布那分鐘 / 發布後 5 分鐘最大（M1 檔案的 spread 欄 = 該分鐘最小點差，實際成交會更差）")
for e in ("NFP", "CPI", "FOMC"):
    X = EV[e]
    P(f"  {e:6s} {X.sp_pre.median():.2f} / {X.sp0.median():.2f} / {X.sp_max5.median():.2f}")

# ---------- 2. 方向：發布前漂移、發布後延續還是反轉 ----------
P("\n## 2. 方向（帶符號的平均，÷ATR；t 值括號）")
P("  2a. 發布前 60 分鐘的漂移，發布後會延續還是反轉？  值 = sign(前60漂移) × 之後報酬")
def signed(X, s, col):
    v = np.sign(X[s]) * X[col]
    return f"{v.mean():+.3f}({tstat(v):+.1f})"
for e in ("NFP", "CPI", "NFP+CPI", "FOMC", "CLAIMS"):
    X = EV[e]
    P(f"  {e:8s} n={len(X):3d}  0→1: {signed(X,'pre60','r1')}  0→15: {signed(X,'pre60','r15')}  "
      f"0→60: {signed(X,'pre60','r60')}  0→收盤: {signed(X,'pre60','r_eod')}")
X = CTL1530
P(f"  對照15:30 n={len(X):3d}  0→1: {signed(X,'pre60','r1')}  0→15: {signed(X,'pre60','r15')}  "
  f"0→60: {signed(X,'pre60','r60')}  0→收盤: {signed(X,'pre60','r_eod')}")

P("\n  2b. 發布後前 k 分鐘的第一波，之後延續還是回吐？  值 = sign(0→k) × (k→k+h)")
for e in ("NFP", "CPI", "NFP+CPI", "FOMC", "CLAIMS"):
    X = EV[e]
    cells = []
    for k, h in ((1, 14), (1, 59), (5, 55), (15, 45), (15, 105), (30, 90)):
        v = np.sign(X[f"r{k}"]) * (X[f"r{k+h}"] - X[f"r{k}"]) if f"r{k+h}" in X else None
        cells.append(f"{k}→{k+h}: {v.mean():+.3f}({tstat(v):+.1f})")
    v = np.sign(X.r15) * (X.r_eod - X.r15)
    cells.append(f"15→收盤: {v.mean():+.3f}({tstat(v):+.1f})")
    P(f"  {e:8s} " + "  ".join(cells))
P("  逐期（23-24 | 25-26）：sign(0→15) × (15→60)、sign(0→15) × (15→收盤)")
for e in ("NFP", "CPI", "NFP+CPI", "FOMC"):
    X = EV[e].copy(); X["era"] = era(X)
    cells = []
    for g, Y in X.groupby("era"):
        a = np.sign(Y.r15) * (Y.r60 - Y.r15); b = np.sign(Y.r15) * (Y.r_eod - Y.r15)
        cells.append(f"{g}: n={len(Y)} {a.mean():+.3f}({tstat(a):+.1f}) / {b.mean():+.3f}({tstat(b):+.1f})")
    P(f"  {e:8s} " + "  |  ".join(cells))

P("\n  2c. 第一波的大小（|0→15|）：大衝擊比較會延續嗎？  sign(0→15) × (15→收盤)，依 |0→15| 二分")
for e in ("NFP+CPI", "FOMC"):
    X = EV[e]; m = X.r15.abs().median()
    for lab, Y in (("小", X[X.r15.abs() < m]), ("大", X[X.r15.abs() >= m])):
        v = np.sign(Y.r15) * (Y.r_eod - Y.r15)
        P(f"  {e:8s} 第一波{lab}（中位 {m:.2f} ATR）n={len(Y)}  {v.mean():+.3f}({tstat(v):+.1f})  勝率 {(v > 0).mean():.2f}")

# ---------- 3. 新聞 IB：發布前 60 分鐘區間的突破 ----------
P("\n## 3. 新聞 IB：發布前 60 分鐘的高低點，發布後的突破")
P("  規則：發布後（0 ~ +60 分）第一次觸碰新聞 IB 高/低 → 順突破方向、觸價進場；停損 IB 另一側；")
P("        出場 = +120 分 / 當天收盤；同時統計 60 分鐘內兩邊都被穿（洗兩邊）的比例。單位 R（IB 區間）與 ATR。每筆扣 2×發布後點差。")
def news_ib(X, horizon):
    R = []
    for r in X.itertuples():
        i = pos(r.t)
        if i >= len(T) or T[i] != np.datetime64(r.t):
            continue
        hi, lo = H[i - 60:i].max(), L[i - 60:i].min()
        rng = hi - lo
        day_end = np.searchsorted(T, np.datetime64(r.day + pd.Timedelta(hours=23, minutes=58)))
        stop_i = min(i + 60, day_end)
        up = np.where(H[i:stop_i] >= hi)[0]; dn = np.where(L[i:stop_i] <= lo)[0]
        both = len(up) > 0 and len(dn) > 0
        if not len(up) and not len(dn):
            R.append(dict(t=r.t, trade=False, both=False)); continue
        side = 1 if (len(up) and (not len(dn) or up[0] <= dn[0])) else -1
        e_i = i + (up[0] if side == 1 else dn[0])
        entry = hi if side == 1 else lo
        if side == 1:
            entry = max(entry, O[e_i])              # 跳空穿過就用開盤價
        else:
            entry = min(entry, O[e_i])
        stop = lo if side == 1 else hi
        x_i = day_end - 1 if horizon == "eod" else min(e_i + horizon, day_end - 1)
        res = None
        for j in range(e_i, x_i + 1):
            if (side == 1 and L[j] <= stop) or (side == -1 and H[j] >= stop):
                res = stop; break
            if j == e_i and ((side == 1 and C[j] < stop) or (side == -1 and C[j] > stop)):
                res = stop; break
        if res is None:
            res = C[x_i]
        cost = 2 * SP[e_i]
        pnl = side * (res - entry) - cost
        R.append(dict(t=r.t, trade=True, both=both, side=side, pnl_atr=pnl / r.atr, R=pnl / abs(entry - stop),
                      ib_atr=rng / r.atr))
    return pd.DataFrame(R)

for e, ctl in (("NFP", CTL1530), ("CPI", CTL1530), ("NFP+CPI", CTL1530), ("FOMC", CTL2100), ("CLAIMS", CTL1530)):
    for hz in (120, "eod"):
        A, B = news_ib(EV[e], hz), news_ib(ctl, hz)
        a, b = A[A.trade], B[B.trade]
        P(f"  {e:8s} 出場{str(hz):>4s}: 觸發 {len(a)}/{len(A)}  洗兩邊 {A.both.mean():.0%}  每筆 {a.R.mean():+.2f}R ({tstat(a.R):+.1f})  "
          f"{a.pnl_atr.mean():+.3f}ATR  勝率 {(a.pnl_atr > 0).mean():.0%}  | 對照：洗兩邊 {B.both.mean():.0%} 每筆 {b.R.mean():+.2f}R {b.pnl_atr.mean():+.3f}ATR")

# ---------- 4. 因子：發布當下的狀態能不能預測之後？ ----------
P("\n## 4. 發布當下的因子 vs 之後的結果（Spearman ρ，括號 t；|t|≥2 才算有東西，且要兩期同號）")
P("  結果 A：|0→60| 波幅大小（能不能預先知道這次會不會大動）")
P("  結果 B：sign(當天開盤→發布) × (0→60)（發布後順著今天方向 > 0、反轉 < 0）")
P("  結果 C：sign(0→15) × (15→收盤)（第一波延續 > 0）")
FACT = ["er30", "er60", "er240", "er_day", "ent60", "pent60", "pent120", "rng_pre60", "absret_pre60",
        "ldn_ib", "ldn_ib_pos", "tpo_pos", "tpo_va_pos", "prev_va_pos", "piv_pos", "piv_dist", "day_move", "dist_ppoc"]
SIGNED = {"er30", "er60", "er240", "er_day", "ldn_ib_pos", "tpo_pos", "tpo_va_pos", "prev_va_pos", "piv_pos", "day_move"}

def rho(x, y):
    ok = x.notna() & y.notna()
    if ok.sum() < 10:
        return np.nan, np.nan
    r = x[ok].rank().corr(y[ok].rank())
    return r, r * np.sqrt((ok.sum() - 2) / max(1 - r * r, 1e-9))

def factor_table(X, name):
    X = X.copy()
    s_day = np.sign(X.day_move)
    X["A"] = X.r60.abs()
    X["B"] = s_day * X.r60
    X["C"] = np.sign(X.r15) * (X.r_eod - X.r15)
    X["era"] = era(X)
    P(f"\n  [{name}] n={len(X)}")
    P(f"  {'因子':14s} {'A |0→60|':>26s} {'B 順今天方向':>26s} {'C 第一波延續':>26s}")
    for f in FACT:
        cells = []
        for out_col in ("A", "B", "C"):
            x = X[f]
            if f in SIGNED and out_col == "A":
                x = x.abs()                                    # 波幅只看絕對值
            if f in SIGNED and out_col == "B":
                x = s_day * x                                  # 帶方向的因子轉成「順今天方向」
            if f in SIGNED and out_col == "C":
                x = x.abs()
            r, t = rho(x, X[out_col])
            e1 = rho(x[X.era == "23-24"], X[out_col][X.era == "23-24"])[0]
            e2 = rho(x[X.era == "25-26"], X[out_col][X.era == "25-26"])[0]
            flag = "*" if abs(t) >= 2 and np.sign(e1) == np.sign(e2) else " "
            cells.append(f"{r:+.2f}({t:+.1f}) {e1:+.2f}|{e2:+.2f}{flag}")
        P(f"  {f:14s} " + "  ".join(f"{c:>26s}" for c in cells))

factor_table(EV["NFP+CPI"], "NFP + CPI（15:30）")
factor_table(EV["FOMC"], "FOMC（21:00）")
factor_table(EV["CLAIMS"], "初領失業金（週四 15:30，次要數據，用來看規律能否複製）")
factor_table(CTL1530, "對照：沒有數據的 15:30")
P("  （帶方向的因子：A、C 用絕對值，B 用「順今天方向」的值；* = |t|≥2 且 2023-24 與 2025-26 同號）")

# ---------- 5. Pivot / TPO 價位：發布後當天碰到的機率 ----------
P("\n## 5. Pivot 與前日 POC：發布後當天會不會碰到（事件日 vs 對照）")
for e, ctl in (("NFP+CPI", CTL1530), ("FOMC", CTL2100)):
    X, Y = EV[e], ctl
    def rate(Z, col, cond):
        z = Z[cond(Z)]; return f"{z[col].mean():.0%}(n={len(z)})"
    P(f"  {e:8s} 價在 P 上方 → 碰 R1: {rate(X,'hit_R1',lambda z:(z.piv_pos>0)&(z.piv_pos<1))} 對照 {rate(Y,'hit_R1',lambda z:(z.piv_pos>0)&(z.piv_pos<1))}"
      f" | 價在 P 下方 → 碰 S1: {rate(X,'hit_S1',lambda z:(z.piv_pos<0)&(z.piv_pos>-1))} 對照 {rate(Y,'hit_S1',lambda z:(z.piv_pos<0)&(z.piv_pos>-1))}")
    P(f"  {'':8s} 回到 P: {rate(X,'hit_P',lambda z:z.piv_pos.abs()<9)} 對照 {rate(Y,'hit_P',lambda z:z.piv_pos.abs()<9)}"
      f" | 碰前日 POC（距離 < 0.5 ATR）: {rate(X,'hit_ppoc',lambda z:z.dist_ppoc<.5)} 對照 {rate(Y,'hit_ppoc',lambda z:z.dist_ppoc<.5)}"
      f" | 距離 ≥ 0.5 ATR: {rate(X,'hit_ppoc',lambda z:z.dist_ppoc>=.5)} 對照 {rate(Y,'hit_ppoc',lambda z:z.dist_ppoc>=.5)}")

pd.concat(EV.values(), ignore_index=True).to_csv("part22_events.csv", index=False)
pd.concat([CTL1530.assign(set="CTRL1530"), CTL2100.assign(set="CTRL2100")]).to_csv("part22_control.csv", index=False)
