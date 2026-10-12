"""第五十六部分：不扣滑價（只扣點差）重算 + 每一層的「加益」+ M30 / H1 版本與組合。

同一套規則套在 M15 / M30 / H1（只換 K 棒週期；ATR = 該週期前 20 根平均波幅、停損 2 ATR、RR 3、收盤平倉）：
  L0 前一根高/低突破（無篩選）
  A  前日收盤在 SMA50 同側      B  今天逆向走不超過 0.17 日ATR      S  SMA50 10 日斜率同向
  CAP 同時最多 8 筆   EQ 權益濾網（影子累積 R < 50 筆均線不做）   VOL 前日 ATR10 / 60 日平均 > 2.2 當天不做
  ENT ATR 熵上升（該週期 20 根 TR 熵 − 10 根前）> 訓練期 80% 分位不做   DS 當天已實現 −15R 後不開新單
IB 核心（core_ib.py）：原版 / + 2R 保本。
「加益」兩種看法：(a) 一層層疊上去；(b) 從完整版拿掉某一層（看那一層現在還貢獻多少）。
成本：只扣點差（slip 0）；最後組合另列 +0.3 滑價對照。訓練 = 2023/8~2024/12，驗證 = 2025~2026/10。數量：1 份 = 0.01 口 = 1 盎司。
"""
import os
os.environ["CORE_IB_DATA"] = "data/XAUUSD_M1_2023_2026.csv"
import numpy as np
import pandas as pd
import part49_safeguards as P
import part24_prev_bar_breakout as B
import core_ib as CI

G, W, IDX = P.G, P.W, P.IDX
C, H, Lo, SP, dayv = W.C, W.H, W.Lo, W.SP, W.dayv
ds = P.ds
TRAIN_END = pd.Timestamp("2025-01-01")
lines = []


def p(s=""):
    print(s, flush=True)
    lines.append(s)


def pf(v):
    v = np.asarray(v)
    return v[v > 0].sum() / -v[v < 0].sum()


# ---------------- 事件與熵 ----------------
def entropy_change(tf):
    pid = B.period_ids(tf)
    g = pd.DataFrame({"pid": pid, "H": H, "L": Lo, "C": C})
    b = g.groupby("pid").agg(H=("H", "max"), L=("L", "min"), C=("C", "last"))
    tr = np.nan_to_num(np.maximum(b.H - b.L, np.maximum((b.H - b.C.shift(1)).abs(), (b.L - b.C.shift(1)).abs())).values)
    v = np.lib.stride_tricks.sliding_window_view(tr, 20)
    pp = v / v.sum(1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = np.r_[np.full(19, np.nan), -np.nansum(np.where(pp > 0, pp * np.log(pp), 0), 1) / np.log(20)]
    dent = ent - np.r_[np.full(10, np.nan), ent[:-10]]
    return pid, dent


EVS = {}
for tf in (15, 30, 60):
    raw = G.ev if tf == 15 else W.events(tf)
    pid, dent = entropy_change(tf)
    rows = []
    for i, L, side, a in raw:
        if IDX[i] < pd.Timestamp("2023-08-24"):
            continue
        s = ds.loc[dayv[i]]
        if not (np.isfinite(s.ma) and np.isfinite(s.ma10) and np.isfinite(s.atr)):
            continue
        rows.append((i, L, side, a, s.volrel, (s.pc - s.ma) * side > 0, (L - s.o) * side / s.atr > -0.17,
                     (s.ma - s.ma10) * side > 0, dent[max(pid[i] - 1, 0)]))
    EVS[tf] = pd.DataFrame(rows, columns=["i", "L", "side", "a", "volrel", "A", "B", "S", "dent"])


def sim(E, slip, t=3):
    rows = []
    for r in E.itertuples():
        risk = 2.0 * r.a
        x = G.sim_detail(r.i, r.L, r.side, r.L - r.side * risk, r.L + r.side * t * risk)
        if x is None:
            continue
        j, px, why = x
        net = (px - r.L) * r.side - SP[r.i] - slip
        rows.append((IDX[r.i], IDX[j], r.i, j, r.side, r.L, net, net / risk, risk, r.volrel, r.dent))
    x = pd.DataFrame(rows, columns=["進場時間", "出場時間", "i", "j", "dir", "entry", "net", "R", "risk", "volrel", "dent"])
    return x.sort_values("進場時間").reset_index(drop=True)


def eq_filter(x):
    ex = x.sort_values("出場時間"); cum = ex.R.cumsum().values
    ma = pd.Series(cum).rolling(50, min_periods=50).mean().values; et = ex.出場時間.values
    keep = [not (k >= 50 and cum[k] < ma[k]) for k in (np.searchsorted(et, tt, side="right") - 1 for tt in x.進場時間.values)]
    return x[np.array(keep)].reset_index(drop=True)


def daily_stop(x, K=15):
    keep, real, cur = [], [], None
    for r in x.itertuples():
        day = r.進場時間.normalize()
        if day != cur:
            cur, real = day, []
        lost = sum(v for (te, v) in real if te <= r.進場時間)
        if lost <= -K:
            keep.append(False); continue
        keep.append(True); real.append((r.出場時間, r.R))
    return x[np.array(keep)].reset_index(drop=True)


ALL = ["A", "B", "S", "CAP", "EQ", "VOL", "ENT", "DS"]
CUT = {}


def pipeline(tf, layers, slip=0.0):
    E = EVS[tf]
    for f in ("A", "B", "S"):
        if f in layers:
            E = E[E[f]]
    x = sim(E, slip)
    if "CAP" in layers:
        x = P.cap_total(x, 8).reset_index(drop=True)
    if "EQ" in layers:
        x = eq_filter(x)
    if "VOL" in layers:
        x = x[~(x.volrel > 2.2).values].reset_index(drop=True)
    if "ENT" in layers:
        x = x[~(x.dent > CUT[tf]).values].reset_index(drop=True)
    if "DS" in layers:
        x = daily_stop(x)
    return x


for tf in (15, 30, 60):     # 熵門檻：完整版（不含熵、不含 DS）訓練期的 80% 分位，固定給所有變體使用
    x0 = pipeline(tf, ["A", "B", "S", "CAP", "EQ", "VOL"])
    CUT[tf] = np.nanquantile(x0.dent[x0.進場時間 < TRAIN_END], 0.8)


# ---------------- 逐分鐘權益 ----------------
def minute_eq(parts):
    n = len(IDX)
    D, E, R = np.zeros(n + 1), np.zeros(n + 1), np.zeros(n + 1)
    for (ii, jj, dd, en, net, q) in parts:
        np.add.at(D, ii, dd * q); np.add.at(D, jj, -dd * q)
        np.add.at(E, ii, dd * en * q); np.add.at(E, jj, -dd * en * q)
        np.add.at(R, jj, net * q)
    D, E, R = np.cumsum(D)[:n], np.cumsum(E)[:n], np.cumsum(R)[:n]
    return R + C * D - E


def mpart(x, q):
    return (x.i.values, x.j.values, x.dir.values.astype(float), x.entry.values, x.net.values, q)


def daily_of(x, q, key="進場時間"):
    return (x.net * q).groupby(x[key].dt.normalize()).sum()


def show(lab, parts, daily, n=None):
    daily = daily.sort_index()
    eq = minute_eq(parts)
    mm = (np.maximum.accumulate(eq) - eq).max()
    yrs = (daily.index.max() - daily.index.min()).days / 365
    mon = daily.groupby(pd.DatetimeIndex(daily.index).to_period("M")).sum()

    def rat(s):
        e = s.cumsum(); m = (e.cummax() - e).max(); return s.sum() / m if m > 0 else np.nan
    tr = daily.index < TRAIN_END
    yr = " / ".join(f"{daily[pd.DatetimeIndex(daily.index).year == y].sum():+,.0f}" for y in (2023, 2024, 2025, 2026))
    p(f"  {lab:30s} {'' if n is None else f'{n:5d} 筆 '}總 {daily.sum():+8,.0f} | MDD 含浮虧 {mm:6,.0f} | 年獲利/MDD {daily.sum() / yrs / mm:4.2f} | "
      f"最差月 {mon.min():+6,.0f} 正月 {(mon > 0).sum()}/{len(mon)} | 訓練 {rat(daily[tr]):4.1f} 驗證 {rat(daily[~tr]):4.1f} | {yr}")
    return daily


# ---------------- IB 核心 ----------------
def ib_prep(X, slip):
    X = X[(X.peff <= CI.PEFF_MAX) & (X.day >= "2023-06-15")].copy()
    gi, gj = [], []
    for r in X.itertuples():
        t = CI.ARR[r.day][0]
        gi.append(IDX.get_indexer([t[r.i0]])[0]); gj.append(IDX.get_indexer([t[int(r.j)]])[0])
    X["i"], X["j"] = gi, gj
    X = X[(X.i >= 0) & (X.j >= 0)].copy()
    X["net"] = X.pnl - slip
    X["dir"], X["entry"], X["進場時間"] = X.side, X.lvl, pd.to_datetime(X.day)
    return X




if __name__ == "__main__":
    p("第五十六部分：不扣滑價（只扣點差）+ 每層加益 + M30 / H1（0.02 口 = 每筆 2 盎司）")
    p("ATR 熵門檻（訓練期 80% 分位）：" + "、".join(f"{ {15: 'M15', 30: 'M30', 60: 'H1'}[k]} {v:+.3f}" for k, v in CUT.items()))
    FULL = {}
    for tf, nm in ((15, "M15"), (30, "M30"), (60, "H1")):
        p(f"\n=== {nm}：(a) 一層層疊上去（只扣點差）===")
        acc = []
        steps = [("L0 前一根突破（無篩選）", [])] + [(f"+ {f}", None) for f in ALL]
        for lab, _ in steps:
            if lab.startswith("+ "):
                acc = acc + [lab[2:]]
            x = pipeline(tf, acc)
            show(f"{lab}（{len(x) / x.進場時間.dt.normalize().nunique():.1f}筆/天 PF {pf(x.net):.2f}）", [mpart(x, 2)], daily_of(x, 2), len(x))
        FULL[tf] = pipeline(tf, ALL)
        p(f"=== {nm}：(b) 從完整版拿掉一層（拿掉後變差越多 = 這層越有用）===")
        for f in ALL:
            x = pipeline(tf, [g for g in ALL if g != f])
            show(f"完整版 − {f}", [mpart(x, 2)], daily_of(x, 2), len(x))

    Eib = CI.events()
    IB = {(be, s): ib_prep(CI.simulate(Eib, be=be), s) for be in (None, 2) for s in (0.0, 0.3)}
    p("\n=== IB 核心（0.02 口，只扣點差）===")
    show("IB 核心", [mpart(IB[(None, 0.0)], 2)], daily_of(IB[(None, 0.0)], 2), len(IB[(None, 0.0)]))
    show("IB + 2R 保本", [mpart(IB[(2, 0.0)], 2)], daily_of(IB[(2, 0.0)], 2), len(IB[(2, 0.0)]))

    # ---------------- 組合 ----------------
    D = pd.DataFrame({"IB": daily_of(IB[(2, 0.0)], 1), "M15": daily_of(FULL[15], 1), "M30": daily_of(FULL[30], 1),
                      "H1": daily_of(FULL[60], 1)}).fillna(0)
    p("\n日損益相關（每筆 1 盎司，只扣點差）：\n" + D.corr().round(2).to_string())
    both = {}
    for a, b_ in (("M15", "M30"), ("M15", "H1"), ("M30", "H1")):
        xa, xb = FULL[{"M15": 15, "M30": 30, "H1": 60}[a]], FULL[{"M15": 15, "M30": 30, "H1": 60}[b_]]
        same = pd.merge(xa[["進場時間", "dir"]].assign(d=xa.進場時間.dt.normalize()), xb[["進場時間", "dir"]].assign(d=xb.進場時間.dt.normalize()),
                        on=["d", "dir"]).d.nunique()
        both[(a, b_)] = same
    p("同一天同方向都有單的天數：" + "、".join(f"{a}&{b_} {v}" for (a, b_), v in both.items()))


    def combo(lab, spec, slip=0.0):
        src = {"IB": IB[(2, slip)], "M15": FULL[15] if slip == 0 else FULL_S[15], "M30": FULL[30] if slip == 0 else FULL_S[30],
               "H1": FULL[60] if slip == 0 else FULL_S[60]}
        parts = [mpart(src[k], q) for k, q in spec.items() if q]
        daily = pd.concat([daily_of(src[k], q) for k, q in spec.items() if q], axis=1).fillna(0).sum(1)
        show(lab, parts, daily)


    p("\n=== 組合（份數 = 0.01 口；只扣點差）===")
    SPECS = [("M15 2", {"M15": 2}), ("M30 2", {"M30": 2}), ("H1 2", {"H1": 2}),
             ("M15 1 + M30 1", {"M15": 1, "M30": 1}), ("M15 1 + H1 1", {"M15": 1, "H1": 1}),
             ("M15 2 + M30 1", {"M15": 2, "M30": 1}), ("M15 2 + H1 1", {"M15": 2, "H1": 1}), ("M15 2 + M30 1 + H1 1", {"M15": 2, "M30": 1, "H1": 1}),
             ("IB 1 + M15 2（目前）", {"IB": 1, "M15": 2}),
             ("IB 1 + M15 2 + M30 1", {"IB": 1, "M15": 2, "M30": 1}), ("IB 1 + M15 2 + H1 1", {"IB": 1, "M15": 2, "H1": 1}),
             ("IB 1 + M15 2 + M30 1 + H1 1", {"IB": 1, "M15": 2, "M30": 1, "H1": 1}),
             ("IB 1 + M15 1 + M30 1 + H1 1", {"IB": 1, "M15": 1, "M30": 1, "H1": 1}),
             ("IB 2 + M15 2 + M30 1 + H1 1", {"IB": 2, "M15": 2, "M30": 1, "H1": 1})]
    for lab, spec in SPECS:
        combo(lab, spec)
    FULL_S = {tf: pipeline(tf, ALL, slip=0.3) for tf in (15, 30, 60)}
    p("\n=== 對照：同樣組合扣 0.3 滑價 ===")
    for lab, spec in SPECS:
        combo(lab, spec, slip=0.3)
    open("results_part56.txt", "w").write("\n".join(lines) + "\n")
