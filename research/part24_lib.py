"""第二十四部分：回復力切換策略的優化 —— 共用函式。

事件 = |D| 第一次 ≥ d0（D 回到 rearm 以內才重新武裝），EMA20、M5；每個事件記下訊號當下的全部因子，
並對「回歸（反向）」「動能（順向）」各算多種出場的結果（每筆獨立，不管持倉限制；持倉限制在之後套用）。
"""
import numpy as np
import pandas as pd
from part22_lib import load_tf, means, atr, events, CUT
from part22_beta import frame
from part22_final import universal_selection, rfi_u

CUT1 = pd.Timestamp("2026-04-01")
SESS = lambda h: "亞洲" if h < 10 else "倫敦" if h < 15 else "紐約" if h < 20 else "尾盤"


def load_all():
    keep = universal_selection(pd.read_csv("part22_beta.csv"))
    d = load_tf("M5"); A = atr(d, 14); M = means(d)
    X = frame(d, A, M, "ema20"); X["rfi"] = rfi_u(X, keep)
    X = X.reindex(d.index)
    return d, A, M, X


def trade(o, h, l, c, spr, seg, t, s, a, H, stop, target=None):
    """回傳 (損益_美元, 出場 index, 最大浮虧_美元, 原因)。下一根開盤進場。"""
    n = len(c); e = t + 1; ep = o[e]; st = ep - s * stop * a
    mae = 0.0
    for j in range(e, min(e + H, n)):
        if seg[j] != seg[t]:
            return s * (c[j - 1] - ep) - spr[e], j - 1, mae, "斷線"
        if (l[j] <= st) if s > 0 else (h[j] >= st):
            xp = (min(st, o[j]) if s > 0 else max(st, o[j])) if j > e else st
            return s * (xp - ep) - spr[e], j, max(mae, s * (ep - xp)), "停損"
        if target is not None and j > e and ((h[j] >= target) if s > 0 else (l[j] <= target)):
            xp = max(target, o[j]) if s > 0 else min(target, o[j])
            return s * (xp - ep) - spr[e], j, mae, "停利"
        mae = max(mae, (ep - l[j]) if s > 0 else (h[j] - ep))
    j = min(e + H - 1, n - 1)
    return s * (c[j] - ep) - spr[e], j, mae, "時間"


def event_frame(d, A, M, X, d0, rearm_frac):
    D = X.D
    ev = events(D, d.seg, d0=d0, rearm=d0 * rearm_frac)
    ev = ev[(ev > 600) & (ev + 2 < len(d))]
    E = X.iloc[ev].copy()
    E["i"] = ev
    E["t"] = d.index[ev]
    E["atr"] = A.to_numpy()[ev]
    E["ema_at"] = M.ema20.to_numpy()[ev]
    E["sess"] = [SESS(t.hour) for t in E.t]
    E["dow"] = E.t.dt.dayofweek
    E["per"] = np.where(E.t < CUT1, "IS1", np.where(E.t < CUT, "IS2", "OOS"))
    return E.reset_index(drop=True).dropna(subset=["rfi"])


def add_outcomes(d, E, specs):
    """specs: {名稱: (leg, H, stop, use_target)}；leg = 'mr'（反向）或 'mom'（順向）。
    為每個 spec 加三欄：pnl_<名稱>、exit_<名稱>（出場 index）、why_<名稱>。"""
    o, h, l, c = (d[k].to_numpy() for k in ("open", "high", "low", "close"))
    spr = d.spread.to_numpy() / 100; seg = d.seg.to_numpy()
    for name, (leg, H, stop, tgt) in specs.items():
        P, X_, W = [], [], []
        for r in E.itertuples():
            s = -np.sign(r.D) if leg == "mr" else np.sign(r.D)
            pnl, xi, _, why = trade(o, h, l, c, spr, seg, r.i, s, r.atr, H, stop, r.ema_at if tgt else None)
            P.append(pnl); X_.append(xi); W.append(why)
        E[f"pnl_{name}"], E[f"exit_{name}"], E[f"why_{name}"] = P, X_, W
    return E


def apply_cap(entries, exits, cap):
    """依進場順序，同時持倉數 ≤ cap 才接受（cap=None 不限）。回傳布林陣列與每筆進場時的持倉數。"""
    order = np.argsort(entries, kind="stable")
    ok = np.zeros(len(entries), bool); opn = []; conc = np.zeros(len(entries), int)
    for k in order:
        opn = [x for x in opn if x >= entries[k]]       # 出場那根 index ≥ 新單進場那根 → 仍在持倉
        if cap is None or len(opn) < cap:
            ok[k] = True; opn.append(exits[k]); conc[k] = len(opn)
    return ok, conc


def summarize(pnl, t, atr=None):
    """pnl：美元/筆；t：進場時間。回傳 dict。"""
    p = np.asarray(pnl, float)
    if len(p) == 0:
        return dict(n=0, avg=np.nan, pf=np.nan, win=np.nan, total=0.0, dd=0.0, sharpe=np.nan, ratio=np.nan)
    day = pd.Series(p, index=pd.DatetimeIndex(t)).groupby(pd.DatetimeIndex(t).normalize()).sum()
    eq = np.cumsum(p[np.argsort(t)]); dd = float((eq - np.maximum.accumulate(eq)).min())
    neg = -p[p <= 0].sum()
    return dict(n=len(p), avg=p.mean(), pf=p[p > 0].sum() / neg if neg > 0 else np.inf, win=np.mean(p > 0),
                total=p.sum(), dd=dd, sharpe=day.mean() / day.std() * np.sqrt(252) if day.std() > 0 else np.nan,
                ratio=p.sum() / -dd if dd < 0 else np.inf)


def fmt(s):
    if s["n"] == 0:
        return "   0 筆"
    return (f"{s['n']:4d} 筆 勝率 {s['win']:.0%} 每筆 {s['avg']:+5.2f} PF {s['pf']:4.2f} 總 {s['total']:+7.0f} "
            f"回撤 {s['dd']:+6.0f} 日Sharpe {s['sharpe']:+5.2f}")
