"""第十二部分：IB 突破延續 → 單一淨部位策略（XAUUSD M1，broker 時間）。

訊號：HOURS 裡每個整點 h 起 IB_MIN 分鐘當 IB；IB 結束後 WIN 小時內，該 IB 第一次被突破
      （上破 = 多、下破 = 空）就是一個訊號，每個 IB 最多一個訊號。
部位：同時只有一個部位。
      空手 → 進場；同方向訊號 → 忽略；反方向訊號 → 平倉並反手（reverse=False 則忽略，只靠停損/收盤出場）。
進場：在 IB 邊成交（K 棒開盤已跳過去就用開盤價）。
停損：stop="opp" IB 另一側 / "mid" IB 中點 / 數字 k = k × IB 區間。
保本：be=k → 浮盈達 k × IB 區間後，停損移到進場價（None = 不用）。
出場：停損、反手，或當天最後一根 K 棒收盤。
過濾：flt=None / "prevday"（只做前一交易日 close−open 的方向）/ "trend24"（只做與 24 小時前收盤相比的方向）。
成本：每筆扣進場那根 K 棒的點差（約一個點差，涵蓋進出）。
同根 K 棒同時碰到停損與其他事件時，一律先算停損（保守）。
"""
import numpy as np
import pandas as pd
from tpo import load_m1

d = load_m1()
d["sp"] = d.spread * 0.01
DAYS = [k for k, g in d.groupby("day") if len(g) >= 1000]
G = {k: g for k, g in d.groupby("day") if k in set(DAYS)}
DD = pd.DataFrame({k: dict(o=g.open.iloc[0], c=g.close.iloc[-1], h=g.high.max(), l=g.low.min()) for k, g in G.items()}).T
DD["prevdir"] = np.sign(DD.c - DD.o).shift(1)
CLOSE_ALL = d.close
SPLIT = pd.Timestamp("2026-06-01")


def day_signals(g, hours, ib_min, win):
    """回傳 [(bar_index, side, level, ib_high, ib_low)]，依時間排序。"""
    t = g.index; H, L = g.high.values, g.low.values
    day = g.day.iloc[0]
    sig = []
    for h in hours:
        t0 = day + pd.Timedelta(hours=h); t1 = t0 + pd.Timedelta(minutes=ib_min)
        m = (t >= t0) & (t < t1)
        if m.sum() < ib_min * 0.8:
            continue
        ibh, ibl = H[m].max(), L[m].min()
        w = np.where((t >= t1) & (t < t1 + pd.Timedelta(hours=win)))[0]
        if not len(w):
            continue
        up = w[H[w] > ibh]; dn = w[L[w] < ibl]
        if len(up) and (not len(dn) or up[0] < dn[0]):
            sig.append((up[0], 1, ibh, ibh, ibl))
        elif len(dn):
            sig.append((dn[0], -1, ibl, ibh, ibl))
    return sorted(sig)


def backtest(hours=range(1, 11), ib_min=60, win=3, stop="opp", be=None, flt=None, days=None, reverse=True):
    trades = []
    for day in (days or DAYS):
        g = G[day]
        O, H, L, C, S = g.open.values, g.high.values, g.low.values, g.close.values, g.sp.values
        sigs = day_signals(g, hours, ib_min, win)
        if flt == "prevday":
            pdir = DD.prevdir.get(day)
            sigs = [s for s in sigs if s[1] == pdir]
        elif flt == "trend24":
            keep = []
            for s in sigs:
                ref_i = CLOSE_ALL.asof(g.index[s[0]] - pd.Timedelta(hours=24))
                if np.sign(s[2] - ref_i) == s[1]:
                    keep.append(s)
            sigs = keep
        # 同一根 K 棒兩個方向的訊號互相抵消
        by_bar = {}
        for s in sigs:
            by_bar.setdefault(s[0], []).append(s)
        pos = None   # dict(side, entry, stop, i0, rng, be_done)
        for i in range(len(C)):
            # 1. 停損
            if pos is not None and i > pos["i0"]:
                hit = L[i] <= pos["stop"] if pos["side"] == 1 else H[i] >= pos["stop"]
                if hit:
                    px = min(O[i], pos["stop"]) if pos["side"] == 1 else max(O[i], pos["stop"])
                    trades.append(dict(day=day, side=pos["side"], entry=pos["entry"], exit=px, cost=pos["cost"],
                                       t_in=g.index[pos["i0"]], t_out=g.index[i], why="stop"))
                    pos = None
            # 2. 保本
            if pos is not None and be is not None and not pos["be_done"]:
                fav = (H[i] - pos["entry"]) if pos["side"] == 1 else (pos["entry"] - L[i])
                if fav >= be * pos["rng"]:
                    pos["stop"] = pos["entry"]; pos["be_done"] = True
            # 3. 新訊號
            if i in by_bar:
                ss = by_bar[i]
                if len({s[1] for s in ss}) > 1:
                    continue
                _, side, lvl, ibh, ibl = ss[0]
                if pos is not None and (pos["side"] == side or not reverse):
                    continue
                fill = max(O[i], lvl) if side == 1 else min(O[i], lvl)
                if pos is not None:   # 反手：以突破價平掉舊部位
                    trades.append(dict(day=day, side=pos["side"], entry=pos["entry"], exit=fill, cost=pos["cost"],
                                       t_in=g.index[pos["i0"]], t_out=g.index[i], why="reverse"))
                rng = ibh - ibl
                if stop == "opp": st = ibl if side == 1 else ibh
                elif stop == "mid": st = (ibh + ibl) / 2
                else: st = fill - side * stop * rng
                pos = dict(side=side, entry=fill, stop=st, i0=i, rng=rng, be_done=False, cost=S[i])
                # 進場那根 K 棒也碰到停損：無法知道先後，保守算停損
                if (L[i] <= st) if side == 1 else (H[i] >= st):
                    trades.append(dict(day=day, side=side, entry=fill, exit=st, cost=S[i],
                                       t_in=g.index[i], t_out=g.index[i], why="stop"))
                    pos = None
        if pos is not None:
            trades.append(dict(day=day, side=pos["side"], entry=pos["entry"], exit=C[-1], cost=pos["cost"],
                               t_in=g.index[pos["i0"]], t_out=g.index[-1], why="close"))
    T = pd.DataFrame(trades)
    if len(T):
        T["pnl"] = T.side * (T.exit - T.entry) - T.cost       # 美元/盎司（= 0.01 手的美元）
    return T


def stats(T, lab=""):
    if not len(T):
        return dict(lab=lab, n=0)
    p = T.pnl; w, l = p[p > 0], p[p <= 0]
    eq = p.cumsum(); dd = (eq - eq.cummax()).min()
    nd = T.day.nunique()
    return dict(lab=lab, n=len(T), per_day=len(T) / max(nd, 1), win=len(w) / len(p) * 100,
                avg=p.mean(), avg_w=w.mean() if len(w) else 0, avg_l=l.mean() if len(l) else 0,
                pf=w.sum() / -l.sum() if len(l) and l.sum() < 0 else np.inf, total=p.sum(), maxdd=dd,
                ret_dd=p.sum() / -dd if dd < 0 else np.inf)


def split_stats(T, lab):
    a = stats(T[T.day < SPLIT], lab); b = stats(T[T.day >= SPLIT], lab)
    return a, b
