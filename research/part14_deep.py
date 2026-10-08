"""第十四部分：規則 ② 深入研究。

規則 ②（基準）：broker 02~10 點每個整點起 60 分鐘當 IB；IB 結束後 3 小時內第一次突破，在 IB 邊進場；
  不做「突破方向 = 前一交易日 (close−open) 方向 且 IB 區間 < 0.13 × ATR10」的單；
  停損 IB 另一側；當天收盤出場；每個 IB 各自一筆（可同時持有多筆）；扣進場 K 棒點差。
單位：美元/盎司（= 0.01 手美元）。1~5 月 = 樣本內（IS，規則在這裡找到），6~10 月 = 樣本外（OOS）。
"""
import numpy as np
import pandas as pd
from tpo import load_m1, build_days

d = load_m1()
D, _ = build_days(d)
G = {k: g for k, g in d.groupby("day") if k in D.index}
DAYS = list(D.index)
SPLIT = pd.Timestamp("2026-06-01")
ARR = {k: (g.index, g.open.values, g.high.values, g.low.values, g.close.values, g.spread.values * 0.01) for k, g in G.items()}


def signals(hours=range(2, 11), ib_min=60, win=3):
    out = []
    for di, day in enumerate(DAYS):
        if di == 0:
            continue
        atr = D.atr10.iloc[di]
        if not np.isfinite(atr):
            continue
        prev = D.iloc[di - 1]; pdir = np.sign(prev.close - prev.open)
        t, O, H, L, C, S = ARR[day]
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
            if len(up) and (not len(dn) or up[0] < dn[0]): side, i, lvl = 1, up[0], ibh
            elif len(dn): side, i, lvl = -1, dn[0], ibl
            else: continue
            out.append(dict(day=day, h=h, i=i, side=side, lvl=lvl, ibh=ibh, ibl=ibl, rng=ibh - ibl, atr=atr,
                            ib_atr=(ibh - ibl) / atr, aligned=side == pdir, t_in=t[i], spread=S[i]))
    return pd.DataFrame(out)


def exits(sig, stop="opp", be=None, hold=None, cost=0.0, trail=None):
    """逐筆計算出場。stop: opp / mid / 數字 k（k × IB 區間）；be: 浮盈 k×IB 區間後移保本；
    hold: 持有幾小時（None = 收盤）；trail: 浮盈超過 1×IB 後，以 trail×IB 區間的距離移動停損。"""
    pnl, tout = np.empty(len(sig)), []
    for n, r in enumerate(sig.itertuples()):
        t, O, H, L, C, S = ARR[r.day]
        side, lvl, rng = r.side, r.lvl, r.rng
        if stop == "opp": st = r.ibl if side == 1 else r.ibh
        elif stop == "mid": st = (r.ibh + r.ibl) / 2
        else: st = lvl - side * stop * rng
        end = len(C) - 1 if hold is None else min(len(C) - 1, r.i + int(hold * 60))
        px, j = C[end], end
        best = lvl
        for k in range(r.i, end + 1):
            if (L[k] <= st) if side == 1 else (H[k] >= st):
                px = st if k == r.i else (min(O[k], st) if side == 1 else max(O[k], st)); j = k; break
            best = max(best, H[k]) if side == 1 else min(best, L[k])
            fav = side * (best - lvl)
            if be is not None and fav >= be * rng:
                st = max(st, lvl) if side == 1 else min(st, lvl)
            if trail is not None and fav >= rng:
                nst = best - side * trail * rng
                st = max(st, nst) if side == 1 else min(st, nst)
        pnl[n] = side * (px - lvl) - r.spread - cost
        tout.append(t[j])
    s = sig.copy(); s["pnl"] = pnl; s["t_out"] = tout
    return s


def rule2(s, thr=0.13):
    return s[~(s.aligned & (s.ib_atr < thr))]


def cap(s, n):
    """同時最多持有 n 筆（依進場時間先到先做）。"""
    s = s.sort_values("t_in"); keep, open_ = [], []
    for r in s.itertuples():
        open_ = [x for x in open_ if x > r.t_in]
        if len(open_) < n:
            keep.append(r.Index); open_.append(r.t_out)
    return s.loc[keep]


def st(x):
    p = x.pnl
    if len(p) < 5:
        return dict(n=len(p))
    w, l = p[p > 0], p[p <= 0]
    eq = x.sort_values("t_out").pnl.cumsum()
    nd = x.day.nunique()
    return dict(n=len(p), per_day=len(p) / max(nd, 1), avg=p.mean(), avg_atr=(p / x.atr).mean(), pf=w.sum() / -l.sum(),
                win=len(w) / len(p) * 100, total=p.sum(), maxdd=(eq - eq.cummax()).min())


def fmt(x):
    s = st(x)
    if "pf" not in s:
        return f"n={s['n']}"
    return (f"{s['per_day']:4.1f}/天 avg {s['avg']:+6.2f} ({s['avg_atr']:+.3f}ATR) PF {s['pf']:.2f} win {s['win']:3.0f}%"
            f" 總 {s['total']:+6.0f} DD {s['maxdd']:6.0f}")


def both(x):
    return f"IS {fmt(x[x.day < SPLIT])} || OOS {fmt(x[x.day >= SPLIT])}"


if __name__ == "__main__":
    out = open("results_part14.txt", "w")
    def P(s=""): print(s); out.write(s + "\n"); out.flush()

    base_sig = signals()
    B = exits(base_sig)
    R2 = rule2(B)
    P("規則 ② 基準")
    P(f"  全期 {fmt(R2)}")
    P(f"  {both(R2)}")
    P(f"  對照：不篩 {both(B)}")

    # ---------------- 1. 參數穩健性 ----------------
    P("\n1. 參數穩健性（只改一個參數，其他維持基準）")
    P("  門檻 thr（IB/ATR）：")
    for thr in [0.0, 0.08, 0.10, 0.12, 0.13, 0.14, 0.16, 0.18, 0.20, 0.25, 9]:
        P(f"    thr={thr:<5} {both(rule2(B, thr))}")
    P("  IB 長度 / 時間窗：")
    for ib_min in (30, 45, 60, 90):
        for win in (2, 3, 5):
            s = exits(signals(ib_min=ib_min, win=win))
            P(f"    IB {ib_min:2d} 分 窗 {win}h  {both(rule2(s))}")
    P("  時段：")
    for lab, hrs in [("01~10", range(1, 11)), ("02~10", range(2, 11)), ("02~08", range(2, 9)), ("03~12", range(3, 13)), ("02~14", range(2, 15))]:
        P(f"    {lab}  {both(rule2(exits(signals(hours=hrs))))}")

    # ---------------- 2. 篩選是否優於隨機 ----------------
    P("\n2. 篩選 vs 隨機刪掉同樣比例的單（2000 次）")
    rng_ = np.random.default_rng(0)
    for lab, X in [("IS", B[B.day < SPLIT]), ("OOS", B[B.day >= SPLIT])]:
        keep = rule2(X); k = len(keep); real = keep.pnl.mean()
        sims = np.array([X.pnl.values[rng_.choice(len(X), k, replace=False)].mean() for _ in range(2000)])
        P(f"  {lab}: 規則② 每筆 {real:+.2f}；隨機保留 {k}/{len(X)} 筆 每筆平均 {sims.mean():+.2f}，"
          f"規則② 勝過 {(sims < real).mean() * 100:.1f}% 的隨機版本")
    rem = B[B.aligned & (B.ib_atr < 0.13)]
    P(f"  被刪掉的單本身：{both(rem)}")

    # ---------------- 3. 同時持倉上限 ----------------
    P("\n3. 同時持倉上限")
    for n in (1, 2, 3, 4, 6, 99):
        P(f"  最多 {n:2d} 筆  {both(cap(R2, n))}")

    # ---------------- 4. 出場方式 ----------------
    P("\n4. 出場方式（套規則②）")
    for lab, kw in [("基準：IB 另一側停損、收盤出", {}), ("停損 IB 中點", dict(stop="mid")), ("停損 0.75×IB", dict(stop=0.75)),
                    ("保本 1×IB", dict(be=1.0)), ("保本 2×IB", dict(be=2.0)), ("移動停損 1×IB", dict(trail=1.0)),
                    ("移動停損 2×IB", dict(trail=2.0)), ("持有 2 小時", dict(hold=2)), ("持有 4 小時", dict(hold=4)),
                    ("持有 8 小時", dict(hold=8)), ("中點停損 + 保本 2×IB", dict(stop="mid", be=2.0))]:
        P(f"  {lab:24s} {both(rule2(exits(base_sig, **kw)))}")

    # ---------------- 5. 拆解 ----------------
    P("\n5. 拆解（規則②）")
    for lab, m in [("做多", R2.side == 1), ("做空", R2.side == -1), ("順前日方向", R2.aligned), ("逆前日方向", ~R2.aligned)]:
        P(f"  {lab:8s} {both(R2[m])}")
    for wd, nm in enumerate("一二三四五"):
        P(f"  星期{nm}    {both(R2[R2.day.dt.dayofweek == wd])}")
    for h in range(2, 11):
        P(f"  {h:02d} 點     {both(R2[R2.h == h])}")
    mm = R2.groupby(R2.day.dt.to_period("M")).pnl.agg(["count", "sum", "mean"])
    P("  逐月：" + "  ".join(f"{k.strftime('%m')}月 {v['sum']:+.0f}/{int(v['count'])}筆" for k, v in mm.iterrows()))
    wk = R2.groupby(R2.day.dt.to_period("W")).pnl.sum()
    P(f"  週損益：正 {(wk > 0).mean() * 100:.0f}% 的週；最差週 {wk.min():+.0f}，最好週 {wk.max():+.0f}")
    dd = R2.groupby("day").pnl.sum()
    top = dd.sort_values(ascending=False)
    P(f"  獲利集中度：最好的 5 天貢獻 {top.head(5).sum():+.0f} / 總 {dd.sum():+.0f}；拿掉最好 10 天後總計 {top.iloc[10:].sum():+.0f}")

    # ---------------- 6. Bootstrap ----------------
    P("\n6. Bootstrap（以「天」為單位重抽 5000 次，保留同一天內多筆的相關性）")
    days_ = R2.day.unique(); byday = {k: g.pnl.values for k, g in R2.groupby("day")}
    pfs, dds, tots = [], [], []
    for _ in range(5000):
        pick = rng_.choice(days_, len(days_), replace=True)
        p = np.concatenate([byday[k] for k in pick])
        pfs.append(p[p > 0].sum() / -p[p <= 0].sum())
        eq = np.cumsum([byday[k].sum() for k in pick]); dds.append((eq - np.maximum.accumulate(eq)).min()); tots.append(p.sum())
    pfs, dds, tots = map(np.array, (pfs, dds, tots))
    P(f"  PF 5%/50%/95%：{np.percentile(pfs, 5):.2f} / {np.median(pfs):.2f} / {np.percentile(pfs, 95):.2f}；PF<1 的機率 {(pfs < 1).mean() * 100:.1f}%")
    P(f"  最大回撤（日損益序列）5%/50%：{np.percentile(dds, 5):.0f} / {np.median(dds):.0f}；總損益 5%：{np.percentile(tots, 5):+.0f}")

    # ---------------- 7. 成本壓力 ----------------
    P("\n7. 成本壓力（每筆額外扣美元/盎司）")
    for c in (0, 0.2, 0.5, 1.0, 2.0):
        x = R2.copy(); x["pnl"] -= c
        P(f"  +{c:<4} {both(x)}")

    # ---------------- 8. 資金管理 ----------------
    P("\n8. 資金管理模擬：起始 10,000 美元，每筆風險 = 帳戶 r%（停損距離 = IB 區間 + 點差），依出場順序結算")
    R2s = R2.sort_values("t_out")
    for cap_n in (99, 3):
        X = cap(R2, cap_n).sort_values("t_out") if cap_n < 99 else R2s
        for r in (0.25, 0.5, 1.0):
            eq, peak, mdd, lots = 10000.0, 10000.0, 0.0, []
            for x in X.itertuples():
                oz = eq * r / 100 / (x.rng + x.spread)
                lots.append(oz / 100)
                eq += oz * x.pnl
                peak = max(peak, eq); mdd = min(mdd, eq / peak - 1)
            P(f"  持倉上限 {cap_n if cap_n < 99 else '無':>2} 每筆風險 {r:4}%：期末 {eq:9,.0f}（{(eq / 10000 - 1) * 100:+.0f}%），"
              f"最大回撤 {mdd * 100:.1f}%，平均每筆 {np.mean(lots):.2f} 手")
    out.close()
