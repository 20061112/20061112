"""第五十四部分：使用者分支的「IB 突破核心」（core_ib.py，第 62~68 部分）× M15 最終版。
IB 核心規則（不改）：IB 起點 02:00~10:30 每 30 分、長度 60 分；IB 結束後 3 小時內 M5 收盤 > IB 高 + 0.05 ATR10 進場（空鏡像）；
停損 IB 另一側；收盤平倉；前一天日效率 > 0.633 不做；週一照做；不限疊單。
資料：data/XAUUSD_M1_2023_2026.csv（2023/6 ~ 2026/10）。原分支：2023/6~2024/10 為它的樣本外、2024/11~2025/12 研究期、2026 驗證。
兩個策略都是 0.02 口，扣點差 + 每筆 0.3 美元滑價。M15 = 最終版 + 兩個防護（RR 3）。"""
import os
os.environ["CORE_IB_DATA"] = "data/XAUUSD_M1_2023_2026.csv"
import numpy as np
import pandas as pd
import core_ib as CI
import part49_safeguards as P

SLIP = 0.3
IDX = P.IDX
C = P.W.C

# ---- IB 核心 ----
X = CI.simulate(CI.events())
X = X[(X.peff <= CI.PEFF_MAX) & (X.day >= "2023-06-15")].copy()
gi, gj = [], []
for r in X.itertuples():
    t = CI.ARR[r.day][0]
    gi.append(IDX.get_indexer([t[r.i0]])[0]); gj.append(IDX.get_indexer([t[int(r.j)]])[0])
X["gi"], X["gj"] = gi, gj
X = X[(X.gi >= 0) & (X.gj >= 0)]
X["net"] = X.pnl - SLIP
X["date"] = X.day
X["entry"] = X.lvl
# ---- M15 ----
M = P.daily_stop(P.run(3).pipe(lambda x: x[~(x.volrel > 2.2).values].reset_index(drop=True)), 15)
M["date"] = M.進場時間.dt.normalize()


def minute_eq(parts):
    n = len(IDX)
    D, E, R = np.zeros(n + 1), np.zeros(n + 1), np.zeros(n + 1)
    for (ii, jj, dd, en, net, q) in parts:
        for a, b, s, e, nt in zip(ii, jj, dd, en, net):
            D[a] += s * q; D[b] -= s * q
            E[a] += s * e * q; E[b] -= s * e * q
            R[b] += nt * q
    D, E, R = np.cumsum(D)[:n], np.cumsum(E)[:n], np.cumsum(R)[:n]
    return R + C * D - E


ib_part = lambda q: (X.gi.values, X.gj.values, X.side.values, X.entry.values, X.net.values, q)
m_part = lambda q: (M.i.values, M.j.values, M.dir.values, M.entry.values, M.net.values, q)
lines = []


def p(s=""):
    print(s, flush=True)
    lines.append(s)


def pf(v):
    return v[v > 0].sum() / -v[v < 0].sum()


p("第五十四部分：IB 突破核心 × M15 最終版（0.02 口，點差 + 0.3 滑價）")
p(f"IB 核心：{len(X)} 筆（{X.day.min().date()} ~ {X.day.max().date()}），每天 {len(X) / X.day.nunique():.1f} 筆")
p("IB 核心 逐年（每筆 1 盎司，扣滑價）PF / 損益：" + "；".join(
    f"{y} PF {pf(g.net):.2f} {g.net.sum():+,.0f}（不扣滑價 PF {pf(g.pnl):.2f}）" for y, g in X.groupby(X.day.dt.year)))
ib_d = (X.net * 2).groupby(X.date).sum()
m_d = (M.net * 2).groupby(M.date).sum()
idx = sorted(set(ib_d.index) | set(m_d.index))
Dd = pd.DataFrame({"IB": ib_d.reindex(idx, fill_value=0), "M15": m_d.reindex(idx, fill_value=0)})
both = Dd[(Dd.IB != 0) & (Dd.M15 != 0)]
p(f"日損益相關：兩者都有交易的日子 {both.IB.corr(both.M15):+.2f}（{len(both)} 天）；全部 {Dd.IB.corr(Dd.M15):+.2f}")


def show(lab, parts, daily):
    eq = minute_eq(parts)
    mm = (np.maximum.accumulate(eq) - eq).max()
    e_d = daily.cumsum(); md = (e_d.cummax() - e_d).max()
    yrs = (daily.index.max() - daily.index.min()).days / 365
    mon = daily.groupby(pd.DatetimeIndex(daily.index).to_period("M")).sum()
    sub = lambda a, b: daily[(daily.index >= pd.Timestamp(a)) & (daily.index < pd.Timestamp(b))]

    def rat(s):
        e = s.cumsum(); m = (e.cummax() - e).max(); return s.sum() / m if m > 0 else np.nan
    yr = " / ".join(f"{daily[pd.DatetimeIndex(daily.index).year == y].sum():+,.0f}" for y in (2023, 2024, 2025, 2026))
    p(f"  {lab:28s} 總 {daily.sum():+8,.0f} 每年 {daily.sum() / yrs:+7,.0f} | MDD 含浮虧 {mm:6,.0f} 日結 {md:6,.0f} | 年獲利/MDD {daily.sum() / yrs / mm:4.2f} | "
      f"最差月 {mon.min():+7,.0f} 正月 {(mon > 0).sum()}/{len(mon)} | 比值 23/6~24/10 {rat(sub('2023-06-01', '2024-11-01')):4.1f}、24/11~25/12 {rat(sub('2024-11-01', '2026-01-01')):4.1f}、2026 {rat(sub('2026-01-01', '2027-01-01')):4.1f} | {yr}")


p("\n=== 單獨與組合（0.02 口 = 每筆 2 盎司）===")
show("IB 核心 0.02 口", [ib_part(2)], Dd.IB)
show("M15 最終版 0.02 口", [m_part(2)], Dd.M15)
show("兩者各 0.02 口", [ib_part(2), m_part(2)], Dd.IB + Dd.M15)
show("兩者各 0.01 口（總曝險同單一）", [ib_part(1), m_part(1)], (Dd.IB + Dd.M15) / 2)
show("IB 0.01 + M15 0.02", [ib_part(1), m_part(2)], Dd.IB / 2 + Dd.M15)
open("results_part54.txt", "w").write("\n".join(lines) + "\n")
