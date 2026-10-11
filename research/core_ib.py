"""IB 突破核心引擎（第六十三部分起使用）—— 可調每個面向，速度比 part17_balance.gen 快。

核心（= 第六十二部分 L1）：
  交易日 = broker 01:00 ~ 23:59（M1 ≥ 1000 根才算）；ATR10 = 前 10 天日線真實波幅平均（不含當天）
  IB：起點 02:00 ~ 10:30 每 30 分，長度 60 分（M1 至少 80% 才算）
  突破：IB 結束後 3 小時內，M5 收盤（每 5 分鐘最後一根 M1 收盤）> IB 高 + 0.05 × ATR10（空單鏡像）；先發生的方向
  進場：該根 M1 收盤價；停損 IB 另一側；當天最後一根收盤出場；同一根同方向只留一筆（最早的 IB）
  濾網：前一天日效率 |收−開|/(高−低) > 0.633 的日子不做
每筆扣進場那根 M1 的點差。金額單位 = 美元/盎司（0.01 手）。
"""
import os
import numpy as np
import pandas as pd
from load import load

DATA = "data/pseudo/XAUUSD_M1_2024_2026.csv"
RESEARCH = ("2024-11-01", "2026-01-01")
VALID = ("2026-01-01", "2026-11-01")
SUBS = [("24/11~12", "2024-11-01", "2025-01-01"), ("25H1", "2025-01-01", "2025-07-01"), ("25H2", "2025-07-01", "2026-01-01")]
PEFF_MAX = 0.633


def _load():
    if not os.path.exists(DATA):
        import part60_A_m1_oos
        part60_A_m1_oos.build()
    m = load(DATA)
    m["day"] = (m.index - pd.Timedelta(hours=1)).normalize()
    days, arr, rows = [], {}, []
    for day, g in m.groupby("day"):
        if len(g) < 1000:
            continue
        mins = ((g.index - day).total_seconds() // 60).astype(int).to_numpy()
        arr[day] = (g.index, mins, g.open.to_numpy(), g.high.to_numpy(), g.low.to_numpy(), g.close.to_numpy(),
                    g.spread.to_numpy() * 0.01)
        rows.append(dict(day=day, o=g.open.iloc[0], h=g.high.max(), l=g.low.min(), c=g.close.iloc[-1]))
        days.append(day)
    D = pd.DataFrame(rows).set_index("day")
    pc = D.c.shift()
    D["tr"] = np.maximum(D.h - D.l, np.maximum((D.h - pc).abs(), (D.l - pc).abs()))
    D["atr10"] = D.tr.rolling(10).mean().shift()
    eff = (D.c - D.o).abs() / (D.h - D.l)
    D["peff"] = eff.shift()
    D["pdir"] = np.sign(D.c - D.o).shift()
    D["trend20"] = ((D.c - D.c.shift(20)) / (D.tr.rolling(20).mean() * np.sqrt(20))).shift()   # 帶方向
    D["eff20"] = eff.rolling(20).mean().shift()
    return days, arr, D


DAYS, ARR, D = _load()


def events(ib_min=60, hours=(2, 10.5), step=30, win_h=3.0, confirm="close5", buf=0.05, monday=True):
    """回傳突破事件 DataFrame（每個 IB 最多一個）。win_h=None 表示到當天結束。"""
    out = []
    for day in DAYS:
        if day.dayofweek == 0 and not monday:
            continue
        atr = D.at[day, "atr10"]
        if not np.isfinite(atr):
            continue
        t, mins, O, H, L, C, S = ARR[day]
        s0 = int(hours[0] * 60)
        while s0 <= hours[1] * 60:
            m = (mins >= s0) & (mins < s0 + ib_min)
            if m.sum() >= ib_min * 0.8:
                ibh, ibl = H[m].max(), L[m].min()
                up_l, dn_l = ibh + buf * atr, ibl - buf * atr
                end = s0 + ib_min + (win_h * 60 if win_h else 10 ** 6)
                w = np.flatnonzero((mins >= s0 + ib_min) & (mins < end))
                if confirm in ("close1", "close5", "close15"):
                    k = int(confirm[5:])
                    w = w[(mins[w] + 1) % k == 0]
                    up, dn = w[C[w] > up_l], w[C[w] < dn_l]
                else:
                    up, dn = w[H[w] > up_l], w[L[w] < dn_l]
                if len(up) and (not len(dn) or up[0] < dn[0]):
                    side, i = 1, up[0]
                elif len(dn):
                    side, i = -1, dn[0]
                else:
                    side = 0
                if side:
                    if confirm == "touch":
                        lvl = max(up_l, O[i]) if side == 1 else min(dn_l, O[i])
                        i0 = i
                    else:
                        lvl, i0 = C[i], i + 1
                    out.append(dict(day=day, s0=s0, i=i, i0=min(i0, len(C) - 1), side=side, lvl=lvl, ibh=ibh, ibl=ibl,
                                    atr=atr, t_in=t[i], spread=S[i], min_in=mins[i]))
            s0 += step
    E = pd.DataFrame(out)
    E = E.join(D[["peff", "pdir", "trend20", "eff20"]], on="day")
    E["aligned"] = E.side == E.pdir
    return E.sort_values(["t_in", "s0"]).drop_duplicates(["t_in", "side"]).reset_index(drop=True)


def simulate(E, stop="ib", exit_min=None, tp=None, hold_min=None, be=None, entry="market", retest_bars=30):
    """stop: 'ib' 另一側 / 'mid' IB 中點 / ('atr', k) 進場價 ∓ k×ATR / ('cap', k) IB 另一側但最多 k×ATR
       exit_min: 當天第幾分鐘（從 00:00 起算）強制平倉；tp: R 倍數停利；hold_min: 最長持有分鐘；be: 到 be×R 後停損移到進場價
       entry: 'market' 突破確認後進場 / 'retest' 確認後掛限價在 IB 邊（retest_bars 根內沒回來就取消）"""
    res = []
    for r in E.itertuples():
        t, mins, O, H, L, C, S = ARR[r.day]
        s = r.side
        n = len(C)
        if entry == "retest":
            edge = r.ibh if s == 1 else r.ibl
            seg = range(r.i0, min(r.i0 + retest_bars, n))
            k = next((k for k in seg if (L[k] <= edge if s == 1 else H[k] >= edge)), None)
            if k is None:
                res.append((np.nan, np.nan, -1, ""))
                continue
            lvl, start = (min(edge, O[k]) if s == 1 else max(edge, O[k])), k
        else:
            lvl, start = r.lvl, r.i0
        if stop == "ib":
            st = r.ibl if s == 1 else r.ibh
        elif stop == "mid":
            st = (r.ibh + r.ibl) / 2
        elif stop[0] == "atr":
            st = lvl - s * stop[1] * r.atr
        else:                                                   # ("cap", k)
            st = r.ibl if s == 1 else r.ibh
            if s * (lvl - st) > stop[1] * r.atr:
                st = lvl - s * stop[1] * r.atr
        risk = s * (lvl - st)
        if not risk > 0:
            res.append((np.nan, np.nan, -1, ""))
            continue
        last = n - 1
        if exit_min is not None:
            cut = np.flatnonzero(mins >= exit_min)
            last = min(last, cut[0]) if len(cut) else last
        if hold_min is not None:
            last = min(last, int(np.searchsorted(mins, mins[start] + hold_min)))
        last = min(last, n - 1)
        if start > last:
            res.append((np.nan, np.nan, -1, ""))
            continue
        tgt = lvl + s * tp * risk if tp else None
        px, j, why = C[last], last, "時間"
        cur = st
        for k in range(start, last + 1):
            if (L[k] <= cur) if s == 1 else (H[k] >= cur):
                px = (min(O[k], cur) if s == 1 else max(O[k], cur)) if k > start else cur
                j, why = k, "停損"
                break
            if tgt is not None and ((H[k] >= tgt) if s == 1 else (L[k] <= tgt)):
                px, j, why = tgt, k, "停利"
                break
            if be is not None and ((H[k] >= lvl + be * risk) if s == 1 else (L[k] <= lvl - be * risk)):
                cur = max(cur, lvl) if s == 1 else min(cur, lvl)
        pnl = s * (px - lvl) - r.spread
        res.append((pnl, pnl / risk, j, why))
    X = E.copy()
    X["pnl"], X["R"], X["j"], X["why"] = zip(*res) if res else ([], [], [], [])
    return X[X.pnl.notna()]


def stats(X):
    p = X.pnl
    if len(p) < 5:
        return dict(n=len(p))
    d = X.groupby("day").pnl.sum()
    return dict(n=len(p), 每天=round(len(p) / X.day.nunique(), 1), PF=round(p[p > 0].sum() / -p[p <= 0].sum(), 2),
                每筆=round(p.mean(), 2), 每筆R=round(X.R.mean(), 3), 日t=round(d.mean() / d.std(ddof=1) * np.sqrt(len(d)), 2),
                總=round(p.sum()))


def report(X, filt=True):
    """研究期（含三個子期間）與驗證期的 PF / 每筆 / 日 t。filt=True 時套用 peff 濾網。"""
    if filt:
        X = X[X.peff <= PEFF_MAX]
    row = {}
    for lab, a, b in SUBS + [("研究", *RESEARCH), ("驗證26", *VALID)]:
        s = stats(X[(X.day >= a) & (X.day < b)])
        row[f"{lab} PF"] = s.get("PF")
        if lab in ("研究", "驗證26"):
            row[f"{lab} 每筆"] = s.get("每筆")
            row[f"{lab} R"] = s.get("每筆R")
            row[f"{lab} 日t"] = s.get("日t")
            row[f"{lab} 筆/天"] = s.get("每天")
            row[f"{lab} 總"] = s.get("總")
    return row
