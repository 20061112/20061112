"""第二十二部分：用 M5 / M15 K 棒重建版本 A，做真正的樣本外驗證（2023~2025 年從沒看過的資料）。

和 M1 版本（final_A.py）規則一模一樣，只有資料粒度不同：
  - IB 高低：IB 60 分鐘內的 K 棒高低（M5 / M15 都和 M1 的結果完全一樣，因為 IB 起點落在 30 分整點）
  - 突破確認：K 棒收盤 > IB 高 + 0.05×ATR10（M5 資料 = 與 M1 版的「M5 收盤」相同；M15 資料 = 改用 M15 收盤，較晚確認）
  - 進場：確認那根 K 棒的收盤價；停損從下一根開始檢查；跳空用開盤價成交
  - TPO 字母：15 分鐘一個字母（M5 / M15 都能精確得到每個字母的高低）
  - ATR10：前 10 個完整交易日的真實波幅平均（不含當天）
篩選門檻：每月月初用「之前所有訊號」重算分位數（IB/ATR 40%、tail 90%、ext_letter 90%），前 2 個月只當暖身。
"""
import numpy as np
import pandas as pd
from load import load

Q_IB, Q_TAIL, Q_EXT = 0.4, 0.9, 0.9


def load_bars(path, bar_min):
    d = load(path)
    d["sp"] = d.spread * 0.01
    d["day"] = (d.index - pd.Timedelta(hours=1)).normalize()
    full = int(23 * 60 / bar_min * 0.7)                    # 完整交易日至少 70% 的 K 棒
    cnt = d.groupby("day").size()
    days = sorted(cnt[cnt >= full].index)
    G = {k: g for k, g in d.groupby("day") if k in set(days)}
    D = pd.DataFrame({k: dict(open=g.open.iloc[0], high=g.high.max(), low=g.low.min(), close=g.close.iloc[-1]) for k, g in G.items()}).T
    D["tr"] = np.maximum(D.high - D.low, np.maximum((D.high - D.close.shift()).abs(), (D.low - D.close.shift()).abs()))
    D["atr10"] = D.tr.rolling(10).mean().shift(1)
    ARR = {k: (g.index, g.open.values, g.high.values, g.low.values, g.close.values, g.sp.values,
               ((g.index - k).total_seconds().values // 60).astype(int)) for k, g in G.items()}
    return days, D, ARR


def profile_feats(H, L, mins, upto, side, atr):
    """當天 01:00 到 upto（含）的 15 分字母剖面 → tail（反向單印尾巴 / ATR）、ext_letter（反向極值字母位置）。"""
    Hh, Ll, mm = H[:upto], L[:upto], mins[:upto]
    per = mm // 15
    ks = np.unique(per)
    lo0 = np.floor(Ll.min()); n = int(np.floor(Hh.max()) - lo0) + 1
    diff = np.zeros(n + 1)
    for k in ks:
        m = per == k
        diff[int(np.floor(Ll[m].min()) - lo0)] += 1; diff[int(np.floor(Hh[m].max()) - lo0) + 1] -= 1
    cnt = np.cumsum(diff)[:n]
    arr = cnt if side == 1 else cnt[::-1]
    tail = 0
    for c in arr:
        if c == 1: tail += 1
        else: break
    j = np.argmin(Ll) if side == 1 else np.argmax(Hh)
    ext = np.searchsorted(ks, per[j]) / max(len(ks) - 1, 1)
    return tail / atr, ext


def signals(days, D, ARR, bar_min, confirm="close", buf=0.05, ib_min=60, step=30, hours=(2, 10), win=3, monday=False):
    out = []
    for di, day in enumerate(days):
        if di == 0 or (day.dayofweek == 0 and not monday):
            continue
        atr = D.atr10[day]
        if not np.isfinite(atr):
            continue
        prev = D.loc[days[di - 1]]; pdir = np.sign(prev.close - prev.open)
        t, O, H, L, C, S, mins = ARR[day]
        for s0 in range(hours[0] * 60, hours[1] * 60 + 60, step):
            m = (mins >= s0) & (mins < s0 + ib_min)
            if m.sum() < ib_min / bar_min * 0.8:
                continue
            ibh, ibl = H[m].max(), L[m].min()
            up_l, dn_l = ibh + buf * atr, ibl - buf * atr
            w = np.where((mins >= s0 + ib_min) & (mins < s0 + ib_min + win * 60))[0]
            if confirm == "close":
                up = w[C[w] > up_l]; dn = w[C[w] < dn_l]
            else:
                up = w[H[w] > up_l]; dn = w[L[w] < dn_l]
            if len(up) and (not len(dn) or up[0] < dn[0]): side, i = 1, up[0]
            elif len(dn): side, i = -1, dn[0]
            else: continue
            if confirm == "close":
                lvl, i_ex, t_in = C[i], i + 1, t[i] + pd.Timedelta(minutes=bar_min)
            else:
                lvl = max(up_l, O[i]) if side == 1 else min(dn_l, O[i]); i_ex, t_in = i, t[i]
            tail, ext = profile_feats(H, L, mins, i + 1, side, atr)
            out.append(dict(day=day, h=s0 / 60, i=i_ex, side=side, lvl=lvl, ibh=ibh, ibl=ibl, rng=ibh - ibl, atr=atr,
                            ib_atr=(ibh - ibl) / atr, aligned=bool(side == pdir), t_in=t_in, spread=S[i], tail=tail, ext_letter=ext))
    return pd.DataFrame(out)


def simulate(s, ARR, bar_min, stop_mode="opp"):
    rows = []
    for r in s.itertuples():
        t, O, H, L, C, S, mins = ARR[r.day]
        side, lvl = r.side, r.lvl
        stop = (r.ibl if side == 1 else r.ibh) if stop_mode == "opp" else (r.ibh + r.ibl) / 2
        risk = side * (lvl - stop)
        if risk <= 0:
            stop = r.ibl if side == 1 else r.ibh; risk = side * (lvl - stop)
        px, j, why = C[-1], len(C) - 1, "收盤"
        for k in range(r.i, len(C)):
            if (L[k] <= stop) if side == 1 else (H[k] >= stop):
                px = min(O[k], stop) if side == 1 else max(O[k], stop); j, why = k, "停損"; break
        pnl = side * (px - lvl) - r.spread
        rows.append(dict(exit_time=t[j] + pd.Timedelta(minutes=bar_min), exit_px=px, exit_reason=why, risk=risk, pnl=pnl,
                         R=pnl / risk if risk > 0 else np.nan))
    return pd.concat([s.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def walk_forward(X, q_tail=Q_TAIL, q_ext=Q_EXT, q_ib=Q_IB, warmup=2):
    X = X.copy(); X["ym"] = X.day.dt.to_period("M"); parts = []
    for m in sorted(X.ym.unique())[warmup:]:
        past, cur = X[X.ym < m], X[X.ym == m]
        k = ~(cur.aligned & (cur.ib_atr < past.ib_atr.quantile(q_ib))) & (cur["tail"] <= past["tail"].quantile(q_tail)) \
            & (cur.ext_letter.fillna(0) <= past.ext_letter.quantile(q_ext))
        parts.append(cur[k])
    return pd.concat(parts).sort_values(["t_in", "h"]).drop_duplicates(["t_in", "side"]).sort_values("t_in").reset_index(drop=True)


def stats(T, days):
    if not len(T):
        return dict(n=0)
    p = T.pnl; w, l = p[p > 0], p[p <= 0]
    td = [d for d in days if T.day.min() <= d <= T.day.max() and d.dayofweek != 0]
    daily = pd.Series(0.0, index=td).add(T.groupby("day").pnl.sum(), fill_value=0)
    eq = T.sort_values("exit_time").pnl.cumsum(); dd = (eq - eq.cummax()).min()
    eqR = (T.sort_values("exit_time").R * 0.5).cumsum()          # 每筆 0.5% 風險、不複利（%）
    mon = T.groupby(T.day.dt.to_period("M")).pnl.sum()
    lose = (p <= 0).astype(int)
    return dict(n=len(T), per_day=len(T) / max(len(td), 1), win=len(w) / len(p) * 100, avg=p.mean(), avgR=T.R.mean(),
                pf=w.sum() / -l.sum() if len(l) else np.inf, total=p.sum(), dd=dd, ret_pct=eqR.iloc[-1],
                dd_pct=(eqR - eqR.cummax()).min(), sharpe=daily.mean() / daily.std() * np.sqrt(250) if daily.std() > 0 else np.nan,
                mon_pos=(mon > 0).mean() * 100, n_mon=len(mon),
                streak=lose.groupby((lose != lose.shift()).cumsum()).sum().max(), atr=T.atr.median())
