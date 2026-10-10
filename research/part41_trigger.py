"""第四十一部分：「突破」這個進場訊號到底有沒有用？

同樣的 OW 條件（整個 IB 在前日價值區外、順方向）、同樣停損（IB 另一側）、2R 保本、收盤出場：
  突破進場（現行）：IB 結束後 3 小時內 收盤 > IB 高 + 0.05ATR（空單反之）才進場
  時間進場：IB 一結束就在下一根開盤進場，不等突破
  時間進場 + 3h 內沒突破（收盤越過 IB 邊 + 0.05ATR）就在窗口結束時出場（不看未來）
  時間進場、但只做「之後有突破」的 IB（看未來，只作對照）
比較 每筆 損益÷ATR（與規模無關）、勝率、0.01 手最多 5 張美元 / Sharpe。
"""
import numpy as np
import pandas as pd
from part27_ow import data, run, pf, PER, prev_profiles

out = open("results_part41.txt", "w")
def P(s=""): print(s); out.write(s + "\n"); out.flush()
SRC = {"M1": ("data/XAUUSD_M1_2026.csv", 1), "M5": ("data/XAUUSD_M5_2025_2026.csv", 5), "M15": ("data/XAUUSD_M15_2023_2026.csv", 15)}


def manage(day, i, s, e, st, ARR, bm, dead=None, lvl=None):
    """dead / lvl：到 dead 這根 K 棒收盤前，若沒有任何收盤越過 lvl（突破價），就在 dead 收盤出場。"""
    t, O, H, L, C, S, mins = ARR[day]
    R = s * (e - st); best = e; be = False; ok = dead is None
    for k in range(i, len(C)):
        if (L[k] <= st) if s == 1 else (H[k] >= st):
            px = min(O[k], st) if s == 1 else max(O[k], st)
            return px, t[k] + pd.Timedelta(minutes=bm)
        best = max(best, H[k]) if s == 1 else min(best, L[k])
        if not be and s * (best - e) >= 2 * R:
            be = True; st = e
        if not ok:
            if s * (C[k] - lvl) > 0:
                ok = True
            elif k >= dead:
                return C[k], t[k] + pd.Timedelta(minutes=bm)
    return C[-1], t[-1] + pd.Timedelta(minutes=bm)


def cap(T, n=5):
    keep, open_ = [], []
    for r in T.sort_values("t_in").itertuples():
        open_ = [x for x in open_ if x > r.t_in]
        if len(open_) < n:
            keep.append(r.Index); open_.append(r.exit_time)
    return T.loc[keep]


def show(name, T, alldays, multi):
    C5 = cap(T).sort_values("exit_time"); eq = C5.pnl.cumsum()
    dl = C5.groupby("day").pnl.sum().reindex(alldays, fill_value=0)
    cells = ("  " + "  ".join(f"{n} {x.u.mean() * 100:+.1f}%" for n, a, b in PER for x in [T[(T.day >= a) & (T.day <= b)]] if len(x))) if multi else ""
    P(f"  {name:30s} {len(T):5d}筆 勝{(T.pnl > 0.05 * T.risk).mean() * 100:3.0f}% 每筆損益÷ATR {T.u.mean() * 100:+5.1f}% PF{pf(T.pnl):.2f} | "
      f"5張 {C5.pnl.sum():+6,.0f}美元 回撤{(eq - eq.cummax()).min():+6,.0f} Sharpe{dl.mean() / dl.std() * np.sqrt(250):.2f}" + cells)


for tf, (path, bm) in SRC.items():
    days, D, ARR = data(path, bm)
    PV = prev_profiles(days, ARR)
    X, _ = run(path, bm)
    B = X[X.ow & (X.day >= "2023-03-01")].sort_values(["t_in", "h"]).drop_duplicates(["t_in", "side"]).reset_index(drop=True)
    ex = [manage(r.day, r.i, r.side, r.lvl, r.ibl if r.side == 1 else r.ibh, ARR, bm) for r in B.itertuples()]
    B["exit_time"] = [b for a, b in ex]; B["pnl"] = B.side * (np.array([a for a, b in ex]) - B.lvl) - B.spread; B["u"] = B.pnl / B.atr
    broke = set(zip(B.day, B.h, B.side))
    rows = []
    for d in days:
        if d < pd.Timestamp("2023-03-01") or d not in PV or not np.isfinite(D.atr10.get(d, np.nan)):
            continue
        t, O, H, L, C, S, mins = ARR[d]; poc, vah, val = PV[d]; atr = D.atr10[d]
        for h in np.arange(2, 11, 0.5):
            m = (mins >= h * 60) & (mins < h * 60 + 60)
            if m.sum() < 60 / bm * 0.8:
                continue
            ibh, ibl = H[m].max(), L[m].min()
            side = 1 if ibl > vah else (-1 if ibh < val else 0)
            if side == 0:
                continue
            nx = np.where(mins >= h * 60 + 60)[0]
            if not len(nx):
                continue
            i = nx[0]; e = O[i]; st = ibl if side == 1 else ibh
            if side * (e - st) <= 0:
                continue
            px, te = manage(d, i, side, e, st, ARR, bm)
            pnl = side * (px - e) - S[i]
            lv = ibh + 0.05 * atr if side == 1 else ibl - 0.05 * atr
            ww = np.where(mins < h * 60 + 60 + 180)[0]
            px2, te2 = manage(d, i, side, e, st, ARR, bm, dead=ww[-1], lvl=lv)
            pnl2 = side * (px2 - e) - S[i]
            rows.append(dict(day=d, h=h, side=side, t_in=t[i], exit_time=te, risk=side * (e - st), pnl=pnl, u=pnl / atr,
                             broke=(d, h, side) in broke, exit2=te2, pnl2=pnl2))
    Tm = pd.DataFrame(rows)
    alldays = [d for d in days if d >= B.day.min()]
    P("\n" + "=" * 130 + f"\n[{tf}]  （右側 M15 各期每筆 損益÷ATR）")
    show("突破進場（現行）", B, alldays, tf == "M15")
    show("時間進場：IB 結束就進", Tm, alldays, tf == "M15")
    show("  其中之後沒有突破的 IB", Tm[~Tm.broke], alldays, tf == "M15")
    show("  其中之後有突破的 IB（看未來）", Tm[Tm.broke], alldays, tf == "M15")
    show("時間進場 + 3h 內沒突破就出場", Tm.assign(exit_time=Tm.exit2, pnl=Tm.pnl2, u=Tm.pnl2 / (Tm.pnl / Tm.u)), alldays, tf == "M15")
out.close()
