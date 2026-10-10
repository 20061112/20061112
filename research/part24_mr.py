"""第二十四部分 A：IB 區間內的均值回歸（M15，2023/3~2026/10），與突破策略互補嗎？

MR1「假突破回歸」：每個 IB（02:00~10:30 每 30 分起 60 分鐘），IB 結束後 3 小時內：
   先有一根 M15 收盤站到 IB 外 + 0.05×ATR（有效突破），之後 lookback 根以內又有一根收盤回到 IB 內
   → 在那根收盤反向進場；停損 = 突破後到回歸前的極值（再加 0.02×ATR）；
   目標 tgt：mid（IB 中點）/ opp（IB 另一側）/ none（只抱到收盤）；沒到目標就收盤出場。
MR2「IB 邊緣反轉（對照）」：IB 結束後 3 小時內第一次碰到 IB 高 / 低就反向做，停損 = 0.5×IB 區間外，目標 IB 中點。
突破基準（BO）：同一組 IB，M15 收盤確認 +0.05ATR、停損 IB 另一側、收盤出場、不篩選、含週一。
"""
import numpy as np
import pandas as pd
from oos_engine import load_bars, signals, simulate

out = open("results_part24.txt", "w")
def P(s=""): print(s); out.write(s + "\n"); out.flush()

days, D, ARR = load_bars("data/XAUUSD_M15_2023_2026.csv", 15)
START = pd.Timestamp("2023-03-01")
PER = [("2023", "2023-03-01", "2023-12-31"), ("2024", "2024-01-01", "2024-12-31"), ("25H1", "2025-01-01", "2025-06-30"),
       ("25H2", "2025-07-01", "2025-12-31"), ("2026", "2026-01-01", "2026-12-31")]


def run_exit(t, O, H, L, C, k0, side, e, stop, tgt, hold=None):
    end = len(C) - 1 if hold is None else min(len(C) - 1, k0 + hold)
    for k in range(k0, end + 1):
        if (L[k] <= stop) if side == 1 else (H[k] >= stop):
            return (min(O[k], stop) if side == 1 else max(O[k], stop)), k, "停損"
        if tgt is not None and ((H[k] >= tgt) if side == 1 else (L[k] <= tgt)):
            return (max(O[k], tgt) if side == 1 else min(O[k], tgt)), k, "目標"
    return C[end], end, "時間"


def mr1(tgt_mode="mid", lookback=4, buf=0.05, sbuf=0.02, hold=None):
    rows = []
    for di, day in enumerate(days):
        if di == 0 or day < START:
            continue
        atr = D.atr10[day]
        if not np.isfinite(atr):
            continue
        t, O, H, L, C, S, mins = ARR[day]
        for s0 in range(120, 660, 30):
            m = (mins >= s0) & (mins < s0 + 60)
            if m.sum() < 3:
                continue
            ibh, ibl = H[m].max(), L[m].min(); mid = (ibh + ibl) / 2
            w = np.where((mins >= s0 + 60) & (mins < s0 + 240))[0]
            up = w[C[w] > ibh + buf * atr]; dn = w[C[w] < ibl - buf * atr]
            if len(up) and (not len(dn) or up[0] < dn[0]): bside, b = 1, up[0]
            elif len(dn): bside, b = -1, dn[0]
            else: continue
            back = [k for k in range(b + 1, min(b + 1 + lookback, len(C))) if (C[k] < ibh if bside == 1 else C[k] > ibl)]
            if not back:
                continue
            k = back[0]; side = -bside; e = C[k]
            ext = H[b:k + 1].max() if bside == 1 else L[b:k + 1].min()
            stop = ext + bside * sbuf * atr
            tgt = mid if tgt_mode == "mid" else (ibl if side == -1 else ibh) if tgt_mode == "opp" else None
            if tgt is not None and side * (tgt - e) <= 0:
                continue
            px, j, why = run_exit(t, O, H, L, C, k + 1, side, e, stop, tgt, hold)
            risk = side * (e - stop)
            pnl = side * (px - e) - S[k]
            rows.append(dict(day=day, h=s0 / 60, side=side, t_in=t[k], exit_time=t[j], risk=risk, pnl=pnl, R=pnl / risk, why=why))
    T = pd.DataFrame(rows)
    return T.sort_values(["t_in", "h"]).drop_duplicates(["t_in", "side"]).reset_index(drop=True)


def mr2():
    rows = []
    for di, day in enumerate(days):
        if di == 0 or day < START:
            continue
        atr = D.atr10[day]
        if not np.isfinite(atr):
            continue
        t, O, H, L, C, S, mins = ARR[day]
        for s0 in range(120, 660, 30):
            m = (mins >= s0) & (mins < s0 + 60)
            if m.sum() < 3:
                continue
            ibh, ibl = H[m].max(), L[m].min(); rg = ibh - ibl; mid = (ibh + ibl) / 2
            w = np.where((mins >= s0 + 60) & (mins < s0 + 240))[0]
            up = w[H[w] >= ibh]; dn = w[L[w] <= ibl]
            if len(up) and (not len(dn) or up[0] < dn[0]): side, k, e = -1, up[0], ibh
            elif len(dn): side, k, e = 1, dn[0], ibl
            else: continue
            stop = e - side * 0.5 * rg
            if (L[k] <= stop) if side == 1 else (H[k] >= stop):
                px, j, why = stop, k, "停損"
            else:
                px, j, why = run_exit(t, O, H, L, C, k + 1, side, e, stop, mid)
            risk = 0.5 * rg; pnl = side * (px - e) - S[k]
            rows.append(dict(day=day, h=s0 / 60, side=side, t_in=t[k], exit_time=t[j], risk=risk, pnl=pnl, R=pnl / risk, why=why))
    return pd.DataFrame(rows).sort_values(["t_in", "h"]).drop_duplicates(["t_in", "side"]).reset_index(drop=True)


pf = lambda p: p[p > 0].sum() / max(-p[p <= 0].sum(), 1e-9)


def line(T, lab):
    cells = []
    for n, a, b in PER:
        x = T[(T.day >= a) & (T.day <= b)]
        cells.append(f"{n} {x.R.mean():+.2f}R/PF{pf(x.pnl):.2f}/勝{(x.pnl > 0).mean() * 100:.0f}%({len(x)})")
    P(f"  {lab:34s} " + "  ".join(cells))


BO = simulate(signals(days, D, ARR, 15, monday=True), ARR, 15)
BO = BO[BO.day >= START].sort_values(["t_in", "h"]).drop_duplicates(["t_in", "side"]).reset_index(drop=True)
P("第二十四部分 A：IB 均值回歸（每格 = 每筆 R / PF / 勝率（筆數））")
line(BO, "突破基準 BO（不篩、含週一）")
res = {}
for tg in ("mid", "opp", "none"):
    for lb in (2, 4):
        T = mr1(tg, lb); res[(tg, lb)] = T
        line(T, f"MR1 假突破回歸 目標={tg} 回歸窗={lb}根")
T = mr1("mid", 4, hold=16); res["hold"] = T
line(T, "MR1 目標=mid 回歸窗=4 最多抱 4 小時")
M2 = mr2(); line(M2, "MR2 IB 邊緣直接反轉（對照）")

P("\n與突破策略的日損益相關（以 R 加總、每天）與合併")
bo_d = BO.groupby("day").R.sum()
for key, lab in [(("mid", 4), "MR1 mid/4"), (("opp", 4), "MR1 opp/4"), (("none", 4), "MR1 none/4")]:
    mr_d = res[key].groupby("day").R.sum()
    J = pd.concat([bo_d.rename("bo"), mr_d.rename("mr")], axis=1).fillna(0)
    J["sum"] = J.bo + J.mr
    cells = []
    for n, a, b in PER:
        x = J[(J.index >= a) & (J.index <= b)]
        sh = lambda s: s.mean() / s.std() * np.sqrt(250)
        cells.append(f"{n} BO {sh(x.bo):+.2f} MR {sh(x.mr):+.2f} 合 {sh(x['sum']):+.2f}")
    P(f"  {lab:12s} 相關 {J.bo.corr(J.mr):+.2f} | 日 Sharpe：" + "  ".join(cells))
for key, T in res.items():
    T.to_csv(f"part24_mr1_{key[0] if isinstance(key, tuple) else key}_{key[1] if isinstance(key, tuple) else 'h16'}.csv", index=False)
BO.to_csv("part24_bo_base.csv", index=False)
out.close()
