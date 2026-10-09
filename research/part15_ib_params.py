"""第十五部分 A：IB 固定參數掃描（基準 = 規則 ②+：不做順前日方向的小 IB、不做星期一、收盤出場、IB 另一側停損）。

掃描：
  IB 長度 ib_min：15 / 30 / 45 / 60 / 90 / 120 分
  起點間隔 step：60 分（每個整點）/ 30 分（每個整點與半點）
  突破確認 confirm：touch（碰到 IB 邊就在 IB 邊成交）/ close1（M1 收盤在 IB 外，以收盤價進場）/ close5（M5 收盤在外）
  緩衝 buf：突破要超過 IB 邊 buf × ATR 才算（進場價 = IB 邊 + buf×ATR）
小 IB 門檻：用 1~5 月該 IB 長度的 ib_atr 40% 分位（60 分時 ≈ 0.13）。
"""
import numpy as np
import pandas as pd
from part14_deep import ARR, D, DAYS, SPLIT, exits, st, both

START_H, END_H = 2, 10          # IB 起點範圍（broker 小時，含 10:xx）


def signals(ib_min=60, step=60, win=3, confirm="touch", buf=0.0):
    out = []
    for di, day in enumerate(DAYS):
        if di == 0 or day.dayofweek == 0:
            continue
        atr = D.atr10.iloc[di]
        if not np.isfinite(atr):
            continue
        prev = D.iloc[di - 1]; pdir = np.sign(prev.close - prev.open)
        t, O, H, L, C, S = ARR[day]
        mins = (t - day).total_seconds().values // 60
        for s0 in range(START_H * 60, END_H * 60 + 60, step):
            m = (mins >= s0) & (mins < s0 + ib_min)
            if m.sum() < ib_min * 0.8:
                continue
            ibh, ibl = H[m].max(), L[m].min()
            up_l, dn_l = ibh + buf * atr, ibl - buf * atr
            w = np.where((mins >= s0 + ib_min) & (mins < s0 + ib_min + win * 60))[0]
            if not len(w):
                continue
            if confirm == "touch":
                up = w[H[w] > up_l]; dn = w[L[w] < dn_l]
            elif confirm == "close1":
                up = w[C[w] > up_l]; dn = w[C[w] < dn_l]
            else:   # close5：只看每 5 分鐘最後一根 M1 的收盤
                w5 = w[(mins[w] + 1) % 5 == 0]
                up = w5[C[w5] > up_l]; dn = w5[C[w5] < dn_l]
            if len(up) and (not len(dn) or up[0] < dn[0]): side, i = 1, up[0]
            elif len(dn): side, i = -1, dn[0]
            else: continue
            lvl = (up_l if side == 1 else dn_l) if confirm == "touch" else C[i]
            if confirm == "touch":
                lvl = max(lvl, O[i]) if side == 1 else min(lvl, O[i])
            i_ex = i if confirm == "touch" else min(i + 1, len(C) - 1)   # 收盤進場：停損從下一根開始檢查
            out.append(dict(day=day, h=s0 // 60, i=i_ex, side=side, lvl=lvl, ibh=ibh, ibl=ibl, rng=ibh - ibl, atr=atr,
                            ib_atr=(ibh - ibl) / atr, aligned=side == pdir, t_in=t[i], spread=S[i]))
    s = pd.DataFrame(out)
    thr = s[s.day < SPLIT].ib_atr.quantile(0.4)
    s = s[~(s.aligned & (s.ib_atr < thr))]
    return s, thr


if __name__ == "__main__":
    out = open("results_part15.txt", "w")
    def P(x=""): print(x); out.write(x + "\n"); out.flush()
    P("第十五部分 A：IB 固定參數（基準 ②+；單位 美元/盎司；IS = 1~5 月，OOS = 6~10 月）")
    rows = []
    for ib_min in (15, 30, 45, 60, 90, 120):
        for step in (60, 30):
            s, thr = signals(ib_min, step)
            x = exits(s)
            a, b = st(x[x.day < SPLIT]), st(x[x.day >= SPLIT])
            rows.append(dict(ib=ib_min, step=step, thr=round(thr, 3), is_day=a["per_day"], is_avg=a["avg"], is_pf=a["pf"],
                             oos_day=b["per_day"], oos_avg=b["avg"], oos_pf=b["pf"], oos_dd=b["maxdd"]))
    R = pd.DataFrame(rows)
    P("\n1) IB 長度 × 起點間隔（觸價進場、無緩衝、3 小時窗）")
    P(R.round(2).to_string(index=False))

    P("\n2) 突破確認方式 × 緩衝（IB 60 分、每整點）")
    for confirm in ("touch", "close1", "close5"):
        for buf in (0.0, 0.02, 0.05):
            s, _ = signals(60, 60, confirm=confirm, buf=buf)
            P(f"  {confirm:6s} buf={buf:<4} {both(exits(s))}")
    out.close()
