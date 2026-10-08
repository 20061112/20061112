"""第十七部分 A：交易筆數 vs 每筆品質的取捨（全部用滾動前推門檻：每月只用過去資料重算分位數）。

可調的開關：
  hours   IB 起點範圍（02~10 / 01~12 / 02~14）
  step    起點間隔 60 分 / 30 分
  confirm 觸價 touch / M5 收盤 close5（+ buf × ATR 緩衝）
  q_tail  tail 篩選的分位數（0.8 = 刪最長 20%；1.0 = 不篩）
  q_ext   ext_letter 篩選的分位數（同上）
  monday  是否做星期一
固定：小 IB 順前日方向的篩選（40% 分位）、IB 另一側停損、收盤出場、同根同向只開一筆。
評分不只看每筆：總損益、總損益/最大回撤、日 Sharpe、每月 PF>1 比例、拿掉最好 10 天後的總損益。
"""
import itertools
import numpy as np
import pandas as pd
from part14_deep import ARR, D, DAYS, SPLIT
from part15_letters import features as basic_features
from part16_adv_letters import adv
from final_system import simulate

CACHE = {}


def gen(confirm="close5", buf=0.05, hours=(2, 10), step=60, monday=False, ib_min=60, win=3):
    key = (confirm, buf, hours, step, monday)
    if key in CACHE:
        return CACHE[key]
    out = []
    for di, day in enumerate(DAYS):
        if di == 0 or (day.dayofweek == 0 and not monday):
            continue
        atr = D.atr10.iloc[di]
        if not np.isfinite(atr):
            continue
        prev = D.iloc[di - 1]; pdir = np.sign(prev.close - prev.open)
        t, O, H, L, C, S = ARR[day]
        mins = (t - day).total_seconds().values // 60
        for s0 in range(hours[0] * 60, hours[1] * 60 + 60, step):
            m = (mins >= s0) & (mins < s0 + ib_min)
            if m.sum() < ib_min * 0.8:
                continue
            ibh, ibl = H[m].max(), L[m].min()
            up_l, dn_l = ibh + buf * atr, ibl - buf * atr
            w = np.where((mins >= s0 + ib_min) & (mins < s0 + ib_min + win * 60))[0]
            if confirm == "close5":
                w = w[(mins[w] + 1) % 5 == 0]
                up = w[C[w] > up_l]; dn = w[C[w] < dn_l]
            else:
                up = w[H[w] > up_l]; dn = w[L[w] < dn_l]
            if len(up) and (not len(dn) or up[0] < dn[0]): side, i = 1, up[0]
            elif len(dn): side, i = -1, dn[0]
            else: continue
            if confirm == "close5":
                lvl, i_ex = C[i], min(i + 1, len(C) - 1)
            else:
                lvl = max(up_l, O[i]) if side == 1 else min(dn_l, O[i]); i_ex = i
            out.append(dict(day=day, h=s0 / 60, i=i_ex, side=side, lvl=lvl, ibh=ibh, ibl=ibl, rng=ibh - ibl, atr=atr,
                            ib_atr=(ibh - ibl) / atr, aligned=bool(side == pdir), t_in=t[i], spread=S[i]))
    s = pd.DataFrame(out)
    s["tail"] = [basic_features(r, 15)["tail"] for r in s.itertuples()]
    s["ext_letter"] = [adv(r).get("ext_letter", np.nan) for r in s.itertuples()]
    X = simulate(s)
    CACHE[key] = X
    return X


def wf(X, q_tail=0.8, q_ext=0.8, q_ib=0.4):
    X = X.copy(); X["ym"] = X.day.dt.to_period("M")
    parts = []
    for m in sorted(X.ym.unique())[2:]:
        past, cur = X[X.ym < m], X[X.ym == m]
        ti = past.ib_atr.quantile(q_ib); tt = past["tail"].quantile(q_tail); te = past.ext_letter.quantile(q_ext)
        keep = ~(cur.aligned & (cur.ib_atr < ti)) & (cur["tail"] <= tt) & (cur.ext_letter.fillna(0) <= te)
        parts.append(cur[keep])
    Y = pd.concat(parts)
    return Y.sort_values(["t_in", "h"]).drop_duplicates(["t_in", "side"])


def score(T):
    p = T.pnl; T = T.sort_values("exit_time"); eq = T.pnl.cumsum(); dd = (eq - eq.cummax()).min()
    days = [d for d in DAYS if T.day.min() <= d <= T.day.max()]
    daily = pd.Series(0.0, index=days).add(T.groupby("day").pnl.sum(), fill_value=0)
    mon = T.groupby(T.day.dt.to_period("M")).pnl.apply(lambda x: x[x > 0].sum() / max(-x[x <= 0].sum(), 1e-9))
    top = T.groupby("day").pnl.sum().sort_values(ascending=False)
    return dict(per_day=len(T) / len(days), n=len(T), avg=p.mean(), avgR=T.R.mean(), pf=p[p > 0].sum() / -p[p <= 0].sum(),
                win=(p > 0).mean() * 100, total=p.sum(), dd=dd, ret_dd=p.sum() / -dd,
                sharpe=daily.mean() / daily.std() * np.sqrt(250), mon_ok=(mon > 1).mean() * 100, ex10=top.iloc[10:].sum())


if __name__ == "__main__":
    out = open("results_part17.txt", "w")
    def P(s=""): print(s); out.write(s + "\n"); out.flush()
    rows = []
    for (confirm, buf), hours, step, (qt, qe), monday in itertools.product(
            [("close5", 0.05), ("close5", 0.02), ("touch", 0.0)], [(2, 10), (1, 12), (2, 14)], [60, 30],
            [(0.8, 0.8), (0.9, 0.9), (0.8, 1.0), (1.0, 1.0)], [False, True]):
        X = gen(confirm, buf, hours, step, monday)
        s = score(wf(X, qt, qe))
        rows.append(dict(confirm=f"{confirm}+{buf}", hours=f"{hours[0]:02d}-{hours[1]:02d}", step=step,
                         filt=f"t{qt}/e{qe}", mon=int(monday), **s))
    R = pd.DataFrame(rows)
    R.to_csv("part17_frontier.csv", index=False)
    pd.set_option("display.width", 250)
    cols = ["confirm", "hours", "step", "filt", "mon", "per_day", "avg", "avgR", "pf", "win", "total", "dd", "ret_dd", "sharpe", "mon_ok", "ex10"]
    P("第十七部分 A：筆數 vs 品質（滾動前推 3~10 月；金額 美元/盎司）")
    P(f"\n目前最佳系統：\n{R[(R.confirm == 'close5+0.05') & (R.hours == '02-10') & (R.step == 60) & (R.filt == 't0.8/e0.8') & (R.mon == 0)][cols].round(2).to_string(index=False)}")
    # 帕累托前緣：在每個筆數區間裡，總損益/回撤 最好的
    R["bucket"] = pd.cut(R.per_day, [0, 3, 4, 5, 6, 8, 10, 13, 40])
    P("\n各筆數區間裡「總損益/回撤」最好的設定（且 PF ≥ 1.5、每月 PF>1 ≥ 75%）")
    ok = R[(R.pf >= 1.5) & (R.mon_ok >= 75)]
    best = ok.loc[ok.groupby("bucket", observed=True).ret_dd.idxmax()]
    P(best[cols].round(2).to_string(index=False))
    P("\n總損益前 10 名")
    P(R.sort_values("total", ascending=False)[cols].head(10).round(2).to_string(index=False))
    out.close()
