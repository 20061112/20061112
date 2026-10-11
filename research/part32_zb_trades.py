"""第三十二部分：第三十部分 S2「IB 高上方抱多、IB 低下方抱空」（ZB，亞洲 IB 01:00~02:00、格距 0.1 ATR）的逐筆交易統計。

規則（與 part29/part30 的 ZB 完全相同，只是逐筆記錄）：
  多：價格碰到 IB 高 → 買 1 單位（IB 結束時已在 IB 高上方 → 02:00 開盤市價買）；跌到 IB 高 − 0.1 ATR10 → 賣出。可重複進場。
  空：IB 低鏡像。23:50 全平。成交在線價（模型同前），買進付點差。1 單位 = 1 盎司（0.01 手）。
輸出：results_part32.txt、part32_zb_trades.csv
"""
import numpy as np, pandas as pd
import part29_va_grid as V
from part28_ib_grid import DAYS, CUT, FLAT

S0, IB_MIN, M = 60, 60, 0.10
T = []
for day, atr, mins, ohlc, sp in DAYS:
    ib = (mins >= S0) & (mins < S0 + IB_MIN)
    if ib.sum() < IB_MIN * 0.8: continue
    ibh, ibl = ohlc[ib, 1].max(), ohlc[ib, 2].min(); g = M * atr
    idx = np.where((mins >= S0 + IB_MIN) & (mins < FLAT))[0]
    Z = {1: dict(ref=ibh), -1: dict(ref=ibl)}
    j0 = idx[0]; x0 = ohlc[j0, 0]
    for sd, z in Z.items():
        z["k"] = int(np.floor(sd * (x0 - z["ref"]) / g)); z["open"] = None
        if z["k"] >= 0: z["open"] = (j0, x0 + (sp[j0] if sd == 1 else 0))
    for j in idx:
        for x in V.path(ohlc, j)[(1 if j == j0 else 0):]:
            for sd, z in Z.items():
                u = sd * (x - z["ref"]) / g
                while u >= z["k"] + 1 or u <= z["k"] - 1:
                    z["k"] += 1 if u >= z["k"] + 1 else -1
                    px = z["ref"] + sd * z["k"] * g
                    if z["k"] == 0 and z["open"] is None:
                        z["open"] = (j, px + (sp[j] if sd == 1 else 0))
                    elif z["k"] == -1 and z["open"] is not None:
                        ji, pi = z["open"]; po = px + (sp[j] if sd == -1 else 0)
                        T.append(dict(day=day, side=sd, t_in=mins[ji], t_out=mins[j], p_in=pi, p_out=po,
                                      pnl=sd * (po - pi), atr=atr, ibw=ibh - ibl, exit="回到 IB")); z["open"] = None
    jl = idx[-1]
    for sd, z in Z.items():
        if z["open"] is not None:
            ji, pi = z["open"]; po = ohlc[jl, 3] + (sp[jl] if sd == -1 else 0)
            T.append(dict(day=day, side=sd, t_in=mins[ji], t_out=mins[jl], p_in=pi, p_out=po,
                          pnl=sd * (po - pi), atr=atr, ibw=ibh - ibl, exit="收盤"))
T = pd.DataFrame(T); T["R_atr"] = T.pnl / T.atr; T["hold_min"] = T.t_out - T.t_in; T["year"] = T.day.dt.year
alldays = pd.Series([d for d, *_ in DAYS])
fmt = lambda m: f"{int(m // 60):02d}:{int(m % 60):02d}"
T["in"] = T.t_in.map(fmt); T["out"] = T.t_out.map(fmt)
T.drop(columns=["t_in", "t_out"]).to_csv("part32_zb_trades.csv", index=False, float_format="%.3f")

out = open("results_part32.txt", "w")
def P(s=""): print(s); out.write(s + "\n")
nd = T.groupby("day").size().reindex(alldays, fill_value=0)
daily = T.groupby("day").R_atr.sum().reindex(alldays, fill_value=0)
P(f"第三十二部分：ZB 逐筆統計（亞洲 IB、格距 0.1 ATR）｜{alldays.min().date()} ~ {alldays.max().date()}，{len(alldays)} 個交易日")
P(f"檢查：每日損益平均 {daily.mean():+.4f} ATR（第三十部分 ZB = 全期約 +0.062）")
P(f"\n總筆數 {len(T)}，平均每交易日 {nd.mean():.2f} 筆（中位 {nd.median():.0f}，P90 {nd.quantile(.9):.0f}，最多 {nd.max()}）")
P("每天筆數分布：" + "  ".join(f"{k}筆:{v}天" for k, v in nd.value_counts().sort_index().items()))
P(f"多 {(T.side == 1).sum()} 筆、空 {(T.side == -1).sum()} 筆；出場：" + "、".join(f"{k} {v}" for k, v in T.exit.value_counts().items()))
w, l = T[T.pnl > 0], T[T.pnl <= 0]
P(f"勝率 {len(w) / len(T):.0%}｜平均獲利 {w.R_atr.mean():+.3f} ATR、平均虧損 {l.R_atr.mean():+.3f} ATR、賺賠比 {w.R_atr.mean() / -l.R_atr.mean():.1f}"
  f"｜每筆 {T.R_atr.mean():+.4f} ATR｜PF {w.pnl.sum() / -l.pnl.sum():.2f}")
P(f"持倉時間 中位 {T.hold_min.median():.0f} 分、平均 {T.hold_min.mean():.0f} 分；收盤出場的單 中位 {T[T.exit == '收盤'].hold_min.median():.0f} 分")
P(f"獲利日 {(daily > 0).mean():.0%}、虧損日 {(daily < 0).mean():.0%}、沒交易 {(nd == 0).mean():.0%}")
P("\n逐年（美元為每筆 1 盎司 = 0.01 手）")
P(f"  {'年':>4s} {'交易日':>5s} {'筆數':>5s} {'每日筆':>6s} {'勝率':>5s} {'每筆ATR':>8s} {'PF':>5s} {'ATR中位$':>8s} {'每筆$':>7s} {'年合計$':>8s} {'獲利$':>7s} {'虧損$':>7s}")
for y, x in T.groupby("year"):
    dd = alldays[alldays.dt.year == y]
    P(f"  {y:4d} {len(dd):5d} {len(x):5d} {len(x) / len(dd):6.2f} {(x.pnl > 0).mean():5.0%} {x.R_atr.mean():+8.4f} "
      f"{x.pnl[x.pnl > 0].sum() / -x.pnl[x.pnl <= 0].sum():5.2f} {x.atr.median():8.1f} {x.pnl.mean():+7.2f} {x.pnl.sum():+8.0f} "
      f"{x.pnl[x.pnl > 0].mean():+7.2f} {x.pnl[x.pnl <= 0].mean():+7.2f}")
P("\n進場時間（broker 時間，小時）筆數：" + "  ".join(f"{h:02d}:{v}" for h, v in T["in"].str[:2].astype(int).value_counts().sort_index().items()))
T["n_in_day"] = T.groupby("day").cumcount() + 1
P("當天第幾筆的每筆 ATR：" + "  ".join(f"第{k}筆:{v.R_atr.mean():+.3f}（{len(v)}）" for k, v in T.groupby(T.n_in_day.clip(upper=6))))
out.close()
