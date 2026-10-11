"""第四十八部分：M1 2025 下半年拆解 —— 回撤從哪裡來。

用 part47_m1_trades.csv（最終版 OW + 2R 保本），最多 5 筆、分倉 3/2/1。
逐月、多空、突破快慢、回撤區間的逐日明細、當時金價行情（月漲跌、ATR、日內趨勢比例）。
"""
import numpy as np
import pandas as pd
from part27_ow import data, pf

out = open("results_part48.txt", "w")
def P(s=""): print(s); out.write(s + "\n"); out.flush()
days, D, ARR = data("data/XAUUSD_M1_2024_2026.csv", 1)
T = pd.read_csv("part47_m1_trades.csv", parse_dates=["day", "t_in", "exit_time"])


def cap(T, n=5):
    keep, open_ = [], []
    for r in T.sort_values("t_in").itertuples():
        open_ = [x for x in open_ if x > r.t_in]
        if len(open_) < n:
            keep.append(r.Index); open_.append(r.exit_time)
    return T.loc[keep]


C = cap(T).sort_values("exit_time").copy()
C["usd"] = C.pnl * C.lots
C["why"] = np.where(C.R < -0.9, "停損", np.where(C.R.abs() <= 0.05, "保本", "其他"))
Dm = D.copy(); Dm["ret"] = Dm.close / Dm.open - 1; Dm["ap"] = Dm.atr10 / Dm.close * 100
Dm["trend"] = (Dm.close - Dm.open).abs() / (Dm.high - Dm.low)                 # 日內單邊程度（收−開 ÷ 高−低）

P("A. 逐月（2024/9 ~ 2026/10）：筆數、勝率、每筆R、分倉美元 ｜ 金價：月漲跌、ATR%、日內單邊比例中位、價值區外開盤日比例")
for p, g in C.groupby(C.day.dt.to_period("M")):
    dm = Dm[(Dm.index >= p.start_time) & (Dm.index <= p.end_time)]
    P(f"  {p}  {len(g):3d}筆 勝{(g.R > 0.05).mean() * 100:3.0f}% 每筆{g.R.mean():+.2f}R 美元{g.usd.sum():+6,.0f} 多{(g.side == 1).sum():2d}/空{(g.side == -1).sum():2d} ｜ "
      f"金價 {dm.close.iloc[-1] / dm.open.iloc[0] * 100 - 100:+5.1f}%  ATR {dm.ap.mean():.2f}%  單邊 {dm.trend.median():.2f}")

H = C[(C.day >= "2025-07-01") & (C.day <= "2025-12-31")]
P(f"\nB. 2025 下半年：{len(H)} 筆、每筆 {H.R.mean():+.3f}R、PF {pf(H.R):.2f}、美元 {H.usd.sum():+,.0f}")
for lab, g in [("多單", H[H.side == 1]), ("空單", H[H.side == -1])]:
    P(f"  {lab} {len(g)} 筆 每筆 {g.R.mean():+.2f}R 美元 {g.usd.sum():+,.0f}")
for l, lab in [(3, "≤0.5h（0.03）"), (2, "0.5~1h（0.02）"), (1, "1~3h（0.01）")]:
    g = H[H.lots == l]
    P(f"  {lab} {len(g)} 筆 每筆 {g.R.mean():+.2f}R 美元 {g.usd.sum():+,.0f}")
P("  出場：" + "  ".join(f"{k} {v}" for k, v in H.why.value_counts().items()))

eq = C.usd.cumsum().values; peak = np.maximum.accumulate(eq); dd = eq - peak
i1 = int(np.argmin(np.where(C.day.between("2025-07-01", "2025-12-31").values, dd, 0)))
i0 = int(np.argmax(eq[:i1 + 1]))
a, b = C.exit_time.iloc[i0], C.exit_time.iloc[i1]
rec = np.where(eq[i1:] >= peak[i1])[0]
P(f"\nC. 2025 下半年最大回撤：{a:%Y-%m-%d} 高點 → {b:%Y-%m-%d} 低點，{dd[i1]:+,.0f} 美元；"
  + (f"{C.exit_time.iloc[i1 + rec[0]]:%Y-%m-%d} 回到高點" if len(rec) else "之後未回到高點"))
W = C[(C.exit_time > a) & (C.exit_time <= b)]
P(f"  期間 {len(W)} 筆、勝率 {(W.R > 0.05).mean() * 100:.0f}%、每筆 {W.R.mean():+.2f}R；分倉美元 {W.usd.sum():+,.0f}（若全部 0.01：{W.pnl.sum():+,.0f}）")
P("  逐日（有交易的日子）：")
for d, g in W.groupby("day"):
    dm = Dm.loc[d]
    P(f"   {d:%m-%d} {'多' if g.side.iloc[0] == 1 else '空'} {len(g)}筆 R合計{g.R.sum():+5.1f} 美元{g.usd.sum():+6,.0f}（0.03手{(g.lots == 3).sum()}筆）｜ 當日 開{dm.open:.0f} 高{dm.high:.0f} 低{dm.low:.0f} 收{dm.close:.0f} 漲跌{dm.ret * 100:+.1f}% 單邊{dm.trend:.2f}")
big = W.nsmallest(5, "usd")
P("  虧最多的 5 筆：" + "；".join(f"{r.t_in:%m-%d %H:%M} {'多' if r.side == 1 else '空'} {r.lots:.0f}×0.01手 停損距{r.risk:.0f}美元 {r.usd:+.0f}" for r in big.itertuples()))
out.close()


# ---------- D. 能不能事先看出來？用「前 N 天日內單邊比例中位數」當盤整偵測（只用過去資料） ----------
out = open("results_part48.txt", "a")
P("\nD. 盤整偵測：前 N 天（不含今天）日內單邊比例 |收−開|÷(高−低) 的中位數 < 門檻 → 今天不做（每格：筆數 / 每筆R / 分倉美元 / 回撤 / 日Sharpe）")
from part27_ow import PER as PER15
for lab, path, bm, TT in [("M1 2024/9~2026/10", "data/XAUUSD_M1_2024_2026.csv", 1, T),
                          ("M15 2023/3~2026/10", "data/XAUUSD_M15_2023_2026.csv", 15,
                           pd.read_csv("part37_trades_M15.csv", parse_dates=["day", "t_in", "exit_time"]))]:
    dd_, DD_, AR_ = data(path, bm)
    tr = ((DD_.close - DD_.open).abs() / (DD_.high - DD_.low))
    TT = TT.copy()
    if "lots" not in TT:
        dly = (TT.t_in - TT.day - pd.to_timedelta(TT.h + 1, unit="h")).dt.total_seconds() / 3600
        TT["lots"] = np.where(dly <= 0.5, 3.0, np.where(dly <= 1, 2.0, 1.0))
    alld = [d for d in dd_ if d >= TT.day.min()]
    P(f"  [{lab}]")
    for n in (5, 10, 20):
        med = tr.rolling(n).median().shift(1)
        for th in ([None] if n == 5 else []) + [0.3, 0.35, 0.4]:
            Y = TT if th is None else TT[TT.day.map(med).fillna(1) >= th]
            Cc = cap(Y).sort_values("exit_time"); u = Cc.pnl * Cc.lots; e = u.cumsum()
            dl = (Cc.assign(u=u).groupby("day").u.sum()).reindex(alld, fill_value=0)
            per = "  ".join(f"{p} {x.R.mean():+.2f}" for p, a0, b0 in PER15 for x in [Cc[(Cc.day >= a0) & (Cc.day <= b0)]] if len(x))
            P(f"   {'全部（現行）' if th is None else f'前{n}天中位 ≥ {th}':16s} {len(Cc):5d}筆 {Cc.R.mean():+.3f}R {u.sum():+7,.0f}$ 回撤{(e - e.cummax()).min():+6,.0f} Sharpe{dl.mean() / dl.std() * np.sqrt(250):.2f} ｜ {per}")
out.close()
