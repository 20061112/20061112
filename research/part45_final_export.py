"""第四十五部分：最終版逐筆明細與統計（0.02 口），給實盤對照用。

來源：final_m15_signals.csv（part32_final_m15.py 產生；實際下單 = 權益濾網通過的單）。
輸出：
  final_m15_trades_002.csv   逐筆：進出場、停損停利、出場原因、持有時間、R、
                             損益（0.02 口，只扣點差 / 再扣 0.3 美元滑價）、可選「漲過頭減半」的口數與損益、權益濾網狀態
  final_m15_monthly_002.csv  逐月：筆數、勝率、損益、月底權益、月內回撤
  results_part45.txt         統計
"""
import numpy as np
import pandas as pd
import part24_prev_bar_breakout as B

SLIP = 0.3
S = pd.read_csv("final_m15_signals.csv", parse_dates=["進場時間", "出場時間"])
T = S[S.實際下單].copy().reset_index(drop=True)
dl = B.daily.copy()
dl.index = pd.DatetimeIndex(dl.index)
sma50 = dl.C.rolling(50).mean()
atrd = (dl.H - dl.L).rolling(10).mean()
ext = ((dl.C - sma50) / atrd).shift(1)
T["date"] = T.進場時間.dt.normalize()
dirv = np.where(T.方向 == "多", 1, -1)
T["前日離50MA(ATR,順方向)"] = (ext.reindex(T.date).values * dirv).round(2)
T["口數"] = 0.02
T["損益_只扣點差"] = (T.淨利_每盎司 * 2).round(2)
T["損益_含滑價"] = ((T.淨利_每盎司 - SLIP) * 2).round(2)
T["R_含滑價"] = ((T.淨利_每盎司 - SLIP) / T.停損距離).round(3)
T["可選_漲過頭減半口數"] = np.where(T["前日離50MA(ATR,順方向)"] > 3, 0.01, 0.02)
T["可選_漲過頭減半損益_含滑價"] = ((T.淨利_每盎司 - SLIP) * T.可選_漲過頭減半口數 * 100).round(2)
T["累積損益_含滑價"] = T.sort_values("出場時間").損益_含滑價.cumsum().reindex(T.index).round(2)
cols = ["進場時間", "方向", "前一根M15高", "前一根M15低", "進場價(=點位)", "ATR15", "停損價", "停利價", "停損距離",
        "前日收盤", "前日SMA50", "SMA50_10日斜率", "今日開盤", "ATR日", "今日漲跌(ATR日,順方向)", "前日離50MA(ATR,順方向)",
        "影子累積R", "影子累積R的50筆均線", "出場時間", "出場價", "出場原因", "持有分鐘", "點差", "淨利_每盎司",
        "R", "R_含滑價", "口數", "損益_只扣點差", "損益_含滑價", "可選_漲過頭減半口數", "可選_漲過頭減半損益_含滑價", "累積損益_含滑價"]
T[cols].to_csv("final_m15_trades_002.csv", index=False, encoding="utf-8-sig")

lines = []


def p(s=""):
    print(s)
    lines.append(s)


def streak(v):
    best = cur = 0
    for x in v:
        cur = cur + 1 if x < 0 else 0
        best = max(best, cur)
    return best


def block(pnl, lab):
    x = T.assign(pnl=pnl)
    d = x.groupby("date").pnl.sum(); e = d.cumsum(); mdd = (e.cummax() - e).max()
    yrs = (x.進場時間.max() - x.進場時間.min()).days / 365
    w, l = x.pnl[x.pnl > 0], x.pnl[x.pnl <= 0]
    mon = x.groupby(x.進場時間.dt.to_period("M")).pnl.sum()
    p(f"--- {lab} ---")
    p(f"  總損益 {x.pnl.sum():+,.0f}　每年 {x.pnl.sum() / yrs:+,.0f}　每月平均 {mon.mean():+,.0f}（中位數 {mon.median():+,.0f}）　正報酬月 {(mon > 0).sum()}/{len(mon)}")
    p(f"  每筆 {x.pnl.mean():+.2f}　勝率 {(x.pnl > 0).mean():.1%}　平均賺 {w.mean():+.2f} / 平均賠 {l.mean():+.2f}　PF {w.sum() / -l.sum():.2f}")
    p(f"  日結 MDD {mdd:,.0f}　最差一天 {d.min():+,.0f}（{d.idxmin().date()}）　最好一天 {d.max():+,.0f}　最差月 {mon.min():+,.0f}（{mon.idxmin()}）　最長連虧 {streak(x.sort_values('出場時間').pnl.values)} 筆")
    yr = x.groupby(x.進場時間.dt.year).agg(n=("pnl", "size"), s=("pnl", "sum"), m=("pnl", "mean"), w=("pnl", lambda v: (v > 0).mean()))
    p("  逐年：" + "；".join(f"{k} {int(v.n)} 筆 {v.s:+,.0f}（每筆 {v.m:+.2f}，勝率 {v.w:.0%}）" for k, v in yr.iterrows()))


p("最終版 M15 前高/前低突破 —— 0.02 口逐筆統計（2023/8/24 ~ 2026/10/8）\n")
p(f"實際下單 {len(T)} 筆，有交易的日子 {T.date.nunique()} 天（平均每天 {len(T) / T.date.nunique():.1f} 筆）；"
  f"影子訊號 {len(S)} 筆，權益濾網停手 {1 - S.實際下單.mean():.0%}")
p(f"多單 {(T.方向 == '多').sum()} 筆、空單 {(T.方向 == '空').sum()} 筆；出場：停利 {(T.出場原因 == '停利').mean():.0%}、停損 {(T.出場原因 == '停損').mean():.0%}、收盤 {(T.出場原因 == '收盤').mean():.0%}")
p(f"持有時間中位數 {T.持有分鐘.median():.0f} 分（停利單 {T[T.出場原因 == '停利'].持有分鐘.median():.0f}、停損單 {T[T.出場原因 == '停損'].持有分鐘.median():.0f}）；"
  f"停損距離中位數 {T.停損距離.median():.2f} 美元/盎司（0.02 口 = 每筆停損約 {T.停損距離.median() * 2:.0f} 美元）\n")
block(T.損益_只扣點差, "A. 只扣點差")
block(T.損益_含滑價, "B. 點差 + 每筆 0.3 美元/盎司滑價（建議以此為準）")
block(T.可選_漲過頭減半損益_含滑價, "C. B + 漲過頭減半（可選）")
p("\n時段（B）：")
for a, b in [(1, 4), (4, 8), (8, 11), (11, 15), (15, 18), (18, 22)]:
    m = T.進場時間.dt.hour.between(a, b - 1)
    p(f"  {a:02d}-{b:02d} 點：{m.sum():5d} 筆（{m.mean():.0%}）每筆 {T.損益_含滑價[m].mean():+.2f}")

mon = T.groupby(T.進場時間.dt.to_period("M")).agg(筆數=("損益_含滑價", "size"), 勝率=("R", lambda v: round((v > 0).mean(), 3)),
                                               損益_只扣點差=("損益_只扣點差", "sum"), 損益_含滑價=("損益_含滑價", "sum"),
                                               可選_漲過頭減半=("可選_漲過頭減半損益_含滑價", "sum"))
mon["月底累積_含滑價"] = mon.損益_含滑價.cumsum()
mon.round(2).to_csv("final_m15_monthly_002.csv", encoding="utf-8-sig")
p("\n逐月（含滑價）：")
for k, v in mon.iterrows():
    p(f"  {k}: {int(v.筆數):4d} 筆  勝率 {v.勝率:.0%}  {v.損益_含滑價:+7,.0f}  累積 {v.月底累積_含滑價:+8,.0f}")
open("results_part45.txt", "w").write("\n".join(lines) + "\n")
