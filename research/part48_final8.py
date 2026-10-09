"""第四十八部分：8 組版本（M5 / M15 × 回看 70 / 100 / 130 / 200 小時）+ 波動比 < 1 加倍 — 詳細交易統計（固定手數）。

規則：OU |z| 由 < 3 穿越到 ≥ 3 → 下一根開盤順勢市價；停損 1.5 ATR14；持有滿 40 根或週五最後一根收盤出場；
      週五 20 點後不開新單；每組同時最多 1 筆；vr = ATR14/ATR200 < 1 → 0.02 手，否則 0.01 手（0.01 手 = 1 盎司）。
成本：進場那根點差、隔夜 $0.7/盎司/晚（週三 ×3）。
"""
import numpy as np
import pandas as pd
from load import load
from part39_ou_vs_er import SPLIT
from part43_twoside_vol_tf import TFrame, signals
from part45_final_ou import trades, streak

LEGS = [(tf, m, h) for tf, m in (("5min", 5), ("15min", 15)) for h in (70, 100, 130, 200)]


def main():
    m1 = load("data/XAUUSD_M1_2026.csv")
    days = sorted(set(m1.index.date))
    nd = len(days)
    first = [d for d in days if pd.Timestamp(d) < SPLIT]
    frames, rows = {}, []
    for tf, mins, h in LEGS:
        F = frames.setdefault(tf, TFrame(m1, tf, mins))
        rows += trades(F, *signals(F, int(h * 60 / mins)), f"{tf}|{h}h")
    T = pd.DataFrame(rows).sort_values("entry").reset_index(drop=True)
    T["lot"] = np.where(T.vr < 1, 0.02, 0.01)
    T["oz"] = T.lot * 100
    T["pnl"] = T.usd * T.oz
    T["risk"] = T.risk_usd * T.oz
    T["R"] = T.usd / T.risk_usd
    T["xday"] = (T.exit - pd.Timedelta(minutes=1)).dt.date
    T["回看"] = T.cfg.str.split("|").str[1]
    T.to_csv("part48_trades.csv", index=False)
    d = pd.Series(T.pnl.to_numpy(), index=T.xday).groupby(level=0).sum().reindex(days, fill_value=0.0)
    eq = d.cumsum()
    dd = eq - np.maximum.accumulate(np.r_[0, eq.to_numpy()])[1:]
    mdd = -dd.min()
    sh = lambda s: s.mean() / s.std() * np.sqrt(252)
    win, loss = T[T.pnl > 0], T[T.pnl <= 0]
    out = []
    P = out.append
    P("=" * 100)
    P("OU 順勢 8 組（M5 / M15 × 70/100/130/200 小時）+ 波動比 < 1 時 0.02 手 — 2026/1/2 ~ 10/8，192 個交易日")
    P("=" * 100)
    ev, lt, ls = 0, None, None
    for r in T.itertuples():
        if lt is None or r.side != ls or (r.entry - lt) > pd.Timedelta(hours=2):
            ev += 1
        lt, ls = r.entry, r.side
    P("\n【總覽】")
    P(f"  交易筆數       {len(T)} 筆（每天 {len(T) / nd:.2f}）；0.02 手 {int((T.lot == 0.02).sum())} 筆（{np.mean(T.lot == 0.02):.0%}）")
    P(f"  獨立行情       {ev} 次（每天 {ev / nd:.2f}；平均每次 {len(T) / ev:.1f} 組一起進場）；有交易的日子 {T.xday.nunique()} / {nd}")
    P(f"  總損益         ${T.pnl.sum():+,.0f}（毛利 ${(T.gross_usd * T.oz).sum():+,.0f}、點差 −${(T.spread_usd * T.oz).sum():,.0f}、隔夜 −${(T.swap_usd * T.oz).sum():,.0f}）")
    P(f"  勝率 / PF      {len(win) / len(T):.1%} / {win.pnl.sum() / -loss.pnl.sum():.2f}")
    P(f"  平均賺 / 賠    +${win.pnl.mean():.1f} / ${loss.pnl.mean():.1f}（R：+{win.R.mean():.2f} / {loss.R.mean():.2f}）；每筆平均 ${T.pnl.mean():+.2f}")
    P(f"  單筆最大       +${T.pnl.max():.0f} / ${T.pnl.min():.0f}；單筆停損金額中位數 ${T.risk.median():.1f}、最大 ${T.risk.max():.0f}")
    P(f"  最大回撤       ${mdd:,.0f}（{eq.index[int(np.argmin(dd.to_numpy()))]}）；獲利/回撤 {d.sum() / mdd:.1f}")
    P(f"  Sharpe         {sh(d):.2f}（前半 {sh(d.loc[first]):.2f} / 後半 {sh(d.drop(first)):.2f}）")
    P(f"  最好 / 最差一天 ${d.max():+,.0f}（{d.idxmax()}）/ ${d.min():+,.0f}（{d.idxmin()}）；有交易的日子正報酬 {np.mean(d[d != 0] > 0):.0%}")
    P(f"  最長連虧 / 連勝 {streak(T.pnl <= 0)} / {streak(T.pnl > 0)} 筆")
    lots = pd.concat([pd.Series(T.lot.to_numpy(), index=T.entry), pd.Series(-T.lot.to_numpy(), index=T.exit)]).sort_index().cumsum()
    rk = pd.concat([pd.Series(T.risk.to_numpy(), index=T.entry), pd.Series(-T.risk.to_numpy(), index=T.exit)]).sort_index().cumsum()
    P(f"  同時持倉       最多 {lots.max():.2f} 手（同時停損風險最多 ${rk.max():,.0f}）；有持倉時中位數 {lots[lots > 1e-9].median():.2f} 手")
    P(f"  需要資金       回撤 ≤ 20%：約 ${mdd / 0.2:,.0f}；≤ 30%：約 ${mdd / 0.3:,.0f}（以最壞起始點計）")

    P("\n【出場與持有】")
    for k, s in T.groupby("why"):
        P(f"  {k:<6} {len(s):4d} 筆（{len(s) / len(T):.0%}） 平均 ${s.pnl.mean():+.1f}（{s.R.mean():+.2f}R） 平均持有 {s.hours.mean():.1f}h")
    P(f"  持有中位數 {T.hours.median():.1f}h（M5 {T[T.tf == '5min'].hours.median():.1f}h、M15 {T[T.tf == '15min'].hours.median():.1f}h）；過夜 {np.mean(T.nights > 0):.0%}")
    P(f"  輸單中曾經浮盈 ≥ 1R 的 {np.mean(loss.mfe_R >= 1):.0%}；贏單最大浮虧中位數 {win.mae_R.median():.2f}R")
    lab = ["−1R（停損）", "−1~0", "0~1", "1~3", "3~5", ">5R"]
    c = pd.cut(T.R, [-np.inf, -0.999, 0, 1, 3, 5, np.inf], labels=lab).value_counts(normalize=True).reindex(lab)
    P("  R 分佈：" + "｜".join(f"{k} {v:.0%}" for k, v in c.items()) + f"；最大單筆 {T.R.max():+.1f}R")

    P("\n【獲利集中度】")
    top = d.sort_values(ascending=False)
    P(f"  最好 5 天 ${top.head(5).sum():+,.0f}（{top.head(5).sum() / d.sum():.0%}）、最好 10 天 {top.head(10).sum() / d.sum():.0%}")
    P(f"  拿掉最好 5 天 ${d.sum() - top.head(5).sum():+,.0f}；拿掉最好 10 天 ${d.sum() - top.head(10).sum():+,.0f}")
    P("  最好 5 天：" + "、".join(f"{k} ${v:+,.0f}" for k, v in top.head(5).items()))

    def grp(col, order):
        g = T.groupby(col).agg(筆數=("pnl", "size"), 勝率=("pnl", lambda x: np.mean(x > 0)), 每筆美元=("pnl", "mean"),
                               總損益=("pnl", "sum")).reindex(order)
        g["占比"] = g.總損益 / T.pnl.sum()
        return g.round(2).to_string()

    P("\n【依組別】")
    P(grp("cfg", [f"{tf}|{h}h" for tf, _, h in LEGS]))
    P("\n【依時框】")
    P(grp("tf", ["5min", "15min"]))
    P("\n【依回看】")
    P(grp("回看", ["70h", "100h", "130h", "200h"]))
    P("\n【依方向】")
    P(grp("side", ["多", "空"]))
    T["波動"] = np.where(T.vr < 1, "vr<1（0.02 手）", "vr≥1（0.01 手）")
    P("\n【依波動】")
    P(grp("波動", ["vr<1（0.02 手）", "vr≥1（0.01 手）"]))
    T["時段"] = pd.cut(T.entry.dt.hour, [-1, 7, 14, 19, 24], labels=["亞洲 0-7", "歐洲 8-14", "美盤 15-19", "美盤後 20-23"])
    P("\n【依進場時段（broker）】")
    P(grp("時段", ["亞洲 0-7", "歐洲 8-14", "美盤 15-19", "美盤後 20-23"]))
    T["星期"] = T.entry.dt.dayofweek.map(dict(enumerate("一二三四五六日")))
    P("\n【依星期】")
    P(grp("星期", list("一二三四五")))
    P("\n【逐月】")
    T["月"] = T.exit.dt.strftime("%Y-%m")
    mo = T.groupby("月").agg(筆數=("pnl", "size"), 勝率=("pnl", lambda x: np.mean(x > 0)), 損益=("pnl", "sum"))
    P(mo.round(2).to_string())
    wk = d.groupby(pd.to_datetime(pd.Series(d.index)).dt.to_period("W").to_numpy()).sum()
    P(f"  週：有交易的週正報酬 {np.mean(wk[wk != 0] > 0):.0%}；最好週 ${wk.max():+,.0f}、最差週 ${wk.min():+,.0f}")
    P("\n【帳戶規模】（固定手數不複利；最壞起始點 = 剛好從高點開始吃完整個最大回撤）")
    for cap in (1000, 2000, 3000, 5000, 10000):
        e = cap + eq.to_numpy()
        pk = np.maximum.accumulate(np.r_[cap, e])[1:]
        P(f"  ${cap:>6,}：從 1/2 開始 期末 ${cap + d.sum():,.0f}（{d.sum() / cap:+.0%}）回撤 {(e / pk - 1).min():.1%}；"
          f"最壞起始點回撤 {-mdd / cap:.0%}；同時停損風險最多 {rk.max() / cap:.0%}")
    txt = "\n".join(out)
    print(txt)
    open("results_part48.txt", "w").write(txt + "\n")


if __name__ == "__main__":
    main()
