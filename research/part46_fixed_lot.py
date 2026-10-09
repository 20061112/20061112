"""第四十六部分：OU 集成最終版本（第 45 部分）改成固定手數：每筆 0.01 手（1 盎司），vr < 1 加倉為 0.02 手。
損益直接用美元 / 盎司 × 盎司數；點差、隔夜已扣（第 45 部分逐筆紀錄）。
"""
import numpy as np
import pandas as pd

T = pd.read_csv("part45_final_trades.csv", parse_dates=["entry", "exit"])
T["oz"] = np.where(T.w == 2, 2, 1)
T["pnl"] = T.usd * T.oz
T["risk"] = T.risk_usd * T.oz
T["xday"] = (T.exit - pd.Timedelta(minutes=1)).dt.date
days = sorted(pd.read_csv("part42_daily_matrix.csv", index_col=0).index.map(lambda s: pd.Timestamp(s).date()))
d = pd.Series(T.pnl.to_numpy(), index=T.xday).groupby(level=0).sum().reindex(days, fill_value=0.0)
eq = d.cumsum()
dd = eq - np.maximum.accumulate(np.r_[0, eq.to_numpy()])[1:]
out = []
P = out.append
P("固定手數：每筆 0.01 手（1 盎司），波動比 < 1 時 0.02 手；16 組同時跑")
P(f"  交易 {len(T)} 筆（0.01 手 {int((T.oz == 1).sum())}、0.02 手 {int((T.oz == 2).sum())}）")
P(f"  總損益 ${T.pnl.sum():+,.0f}；毛利 ${T.gross_usd.mul(T.oz).sum():+,.0f}、點差 −${T.spread_usd.mul(T.oz).sum():,.0f}、隔夜 −${T.swap_usd.mul(T.oz).sum():,.0f}")
P(f"  每筆平均 ${T.pnl.mean():+.2f}；贏單平均 ${T.pnl[T.pnl > 0].mean():+.2f}、輸單平均 ${T.pnl[T.pnl <= 0].mean():+.2f}；勝率 {np.mean(T.pnl > 0):.0%}；PF {T.pnl[T.pnl > 0].sum() / -T.pnl[T.pnl < 0].sum():.2f}")
P(f"  單筆最大獲利 ${T.pnl.max():+.0f}、單筆最大虧損 ${T.pnl.min():+.0f}")
P(f"  單筆風險（停損金額）中位數 ${T.risk.median():.1f}、最大 ${T.risk.max():.0f}")
P(f"  最大回撤 ${-dd.min():,.0f}（{eq.index[int(np.argmin(dd.to_numpy()))]}）；獲利/回撤 {T.pnl.sum() / -dd.min():.1f}")
P(f"  最好一天 ${d.max():+,.0f}（{d.idxmax()}）、最差一天 ${d.min():+,.0f}（{d.idxmin()}）")
top = d.sort_values(ascending=False)
P(f"  最好 5 天 ${top.head(5).sum():+,.0f}（占 {top.head(5).sum() / d.sum():.0%}）；拿掉最好 10 天 ${d.sum() - top.head(10).sum():+,.0f}")
tl = pd.concat([pd.Series(T.oz.to_numpy() / 100, index=T.entry), pd.Series(-T.oz.to_numpy() / 100, index=T.exit)]).sort_index().cumsum()
rk = pd.concat([pd.Series(T.risk.to_numpy(), index=T.entry), pd.Series(-T.risk.to_numpy(), index=T.exit)]).sort_index().cumsum()
P(f"  同時持倉：最多 {tl.max():.2f} 手（同時停損風險最多 ${rk.max():,.0f}）；有持倉時中位數 {tl[tl > 1e-9].median():.2f} 手")
P(f"  保證金（1:100、金價約 $4,000 時 0.01 手約需 $40）：最多約 ${tl.max() * 100 * 4000 / 100:,.0f}")
P("\n  逐月（美元）")
T["月"] = T.exit.dt.strftime("%Y-%m")
m = T.groupby("月").agg(筆數=("pnl", "size"), 勝率=("pnl", lambda x: np.mean(x > 0)), 損益=("pnl", "sum"),
                       最大單筆虧損=("pnl", "min"))
P(m.round(2).to_string())
P("\n  依時框（美元）")
P(T.groupby("tf").agg(筆數=("pnl", "size"), 損益=("pnl", "sum"), 每筆=("pnl", "mean")).round(2).to_string())
P("\n  帳戶規模對照（固定手數，不複利）")
for cap in (1000, 2000, 3000, 5000, 10000):
    e = cap + eq.to_numpy()
    pk = np.maximum.accumulate(np.r_[cap, e])[1:]
    P(f"    起始 ${cap:>6,}：期末 ${cap + d.sum():>8,.0f}（{d.sum() / cap:+.0%}），最大回撤 {(e / pk - 1).min():.1%}，"
      f"同時停損風險最多占起始資金 {rk.max() / cap:.0%}")
txt = "\n".join(out)
print(txt)
open("results_part46.txt", "w").write(txt + "\n")
