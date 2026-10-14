"""第二十九部分：H1 六合一 —— 2026/7 發生什麼事，以及換算成美元。
輸出：python3 part29_jul_usd.py > results_part29_jul_usd.txt
"""
import numpy as np
import pandas as pd
import part28_h1_combo as P
from load import load

NAME = P.NAME


def daily_context():
    m15 = load("data/XAUUSD_M15_2023_2026.csv")
    d = m15.close.resample("1D").last().dropna().to_frame("close")
    d["sma50"] = d.close.rolling(50).mean()
    d["slope20"] = d.sma50 - d.sma50.shift(20)
    up = (d.close > d.sma50) & (d.slope20 > 0)
    dn = (d.close < d.sma50) & (d.slope20 < 0)
    d["regime_today"] = np.where(up, "多頭", np.where(dn, "空頭", "盤整"))
    d["交易用狀態(前一天)"] = d.regime_today.shift(1)
    return d


def usd_section(df, T):
    print("\n## 2. 換算成美元")
    print("  （損益已扣進場點差；沒有扣隔夜利息 swap，平均持有 32 小時約 1~2 晚）")
    yrs = T.entry_time.dt.year

    def curve(q_per_trade, title):
        X = T.copy()
        X["usd"] = X.pnl_usd * q_per_trade                   # q 盎司
        X["combo_R"] = X.usd
        X["R_usd"] = P.W / q_per_trade if np.isscalar(q_per_trade) else P.W / q_per_trade
        ec, em, _, worst = P.mtm_curve(df, X)
        by = X.groupby(yrs).usd.sum()
        print(f"\n  ### {title}")
        print(f"    總損益 ${X.usd.sum():,.0f}；最大回撤 平倉 ${(ec.cummax() - ec).max():,.0f} / 含浮動 ${(em.cummax() - em).max():,.0f}；"
              f"同時最大浮虧 ${-worst.min():,.0f}")
        print("    逐年 " + "  ".join(f"{y}: ${v:+,.0f}" for y, v in by.items()))
        jul = X[(X.exit_time >= "2026-07-01") & (X.exit_time < "2026-08-01")].usd.sum()
        print(f"    2026/7 單月 ${jul:+,.0f}")

    curve(1.0, "固定手數：每個版本每筆 0.01 手（1 盎司），同一訊號最多 0.06 手")
    curve(10.0, "固定手數：每個版本每筆 0.1 手（10 盎司），同一訊號最多 0.6 手")

    # 風險固定（不複利 / 複利）
    for acct, risk in ((10000, 0.01), (100000, 0.01)):
        unit = acct * risk                                      # 1 組合 R 的美元
        s = T.combo_R * unit
        print(f"\n  ### 帳戶 ${acct:,}，每個訊號總風險 {risk:.0%}（每版本 {risk / 6:.3%}），不複利")
        ec = s.groupby(T.exit_time).sum().cumsum()
        print(f"    總損益 ${s.sum():+,.0f}（{s.sum() / acct:+.1%}）；平倉最大回撤 ${(ec.cummax() - ec).max():,.0f}；"
              f"含浮動最大回撤約 ${5.45 * unit:,.0f}")
        print("    逐年 " + "  ".join(f"{y}: ${v:+,.0f}" for y, v in s.groupby(yrs).sum().items()))
        lots = acct * risk / 6 / (T.R_usd * 100)
        print(f"    需要的手數：中位數 {lots.median():.3f} 手、範圍 {lots.min():.3f} ~ {lots.max():.3f} 手"
              f"（低於 0.01 手的比例 {(lots < 0.01).mean():.0%}）")

    # 複利：以進場當下的帳戶淨值（已實現）計算風險
    for acct in (10000, 100000):
        ev = []
        for r in T.itertuples():
            ev.append((r.entry_time, 0, r.Index))
            ev.append((r.exit_time, 1, r.Index))
        ev.sort()
        eq, risk_amt, curve_pts = acct, {}, []
        for t, kind, i in ev:
            if kind == 0:
                risk_amt[i] = eq * 0.01 / 6
            else:
                eq += T.pnl_R[i] * risk_amt[i]
                curve_pts.append((t, eq))
        c = pd.Series([v for _, v in curve_pts], index=[t for t, _ in curve_pts])
        print(f"\n  ### 帳戶 ${acct:,}，每個訊號 1%，複利")
        print(f"    期末 ${c.iloc[-1]:,.0f}（{c.iloc[-1] / acct - 1:+.1%}）；平倉最大回撤 {((c / c.cummax()) - 1).min():.1%}")
        print("    逐年期末 " + "  ".join(f"{y}: ${v:,.0f}" for y, v in c.groupby(c.index.year).last().items()))


def main():
    df, T = P.build()
    d = daily_context()
    print("## 1. 2026/7 發生什麼事")
    J = T[(T.exit_time >= "2026-06-20") & (T.exit_time < "2026-08-01")].sort_values("entry_time")
    print("\n  ### 日線背景（6/20 ~ 7/31）")
    dd = d.loc["2026-06-15":"2026-08-05"]
    print(dd[["close", "sma50", "slope20", "交易用狀態(前一天)"]].round(1).to_string())
    print("\n  ### 這段期間的交易（同一訊號多版本合併）")
    S = J.groupby(["entry_time", "side"]).agg(版本=("cond", lambda x: ",".join(NAME[k] for k in x)), 進場價=("entry_px", "first"),
                                               出場時間=("exit_time", "max"), 出場=("exit", lambda x: "/".join(sorted(set(P.EXIT[k] for k in x)))),
                                               平均R=("pnl_R", "mean"), 組合R=("combo_R", "sum"), 美元_每版0_01手=("pnl_usd", "sum"))
    print(S.round({"進場價": 2, "平均R": 2, "組合R": 2, "美元_每版0_01手": 2}).to_string())
    jul = T[(T.exit_time >= "2026-07-01") & (T.exit_time < "2026-08-01")]
    print(f"\n  7 月出場 {len(jul)} 筆（空 {(jul.side == -1).sum()}）：勝率 {(jul.pnl_R > 0).mean():.0%}，組合 {jul.combo_R.sum():+.2f}R；"
          f"停損 {(jul.exit == 'stop').sum()} 筆、時間 {(jul.exit == 'time').sum()} 筆、停利 {(jul.exit == 'tp').sum()} 筆")
    low = d.loc["2026-06-01":"2026-08-31"].close
    print(f"  6~8 月日收盤最低 {low.min():.1f}（{low.idxmin():%m/%d}），之後到 8/31 反彈 {low.loc[low.idxmin():].max() / low.min() - 1:+.1%}")
    # 空頭期間空單總覽
    sh = T[(T.side == -1) & (T.entry_time >= "2026-03-01")]
    print(f"  2026 年的空單合計：{len(sh)} 筆，組合 {sh.combo_R.sum():+.2f}R（4~6 月 {sh[sh.exit_time < '2026-07-01'].combo_R.sum():+.2f}R，"
          f"7 月以後 {sh[sh.exit_time >= '2026-07-01'].combo_R.sum():+.2f}R）")
    # 其他年份同樣的「空頭尾段」
    print("\n  ### 歷史上空頭狀態結束前最後一個月的空單（比較用）")
    reg = d["交易用狀態(前一天)"]
    ends = reg[(reg == "空頭") & (reg.shift(-1) != "空頭")].index
    for e in ends:
        g = T[(T.side == -1) & (T.entry_time >= e - pd.Timedelta(days=30)) & (T.entry_time <= e)]
        if len(g):
            print(f"    空頭結束 {e:%Y-%m-%d}：前 30 天空單 {len(g)} 筆，組合 {g.combo_R.sum():+.2f}R")
    usd_section(df, T)


if __name__ == "__main__":
    main()
