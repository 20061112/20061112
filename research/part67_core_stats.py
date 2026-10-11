"""第六十七部分：IB 突破核心（第六十二部分原版）的詳細交易數據 → results_part67.txt、part67_trades.csv

規則：IB 起點 02:00~10:30 每 30 分、長度 60 分；IB 結束後 3 小時內 M5 收盤 > IB 高 + 0.05×ATR10（空單鏡像）進場；
      停損 IB 另一側；當天最後一根收盤出場；同一根同方向只留一筆；前一天日效率 > 0.633 不做；週一照做；不限疊單。
期間：2024/11/1 ~ 2026/10/8（研究期 2024/11~2025/12、2026 驗證）。每筆 1 盎司（= 0.01 手），美元，已扣進場點差。
"""
import numpy as np
import pandas as pd
from core_ib import events, simulate, ARR, PEFF_MAX

PER = [("全期 2024/11~2026/10", "2024-11-01", "2026-11-01"), ("研究期 2024/11~2025/12", "2024-11-01", "2026-01-01"),
       ("2026/1~10", "2026-01-01", "2026-11-01")]


def enrich(X):
    rows = []
    for r in X.itertuples():
        t, mins, O, H, L, C, S = ARR[r.day]
        j = int(r.j)
        seg = slice(r.i0, j + 1)
        hi, lo = H[seg].max(), L[seg].min()
        mfe = r.side * ((hi if r.side == 1 else lo) - r.lvl)
        mae = r.side * (r.lvl - (lo if r.side == 1 else hi))
        risk = abs(r.lvl - (r.ibl if r.side == 1 else r.ibh))
        rows.append(dict(exit_time=t[j], exit_px=r.lvl + r.side * (r.pnl + r.spread), hold_h=(t[j] - r.t_in).total_seconds() / 3600,
                         risk=risk, risk_atr=risk / r.atr, mfe=mfe, mae=min(mae, risk), mfe_R=mfe / risk))
    return pd.concat([X.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def streak(b):
    s = (b != b.shift()).cumsum()
    return b.groupby(s).sum().max() if b.any() else 0


def block(T, days_all):
    p = T.pnl.to_numpy()
    w, l = p[p > 0], p[p <= 0]
    T = T.sort_values("exit_time")
    eq = T.pnl.cumsum().to_numpy()
    peak = np.maximum.accumulate(np.r_[0, eq])[1:]
    dd = eq - peak
    k = dd.argmin()
    start = T.exit_time.iloc[np.flatnonzero(eq[:k + 1] == peak[k])[0]] if (eq[:k + 1] == peak[k]).any() else T.exit_time.iloc[0]
    rec = np.flatnonzero(eq[k:] >= peak[k])
    rec_t = T.exit_time.iloc[k + rec[0]] if len(rec) else None
    days = [d for d in days_all if T.day.min() <= d <= T.day.max()]
    daily = pd.Series(0.0, index=days).add(T.groupby("day").pnl.sum(), fill_value=0)
    traded = T.groupby("day").pnl.sum()
    mo = T.groupby(T.day.dt.to_period("M")).pnl.sum()
    top5 = traded.nlargest(5).sum()
    L = []
    L.append(f"  交易筆數 {len(p)}、交易日 {len(days)}（有下單 {len(traded)} 天，{len(traded) / len(days):.0%}）、平均每天 {len(p) / len(days):.1f} 筆"
             f"（有下單的日子 {len(p) / len(traded):.1f} 筆，最多 {T.groupby('day').size().max()} 筆）")
    L.append(f"  勝率 {len(w) / len(p):.1%}；平均獲利 +{w.mean():.2f}、平均虧損 {l.mean():.2f}、賺賠比 {w.mean() / -l.mean():.2f}")
    L.append(f"  PF {w.sum() / -l.sum():.2f}；每筆期望 {p.mean():+.2f} 美元（{T.R.mean():+.3f}R），中位數 {np.median(p):+.2f}；t 值 {p.mean() / p.std(ddof=1) * np.sqrt(len(p)):.1f}")
    L.append(f"  總損益 {p.sum():+,.0f}；最大單筆獲利 +{p.max():.1f}、最大單筆虧損 {p.min():.1f}")
    L.append(f"  最大回撤 {dd.min():,.0f}（{start:%Y-%m-%d} 開始，{'未回復' if rec_t is None else f'{rec_t:%Y-%m-%d} 回復'}）；總損益 / 最大回撤 {p.sum() / -dd.min():.2f}")
    L.append(f"  最長連虧 {streak(pd.Series(p <= 0))} 筆、最長連賺 {streak(pd.Series(p > 0))} 筆")
    L.append(f"  日損益：賺錢日 {(traded > 0).mean():.0%}、平均 {daily.mean():+.1f}、最好 {traded.max():+.0f}、最差 {traded.min():+.0f}、"
             f"日 Sharpe（年化）{daily.mean() / daily.std() * np.sqrt(250):.2f}、日 t {daily.mean() / daily.std(ddof=1) * np.sqrt(len(daily)):.2f}")
    L.append(f"  最好 5 天佔總損益 {top5 / p.sum():.0%}；月份賺錢 {(mo > 0).sum()}/{len(mo)}、最好月 {mo.max():+,.0f}、最差月 {mo.min():+,.0f}")
    L.append(f"  出場：停損 {(T.why == '停損').mean():.0%}、收盤平倉 {(T.why != '停損').mean():.0%}"
             f"（收盤平倉的單 勝率 {(T.pnl[T.why != '停損'] > 0).mean():.0%}、平均 {T.pnl[T.why != '停損'].mean():+.1f}）")
    L.append(f"  停損距離：中位 {T.risk.median():.1f} 美元（{T.risk_atr.median():.2f} ATR），25%~75% {T.risk.quantile(.25):.1f}~{T.risk.quantile(.75):.1f}、最大 {T.risk.max():.1f}")
    L.append(f"  持有時間：中位 {T.hold_h.median():.1f} 小時、平均 {T.hold_h.mean():.1f}；停損單中位 {T.hold_h[T.why == '停損'].median():.1f} 小時")
    L.append(f"  最大浮盈 MFE：中位 {T.mfe_R.median():.2f}R；到過 +1R 的 {(T.mfe_R >= 1).mean():.0%}、+2R {(T.mfe_R >= 2).mean():.0%}、+3R {(T.mfe_R >= 3).mean():.0%}；"
             f"到過 +1R 最後卻虧損 {((T.mfe_R >= 1) & (T.pnl <= 0)).mean():.0%}")
    L.append(f"  點差：平均 {T.spread.mean():.2f} 美元/筆（佔停損距離 {(T.spread / T.risk).median():.1%}），合計 {T.spread.sum():,.0f}")
    q = np.percentile(T.R, [5, 25, 50, 75, 95])
    L.append(f"  每筆 R 分布：5% {q[0]:+.2f}、25% {q[1]:+.2f}、中位 {q[2]:+.2f}、75% {q[3]:+.2f}、95% {q[4]:+.2f}")
    return "\n".join(L)


def group(T, key, lab):
    g = T.groupby(key, observed=True)
    R = pd.DataFrame(dict(筆數=g.size(), 勝率=g.pnl.apply(lambda x: (x > 0).mean()).round(2),
                          PF=g.pnl.apply(lambda x: x[x > 0].sum() / max(-x[x <= 0].sum(), 1e-9)).round(2),
                          每筆=g.pnl.mean().round(2), 每筆R=g.R.mean().round(3), 總=g.pnl.sum().round(0)))
    return f"\n  [{lab}]\n  " + R.to_string().replace("\n", "\n  ")


def main():
    X = simulate(events())
    X = X[(X.peff <= PEFF_MAX) & (X.day >= "2024-11-01")]
    T = enrich(X)
    T["方向"] = T.side.map({1: "多", -1: "空"})
    T["星期"] = T.day.dt.dayofweek.map(dict(enumerate("一二三四五")))
    T["進場小時"] = (T.min_in // 60).astype(int).map(lambda h: f"{h:02d}")
    T["IB起點"] = (T.s0 // 60).astype(int).map(lambda h: f"{h:02d}:xx")
    T["當天第幾筆"] = T.groupby("day").cumcount().add(1).clip(upper=10).map(lambda k: f"{k}" if k < 10 else "10+")
    days_all = sorted(T.day.unique())
    from core_ib import DAYS
    out = ["第六十七部分：IB 突破核心（原版）詳細交易數據 —— 每筆 1 盎司（0.01 手），美元，已扣點差"]
    for lab, a, b in PER:
        S = T[(T.day >= a) & (T.day < b)]
        out.append(f"\n===== {lab} =====")
        out.append(block(S, [d for d in DAYS if a <= str(d.date()) < b]))
    for key, lab in (("方向", "多 / 空"), ("星期", "星期"), ("IB起點", "IB 起點"), ("進場小時", "進場時間（broker）"),
                     ("當天第幾筆", "當天第幾筆單"), ("why", "出場原因")):
        out.append(group(T[T.day < "2026-01-01"], key, f"{lab} —— 研究期 2024/11~2025/12"))
        out.append(group(T[T.day >= "2026-01-01"], key, f"{lab} —— 2026"))
    mo = T.groupby(T.day.dt.to_period("M")).agg(筆數=("pnl", "size"), 勝率=("pnl", lambda x: (x > 0).mean()),
                                               PF=("pnl", lambda x: x[x > 0].sum() / max(-x[x <= 0].sum(), 1e-9)),
                                               總=("pnl", "sum"), 每筆R=("R", "mean"))
    mo["累積"] = mo.總.cumsum()
    out.append("\n===== 逐月 =====\n" + mo.round(2).to_string())
    cols = ["day", "t_in", "方向", "IB起點", "ibh", "ibl", "lvl", "exit_time", "exit_px", "why", "risk", "risk_atr", "spread",
            "pnl", "R", "mfe", "mae", "mfe_R", "hold_h", "atr", "peff"]
    T[cols].rename(columns=dict(day="交易日", t_in="進場時間", ibh="IB高", ibl="IB低", lvl="進場價", exit_time="出場時間",
                                exit_px="出場價", why="出場原因", risk="停損距離", risk_atr="停損距離_ATR", spread="點差",
                                pnl="損益", mfe="最大浮盈", mae="最大浮虧", mfe_R="最大浮盈_R", hold_h="持有小時", atr="ATR10",
                                peff="前一天效率")).to_csv("part67_trades.csv", index=False)
    txt = "\n".join(out)
    open("results_part67.txt", "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
