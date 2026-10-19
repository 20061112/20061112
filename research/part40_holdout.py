"""第四十部分：凍結後一次打開保留期 2025/10 ~ 2026/10（第三十七~三十九部分完全沒看過這段）→ results_part40.txt、part40_trades.csv
凍結規則（broker 時間 = 紐約 + 7）
  A**（主策略，順勢）：IB 起點 10:30~16:00 每 30 分（= 紐約 03:30~09:00，歐洲盤到美股盤前）、IB 45 分；
       IB 結束後 3 小時內 M5 收盤 > IB 高 + 0.05×ATR10 做多（空單鏡像）；停損 IB 另一側；當天最後一根收盤出場；
       每天每方向最多 2 筆；前一天日效率 |收−開|/(高−低) > 0.75 的日子不做
  C*（觀察，順勢）：01:00 開盤後 30 分鐘當 IB，其餘同上（每天最多 1 筆、不加濾網）
參考：XAU 核心原封不動；B（亞洲盤後段反向，第二階段被淘汰）；A** 的鏡像（同一點做反方向）。
"""
import numpy as np, pandas as pd
from nas_core import run_many, period, HALVES, RESEARCH, HOLDOUT, D
from part38_stage2_signal import cap

PERIODS = HALVES + [("研究期合計", *RESEARCH), ("★保留 25Q4-26Q1", "2025-10-01", "2026-04-01"),
                    ("★保留 26Q2-10月", "2026-04-01", "2026-11-01"), ("★保留期合計", *HOLDOUT), ("全部", "2023-10-01", "2026-11-01")]


def A2(mode=1):
    X = cap(run_many(range(630, 961, 30), ib=45, mode=mode), 2).join(D[["peff"]], on="day")
    return X[X.peff <= 0.75]


STRATS = {
    "A** 主策略": lambda: A2(),
    "C* 開盤 IB30": lambda: run_many([60], ib=30, mode=1),
    "（參考）XAU 核心原封不動": lambda: (lambda X: X[X.peff <= 0.633])(run_many(range(120, 631, 30)).join(D[["peff"]], on="day")),
    "（參考）B 亞洲反向（已淘汰）": lambda: run_many(range(390, 511, 30), ib=60, mode=-1),
    "（對照）A** 鏡像反向": lambda: A2(mode=-1),
}


def st(X, a, b):
    X = period(X, a, b)
    days = D.index[(D.index >= a) & (D.index < b)]
    if len(X) < 5:
        return dict(筆數=len(X))
    A = X.pnl / X.atr
    d = A.groupby(X.day).sum().reindex(days, fill_value=0)
    eq = d.cumsum()
    mo = d.groupby(d.index.to_period("M")).sum()
    return dict(筆數=len(X), 勝率=round((X.R > 0).mean(), 2), PF=round(X.R[X.R > 0].sum() / -X.R[X.R <= 0].sum(), 2),
                每筆R=round(X.R.mean(), 3), 每筆ATR=round(A.mean(), 4), 日t=round(d.mean() / d.std() * np.sqrt(len(d)), 2),
                年化Sharpe=round(d.mean() / d.std() * np.sqrt(252), 2), 總ATR=round(A.sum(), 1), 回撤ATR=round((eq - eq.cummax()).min(), 1),
                月正=f"{(mo > 0).sum()}/{len(mo)}", 多ATR=round(A[X.side == 1].mean(), 4), 空ATR=round(A[X.side == -1].mean(), 4))


if __name__ == "__main__":
    out, keep = [], {}
    for name, f in STRATS.items():
        X = f()
        keep[name] = X
        T = pd.DataFrame([dict(期間=lab, **st(X, a, b)) for lab, a, b in PERIODS]).set_index("期間")
        out.append(f"\n=== {name} ===\n" + T.to_string())
    # A** + C* 日損益合併
    days = D.index[D.index >= RESEARCH[0]]
    dd = {k: (keep[k].pnl / keep[k].atr).groupby(keep[k].day).sum().reindex(days, fill_value=0) for k in ("A** 主策略", "C* 開盤 IB30")}
    comb = dd["A** 主策略"] + dd["C* 開盤 IB30"]
    lines = ["\n=== A** + C*（日損益相加，ATR 單位）==="]
    for lab, a, b in PERIODS:
        c = comb[(comb.index >= a) & (comb.index < b)]
        lines.append(f"{lab:16s} 日t {c.mean() / c.std() * np.sqrt(len(c)):5.2f}  年化Sharpe {c.mean() / c.std() * np.sqrt(252):5.2f}  總 {c.sum():+7.1f} ATR")
    for lab, a, b in (("研究期", *RESEARCH), ("保留期", *HOLDOUT)):
        x, y = dd["A** 主策略"], dd["C* 開盤 IB30"]
        m = (x.index >= a) & (x.index < b)
        lines.append(f"A**/C* 日相關（{lab}）：{x[m].corr(y[m]):.2f}")
    bh = D.c[D.index >= HOLDOUT[0]]
    lines.append(f"\n同期 NAS100 買進持有：保留期 {bh.iloc[-1] / bh.iloc[0] - 1:+.1%}")
    out += lines
    txt = "第四十部分：凍結後打開保留期（ATR = 當天 ATR10，每筆 1 單位名目；R = 停損距離）" + "\n".join(out)
    open("results_part40.txt", "w").write(txt + "\n")
    pd.concat([v.assign(strategy=k) for k, v in keep.items() if not k.startswith("（")]).to_csv("part40_trades.csv", index=False)
    print(txt)
