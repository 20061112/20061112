"""第四十八部分：z-score 反轉 —— 凍結後一次打開保留期 2025/10 ~ 2026/10 → results_part48.txt、part48_trades.csv
凍結 Z1：M15（01:00 起算）；z = (收盤 − 20 根均線) / 20 根標準差；
  前一根 z ≤ −2、這根 z > −2（回到布林下軌內）且這根開始於 broker 09:00~20:00 → 多單候選（空單鏡像）；
  之後 3 根內突破這根高點才進場（先破停損就取消）；停損 = 近 3 根最低 − 0.1×ATR14；停利 2R；最多持有 16 根收盤平倉。
  無濾網（第三階段沒有一個通過）。
逐層 / 對照：XAU 第四部分 D 原封不動（型態 + z、全天）、純 z 全天（第一階段時段之前）、只要型態不看 z。
注意：第二十三部分曾把 XAU 第四部分的 D / 三選二搬到 NAS 2025/10~2026/10 測過（−0.00R / −0.09R），
      所以「D 原封不動」那一列在保留期不是第一次看；Z1（純 z、時段）之前沒有在這段跑過。"""
import numpy as np, pandas as pd
from z_core import run, m1_check, D, HALVES, RESEARCH, HOLDOUT, period

PER = HALVES + [("研究期合計", *RESEARCH), ("★保留 25Q4-26Q1", "2025-10-01", "2026-04-01"),
                ("★保留 26Q2-10月", "2026-04-01", "2026-11-01"), ("★保留期合計", *HOLDOUT)]
STRATS = {
    "Z1（定案）": lambda: run(k=15, src="zcross", hours=(9, 20)),
    "Z1 + 滑價 1 點": lambda: run(k=15, src="zcross", hours=(9, 20), slip=1.0),
    "（逐層）純 z 全天 M15": lambda: run(k=15, src="zcross"),
    "（參考）XAU 第四部分 D 原封不動": lambda: run(k=15),
    "（對照）只要型態、不看 z": lambda: run(k=15, src="pattern_noz", hours=(9, 20)),
}


def st(X, a, b, col="R"):
    X = period(X, a, b)
    days = D.index[(D.index >= a) & (D.index < b)]
    d = X.groupby("day")[col].sum().reindex(days, fill_value=0)
    eq = d.cumsum()
    mo = d.groupby(d.index.to_period("M")).sum()
    r = X[col]
    return dict(筆數=len(X), 勝率=round((r > 0).mean(), 2), PF=round(r[r > 0].sum() / -r[r <= 0].sum(), 2), 每筆R=round(r.mean(), 3),
                日t=round(d.mean() / d.std() * np.sqrt(len(d)), 2), 年化Sharpe=round(d.mean() / d.std() * np.sqrt(252), 2),
                總R=round(r.sum()), 回撤R=round((eq - eq.cummax()).min()), 月正=f"{(mo > 0).sum()}/{len(mo)}",
                多R=round(r[X.side == 1].mean(), 3), 空R=round(r[X.side == -1].mean(), 3))


if __name__ == "__main__":
    out, keep = [], {}
    for name, f in STRATS.items():
        X = f()
        keep[name] = X
        out.append(f"\n=== {name} ===\n" + pd.DataFrame([dict(期間=lab, **st(X, a, b)) for lab, a, b in PER]).set_index("期間").to_string())
    Y = m1_check(period(keep["Z1（定案）"], *HOLDOUT), 15)
    out.append(f"\nZ1 保留期用 M1 逐分鐘重算：每筆 {Y.R_m1.mean():+.3f}R（K 棒層級 {Y.R.mean():+.3f}R），相關 {Y[['R', 'R_m1']].corr().iat[0, 1]:.3f}")
    bh = D.c[D.index >= HOLDOUT[0]]
    out.append(f"同期 NAS100 買進持有 {bh.iloc[-1] / bh.iloc[0] - 1:+.1%}")
    txt = "第四十八部分：z-score 反轉凍結後打開保留期（R = 停損距離，每筆固定風險）" + "\n".join(out)
    open("results_part48.txt", "w").write(txt + "\n")
    keep["Z1（定案）"].to_csv("part48_trades.csv", index=False)
    print(txt)
