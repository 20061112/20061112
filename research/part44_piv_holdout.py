"""第四十四部分：前高前低突破 —— 凍結後一次打開保留期 2025/10 ~ 2026/10 → results_part44.txt、part44_trades.csv
凍結 PIV*：K 線 10 分（01:00 起算），只在 01:00~04:00（broker，= 紐約 18:00~21:00）開始的 K 棒進場；
  方向 = 前一日收盤在 SMA30 上 且 SMA30 比 10 天前高 → 只做多（空單鏡像），其他不做；
  這根 K 棒開盤在前一根高（低）內側、盤中第一次碰到就進場；停損 2×ATR（前 20 根 10 分 K 高低差平均）、停利 3R、當天收盤平倉；
  無 B 條件、無持倉上限、無權益濾網。
逐層：XAU M15 原封不動 → 第一階段 P（10 分、01~04、SMA50）→ 第三階段 PIV*（SMA30/10）；對照 = 同時段每根開盤直接順方向進場。"""
import numpy as np, pandas as pd
from piv_core import run, direction, D, HALVES, RESEARCH, HOLDOUT, period

PER = HALVES + [("研究期合計", *RESEARCH), ("★保留 25Q4-26Q1", "2025-10-01", "2026-04-01"),
                ("★保留 26Q2-10月", "2026-04-01", "2026-11-01"), ("★保留期合計", *HOLDOUT)]
D50, D30 = direction(50, 10), direction(30, 10)
STRATS = {
    "PIV*（定案）": lambda: run(k=10, hours=(1, 4), dirs=D30),
    "（對照）PIV* 時段每根開盤直接順方向": lambda: run(k=10, hours=(1, 4), dirs=D30, ctrl=True),
    "（逐層）第一階段 P：10 分、01~04、SMA50": lambda: run(k=10, hours=(1, 4), dirs=D50),
    "（參考）XAU M15 原封不動（含 B，無上限 / 權益濾網）": lambda: run(k=15, hours=(1, 22), dirs=D50, bthr=0.17),
}


def st(X, a, b):
    X = period(X, a, b)
    days = D.index[(D.index >= a) & (D.index < b)]
    d = X.groupby("day").R.sum().reindex(days, fill_value=0)
    eq = d.cumsum()
    mo = d.groupby(d.index.to_period("M")).sum()
    return dict(筆數=len(X), 勝率=round((X.R > 0).mean(), 2), PF=round(X.R[X.R > 0].sum() / -X.R[X.R <= 0].sum(), 2),
                每筆R=round(X.R.mean(), 3), 日t=round(d.mean() / d.std() * np.sqrt(len(d)), 2),
                年化Sharpe=round(d.mean() / d.std() * np.sqrt(252), 2), 總R=round(X.R.sum()), 回撤R=round((eq - eq.cummax()).min()),
                月正=f"{(mo > 0).sum()}/{len(mo)}", 多R=round(X.R[X.side == 1].mean(), 3), 空R=round(X.R[X.side == -1].mean(), 3),
                多筆=int((X.side == 1).sum()), 空筆=int((X.side == -1).sum()))


if __name__ == "__main__":
    out, keep = [], {}
    for name, f in STRATS.items():
        X = f()
        keep[name] = X
        T = pd.DataFrame([dict(期間=lab, **st(X, a, b)) for lab, a, b in PER]).set_index("期間")
        out.append(f"\n=== {name} ===\n" + T.to_string())
    for lab, a, b in (("研究期", *RESEARCH), ("保留期", *HOLDOUT)):
        s, c = period(keep["PIV*（定案）"], a, b), period(keep["（對照）PIV* 時段每根開盤直接順方向"], a, b)
        out.append(f"突破 − 對照（{lab}）：每筆 {s.R.mean() - c.R.mean():+.3f}R")
    hd = pd.Series(D30, index=D.index)
    hd = hd[(hd.index >= HOLDOUT[0])]
    out.append(f"保留期方向天數：多 {(hd == 1).sum()}、空 {(hd == -1).sum()}、不做 {(hd == 0).sum()}")
    txt = "第四十四部分：前高前低突破凍結後打開保留期（R = 停損距離，每筆固定風險）" + "\n".join(out)
    open("results_part44.txt", "w").write(txt + "\n")
    keep["PIV*（定案）"].to_csv("part44_trades.csv", index=False)
    print(txt)
