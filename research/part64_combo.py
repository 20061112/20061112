"""第六十四部分：把第六十三部分「通過」的面向組合 —— 只用研究期挑，2026 驗證 → results_part64.txt、part64_combo.csv

組合空間（2 × 2 × 2 × 2 × 2 × 3 = 96 組）：
  IB 長度 {45, 60}、突破有效 {2, 3} 小時、起點間隔 {15, 30} 分、出場 {收盤, 18:00}、到 1R 移保本 {否, 是}、
  每天每方向疊單上限 {不限, 3, 2}
濾網 peff ≤ 0.633 一律套用。
挑選分數：研究期逐日 t 值（= 日報酬平均 / 標準差 × √天數，與部位大小無關）。
另外檢查：研究期排名 vs 驗證期排名的相關、研究期前 10 名在驗證期的位置。
"""
import itertools
import numpy as np
import pandas as pd
from core_ib import events, simulate, report

pd.set_option("display.width", 260)


def cap_stack(X, n):
    if n is None:
        return X
    X = X.sort_values("t_in")
    return X[X.groupby(["day", "side"]).cumcount() < n]


def main():
    rows = []
    ev = {}
    for ib, win, step in itertools.product((45, 60), (2, 3), (15, 30)):
        ev[(ib, win, step)] = events(ib_min=ib, win_h=win, step=step)
    for (ib, win, step), E in ev.items():
        for ex, be in itertools.product((None, 18 * 60), (None, 1)):
            X = simulate(E, exit_min=ex, be=be)
            for cap in (None, 3, 2):
                rows.append(dict(IB=ib, 有效=win, 間隔=step, 出場="18:00" if ex else "收盤", 保本="1R" if be else "-",
                                 疊單="不限" if cap is None else cap, **report(cap_stack(X, cap))))
    R = pd.DataFrame(rows)
    R.to_csv("part64_combo.csv", index=False)
    rk_r, rk_v = R["研究 日t"].rank(), R["驗證26 日t"].rank()
    cols = ["IB", "有效", "間隔", "出場", "保本", "疊單", "24/11~12 PF", "25H1 PF", "25H2 PF", "研究 PF", "研究 R", "研究 日t",
            "研究 筆/天", "研究 總", "驗證26 PF", "驗證26 R", "驗證26 日t", "驗證26 總"]
    base = R[(R.IB == 60) & (R.有效 == 3) & (R.間隔 == 30) & (R.出場 == "收盤") & (R.保本 == "-") & (R.疊單 == "不限")]
    out = ["第六十四部分：通過面向的組合（96 組；只用研究期挑）",
           f"研究期日 t 與驗證期日 t 的等級相關：{rk_r.corr(rk_v):+.2f}；研究期日 t 與驗證期 PF：{rk_r.corr(R['驗證26 PF'].rank()):+.2f}",
           "\n基準（核心）：", base[cols].to_string(index=False),
           "\n研究期日 t 前 10 名：", R.sort_values("研究 日t", ascending=False)[cols].head(10).to_string(index=False)]
    top = R.sort_values("研究 日t", ascending=False).head(10)
    out.append(f"\n前 10 名的驗證期日 t 平均 {top['驗證26 日t'].mean():.2f}（96 組平均 {R['驗證26 日t'].mean():.2f}，基準 {base['驗證26 日t'].iloc[0]:.2f}）；"
               f"驗證期 PF 平均 {top['驗證26 PF'].mean():.2f}（全部 {R['驗證26 PF'].mean():.2f}）")
    out.append("\n各選項平均（研究日t / 驗證日t / 研究PF / 驗證PF）：")
    for c in ("IB", "有效", "間隔", "出場", "保本", "疊單"):
        out.append(R.groupby(c)[["研究 日t", "驗證26 日t", "研究 PF", "驗證26 PF", "研究 總", "驗證26 總"]].mean().round(2).to_string())
    txt = "\n".join(out)
    open("results_part64.txt", "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
