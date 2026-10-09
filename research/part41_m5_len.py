"""第四十一部分：M5 回歸長度掃描（短 L 會不會比較好）。

M5 單獨做；L ∈ {100,200,300,400,600,800,1200,1600,2400}；ou / ou_ew；thr 2.5 / 3；
出場：不設目標（停損 1.5 ATR + 時間）或 5 ATR 目標；最多持有 40 根（200 分鐘）或 120 根（= M15 的 40 根）；最多 1 單。
其餘同第 40 部分（點差、隔夜、週五平倉、R = 損益 / 初始停損）。另外記錄訊號方向和過去 L 根漲跌同向的比例（順勢程度）。
"""
import numpy as np
import pandas as pd
from load import load
from part37_wf import zscore
from part39_ou_vs_er import Frame, metr
from part40_target import run

LS = [100, 200, 300, 400, 600, 800, 1200, 1600, 2400]


def main():
    m1 = load("data/XAUUSD_M1_2026.csv")
    days = sorted(set(m1.index.date))
    F = Frame(m1, "5min")
    lines = []
    P = lines.append
    grid = []
    for method in ("ou", "ou_ew"):
        for hold in (40, 120):
            for tgt in (("none", 0), ("atr", 5)):
                P(f"\n=== M5 {method}  持有上限 {hold} 根  目標 {'無' if tgt[0] == 'none' else '5 ATR'}")
                P(f"  {'L':>5}{'≈小時':>6}{'thr':>5}{'筆數':>6}{'每天':>6}{'順勢%':>7}{'每筆R':>8}{'勝率':>6}{'PF':>6}{'總R':>8}"
                  f"{'MDD':>7}{'P/MDD':>7}{'Sharpe':>8}{'前半':>7}{'後半':>7}")
                for L in LS:
                    z, mu = zscore(F.df, method, L)
                    for thr in (2.5, 3.0):
                        T = pd.DataFrame(run(F, method, L, thr, tgt, hold, 1))
                        if len(T) < 10:
                            continue
                        k = F.idx.get_indexer(T.time)
                        trend = np.mean(T.side.to_numpy() == np.sign(F.c[k] - F.c[np.maximum(k - L, 0)]))
                        m = metr(T, days)
                        P(f"  {L:>5}{L * 5 / 60:>6.0f}{thr:>5}{m['n']:>6d}{m['per_day']:>6.2f}{trend:>7.0%}{m['avg']:>+8.3f}"
                          f"{m['win']:>6.0%}{m['PF']:>6.2f}{m['sumR']:>+8.1f}{m['MDD']:>7.1f}{m['P/MDD']:>7.1f}"
                          f"{m['Sharpe']:>+8.2f}{m['前Sharpe']:>+7.2f}{m['後Sharpe']:>+7.2f}")
                        grid.append(dict(method=method, hold=hold, tgt=tgt[0], L=L, thr=thr, trend=trend, **m))
    G = pd.DataFrame(grid)
    G.to_csv("part41_grid.csv", index=False)
    P("\n依 L 平均（8 種 method×hold×目標×2 門檻 = 16 組）")
    P(G.groupby("L")[["trend", "per_day", "avg", "Sharpe", "前Sharpe", "後Sharpe", "P/MDD"]].mean().round(2).to_string())
    txt = "\n".join(lines)
    print(txt)
    open("results_part41.txt", "w").write(txt + "\n")


if __name__ == "__main__":
    main()
