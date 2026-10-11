"""第三十四部分：第三十三部分的完全反向 —— 通道順勢網格。

同樣的線、同樣的觸發價，只是方向相反（part33_band_grid.sim 的 rev=-1）：
  價格 ≥ 中線 + k·u → 第 k 層「買」1 單位（往上突破加碼）；下方鏡像做空。
  出場「往內一層」：第 k 層在價格回到 中線 + (k−1)·u 時賣出（第 1 層 = 回到中線）；「回中線」：全部回中線才平。
  K = 4：碰到 5 單位全平（原本是停損，反過來變成停利）並鎖住到回中線；K = 10：不設上限停利。
點差照樣付（每筆買進付點差），所以反向版 ≠ 原版的負號：反向 ≈ −（原版毛利）− 點差。
"""
import numpy as np, pandas as pd
from multiprocessing import Pool
import part33_band_grid as B
from part28_ib_grid import t


def job(args):
    tf, kind, K, ex, rev, seed, zero = args
    if zero: B.SP = np.zeros_like(B.SP)
    oh = B.OHLC0 if seed is None else B.mirrored(B.OHLC0, np.random.default_rng(seed))
    mid, u = B.bands(oh, tf, kind)
    return args, B.sim(oh, mid, u, K, ex, rev)


if __name__ == "__main__":
    cfg = [(tf, kind, K, ex) for tf in ("5min", "15min", "1h") for kind in ("BB", "KC") for K in (4, 10) for ex in ("inward", "mean")]
    jobs = [c + (-1, sd, False) for c in cfg for sd in [None] + list(range(B.SEEDS))] + \
           [c + (rv, None, True) for c in cfg for rv in (1, -1)]
    with Pool(4) as pool:
        res = dict(pool.map(job, jobs, chunksize=1))
    out = open("results_part34.txt", "w")
    def P(s=""): print(s); out.write(s + "\n")
    P("第三十四部分：通道順勢網格（第三十三部分的反向：突破中線 + k·u 順勢加碼、回一層出）")
    P(f"單位：每週損益 ÷ 日 ATR10；括號 = t；基準 = 鏡像 M1 同模型 {B.SEEDS} 次；樣本內 2023~24 / 樣本外 2025~26。")
    P("「零點差 原版 / 反向」= 兩者都不付點差時的毛利（應互為負號，用來檢查）。")
    P(f"\n  {'週期':5s} {'通道':3s} {'設定':9s} {'出場':6s} | {'反向 樣本內':>14s} {'樣本外':>14s} | {'勝率':>4s} {'最差週':>7s} {'最佳週':>7s} {'每日筆':>6s} | {'基準':>6s} | {'零點差 原版/反向':>16s} | 逐年")
    keep = []
    for c in cfg:
        tf, kind, K, ex = c
        r = res[c + (-1, None, False)]; b = pd.concat([res[c + (-1, sd, False)] for sd in range(B.SEEDS)])
        z1 = res[c + (1, None, True)]; z2 = res[c + (-1, None, True)]
        a = r.pnl / r.atr; ab = b.pnl / b.atr; IS = (r.year < B.CUT).values
        yr = " ".join(f"{y % 100}:{a[(r.year == y).values].mean():+.2f}" for y in range(2023, 2027))
        P(f"  {tf:5s} {kind:3s} {('K4 停利5' if K == 4 else 'K10'):9s} {('往內' if ex == 'inward' else '回中線'):6s} | "
          f"{a[IS].mean():+6.2f}({t(a[IS]):+5.2f}) {a[~IS].mean():+6.2f}({t(a[~IS]):+5.2f}) | {(a > 0).mean():4.0%} {a.min():+7.1f} {a.max():+7.1f} "
          f"{r.trades.mean() / 5:6.1f} | {ab.mean():+6.2f} | {(z1.pnl / z1.atr).mean():+7.2f} / {(z2.pnl / z2.atr).mean():+6.2f} | {yr}")
        keep.append(dict(tf=tf, kind=kind, K=K, exit=ex, IS=a[IS].mean(), t_IS=t(a[IS]), OOS=a[~IS].mean(), t_OOS=t(a[~IS]),
                         win=(a > 0).mean(), worst=a.min(), best=a.max(), per_day=r.trades.mean() / 5, base=ab.mean(),
                         gross_orig=(z1.pnl / z1.atr).mean(), gross_rev=(z2.pnl / z2.atr).mean()))
    pd.DataFrame(keep).to_csv("part34_summary.csv", index=False, float_format="%.4f")
    out.close()
