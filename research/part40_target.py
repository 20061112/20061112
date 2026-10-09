"""第四十部分：把 OU 目標拉近（μ* 太遠幾乎碰不到）。

目標三種寫法（觸價成交；同一根先檢查停損 → 保守）
  frac f  目標 = 進場價 + f·(μ* − 進場價)          f ∈ {0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1}
  sig Δ   目標 = 進場價 + d·Δ·σ_eq（z 退回 Δ）         Δ ∈ {0.25, 0.5, 1, 2}
  atr k   目標 = 進場價 + d·k·ATR（對照組）            k ∈ {1, 2, 3, 5}
其餘同第 39 部分：|z| ≥ 3、方向往 μ*、停損 1.5 ATR、最多 40 根（另測 80 根）、點差、隔夜、週五平倉、週五 20 點後不開新單、
R = 損益 / 初始停損；同一設定最多 1 單（另測 unlimited）。
"""
import numpy as np
import pandas as pd
from load import load
from part37_wf import zscore
from part39_ou_vs_er import Frame, SETS, SWAP, metr, daily

SL = 1.5
TARGETS = [("frac", f) for f in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)] + \
          [("sig", s) for s in (0.25, 0.5, 1.0, 2.0)] + [("atr", k) for k in (1, 2, 3, 5)] + [("none", 0)]


def run(F, method, L, thr, tgt, hold, cap):
    z, mu = zscore(F.df, method, L)
    sd = (F.c - mu) / z
    za = np.abs(z)
    n = len(z)
    rows, open_exit = [], []
    for t in range(1, n - 1):
        if not (za[t] >= thr and za[t - 1] < thr) or np.isnan(F.A[t]):
            continue
        k = t + 1
        if F.dow[k] == 4 and F.hour[k] >= 20:
            continue
        open_exit = [x for x in open_exit if x >= k]
        if cap is not None and len(open_exit) >= cap:
            continue
        d, e, a = -int(np.sign(z[t])), F.o[k], F.A[t]
        if tgt[0] == "frac":
            tp = e + tgt[1] * (mu[t] - e)
        elif tgt[0] == "sig":
            tp = e + d * tgt[1] * abs(sd[t])
        elif tgt[0] == "atr":
            tp = e + d * tgt[1] * a
        else:
            tp = None
        if tp is not None and d * (tp - e) <= 0:
            continue
        risk = SL * a
        stop = e - d * risk
        j_out, x, why = None, None, "time"
        for j in range(k, min(k + hold, n)):
            if (F.l[j] <= stop) if d == 1 else (F.h[j] >= stop):
                j_out, x, why = j, stop, "stop"
                break
            if tp is not None and ((F.h[j] >= tp) if d == 1 else (F.l[j] <= tp)):
                j_out, x, why = j, tp, "tp"
                break
            if F.wk_last[j]:
                j_out, x, why = j, F.c[j], "flat"
                break
        if j_out is None:
            j_out = min(k + hold, n) - 1
            x = F.c[j_out]
        open_exit.append(j_out)
        usd = (x - e) * d - F.spr[k] - SWAP * F.nights(k, j_out)
        rows.append(dict(time=F.idx[t], xtime=F.idx[j_out], side=d, why=why, bars=j_out - k + 1,
                         tdist=abs(tp - e) / a if tp is not None else np.nan, R=usd / risk))
    return rows


def main():
    m1 = load("data/XAUUSD_M1_2026.csv")
    days = sorted(set(m1.index.date))
    frames = {tf: Frame(m1, tf) for tf in ("5min", "15min", "1h")}
    lines = []
    P = lines.append
    grid = []
    P("目標拉近：每列 = 目標寫法；目標距離 = 進場到目標幾個 ATR（中位數）；停損固定 1.5 ATR")
    for hold, cap, mlab in ((40, 1, "最多1單 H40"), (80, 1, "最多1單 H80"), (40, None, "不限單數 H40")):
        for sn, legs in SETS.items():
            P(f"\n=== {sn} [{mlab}]")
            P(f"  {'目標':<12}{'距離ATR':>8}{'碰到目標':>8}{'停損':>6}{'時間/平倉':>9}{'筆數':>6}{'每筆R':>8}{'勝率':>6}{'PF':>6}"
              f"{'總R':>8}{'MDD':>7}{'P/MDD':>7}{'Sharpe':>8}{'前半':>7}{'後半':>7}")
            for tgt in TARGETS:
                rows = []
                for tf, method, L in legs:
                    rows += run(frames[tf], method, L, 3.0, tgt, hold, cap)
                T = pd.DataFrame(rows).sort_values("time")
                m = metr(T, days)
                w = T.why.value_counts(normalize=True)
                lab = f"{tgt[0]} {tgt[1]:g}" if tgt[0] != "none" else "無目標"
                P(f"  {lab:<12}{T.tdist.median():>8.1f}{w.get('tp', 0):>8.0%}{w.get('stop', 0):>6.0%}"
                  f"{w.get('time', 0) + w.get('flat', 0):>9.0%}{m['n']:>6d}{m['avg']:>+8.3f}{m['win']:>6.0%}{m['PF']:>6.2f}"
                  f"{m['sumR']:>+8.1f}{m['MDD']:>7.1f}{m['P/MDD']:>7.1f}{m['Sharpe']:>+8.2f}{m['前Sharpe']:>+7.2f}{m['後Sharpe']:>+7.2f}")
                grid.append(dict(mode=mlab, set=sn, tgt=lab, kind=tgt[0], dist=T.tdist.median(),
                                 hit=w.get("tp", 0), **m))
    G = pd.DataFrame(grid)
    G.to_csv("part40_grid.csv", index=False)
    P("\n" + "=" * 120)
    P("各目標在 4 組設定 × 3 種模式（12 組）的平均")
    P(G.groupby("tgt", sort=False)[["dist", "hit", "avg", "win", "Sharpe", "前Sharpe", "後Sharpe", "P/MDD"]].mean()
      .round(2).to_string())
    txt = "\n".join(lines)
    print(txt)
    open("results_part40.txt", "w").write(txt + "\n")


if __name__ == "__main__":
    main()
