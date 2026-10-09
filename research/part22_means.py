"""第二十二部分 A：哪一種「均值」最有吸引力？

對每個週期（M1 / M5 / M15）、每種均值，D = (收盤 − 均值) / ATR14：
  1. 全部 K 棒：前瞻 H 根報酬（ATR）對 −D 回歸，β = 平均被拉回的比例（> 0 = 均值回歸、< 0 = 延續）；
     每 H 根取一個樣本以避免重疊；另算 Spearman IC。
  2. 事件：|D| 第一次超過 IS 90% 分位（不同均值的距離分佈不同，用分位數讓事件稀有度相同），
     做反向（回歸）持有 12 根，扣點差；race = 先碰事件當下均值 vs 先再往外 1 ATR。
"""
import numpy as np
import pandas as pd
from part22_lib import load_tf, means, atr, fwd_returns, events, race, spearman, tstat_ic, CUT

out = open("results_part22.txt", "w")
def P(s=""): print(s); out.write(s + "\n"); out.flush()

P("=" * 120)
P("第二十二部分 A：均值定義比較（IS = 1/2~5/31，OOS = 6/1~10/8）")
rows = []
for tf in ("M1", "M5", "M15"):
    d = load_tf(tf)
    A = atr(d, 14)
    M = means(d)
    cost = d.spread / 100 / A
    isx = d.index < CUT
    Hs = {"M1": (5, 15, 60), "M5": (3, 12, 48), "M15": (2, 6, 16)}[tf]
    P(f"\n--- {tf}（{len(d)} 根）  前瞻 H = {Hs} 根 ---")
    P(f"{'均值':16s} " + " ".join(f"β{h:<3d}IS   OOS  " for h in Hs) + " | IC(H中) IS / OOS | 事件 n IS/OOS  回歸12根 IS / OOS (t)   race回歸率 IS/OOS")
    for name in M.columns:
        D = (d.close - M[name]) / A
        cells = []
        for H in Hs:
            f = fwd_returns(d, A, H)
            for per in (isx, ~isx):
                sub = pd.DataFrame({"D": D, "f": f})[per].iloc[::H].dropna()
                sub = sub[sub.D.abs() < 8]
                b = np.polyfit(-sub.D, sub.f, 1)[0]
                cells.append(b)
        Hm = Hs[1]
        f = fwd_returns(d, A, Hm)
        ics = []
        for per in (isx, ~isx):
            sub = pd.DataFrame({"D": D, "f": f})[per].iloc[::Hm].dropna()
            ics.append(spearman(-sub.D, sub.f))
        thr = D[isx].abs().quantile(0.90)
        ev = events(D, d.seg, d0=thr, rearm=thr / 3)
        ev = ev[ev + 13 < len(d)]
        s = -np.sign(D.to_numpy()[ev])
        f12 = fwd_returns(d, A, 12).to_numpy()
        # 下一根開盤進場：(C[t+12] − O[t+1]) 近似為 f12 − (O[t+1]−C[t])/A，差異很小，這裡用收盤基準 + 點差
        g = s * f12[ev] - cost.to_numpy()[ev]
        rc = race(d, ev, M[name].to_numpy()[ev], A.to_numpy()[ev], -s)
        isev = d.index[ev] < CUT
        stat = []
        for pm in (isev, ~isev):
            x = g[pm]; x = x[~np.isnan(x)]
            stat.append((len(x), x.mean(), x.mean() / x.std() * np.sqrt(len(x)), (rc[pm] == 1).sum() / max((rc[pm] != 0).sum(), 1)))
        P(f"{name:16s} " + " ".join(f"{cells[2*k]:+.3f} {cells[2*k+1]:+.3f}  " for k in range(len(Hs)))
          + f" | {ics[0]:+.3f} / {ics[1]:+.3f} | {stat[0][0]:4d}/{stat[1][0]:4d}  {stat[0][1]:+.3f} / {stat[1][1]:+.3f} ({stat[0][2]:+.1f}/{stat[1][2]:+.1f})"
          + f"   {stat[0][3]:.2f} / {stat[1][3]:.2f}  (門檻 {thr:.2f} ATR)")
        rows.append(dict(tf=tf, mean=name, **{f"b{h}_{p}": cells[2*k + j] for k, h in enumerate(Hs) for j, p in enumerate(("IS", "OOS"))},
                         ic_IS=ics[0], ic_OOS=ics[1], n_IS=stat[0][0], n_OOS=stat[1][0], fade_IS=stat[0][1], fade_OOS=stat[1][1],
                         t_IS=stat[0][2], t_OOS=stat[1][2], race_IS=stat[0][3], race_OOS=stat[1][3], thr=thr))
pd.DataFrame(rows).to_csv("part22_means.csv", index=False)
