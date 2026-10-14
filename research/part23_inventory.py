"""第二十三部分 第 3 步：session 自己的參考 — 隔夜部位（overnight inventory）。

AMT 說法（Dalton / Steidlmayer）：新的 session 開盤時，如果之前的部位偏向一邊
（開盤價在之前區間的上緣、高於之前的 VWAP），短線部位需要修正 → 開盤後先往反方向走，回到之前區間的中間。
事前登記的預測：
  N1  開盤價相對「之前區間」的位置 inv = (開盤 − 區間中點) / (區間寬/2)，inv 越正 → 開盤後 60 / 120 分越往下
  N2  |inv| 大（> 0.6）時，開盤後 120 分內回到之前區間中點 / VWAP 的比例 > 安慰劑
定義：
  NY  開盤 15:30，之前區間 = 當天 01:00~15:29（亞洲 + 倫敦 = 隔夜）
  LDN 開盤 10:00，之前區間 = 當天 01:00~09:59（亞洲）
  安慰劑：同樣的計算放在其他時間點（12:00、13:00、18:00、20:00，之前區間 = 當天 01:00 到那個時間），
          問「session 開盤是否特別」
  VWAP 用 tick volume 加權（M1 典型價）。報酬用 ATR10、扣點差，方向 = −sign(inv)（修正方向）。
"""
import pickle
import numpy as np
import pandas as pd
from part22_core import load_m1, arrays, race, tstat

YRS = (2023, 2024, 2025, 2026)
POINTS = {"LDN": 10 * 60, "NY": 15 * 60 + 30, "P12": 12 * 60, "P13": 13 * 60, "P18": 18 * 60, "P20": 20 * 60}


def run():
    d = load_m1()
    H, L, C, O, SP = arrays(d)
    V = d.tickvol.values.astype(float)
    TP = (H + L + C) / 3
    D, S = pickle.load(open("part22_sessions.pkl", "rb"))
    atr = D.atr
    rows = []
    for day, g in d.groupby("day"):
        if len(g) < 1000 or day not in atr.index or np.isnan(atr[day]):
            continue
        a = d.index.get_loc(g.index[0])
        mod = g["mod"].values
        end = a + len(g) - 1
        for name, t in POINTS.items():
            k = np.searchsorted(mod, t)
            if k >= len(g) or mod[k] != t or k < 120:
                continue
            j = a + k                                      # 開盤那根
            hi, lo = H[a:j].max(), L[a:j].min()
            mid, half = (hi + lo) / 2, (hi - lo) / 2
            vw = (TP[a:j] * V[a:j]).sum() / V[a:j].sum()
            o, A = O[j], atr[day]
            inv = (o - mid) / half
            dr = -np.sign(inv) if inv != 0 else 1          # 修正方向
            fr = lambda m: ((C[min(j + m, end)] - o) * dr - SP[j]) / A
            # 120 分內是否回到中點 / VWAP（在往外延伸 0.2 ATR 之前）
            lim = min(j + 120, end)
            hit_mid = (L[j:lim].min() <= mid) if inv > 0 else (H[j:lim].max() >= mid)
            hit_vw = (L[j:lim].min() <= vw) if o > vw else (H[j:lim].max() >= vw)
            first_mid = race(H, L, j, mid if inv < 0 else o + 0.2 * A, o - 0.2 * A if inv < 0 else mid, lim)
            first_mid = (first_mid == -1) if inv > 0 else (first_mid == 1)
            rows.append(dict(day=day, year=day.year, pt=name, inv=inv, vw_z=(o - vw) / A, rng=(hi - lo) / A,
                             dist_mid=abs(o - mid) / A, r60=fr(60), r120=fr(120), hit_mid=hit_mid, hit_vw=hit_vw,
                             first_mid=first_mid))
    E = pd.DataFrame(rows)
    E.to_csv("part23_inventory.csv", index=False)
    return E


def summarize(E):
    out = ["=" * 120, "第 3 步：隔夜部位修正（開盤價在之前區間的位置 inv；報酬方向 = 往區間中點修正，ATR，扣點差）",
           "   預測 N1：|inv| 越大，往中點修正的報酬越大；N2：大 |inv| 時 120 分內回到中點的比例 > 安慰劑時間點", "=" * 120]
    E = E.copy()
    E["bin"] = pd.cut(E.inv.abs(), [0, 0.3, 0.6, 0.9, 10], labels=["0~0.3", "0.3~0.6", "0.6~0.9", ">0.9（區間外開）"])
    for pt in POINTS:
        g = E[E.pt == pt]
        out.append(f"\n  [{pt}{'（安慰劑）' if pt.startswith('P') else ''}]  n={len(g)}  inv 與 r120 的相關（修正方向前）"
                   f" {np.corrcoef(g.inv, g.r120 * -np.sign(g.inv))[0, 1]:+.3f}")
        for b, gb in g.groupby("bin", observed=True):
            yr = " ".join(f"{gb[gb.year == y].r120.mean():+.3f}" for y in YRS)
            a, v = gb[gb.year <= 2024].r120, gb[gb.year >= 2025].r120
            out.append(f"    |inv| {b:<14} n={len(gb):4d}  r60 {gb.r60.mean():+.3f}  r120 {gb.r120.mean():+.3f} (t {tstat(gb.r120):+4.1f})  "
                       f"發現 {a.mean():+.3f} / 驗證 {v.mean():+.3f}  逐年 {yr}  "
                       f"120分內回中點 {gb.hit_mid.mean()*100:4.1f}%（先於往外0.2ATR {gb.first_mid.mean()*100:4.1f}%）"
                       f"  回VWAP {gb.hit_vw.mean()*100:4.1f}%  距中點 {gb.dist_mid.median():.2f} ATR")
    return out


if __name__ == "__main__":
    E = run()
    txt = "\n".join(summarize(E))
    print(txt)
    open("results_part23_step3.txt", "w").write(txt + "\n")
