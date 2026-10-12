"""第四十六部分（五）：CPR（TC/BC）、Woodie P、DeMark P 的反彈單，扣成本後能不能賺？
當天第一次碰到點位 → 限價在點位順「反彈方向」進場（上方來做多、下方來做空）；
停損 = 點位反側 s × 日ATR，停利 = t × 停損；進場那根若已穿越停損 → 停損；當根停利不算（順序不明）；之後同根兩邊都碰 → 停損；
240 分鐘未決以收盤計。成本：點差（限價單不加滑價；另列 +0.1 美元）。對照：平移 ±0.15/0.3/0.5 ATR 的假點位。"""
import numpy as np
import pandas as pd
import part46_pivot_methods as PV

SETS = [(0.05, 1), (0.05, 2), (0.1, 1), (0.1, 2)]
TARGET = [("CPR", "TC"), ("CPR", "BC"), ("Woodie", "P"), ("DeMark", "P"), ("Classic日", "P")]
rows = []
for a, b in zip(PV.DAYS[:-1], PV.DAYS[1:]):
    atr = PV.ATR.get(b, np.nan)
    if not np.isfinite(atr):
        continue
    g = PV.days[b]
    H, Lo, C = g.high.values, g.low.values, g.close.values
    S = g.spread.values * 0.01
    hrs = g.index.hour.values
    lv = PV.levels(PV.daily.loc[a], None)
    for key in TARGET:
        for off in (0.0,) + PV.OFFS:
            L = lv[key] + off * atr
            hit = (Lo <= L) & (H >= L) & (hrs < 22)
            hit[0] = False
            if not hit.any():
                continue
            i = int(np.argmax(hit))
            side = np.sign(C[i - 1] - L)
            if side == 0:
                continue
            for s, t in SETS:
                risk = s * atr
                stop, tgt = L - side * risk, L + side * t * risk
                if (Lo[i] <= stop) if side > 0 else (H[i] >= stop):
                    res = -risk
                else:
                    hh, ll = H[i + 1:i + 241], Lo[i + 1:i + 241]
                    hs, ht = ((ll <= stop), (hh >= tgt)) if side > 0 else ((hh >= stop), (ll <= tgt))
                    js = int(np.argmax(hs)) if hs.any() else 10 ** 9
                    jt = int(np.argmax(ht)) if ht.any() else 10 ** 9
                    if js == jt == 10 ** 9:
                        res = (C[min(i + 240, len(C) - 1)] - L) * side
                    else:
                        res = -risk if js <= jt else t * risk
                rows.append(dict(day=b, lvl=f"{key[0]} {key[1]}", off=off, s=s, t=t, R=(res - S[i]) / risk, R01=(res - S[i] - 0.1) / risk))
X = pd.DataFrame(rows)
X["year"] = pd.DatetimeIndex(X.day).year
lines = ["第四十六部分（五）：中軸類點位的反彈單（每筆 R，扣點差；括號為再扣 0.1 美元）"]
for (s, t), g in X.groupby(["s", "t"]):
    lines.append(f"\n停損 {s} 日ATR、RR {t}")
    for lv, h in g.groupby("lvl"):
        r, f = h[h.off == 0], h[h.off != 0]
        yr = " / ".join(f"{r.R[r.year == y].mean():+.3f}" for y in (2023, 2024, 2025, 2026))
        lines.append(f"  {lv:12s} n={len(r):4d} 真 {r.R.mean():+.3f}（{r.R01.mean():+.3f}） t={r.R.mean() / r.R.std() * np.sqrt(len(r)):+.2f} 假 {f.R.mean():+.3f} | 逐年 {yr}")
print("\n".join(lines))
open("results_part46_cpr.txt", "w").write("\n".join(lines) + "\n")
