"""第二十二部分補充：把「碰到點位後偏穿越」直接當突破單，扣點差看有沒有賺。

進場：當天第一次碰到點位（同 part22_pivots.py 的事件），在點位用停損單順「穿越」方向進場。
停損 = 點位反側 s·ATR；停利 = 穿越方向 t·ATR；碰觸那根若也碰到停損 → 算停損（保守）；
之後同一根停損停利都碰 → 停損；240 分鐘未決以收盤計。每筆扣當根點差。
真點位 vs 平移 ±0.15/0.3/0.5 ATR 的假點位；逐年。
"""
import numpy as np
import pandas as pd
import part22_pivots as P  # 重用資料與點位（會順便重跑主檢驗）

d = P.d
d["sp"] = d.spread * 0.01
SETS = [(0.1, 0.1), (0.1, 0.2), (0.2, 0.2), (0.2, 0.4)]
TARGETS = {"PDH": ("PrevDay", "PDH"), "PDL": ("PrevDay", "PDL"), "Classic R1": ("Classic", "R1"),
           "Classic S1": ("Classic", "S1"), "Camarilla H4": ("Camarilla", "H4"), "Camarilla L4": ("Camarilla", "L4")}

rows = []
for a, b in zip(P.DAYS[:-1], P.DAYS[1:]):
    atr = P.ATR.get(b, np.nan)
    if not np.isfinite(atr):
        continue
    g = P.days[b]
    H, Lo, C = g.high.values, g.low.values, g.close.values
    S = (g.spread.values * 0.01)
    hrs = g.index.hour.values
    lv = P.levels_for(P.daily.loc[a], None)
    for lab, key in TARGETS.items():
        for off in (0.0,) + P.OFFS:
            L = lv[key] + off * atr
            hit = (Lo <= L) & (H >= L) & (hrs < 22)
            hit[0] = False
            if not hit.any():
                continue
            i = int(np.argmax(hit))
            side0 = np.sign(C[i - 1] - L)
            if side0 == 0:
                continue
            d_ = -side0  # 穿越方向
            for s, t in SETS:
                stop, tgt = L - d_ * s * atr, L + d_ * t * atr
                st0 = Lo[i] <= stop if d_ > 0 else H[i] >= stop
                if st0:
                    res = -s
                else:
                    res = None
                    for j in range(i + 1, min(i + 241, len(C))):
                        hs = Lo[j] <= stop if d_ > 0 else H[j] >= stop
                        ht = H[j] >= tgt if d_ > 0 else Lo[j] <= tgt
                        if hs:
                            res = -s; break
                        if ht:
                            res = t; break
                    if res is None:
                        res = d_ * (C[min(i + 240, len(C) - 1)] - L) / atr
                rows.append(dict(day=b, lvl=lab, off=off, s=s, t=t, R=(res - S[i] / atr) / s, cost=S[i] / atr / s))

T = pd.DataFrame(rows)
T["year"] = pd.DatetimeIndex(T.day).year
lines = ["", "=== 補充：碰到點位 → 順穿越方向做突破單（每筆 R，已扣點差）==="]
for (s, t), g in T.groupby(["s", "t"]):
    lines.append(f"\n停損 {s} ATR / 停利 {t} ATR（點差平均 {g.cost.mean():.3f}R）")
    lines.append(f"  {'點位':14s} {'真 n':>5s} {'真 R':>7s} {'t':>6s} {'假 R':>7s} | 逐年真 R（2023/2024/2025/2026）")
    for lab in TARGETS:
        x = g[g.lvl == lab]
        r, f = x[x.off == 0].R, x[x.off != 0].R
        yrs = " / ".join(f"{r[x[x.off == 0].year == y].mean():+.3f}" for y in (2023, 2024, 2025, 2026))
        lines.append(f"  {lab:14s} {len(r):5d} {r.mean():+7.3f} {r.mean() / r.std() * np.sqrt(len(r)):+6.2f} {f.mean():+7.3f} | {yrs}")
print("\n".join(lines))
open(P.OUT, "a").write("\n".join(lines) + "\n")
