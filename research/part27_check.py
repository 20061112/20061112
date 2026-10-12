"""第二十七部分補充：放大停損後較好的幾組，逐年 + 以「天」為單位的 t（同一天多筆單持有到收盤、彼此相關）。
全部事件（不抽樣），K 棒顏色推估同根順序，每筆扣點差。也列出每筆的美元損益（以 0.01 手 = 1 盎司計）。"""
import numpy as np
import pandas as pd
import part27_wide_brackets as W

CASES = [(15, "M15", 2.0, 3.0), (15, "M15", 3.0, 2.0), (30, "M30", 2.0, 2.0), (60, "H1", 2.0, 2.0),
         (240, "H4", 1.0, 2.0), (240, "H4", 0.5, 3.0), (1440, "D1", 0.5, 2.0)]
lines = ["", "=== 補充：較好的幾組（全部事件、逐年、以天計的 t）==="]
for P, nm, s, t in CASES:
    ev = W.events(P)
    rows = []
    for i, L, side, a in ev:
        risk = s * a
        pnl = W.sim_full(i, L, side, L - side * risk, L + side * t * risk, True)
        if pnl is None:
            continue
        rows.append((W.dayv[i], side, (pnl - W.SP[i]) / risk, pnl - W.SP[i], W.B.UP50.iat[i]))
    T = pd.DataFrame(rows, columns=["day", "side", "R", "usd", "up50"])
    T["year"] = pd.DatetimeIndex(T.day).year
    dR = T.groupby("day").R.sum()
    du = T.groupby("day").usd.sum()
    yr = " / ".join(f"{T.R[T.year == y].mean():+.3f}" for y in (2023, 2024, 2025, 2026))
    lines.append(f"  {nm:4s} s={s} t={t}: {len(T)} 筆（每天 {len(T) / T.day.nunique():.1f}），每筆 {T.R.mean():+.3f}R、{T.usd.mean():+.2f} 美元，"
                 f"日 t={dR.mean() / dR.std() * np.sqrt(len(dR)):+.2f}，逐年 {yr}")
    lines.append(f"       做多 {T.R[T.side > 0].mean():+.3f} / 做空 {T.R[T.side < 0].mean():+.3f}；"
                 f"均線上方做多 {T.R[(T.side > 0) & (T.up50 == True)].mean():+.3f}、下方做空 {T.R[(T.side < 0) & (T.up50 == False)].mean():+.3f}；"
                 f"日損益最大回撤 {(du.cumsum().cummax() - du.cumsum()).max():.0f} 美元、總計 {du.sum():+.0f} 美元")
print("\n".join(lines))
open("results_part27.txt", "a").write("\n".join(lines) + "\n")
