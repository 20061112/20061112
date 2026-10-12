"""第四十六部分（三）：其他 Pivot 算法 —— 和第二十二部分相同方法（第一次碰觸後先反彈 k·ATR 還是先穿越；對照平移的假點位）。
新增：Woodie、DeMark、CPR（中樞區間 TC/BC）、月 Pivot（Classic）；Classic 日線當參照。
  Woodie : P=(H+L+2C)/4，R1=2P−L，S1=2P−H，R2=P+(H−L)，S2=P−(H−L)
  DeMark : X = H+2L+C（C<O）/ 2H+L+C（C>O）/ H+L+2C（C=O）；P=X/4，R1=X/2−L，S1=X/2−H
  CPR    : P=(H+L+C)/3，BC=(H+L)/2，TC=2P−BC
  月     : 前一個月的 Classic P / R1 / S1 / R2 / S2
"""
import numpy as np
import pandas as pd
from load import load

KS = (0.05, 0.10, 0.20)
OFFS = (-0.5, -0.3, -0.15, 0.15, 0.3, 0.5)
d = load("data/XAUUSD_M1_2023_2026.csv")
d["day"] = (d.index - pd.Timedelta(hours=1)).normalize()
days = {k: g for k, g in d.groupby("day") if len(g) >= 600}
DAYS = sorted(days)
daily = pd.DataFrame({k: dict(H=g.high.max(), L=g.low.min(), C=g.close.iloc[-1], O=g.open.iloc[0]) for k, g in days.items()}).T
daily.index = pd.DatetimeIndex(daily.index)
ATR = (daily.H - daily.L).rolling(10).mean().shift(1)
mon = daily.groupby(daily.index.to_period("M")).agg(H=("H", "max"), L=("L", "min"), C=("C", "last"))
mon_prev = mon.shift(1)


def classic(H, L, C):
    P = (H + L + C) / 3
    return dict(P=P, R1=2 * P - L, S1=2 * P - H, R2=P + (H - L), S2=P - (H - L))


def levels(prev, m):
    H, L, C, O = prev.H, prev.L, prev.C, prev.O
    out = {("Classic日", k): v for k, v in classic(H, L, C).items()}
    P = (H + L + 2 * C) / 4
    out.update({("Woodie", "P"): P, ("Woodie", "R1"): 2 * P - L, ("Woodie", "S1"): 2 * P - H,
                ("Woodie", "R2"): P + (H - L), ("Woodie", "S2"): P - (H - L)})
    X = H + 2 * L + C if C < O else (2 * H + L + C if C > O else H + L + 2 * C)
    out.update({("DeMark", "P"): X / 4, ("DeMark", "R1"): X / 2 - L, ("DeMark", "S1"): X / 2 - H})
    Pc = (H + L + C) / 3
    BC = (H + L) / 2
    out.update({("CPR", "TC"): max(2 * Pc - BC, BC), ("CPR", "BC"): min(2 * Pc - BC, BC)})
    if m is not None and np.isfinite(m.H):
        out.update({("月Pivot", k): v for k, v in classic(m.H, m.L, m.C).items()})
    return out


def touch(H, Lo, C, hrs, L, atr):
    hit = (Lo <= L) & (H >= L) & (hrs < 22)
    hit[0] = False
    if not hit.any():
        return None
    i = int(np.argmax(hit))
    side = np.sign(C[i - 1] - L)
    if side == 0:
        return None
    res = {}
    for k in KS:
        up, dn = L + side * k * atr, L - side * k * atr
        if (Lo[i] <= dn) if side > 0 else (H[i] >= dn):
            res[k] = 0.0
            continue
        hh, ll = H[i + 1:i + 241], Lo[i + 1:i + 241]
        bnc, pen = ((hh >= up), (ll <= dn)) if side > 0 else ((ll <= up), (hh >= dn))
        ib = int(np.argmax(bnc)) if bnc.any() else 10 ** 9
        ip = int(np.argmax(pen)) if pen.any() else 10 ** 9
        res[k] = 0.5 if ib == ip else (1.0 if ib < ip else 0.0)
    return res


rows = []
for a, b in zip(DAYS[:-1], DAYS[1:]):
    atr = ATR.get(b, np.nan)
    if not np.isfinite(atr):
        continue
    g = days[b]
    H, Lo, C = g.high.values, g.low.values, g.close.values
    hrs = g.index.hour.values
    per = pd.Timestamp(b).to_period("M")
    m = mon_prev.loc[per] if per in mon_prev.index else None
    for (fam, nm), L in levels(daily.loc[a], m).items():
        for off in (0.0,) + OFFS:
            r = touch(H, Lo, C, hrs, L + off * atr, atr)
            if r:
                rows.append(dict(day=b, fam=fam, lvl=nm, off=off, **{f"b{k}": v for k, v in r.items()}))
E = pd.DataFrame(rows)
E["year"] = pd.DatetimeIndex(E.day).year
lines = ["第四十六部分（三）：其他 Pivot 算法的反應（真點位反彈率 − 平移假點位；k = 先反彈 / 先穿越的距離，日 ATR 倍數）",
         f"{'點位':16s} {'n':>5s} | " + " | ".join(f"k={k}: 真 / 假 / 差 / 逐年差(23,24,25,26)" for k in KS)]
for fam in ["Classic日", "Woodie", "DeMark", "CPR", "月Pivot"]:
    groups = [(fam + " 全部", E[E.fam == fam])] + [(f"  {fam} {nm}", E[(E.fam == fam) & (E.lvl == nm)]) for nm in sorted(E[E.fam == fam].lvl.unique())]
    for lab, s in groups:
        cells = []
        for k in KS:
            c = f"b{k}"
            r, f = s[s.off == 0][c].mean(), s[s.off != 0][c].mean()
            yd = ",".join(f"{(s[(s.year == y) & (s.off == 0)][c].mean() - s[(s.year == y) & (s.off != 0)][c].mean()) * 100:+.0f}" for y in (2023, 2024, 2025, 2026))
            cells.append(f"{r * 100:4.1f}/{f * 100:4.1f}/{(r - f) * 100:+5.1f} ({yd})")
        lines.append(f"{lab:16s} {(s.off == 0).sum():5d} | " + " | ".join(cells))
print("\n".join(lines))
open("results_part46_pivots.txt", "w").write("\n".join(lines) + "\n")
