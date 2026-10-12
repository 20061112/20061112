"""第二十四部分：把「前日高突破」縮小週期 —— 前一根 D1 / H4 / H1 / M30 / M15 K 棒的高低點突破，多空都測。

規則（每個週期 P 都一樣）：
  - K 棒以 broker 01:00 為起點切（H4 = 01,05,09,…；D1 = 交易日）。
  - 當前這根 P 週期 K 棒開盤在前一根高點下方，期間內第一次碰到前一根高點 → 停損單在高點做多；
    低點對稱做空。
  - 停損 = 點位反側 s × ATR_P，ATR_P = 前 20 根 P 週期 K 棒波幅平均（日線用前 10 日，與第二十三部分相同）；
    停利 2R；最長持有 = max(60 分, min(P, 240 分))，D1 為 240 分。
  - 碰觸那根 M1 若也碰到停損 → 算停損；之後同一根停損停利都碰 → 停損；每筆扣進場那根點差。
  - 對照：點位平移 ±0.15 / ±0.3 ATR_P 的假點位，同規則。
  - 趨勢狀態：前一日收盤在 50 日均線上方 / 下方。
資料：data/XAUUSD_M1_2023_2026.csv（2023/6 ~ 2026/10）。
"""
import numpy as np
import pandas as pd
from load import load

OUT = "results_part24.txt"
d = load("data/XAUUSD_M1_2023_2026.csv")
d["day"] = (d.index - pd.Timedelta(hours=1)).normalize()
cnt = d.groupby("day").size()
d = d[d.day.isin(cnt[cnt >= 600].index)]
H, Lo, C = d.high.values, d.low.values, d.close.values
SP = d.spread.values * 0.01
N = len(d)
dayv = d.day.values
DAYS = np.unique(dayv)
daily = d.groupby("day").agg(H=("high", "max"), L=("low", "min"), C=("close", "last"), O=("open", "first"))
UP50 = (daily.C > daily.C.rolling(50).mean()).shift(1)  # 前一日收盤 vs 50 日均線（當天已知）
UP50 = pd.Series(d.day.map(UP50).values)
mins = ((d.index.hour - 1) * 60 + d.index.minute).values
# 每個交易日最後一根（避免持倉跨日）：用 23:00 前
day_end = pd.Series(np.arange(N)).groupby(dayv).transform(lambda x: x.iloc[-1]).values
lines = []


def p(s=""):
    print(s)
    lines.append(s)


def period_ids(P):
    if P >= 1440:
        return pd.factorize(dayv)[0]
    return pd.factorize(pd.Series(dayv).astype(str) + "_" + pd.Series(mins // P).astype(str))[0]


def simulate(i, entry, dir_, stop, tgt, last):
    """M1 逐根；i = 進場那根。回傳價格損益。"""
    if (Lo[i] <= stop) if dir_ > 0 else (H[i] >= stop):
        return (stop - entry) * dir_
    if last <= i:
        return (C[i] - entry) * dir_
    hh, ll = H[i + 1:last + 1], Lo[i + 1:last + 1]
    if dir_ > 0:
        hs, ht = ll <= stop, hh >= tgt
    else:
        hs, ht = hh >= stop, ll <= tgt
    js = int(np.argmax(hs)) if hs.any() else 10 ** 9
    jt = int(np.argmax(ht)) if ht.any() else 10 ** 9
    if js == 10 ** 9 and jt == 10 ** 9:
        return (C[last] - entry) * dir_
    if js <= jt:
        return (stop - entry) * dir_
    return (tgt - entry) * dir_


def run(P, s=0.2, t=2.0, offs=(0.0,)):
    pid = period_ids(P)
    g = pd.DataFrame({"pid": pid, "H": H, "L": Lo, "O": d.open.values})
    agg = g.groupby("pid").agg(H=("H", "max"), L=("L", "min"), O=("O", "first"))
    rng = agg.H - agg.L
    atrP = rng.rolling(10 if P >= 1440 else 20).mean().shift(1)
    prevH, prevL = agg.H.shift(1), agg.L.shift(1)
    starts = pd.Series(np.arange(N)).groupby(pid).agg(["first", "last"])
    hold = 240 if P >= 1440 else max(60, min(P, 240))
    rows = []
    for k in range(1, len(agg)):
        a = atrP.iat[k]
        if not np.isfinite(a) or a <= 0:
            continue
        i0, i1 = starts["first"].iat[k], starts["last"].iat[k]
        if mins[i0] >= 21 * 60:  # 22 點後不開新單
            continue
        for dir_, base in ((1, prevH.iat[k]), (-1, prevL.iat[k])):
            for off in offs:
                for sg in ((1,) if off == 0 else (1, -1)):
                    L = base + sg * off * a
                    if (agg.O.iat[k] - L) * dir_ >= 0:  # 開盤已在外側
                        continue
                    seg = (Lo[i0:i1 + 1] <= L) & (H[i0:i1 + 1] >= L)
                    seg &= mins[i0:i1 + 1] < 22 * 60
                    if not seg.any():
                        continue
                    i = i0 + int(np.argmax(seg))
                    if i == i0 and i0 > 0 and (C[i0 - 1] - L) * dir_ > 0:
                        continue
                    stop = L - dir_ * s * a
                    risk = s * a
                    last = min(i + hold, day_end[i])
                    pnl = simulate(i, L, dir_, stop, L + dir_ * t * risk, last)
                    rows.append((dayv[i], dir_, off, (pnl - SP[i]) / risk, SP[i] / risk, UP50.iat[i], risk))
    T = pd.DataFrame(rows, columns=["day", "dir", "off", "R", "cost", "up50", "risk"])
    T["year"] = pd.DatetimeIndex(T.day).year
    return T


def st(r):
    r = pd.Series(r)
    if len(r) < 5:
        return f"n={len(r)}"
    gp, gl = r[r > 0].sum(), -r[r < 0].sum()
    return f"n={len(r):6d} {r.mean():+.3f}R t={r.mean() / r.std() * np.sqrt(len(r)):+6.2f} PF {gp / max(gl, 1e-9):.2f}"


def yearly(T):
    return " / ".join(f"{T.R[T.year == y].mean():+.2f}" for y in (2023, 2024, 2025, 2026))


p("第二十四部分：前一根 K 棒高/低點突破，週期縮小（XAUUSD M1 2023/6~2026/10）")
p("停損 0.2 ATR_P，停利 2R；每筆已扣點差；逐年 = 2023 / 2024 / 2025 / 2026 平均 R\n")
PER = [(1440, "D1"), (240, "H4"), (60, "H1"), (30, "M30"), (15, "M15")]
summary = {}
for P, nm in PER:
    T = run(P, offs=(0.0, 0.15, 0.3))
    real, fake = T[T.off == 0], T[T.off != 0]
    summary[nm] = real
    p(f"=== {nm}：每天約 {len(real) / len(DAYS):.1f} 筆，停損中位數 {real.risk.median():.2f} 美元，點差平均 {real.cost.mean():.3f}R ===")
    for dir_, lab in ((1, "做多（突破前高）"), (-1, "做空（跌破前低）")):
        r, f = real[real.dir == dir_], fake[fake.dir == dir_]
        p(f"  {lab}  真 {st(r.R)} | 假點位 {f.R.mean():+.3f}R | 逐年 {yearly(r)}")
        p(f"      50日均線上方 {st(r[r.up50 == True].R)} | 下方 {st(r[r.up50 == False].R)}")
        p(f"      未扣點差 {(r.R + r.cost).mean():+.3f}R")
    both = real
    p(f"  多空合計 {st(both.R)}；順勢版（均線上方只做多、下方只做空）"
      f"{st(both[((both.dir == 1) & (both.up50 == True)) | ((both.dir == -1) & (both.up50 == False))].R)}")
    p()

# 小週期換較寬的停損（點差佔比降低）
p("=== 較小週期改用較寬停損（s = 0.5 / 1.0 ATR_P，停利 2R）===")
for P, nm in PER[2:]:
    for s in (0.5, 1.0):
        T = run(P, s=s)
        for dir_, lab in ((1, "多"), (-1, "空")):
            r = T[T.dir == dir_]
            p(f"  {nm:4s} s={s:<4} {lab} {st(r.R)} 點差 {r.cost.mean():.3f}R | 均線上 {r[r.up50 == True].R.mean():+.3f} 下 {r[r.up50 == False].R.mean():+.3f} | 逐年 {yearly(r)}")
open(OUT, "w").write("\n".join(lines) + "\n")
