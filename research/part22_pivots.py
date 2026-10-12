"""第二十二部分：Pivot 點位的反應 —— 真的比隨便一條線好嗎？

資料：data/XAUUSD_M1_2023_2026.csv（2023/6/1 ~ 2026/10/8，broker 時間 = 紐約 + 7，交易日 01:00~23:59）。
一開始就用完整 3 年多資料、逐年檢查（第十部分的教訓）。

點位（全部只用前一交易日 / 前一週的 H、L、C，當天開盤前就已知）：
  Classic : P=(H+L+C)/3, R1=2P−L, S1=2P−H, R2=P+(H−L), S2=P−(H−L), R3=H+2(P−L), S3=L−2(H−P)
  Fib     : P ± 0.382 / 0.618 / 1.000 × (H−L)
  Camarilla: C ± (H−L) × 1.1/12, 1.1/6, 1.1/4, 1.1/2（H1~H4 / L1~L4）
  Weekly  : 前一週的 Classic P、R1、S1、R2、S2
  參考    : 前日高 PDH、前日低 PDL、前日收 PDC

對照組（安慰劑）：每條真實點位平移 ±0.15 / ±0.3 / ±0.5 ATR 的假點位，套用完全相同的規則。
  如果 Pivot 有「反應」，真點位的反彈率應該 **高於** 附近的假點位。
ATR：前 10 個交易日的日波幅平均（不含當天）。

事件：當天 01:00~22:00 第一次碰到該點位的 M1 K 棒（low ≤ L ≤ high），且上一根收盤在點位某一側。
  上一根收在上方 → 從上面跌下來碰 → 「支撐」→ 反彈 = 往上；反之為壓力。
反應指標（對稱括號，隨機漫步下約 50%）：從 L 起，先到 L + k·ATR（反彈）還是先到 L − k·ATR（穿越），k = 0.05 / 0.10 / 0.20。
  碰觸那根 K 棒若已穿越 k → 算穿越（無法知道先後，保守處理；真假點位一視同仁）；之後同一根兩邊都碰算 0.5；240 分鐘未決算 0.5。
另外：轉折點聚集 —— M1 的 ±30 根擺動高/低點，到最近一條點位的距離分布，看 0 附近有沒有尖峰。
"""
import numpy as np
import pandas as pd
from load import load

KS = (0.05, 0.10, 0.20)
OFFS = (-0.5, -0.3, -0.15, 0.15, 0.3, 0.5)
OUT = "results_part22.txt"

d = load("data/XAUUSD_M1_2023_2026.csv")
d["day"] = (d.index - pd.Timedelta(hours=1)).normalize()
days = {k: g for k, g in d.groupby("day") if len(g) >= 600}
DAYS = sorted(days)
daily = pd.DataFrame({k: dict(H=g.high.max(), L=g.low.min(), C=g.close.iloc[-1], O=g.open.iloc[0]) for k, g in days.items()}).T
daily.index = pd.DatetimeIndex(daily.index)
ATR = (daily.H - daily.L).rolling(10).mean().shift(1)

# 週：以星期一開始的 ISO 週
wk = daily.groupby(daily.index.to_period("W-SUN")).agg(H=("H", "max"), L=("L", "min"), C=("C", "last"))
wk_prev = wk.shift(1)


def classic(H, L, C):
    P = (H + L + C) / 3
    return dict(P=P, R1=2 * P - L, S1=2 * P - H, R2=P + (H - L), S2=P - (H - L), R3=H + 2 * (P - L), S3=L - 2 * (H - P))


def levels_for(prev, w):
    H, L, C = prev.H, prev.L, prev.C
    r = H - L
    out = {}
    for k, v in classic(H, L, C).items():
        out[("Classic", k)] = v
    P = (H + L + C) / 3
    for f, nm in [(0.382, 1), (0.618, 2), (1.0, 3)]:
        out[("Fib", f"R{nm}")] = P + f * r
        out[("Fib", f"S{nm}")] = P - f * r
    for i, f in enumerate([1.1 / 12, 1.1 / 6, 1.1 / 4, 1.1 / 2], 1):
        out[("Camarilla", f"H{i}")] = C + f * r
        out[("Camarilla", f"L{i}")] = C - f * r
    if w is not None and np.isfinite(w.H):
        for k, v in classic(w.H, w.L, w.C).items():
            if k in ("P", "R1", "S1", "R2", "S2"):
                out[("Weekly", k)] = v
    out[("PrevDay", "PDH")] = H
    out[("PrevDay", "PDL")] = L
    out[("PrevDay", "PDC")] = C
    return out


def touch_event(H, Lo, C, hrs, L, atr):
    hit = (Lo <= L) & (H >= L) & (hrs < 22)
    hit[0] = False
    if not hit.any():
        return None
    i = int(np.argmax(hit))
    side = np.sign(C[i - 1] - L)
    if side == 0:
        return None
    res = {"i": i, "side": side}
    n = len(C)
    for k in KS:
        up, dn = L + side * k * atr, L - side * k * atr  # up = 反彈方向, dn = 穿越方向
        # 碰觸那根：已穿越 → 穿越
        pen0 = Lo[i] <= dn if side > 0 else H[i] >= dn
        if pen0:
            res[f"b{k}"] = 0.0
            continue
        j1 = min(i + 241, n)
        hh, ll = H[i + 1:j1], Lo[i + 1:j1]
        if side > 0:
            bnc, pen = hh >= up, ll <= dn
        else:
            bnc, pen = ll <= up, hh >= dn
        ib = int(np.argmax(bnc)) if bnc.any() else 10 ** 9
        ip = int(np.argmax(pen)) if pen.any() else 10 ** 9
        if ib == ip:
            res[f"b{k}"] = 0.5
        else:
            res[f"b{k}"] = 1.0 if ib < ip else 0.0
    for m in (15, 60):
        j = min(i + m, n - 1)
        res[f"fwd{m}"] = side * (C[j] - L) / atr
    res["hour"] = int(hrs[i])
    return res


rows = []
day_idx = {k: i for i, k in enumerate(DAYS)}
for a, b in zip(DAYS[:-1], DAYS[1:]):
    atr = ATR.get(b, np.nan)
    if not np.isfinite(atr):
        continue
    g = days[b]
    H, Lo, C = g.high.values, g.low.values, g.close.values
    hrs = g.index.hour.values
    wper = pd.Timestamp(b).to_period("W-SUN")
    w = wk_prev.loc[wper] if wper in wk_prev.index else None
    lv = levels_for(daily.loc[a], w)
    O = g.open.values[0]
    for (fam, name), L in lv.items():
        for off in (0.0,) + OFFS:
            LL = L + off * atr
            r = touch_event(H, Lo, C, hrs, LL, atr)
            if r is None:
                continue
            rows.append(dict(day=b, fam=fam, lvl=name, off=off, dist_open=abs(LL - O) / atr, **r))

E = pd.DataFrame(rows)
E["year"] = pd.DatetimeIndex(E.day).year
E["real"] = E.off == 0


lines = []


def p(s=""):
    print(s)
    lines.append(s)


def se(x):
    return x.std(ddof=1) / np.sqrt(len(x))


p("第二十二部分：Pivot 點位反應（XAUUSD M1，%s ~ %s，%d 個交易日）" % (DAYS[0].date(), DAYS[-1].date(), len(DAYS)))
p("指標：第一次碰觸後，先反彈 k·ATR 的比例（對稱括號；真點位 vs 平移 ±0.15/0.3/0.5 ATR 的假點位）")
p("差 = 真 − 假（百分點），t 以「天」為群集計算（真假同一天配對）。\n")


def compare(sub, k, label):
    col = f"b{k}"
    real = sub[sub.real]
    fake = sub[~sub.real]
    # 每天配對：真點位當天平均 − 假點位當天平均
    rd = real.groupby("day")[col].mean()
    fd = fake.groupby("day")[col].mean()
    j = rd.index.intersection(fd.index)
    diff = rd[j] - fd[j]
    t = diff.mean() / se(diff) if len(diff) > 2 else np.nan
    return real[col].mean(), fake[col].mean(), len(real), diff.mean(), t


for k in KS:
    p(f"=== k = {k} ATR（2023~2026 平均日波幅約 {(daily.H - daily.L).mean():.0f} 美元 → 約 {k * (daily.H - daily.L).mean():.1f} 美元）===")
    p(f"{'點位':22s} {'n':>5s} {'真':>6s} {'假':>6s} {'差':>6s} {'t':>6s} | 逐年 差（2023 / 2024 / 2025 / 2026）")
    groups = [("全部 Pivot（不含前日HLC）", E[E.fam.isin(["Classic", "Fib", "Camarilla", "Weekly"])])]
    for fam in ["Classic", "Fib", "Camarilla", "Weekly", "PrevDay"]:
        groups.append((fam, E[E.fam == fam]))
    for fam, names in [("Classic", ["P", "R1", "S1", "R2", "S2", "R3", "S3"]),
                       ("Camarilla", ["H3", "L3", "H4", "L4"]),
                       ("Weekly", ["P", "R1", "S1"]),
                       ("PrevDay", ["PDH", "PDL", "PDC"])]:
        for nm in names:
            groups.append((f"  {fam} {nm}", E[(E.fam == fam) & (E.lvl == nm)]))
    for lab, sub in groups:
        r, f, n, dmean, t = compare(sub, k, lab)
        yrs = []
        for y in (2023, 2024, 2025, 2026):
            s = sub[sub.year == y]
            yrs.append(f"{(s[s.real][f'b{k}'].mean() - s[~s.real][f'b{k}'].mean()) * 100:+5.1f}")
        p(f"{lab:22s} {n:5d} {r * 100:5.1f}% {f * 100:5.1f}% {(r - f) * 100:+5.1f} {t:+6.2f} | {' / '.join(yrs)}")
    p()

# 依「離開盤距離」分層的對照（P 通常離開盤很近，會在開盤就被碰到）
p("=== 控制離開盤距離（k=0.10）：只比較同一距離區間內的真/假點位 ===")
piv = E[E.fam.isin(["Classic", "Fib", "Camarilla", "Weekly"])].copy()
piv["dbin"] = pd.cut(piv.dist_open, [0, 0.15, 0.3, 0.5, 0.8, 1.2, 5])
for b, s in piv.groupby("dbin", observed=True):
    r, f = s[s.real]["b0.1"], s[~s.real]["b0.1"]
    p(f"  離開盤 {str(b):12s} 真 n={len(r):5d} {r.mean() * 100:5.1f}%  假 n={len(f):6d} {f.mean() * 100:5.1f}%  差 {(r.mean() - f.mean()) * 100:+5.1f}")
p()

# 支撐 vs 壓力、時段
p("=== 依時段（k=0.10，全部 Pivot）真 − 假 ===")
for a in range(1, 22, 3):
    s = piv[(piv.hour >= a) & (piv.hour < a + 3)]
    r, f = s[s.real]["b0.1"], s[~s.real]["b0.1"]
    p(f"  {a:02d}-{a + 3:02d} 點  真 n={len(r):5d} {r.mean() * 100:5.1f}%  假 {f.mean() * 100:5.1f}%  差 {(r.mean() - f.mean()) * 100:+5.1f}")
p()

p("=== 前瞻報酬（反彈方向為正，ATR，未扣成本）===")
for lab, s in [("全部 Pivot", piv), ("Classic", E[E.fam == "Classic"]), ("Camarilla", E[E.fam == "Camarilla"]), ("PrevDay", E[E.fam == "PrevDay"])]:
    r, f = s[s.real], s[~s.real]
    p(f"  {lab:10s} 15分 真 {r.fwd15.mean():+.4f} 假 {f.fwd15.mean():+.4f} | 60分 真 {r.fwd60.mean():+.4f} (t={r.fwd60.mean() / se(r.fwd60):+.2f}) 假 {f.fwd60.mean():+.4f}")
p()

# ===== 轉折點聚集 =====
p("=== 轉折點聚集：M1 ±30 根擺動高/低點，到最近 Classic 點位的距離（ATR）===")
W = 30
hist_bins = np.arange(-0.30, 0.3001, 0.02)
acc = {o: np.zeros(len(hist_bins) - 1) for o in (0.0, 0.15, 0.3)}
nsw = 0
for a, b in zip(DAYS[:-1], DAYS[1:]):
    atr = ATR.get(b, np.nan)
    if not np.isfinite(atr):
        continue
    g = days[b]
    H, Lo = g.high.values, g.low.values
    hs = pd.Series(H).rolling(2 * W + 1, center=True).max().values == H
    ls = pd.Series(Lo).rolling(2 * W + 1, center=True).min().values == Lo
    pts = np.concatenate([H[hs], Lo[ls]])
    nsw += len(pts)
    lv = np.array(list(classic(daily.loc[a].H, daily.loc[a].L, daily.loc[a].C).values()))
    for o in acc:
        for sgn in ((1,) if o == 0 else (1, -1)):
            L2 = lv + sgn * o * atr
            dd = (pts[:, None] - L2[None, :]) / atr
            near = dd[np.arange(len(pts)), np.abs(dd).argmin(1)]
            acc[o] += np.histogram(near, hist_bins)[0] / (1 if o == 0 else 2)
p(f"  擺動點 {nsw} 個；每格 0.02 ATR 的擺動點數（真點位 / 平移 0.15 / 平移 0.3）")
for i in range(len(hist_bins) - 1):
    c = (hist_bins[i] + hist_bins[i + 1]) / 2
    p(f"  {c:+.2f}  {acc[0.0][i]:7.0f}  {acc[0.15][i]:7.0f}  {acc[0.3][i]:7.0f}")
center = np.abs((hist_bins[:-1] + hist_bins[1:]) / 2) < 0.03
for o in acc:
    p(f"  平移 {o}: |距離|<0.03 ATR 佔 ±0.3 內的 {acc[o][center].sum() / acc[o].sum() * 100:.2f}%（均勻分布為 {center.sum() / len(center) * 100:.2f}%）")

open(OUT, "w").write("\n".join(lines) + "\n")
