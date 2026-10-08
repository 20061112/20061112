"""第十一部分：碰到前日 TPO 價位（POC / VAH / VAL）做反轉單；30 分 vs 5 分 TPO。

價位：前一交易日剖面的 POC、VAH、VAL；TPO 字母期分別用 30 分與 5 分。
事件：當天（broker 01:00~22:00）第一次碰到該價位的 M1 K 棒（low ≤ L ≤ high），
      方向 = 上一根收盤在價位上方 → 做多（支撐），在下方 → 做空（壓力）；限價在 L 成交。
結果：
  fwd15/30/60：進場後 N 分鐘收盤相對 L 的報酬（ATR，+ = 反轉方向），扣一個點差。
  括號單：停損 0.1 ATR、停利 0.2 ATR（約 10 / 20 美元），同根都碰算停損；4 小時沒結束以收盤計。
對照組：同樣規則套在「價位 ± 0.25 / 0.5 ATR」的假價位上，看真的 TPO 價位有沒有比較好。
"""
import numpy as np, pandas as pd
from tpo import load_m1, tpo_profile, poc_va, TICK

d = load_m1()
d["sp"] = d.spread * 0.01
SPLIT = pd.Timestamp("2026-06-01")
days = {k: g for k, g in d.groupby("day") if len(g) >= 1000}
DAYS = sorted(days)


def levels(period_min):
    out = {}
    for k in DAYS:
        g = days[k].copy()
        mins = (g.index.hour - 1) * 60 + g.index.minute
        g["period"] = mins // period_min
        pr, c, _ = tpo_profile(g)
        poc, vah, val = poc_va(pr, c)
        out[k] = dict(POC=poc + TICK / 2, VAH=vah + TICK, VAL=val)
    return pd.DataFrame(out).T


L30, L5 = levels(30), levels(5)
rng = pd.Series({k: days[k].high.max() - days[k].low.min() for k in DAYS})
tr = pd.concat([rng, (pd.Series({k: days[k].close.iloc[-1] for k in DAYS}))], axis=1)
ATR = rng.rolling(10).mean().shift(1)


def event(g, L, atr, stop_k=0.1, tgt_k=0.2):
    H, Lo, C, S = g.high.values, g.low.values, g.close.values, g.sp.values
    hrs = g.index.hour
    hit = np.where((Lo <= L) & (H >= L) & (hrs < 22))[0]
    hit = hit[hit > 0]
    if not len(hit):
        return None
    i = hit[0]
    side = np.sign(C[i - 1] - L)
    if side == 0:
        return None
    r = {}
    for n in (15, 30, 60):
        j = min(i + n, len(C) - 1)
        r[f"fwd{n}"] = (side * (C[j] - L) - S[i]) / atr
    stop, tgt = L - side * stop_k * atr, L + side * tgt_k * atr
    res = np.nan
    for j in range(i, min(i + 240, len(C))):
        hs = Lo[j] <= stop if side > 0 else H[j] >= stop
        ht = H[j] >= tgt if side > 0 else Lo[j] <= tgt
        if hs:
            res = -stop_k; break
        if ht and j > i:
            res = tgt_k; break
    else:
        res = side * (C[min(i + 239, len(C) - 1)] - L) / atr
    r["bracket"] = res - S[i] / atr
    r["win"] = float(res >= tgt_k - 1e-9)
    r["hour"] = g.index[i].hour
    return r


def run(LV, offsets=(0,)):
    rows = []
    for a, b in zip(DAYS[:-1], DAYS[1:]):
        atr = ATR[b]
        if not np.isfinite(atr):
            continue
        for name in ["POC", "VAH", "VAL"]:
            for off in offsets:
                for sgn in ([1] if off == 0 else [1, -1]):
                    r = event(days[b], LV.loc[a, name] + sgn * off * atr, atr)
                    if r:
                        rows.append(dict(day=b, lvl=name, off=off, **r))
    return pd.DataFrame(rows).set_index("day")


def show(E, lab):
    t = lambda x: x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    x = E.bracket
    print(f"  {lab:22s} n={len(E):4d}  fwd15 {E.fwd15.mean():+.3f}  fwd30 {E.fwd30.mean():+.3f}  fwd60 {E.fwd60.mean():+.3f} (t={t(E.fwd60):+.2f})"
          f" | 括號單 勝率 {E.win.mean() * 100:.0f}%（損益兩平 33%）每筆 {x.mean():+.3f} ATR t={t(x):+.2f}"
          f"  H1 {x[x.index < SPLIT].mean():+.3f} H2 {x[x.index >= SPLIT].mean():+.3f}")


print("前日 TPO 價位：30 分 vs 5 分剖面差異（美元，中位數）")
for c in ["POC", "VAH", "VAL"]:
    print(f"  {c}: |30分 − 5分| 中位數 {(L30[c] - L5[c]).abs().median():.1f}，90% 分位 {(L30[c] - L5[c]).abs().quantile(.9):.1f}")
print("  VA 寬度中位數：30 分 %.1f，5 分 %.1f" % ((L30.VAH - L30.VAL).median(), (L5.VAH - L5.VAL).median()))

for lab, LV in [("30 分 TPO", L30), ("5 分 TPO", L5)]:
    print(f"\n=== {lab}：碰到前日價位做反轉 ===")
    E = run(LV, offsets=(0, 0.25, 0.5))
    E.to_csv(f"part11_events_{'30' if LV is L30 else '5'}.csv")
    for name in ["POC", "VAH", "VAL"]:
        show(E[(E.lvl == name) & (E.off == 0)], f"{name}")
    show(E[E.off == 0], "三個價位合計")
    show(E[E.off == 0.25], "對照：價位±0.25ATR")
    show(E[E.off == 0.5], "對照：價位±0.5ATR")
    e0 = E[E.off == 0]
    print("  依時段（三個價位合計，括號單每筆 ATR）：",
          {f"{a:02d}-{a + 3:02d}": round(e0[(e0.hour >= a) & (e0.hour < a + 4)].bracket.mean(), 3) for a in range(1, 22, 4)})
