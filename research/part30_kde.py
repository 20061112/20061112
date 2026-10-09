"""第三十部分：插針 KDE 支撐壓力，碰到時會不會反轉。

資料：data/XAUUSD_M1_2026.csv（2026/1/2 ~ 10/8），重採樣。
- 插針在 pin TF（M15 或 H1）上找，KDE 求價位（kde_levels.py）。
- 價位模式：
    rolling N 天：每天開盤前用過去 N 個日曆天的插針重算（N = 5/10/20/60）
    expanding：用 1/2 至前一天的全部插針
    fixed：只用 1/2~5/31 的插針算一次，固定套用到 6/1~10/8
- 碰觸（M5）：價位在下方且收盤曾離開 >= arm × ATR → 之後 low <= L + tol × ATR 算一次「支撐碰觸」，
  壓力鏡像；碰觸後要再離開 arm × ATR 才重新計算（價位突破後可轉為反向角色）。
- 結果（ATR = M5 ATR14，參考價 = L ± tol·ATR）：
    race k：先往反彈方向走 k ATR（high/low）還是先往突破方向走 k ATR，碰觸棒本身若已突破算突破
    fwd n：n 根後收盤相對參考價（往反彈方向，ATR）
    交易：限價於參考價進場、停損 1 ATR、停利 1R / 2R、最多 24 根，扣點差
- 對照組（placebo）：同一組價位整體平移 ±(3~6)h 的隨機距離，重複 20 次，同樣規則統計。
  → 若 KDE 價位真的有支撐壓力，反彈率應明顯高於 placebo。
"""
import numpy as np
import pandas as pd
from load import load
from td import resample, atr
from kde_levels import find_pins, levels_from_pins

TOL, ARM, H = 0.1, 1.0, 24
SPLIT = pd.Timestamp("2026-06-01")
RNG = np.random.default_rng(0)


def touches(m5, A, day_slices, day_levels):
    """day_levels: dict day -> DataFrame(price, strength)。回傳碰觸事件 DataFrame。"""
    h, l, c = (m5[k].to_numpy(float) for k in ("high", "low", "close"))
    ev = []
    for d, (s, e) in day_slices.items():
        lv = day_levels.get(d)
        if lv is None or len(lv) == 0:
            continue
        for L, st in zip(lv.price.to_numpy(), lv.strength.to_numpy()):
            armed = 0  # +1 支撐待碰、-1 壓力待碰
            for t in range(s, e):
                a = A[t - 1] if t > 0 else np.nan
                if not a > 0:
                    continue
                if armed == 1 and l[t] <= L + TOL * a:
                    ev.append((t, 1, L, st, a))
                    armed = 0
                elif armed == -1 and h[t] >= L - TOL * a:
                    ev.append((t, -1, L, st, a))
                    armed = 0
                if c[t] >= L + ARM * a:
                    armed = 1
                elif c[t] <= L - ARM * a:
                    armed = -1
    return pd.DataFrame(ev, columns=["t", "dir", "L", "strength", "atr"])


def outcomes(m5, ev):
    h, l, c = (m5[k].to_numpy(float) for k in ("high", "low", "close"))
    spr = m5.spread.to_numpy(float) * 0.01
    n = len(c)
    out = {k: [] for k in ("race1", "race2", "f3", "f12", "f24", "tp1", "tp2")}
    for t, d, L, a in zip(ev.t.to_numpy(), ev.dir.to_numpy(), ev.L.to_numpy(), ev.atr.to_numpy()):
        ref = L + d * TOL * a
        end = min(t + H, n - 1)
        hh, ll = h[t:end + 1], l[t:end + 1]
        fav = (hh - ref) / a if d == 1 else (ref - ll) / a
        adv = (ref - ll) / a if d == 1 else (hh - ref) / a
        fav[0] = -np.inf  # 碰觸棒的高點可能在碰觸之前，不算反彈；碰觸棒的突破照算（保守）
        for k, key in ((1, "race1"), (2, "race2")):
            fb = np.flatnonzero(adv >= k)
            fg = np.flatnonzero(fav >= k)
            ib = fb[0] if len(fb) else 10 ** 9
            ig = fg[0] if len(fg) else 10 ** 9
            out[key].append(np.nan if ib == ig == 10 ** 9 else float(ig < ib))
        for k, key in ((3, "f3"), (12, "f12"), (24, "f24")):
            out[key].append((c[t + k] - ref) * d / a if t + k < n else np.nan)
        # 交易：碰觸棒以參考價成交，停損 1 ATR，停利 tp×1 ATR，同棒停損優先
        for tp, key in ((1, "tp1"), (2, "tp2")):
            r = np.nan
            for j in range(len(adv)):
                if adv[j] >= 1:
                    r = -1.0
                    break
                if fav[j] >= tp:
                    r = float(tp)
                    break
            if np.isnan(r):
                r = (c[end] - ref) * d / a
            out[key].append(r - spr[t] / a)
    for k, v in out.items():
        ev[k] = v
    return ev


def summarize(ev):
    r = dict(n=len(ev))
    for k in ("race1", "race2", "f3", "f12", "f24", "tp1", "tp2"):
        v = ev[k].dropna()
        r[k] = v.mean()
        if k.startswith(("f", "tp")):
            r[k + "_t"] = v.mean() / v.std(ddof=1) * np.sqrt(len(v)) if len(v) > 2 else np.nan
    return r


def build_levels(pins, days, mode, N=None, bw=0.5, side_split=False, **kw):
    lv = {}
    if mode == "fixed":
        base = levels_from_pins(pins[pins.index < SPLIT], bw=bw, **kw)
    for d in days:
        if mode == "fixed":
            if d < SPLIT:
                continue
            lv[d] = base
            continue
        if mode == "rolling":
            p = pins[(pins.index >= d - pd.Timedelta(days=N)) & (pins.index < d)]
        else:
            p = pins[pins.index < d]
        lv[d] = levels_from_pins(p, bw=bw, **kw)
    return lv


def shift_levels(lv, lo=3, hi=6):
    out = {}
    for d, df in lv.items():
        if len(df) == 0:
            out[d] = df
            continue
        hsz = df.h.iloc[0]
        off = RNG.uniform(lo, hi) * hsz * RNG.choice([-1, 1])
        x = df.copy()
        x["price"] = x.price + off
        out[d] = x
    return out


def run_config(m5, A5, slices, pins, mode, N, bw, n_placebo=20, **kw):
    days = list(slices)
    lv = build_levels(pins, days, mode, N, bw, **kw)
    ev = outcomes(m5, touches(m5, A5, slices, lv))
    ev["half"] = np.where(m5.index[ev.t] < SPLIT, 1, 2)
    pl = []
    for _ in range(n_placebo):
        pl.append(outcomes(m5, touches(m5, A5, slices, shift_levels(lv))))
    pl = pd.concat(pl)
    pl["half"] = np.where(m5.index[pl.t] < SPLIT, 1, 2)
    return ev, pl, lv


def main():
    m1 = load("data/XAUUSD_M1_2026.csv")
    m5 = resample(m1, "5min")
    A5 = atr(m5)
    day = m5.index.normalize()
    slices = {}
    for d in day.unique():
        idx = np.flatnonzero(day == d)
        slices[d] = (idx[0], idx[-1] + 1)
    # 第一個月當暖身，從 2/2 起統計（fixed 從 6/1 起）
    slices = {d: v for d, v in slices.items() if d >= pd.Timestamp("2026-02-02")}

    lines = []
    P = lines.append
    rows = []
    configs = []
    for ptf in ("15min", "1h"):
        for mode, N in (("rolling", 5), ("rolling", 10), ("rolling", 20), ("rolling", 60),
                        ("expanding", None), ("fixed", None)):
            configs.append((ptf, mode, N, 0.5))
    configs += [("15min", "rolling", 20, 0.25), ("15min", "rolling", 20, 1.0)]
    pin_cache = {}
    all_ev = {}
    for ptf, mode, N, bw in configs:
        if ptf not in pin_cache:
            d = resample(m1, ptf)
            pin_cache[ptf] = find_pins(d, atr(d))
        pins = pin_cache[ptf]
        ev, pl, lv = run_config(m5, A5, slices, pins, mode, N, bw)
        name = f"{ptf} {mode}{'' if N is None else N} bw{bw}"
        nlv = np.mean([len(x) for x in lv.values()])
        all_ev[name] = (ev, pl)
        for grp, e, p in (("all", ev, pl),
                          ("H1", ev[ev.half == 1], pl[pl.half == 1]),
                          ("H2", ev[ev.half == 2], pl[pl.half == 2])):
            s, b = summarize(e), summarize(p)
            rows.append(dict(cfg=name, grp=grp, levels=nlv, n=s["n"], placebo_n=b["n"] / 20,
                             race1=s["race1"], race1_pl=b["race1"], race2=s["race2"], race2_pl=b["race2"],
                             f12=s["f12"], f12_pl=b["f12"], tp1=s["tp1"], tp1_pl=b["tp1"],
                             tp2=s["tp2"], tp2_pl=b["tp2"], tp2_t=s["tp2_t"]))
        print(name, "done", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv("part30_summary.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    P("=== KDE 插針價位 vs placebo（平移價位），M5 碰觸 ===")
    P(res.round(3).to_string())

    # 強度分組、第幾次碰觸、方向（支撐/壓力）— 用 M15 rolling20 bw0.5
    key = "15min rolling20 bw0.5"
    ev, pl = all_ev[key]
    P(f"\n=== {key}：依 KDE 強度分組 ===")
    for nm, e in (("KDE", ev), ("placebo", pl)):
        q = pd.qcut(e.strength, 4, labels=["Q1", "Q2", "Q3", "Q4"])
        P(nm)
        P(e.groupby(q, observed=True)[["race1", "race2", "f12", "tp1", "tp2"]].agg(["mean"]).round(3).to_string()
          + "\n" + e.groupby(q, observed=True).size().to_string())
    P(f"\n=== {key}：支撐 (dir=+1) vs 壓力 (dir=-1) ===")
    for nm, e in (("KDE", ev), ("placebo", pl)):
        P(nm + "\n" + e.groupby("dir")[["race1", "race2", "f12", "tp1", "tp2"]].mean().round(3).to_string())
    open("results_part30.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
