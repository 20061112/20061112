"""第二十二部分補充：用 M1 逐根重跑 ZigZag 事件的交易模擬（2023/6 起才有 M1）。

M15 / H1 K 棒內不知道先碰停損還是先碰價位，part22_zigzag.py 的掛單版（A0/C0）用最保守假設，
37% 的單在事件 K 棒就被算停損。這裡改用 M1 找出實際碰價位的那一分鐘再往後模擬。
停損 / 停利距離用事件前一根的 ATR（掛單時已知）；同一根 M1 同時碰停損與停利算停損。
"""
import numpy as np
import pandas as pd
from load import load
import part22_zigzag as z

M1 = load("data/XAUUSD_M1_2023_2026.csv")
M1 = M1[~M1.index.duplicated()]
T1 = M1.index.values
O1, H1_, L1, C1, SP1 = (M1[x].values for x in ("open", "high", "low", "close", "spread"))
SP1 = SP1 * 0.01


def first_hit(cond):
    if len(cond) == 0:
        return -1
    k = np.argmax(cond)
    return k if cond[k] else -1


def sim_m1(d, E, tf_min, at_level):
    out = np.full(len(E), np.nan)
    a_prev = d.atr.shift().values
    idx = d.index
    for k, (t, dr, lvl, th) in enumerate(zip(E.i.values, E.dir.values, E.level.values, E.through.values)):
        if dr == 0 or t + 1 >= len(d) or idx[t] < M1.index[0]:
            continue
        A = a_prev[t]
        bar_start = idx[t]
        bar_end = bar_start + pd.Timedelta(minutes=tf_min)
        stop_time = bar_start + pd.Timedelta(minutes=tf_min * (z.MAX_HOLD + 1))
        s0 = np.searchsorted(T1, np.datetime64(bar_start))
        s1 = np.searchsorted(T1, np.datetime64(bar_end))
        s2 = np.searchsorted(T1, np.datetime64(stop_time))
        if at_level:
            touch = H1_[s0:s1] >= lvl if th == 1 else L1[s0:s1] <= lvl
            j = first_hit(touch)
            if j < 0:
                continue  # M1 與 M15 資料不一致，略過
            j += s0
            e = max(lvl, O1[j]) if th == 1 else min(lvl, O1[j])
            sp = SP1[j]
        else:
            j = s1 - 1           # 事件 K 棒最後一分鐘收盤進場
            e = d.close.values[t]
            sp = d.sp.values[t]
        st = e - dr * z.STOP_ATR * A
        tg = e + dr * z.TARGET_R * z.STOP_ATR * A
        hh, ll = H1_[j:s2].copy(), L1[j:s2].copy()
        if not at_level:
            hh, ll = hh[1:], ll[1:]
        if len(hh) == 0:
            continue
        hit_s = ll <= st if dr == 1 else hh >= st
        hit_t = hh >= tg if dr == 1 else ll <= tg
        if at_level:
            hit_t[0] = False  # 進場那一分鐘不算停利（不知道先後）
        ks, kt = first_hit(hit_s), first_hit(hit_t)
        if ks >= 0 and (kt < 0 or ks <= kt):
            r = -1.0
        elif kt >= 0:
            r = z.TARGET_R
        else:
            r = dr * (C1[s2 - 1] - e) / (z.STOP_ATR * A)
        out[k] = r - sp / (z.STOP_ATR * A)
    return out


def line(name, R, years):
    m = ~np.isnan(R)
    r, y = R[m], years[m]
    gp, gl = r[r > 0].sum(), -r[r < 0].sum()
    yr = "  ".join(f"{yy}:{r[y == yy].mean():+.2f}({(y == yy).sum()})" for yy in sorted(set(y)))
    print(f"  {name:6s} n={len(r):4d} 期望 {r.mean():+.3f}R t={r.mean() / r.std() * np.sqrt(len(r)):+.2f} "
          f"勝率 {(r > 0).mean():.2f} PF {gp / gl:.2f} | {yr}")


if __name__ == "__main__":
    for tf, N, tf_min in (("M15", 192, 15), ("H1", 120, 60)):
        d, P, E, dr = z.run(tf, 3, N, verbose=False)
        E = E[E.time >= M1.index[0]]
        print(f"\n### {tf} k=3 N={N}（2023/6 ~ 2026/10，M1 逐根）")
        for s in "ABCD":
            S = E[E.setup == s]
            line(s, sim_m1(d, S, tf_min, False), S.year.values)
        E0 = E.copy()
        E0["dir"] = np.where(E0.ext_lvl | E0.pb_lvl, E0.trend, 0)
        for s, m in (("A0", E0.ext_lvl), ("C0", E0.pb_lvl)):
            S = E0[m]
            line(s, sim_m1(d, S, tf_min, True), S.year.values)
        # 逆勢掛單版：碰 HH/LL 直接反向（限價做頂底），碰 HL/LH 直接追破（停損單）
        E1 = E.copy()
        E1["dir"] = np.where(E1.ext_lvl | E1.pb_lvl, -E1.trend, 0)
        for s, m in (("B0", E1.ext_lvl), ("D0", E1.pb_lvl)):
            S = E1[m]
            line(s, sim_m1(d, S, tf_min, True), S.year.values)
        # 多空拆開（排除黃金長期上漲的影響）
        for s, sg in (("A0", 1), ("B0", -1)):
            S = E[E.ext_lvl].copy()
            S["dir"] = sg * S.trend
            R = sim_m1(d, S, tf_min, True)
            for dd, nm in ((1, "多"), (-1, "空")):
                m = (S.dir.values == dd) & ~np.isnan(R)
                print(f"    {s} 只做{nm}: n={m.sum()} 期望 {R[m].mean():+.3f}R")
        # 收斂結構（LH+HL）收盤突破就跟
        G = E[(E.trend == 0) & (E.lastH == "LH") & (E.lastL == "HL") & E.beyond].copy()
        G["dir"] = G.through
        line("收斂破", sim_m1(d, G, tf_min, False), G.year.values)
