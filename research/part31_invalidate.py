"""第三十一部分：碰到沒反應的價位要不要剔除（價位失效規則）。

延續第三十部分（part30_kde.py）的價位與碰觸定義。
「沒反應」= 碰觸後沒有先反彈 1 ATR（先突破 1 ATR，或 24 根內都沒走到 1 ATR）。
結果在 res_t（揭曉的那根）之後才能使用，價位狀態以價格距離 <= h 的範圍認定
（滾動模式每天重算的價位會略有移動，所以用價格範圍而不是同一條線）。

規則（依時間順序逐筆處理，所有碰觸、包括被略過的都會更新狀態）：
  base      全部都做
  kill      附近曾經失效過一次 → 永久剔除
  kill20    附近 20 天內失效過 → 剔除（20 天後恢復）
  kill2     附近累計失效 2 次 → 永久剔除
  last_ok   附近「最近一次碰觸」不是失效才做（失效後再被碰且有反彈 → 恢復）
  proven    附近「最近一次碰觸」有反彈才做（第一次碰觸不做）
placebo 用同樣規則。
"""
import numpy as np
import pandas as pd
from load import load
from td import resample, atr
from kde_levels import find_pins
from part30_kde import (touches, outcomes, summarize, build_levels, shift_levels, SPLIT)

RULES = ["base", "kill", "kill20", "kill2", "last_ok", "proven"]
DAY5 = 288  # M5 一天的根數


def apply_rules(ev, hsz):
    """ev 依 t 排序，hsz：每筆事件的頻寬 h。回傳 dict rule -> bool mask。"""
    ev = ev.sort_values("t").reset_index(drop=True)
    t = ev.t.to_numpy()
    L = ev.L.to_numpy()
    rt = ev.res_t.to_numpy()
    fail = ~(ev.race1.to_numpy() == 1)
    h = hsz
    masks = {r: np.zeros(len(ev), bool) for r in RULES}
    for i in range(len(ev)):
        prev = np.flatnonzero((rt[:i] < t[i]) & (np.abs(L[:i] - L[i]) <= h[i]))
        if len(prev) == 0:
            masks["base"][i] = masks["kill"][i] = masks["kill20"][i] = masks["kill2"][i] = True
            masks["last_ok"][i] = True
            continue
        pf = prev[fail[prev]]
        last = prev[np.argmax(rt[prev])]
        masks["base"][i] = True
        masks["kill"][i] = len(pf) == 0
        masks["kill20"][i] = not np.any(t[i] - rt[pf] <= 20 * DAY5)
        masks["kill2"][i] = len(pf) < 2
        masks["last_ok"][i] = not fail[last]
        masks["proven"][i] = not fail[last]
    return ev, masks


def add_h(ev, lv, m5):
    days = m5.index[ev.t].normalize()
    return np.array([lv[d].h.iloc[0] if d in lv and len(lv[d]) else np.nan for d in days])


def main():
    m1 = load("data/XAUUSD_M1_2026.csv")
    m5 = resample(m1, "5min")
    A5 = atr(m5)
    day = m5.index.normalize()
    slices = {}
    for d in day.unique():
        idx = np.flatnonzero(day == d)
        slices[d] = (idx[0], idx[-1] + 1)
    slices = {d: v for d, v in slices.items() if d >= pd.Timestamp("2026-02-02")}
    rng = np.random.default_rng(1)
    import part30_kde
    part30_kde.RNG = rng

    rows = []
    for ptf in ("15min", "1h"):
        dd = resample(m1, ptf)
        pins = find_pins(dd, atr(dd))
        for mode, N in (("rolling", 20), ("rolling", 60), ("expanding", None), ("fixed", None)):
            name = f"{ptf} {mode}{'' if N is None else N}"
            lv = build_levels(pins, list(slices), mode, N, 0.5)
            sets = [("KDE", lv)] + [("placebo", shift_levels(lv)) for _ in range(20)]
            acc = {}
            for kind, L in sets:
                ev = outcomes(m5, touches(m5, A5, slices, L))
                if len(ev) == 0:
                    continue
                ev, masks = apply_rules(ev, add_h(ev, L, m5))
                half = np.where(m5.index[ev.t] < SPLIT, 1, 2)
                for r, m in masks.items():
                    for grp, g in (("all", m), ("H1", m & (half == 1)), ("H2", m & (half == 2))):
                        acc.setdefault((kind, r, grp), []).append(ev[g])
            for r in RULES:
                for grp in ("all", "H1", "H2"):
                    k = acc.get(("KDE", r, grp), [pd.DataFrame()])
                    e = pd.concat(k)
                    p = pd.concat(acc.get(("placebo", r, grp), [pd.DataFrame()]))
                    if len(e) < 5:
                        continue
                    s, b = summarize(e), summarize(p)
                    rows.append(dict(cfg=name, rule=r, grp=grp, n=s["n"], placebo_n=b["n"] / 20,
                                     race1=s["race1"], race1_pl=b["race1"], race2=s["race2"],
                                     race2_pl=b["race2"], tp1=s["tp1"], tp1_pl=b["tp1"],
                                     tp2=s["tp2"], tp2_pl=b["tp2"], tp2_t=s["tp2_t"]))
            print(name, "done", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv("part31_summary.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    txt = "=== 價位失效規則：KDE vs placebo（M5 碰觸） ===\n" + res.round(3).to_string()
    # 失效之後的碰觸：被 kill 剔除掉的那些到底表現如何
    open("results_part31.txt", "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
