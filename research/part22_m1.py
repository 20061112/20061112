"""第二十二部分（續 2）：
A. M1 短窗口是不是該做逆勢？—— 短陡（單獨 / 共振 / 逆長）在 M1 短窗口下，順 vs 逆。
B. M1 用 H1 等長窗口（480 / 3840 根）時，交易能不能變多？—— 縮短持有、訊號一出現就進（可疊單）、持續期間每小時加碼。
損益以 bp（萬分之一，pnl / 進場價 × 1e4）表示，不同年份金價不同也能比；成本 = 當根點差。
疊單版本以「日」為單位彙總算 t（同一天的單彼此相關）。
"""
import numpy as np
import pandas as pd
from load import load
from part22_mtf import prep, rank_abs


def ev(P, mask, sign, h, dedup=True, stride=None):
    idx = np.flatnonzero(mask)
    idx = idx[idx + h < len(P["c"])]
    if dedup:
        keep, last = [], -10 ** 9
        for p in idx:
            if p - last >= (stride or h):
                keep.append(p)
                last = p
        idx = np.array(keep, int)
    c = P["c"]
    s = sign[idx]
    g = s * (c[idx + h] - c[idx]) / c[idx] * 1e4
    cost = P["spr"][idx] / c[idx] * 1e4
    return idx, g, cost


def line(P, label, idx, g, cost):
    net = g - cost
    if len(net) < 20:
        print(f"  {label:<44s} 筆數 {len(net)}（太少）")
        return
    day = pd.Series(net, index=P["time"][idx].normalize()).groupby(level=0).sum()
    yr = P["year"][idx]
    weeks = (P["time"][-1] - P["time"][0]).days / 7
    ys = " ".join(f"{y % 100}:{net[yr == y].mean():+.1f}" for y in np.unique(yr))
    print(f"  {label:<44s} 筆數 {len(net):6d} ({len(net) / weeks:5.1f}/週) 毛 {g.mean():+6.2f} 成本 {cost.mean():5.2f} 淨 {net.mean():+6.2f}bp "
          f"t(日) {day.mean() / day.std() * np.sqrt(len(day)):+5.2f} 勝率 {(net > 0).mean():.2f} 總淨 {net.sum():+8.0f}bp | {ys}")


if __name__ == "__main__":
    m1 = load("data/XAUUSD_M1_2023_2026.csv")
    P = prep(m1, 1320)
    P["spr"] = m1.spread.to_numpy() * 0.01
    P["time"] = m1.index

    print("A. M1 短窗口：順勢 vs 逆勢（短窗口 = 長度 n，持有 = n 根；長窗口 = 8n）")
    for ns in (5, 15, 30, 60):
        rs, ss = rank_abs(P, ns)
        rl, sl = rank_abs(P, ns * 8)
        print(f" 短 {ns} 分 / 長 {ns * 8} 分")
        for th in (.8, .95):
            m = rs >= th
            line(P, f"短陡≥{th} 單獨 順", *ev(P, m, ss, ns))
            line(P, f"短陡≥{th} 單獨 逆", *ev(P, m, -ss, ns))
        co = (rs >= .8) & (rl >= .8) & (ss == sl)
        line(P, "共振 順", *ev(P, co, ss, ns))
        line(P, "共振 逆", *ev(P, co, -ss, ns))
        op = (rs >= .8) & (rl >= .6) & (ss != sl)
        line(P, "短陡逆長 順短", *ev(P, op, ss, ns))
        line(P, "短陡逆長 逆短（=順長）", *ev(P, op, -ss, ns))
    print(" 短陡逆 H1 等長大週期（長 3840 分 ≈ 64 小時）：M1 短線急動、但跟大方向相反 → 逆短")
    rL, sL = rank_abs(P, 3840)
    for ns in (15, 30, 60):
        rs, ss = rank_abs(P, ns)
        m = (rs >= .9) & (rL >= .6) & (ss != sL)
        line(P, f"短 {ns} 分 ≥0.9 逆大週期 → 逆短", *ev(P, m, -ss, ns))
        line(P, f"短 {ns} 分 ≥0.9 逆大週期 → 逆短 持 4n", *ev(P, m, -ss, 4 * ns))

    print("\nB. M1 480 / 3840（= H1 8/64）共振，交易次數怎麼變")
    rs, ss = rank_abs(P, 480)
    rl, sl = rank_abs(P, 3840)
    co = (rs >= .8) & (rl >= .8) & (ss == sl)
    onset = co & ~np.r_[False, co[:-1]]
    print(f"  訊號成立的分鐘佔全部 {co.mean():.3%}；每週平均 {onset.sum() / ((P['time'][-1] - P['time'][0]).days / 7):.1f} 次「新出現」")
    line(P, "原版：持有 480 分、不疊單", *ev(P, co, ss, 480))
    for h in (240, 120, 60, 30):
        line(P, f"持有 {h} 分、不疊單（訊號還在就再進）", *ev(P, co, ss, h))
    line(P, "訊號新出現就進、持 480、可疊單", *ev(P, onset, ss, 480, dedup=False))
    for st in (60, 120):
        line(P, f"訊號持續期間每 {st} 分加 1 筆、持 480", *ev(P, co, ss, 480, stride=st))
    print("  放寬門檻（持有 480、不疊單）")
    for a in (.6, .7):
        m = (rs >= a) & (rl >= a) & (ss == sl)
        line(P, f"短≥{a} 長≥{a}", *ev(P, m, ss, 480))
    line(P, "基準：每 480 分順長角度方向", *ev(P, ~np.isnan(rl), sl, 480))
    line(P, "基準：每 480 分做多", *ev(P, ~np.isnan(rl), np.ones(len(sl)), 480))
