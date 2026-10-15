"""第二十二部分 A：前半天的 Hawkes ρ 能不能分辨「來回盤」與「單邊盤」？

可推翻的預測：前半天（broker 01:00~13:00）估出的 ρ 越高 → 當天越像窄幅來回日。
  1. 同時段檢查：前半天 ρ 和前半天自己的 ER 應該負相關（否則 ρ 根本沒量到「來回」）。
  2. 預測：前半天 ρ 和後半天（13:00~23:59）的 ER / 變異數比 應該負相關，
     而且要比「前半天 ER」這種免費指標多提供資訊（偏相關）。
  3. 分類：全天 ER 前 1/3（單邊日）vs 後 1/3（來回日），前半天 ρ 的 AUC，對照前半天 ER 的 AUC。
資料：XAUUSD M1 2023/6/1 ~ 2026/10/8（860 個交易日）。
"""
import numpy as np
import pandas as pd
from load import load
from hawkes import trading_clock, day_scale, make_events, batch_em_days, online_em, to_bars

CUT = 720.0                         # 日內分鐘 720 = broker 13:00
KINDS = ["step_1", "step_2", "step_4", "bar_1", "bar_2"]
BETAS = [0.05, 0.1, 0.3]


def spearman(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    return pd.Series(x[ok]).rank().corr(pd.Series(y[ok]).rank()), ok.sum()


def partial_spearman(x, y, controls):
    ok = np.isfinite(x) & np.isfinite(y)
    for c in controls:
        ok &= np.isfinite(c)
    R = lambda v: pd.Series(v[ok]).rank().values
    Z = np.column_stack([np.ones(ok.sum())] + [R(c) for c in controls])
    rx = R(x) - Z @ np.linalg.lstsq(Z, R(x), rcond=None)[0]
    ry = R(y) - Z @ np.linalg.lstsq(Z, R(y), rcond=None)[0]
    return np.corrcoef(rx, ry)[0, 1]


def auc(score, label):
    ok = np.isfinite(score) & (label >= 0)
    s, l = score[ok], label[ok]
    r = pd.Series(s).rank().values
    n1 = (l == 1).sum(); n0 = (l == 0).sum()
    return (r[l == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def day_targets(m, day_id, minute, nd):
    c = m.close.values; o = m.open.values; h = m.high.values; l = m.low.values
    D = pd.DataFrame({"d": day_id, "mi": minute, "c": c, "o": o, "h": h, "l": l})
    D["dc"] = D.groupby("d").c.diff().abs()
    out = {}
    for name, sub in (("A", D[D.mi < CUT]), ("B", D[D.mi >= CUT]), ("day", D)):
        g = sub.groupby("d")
        path = g.dc.sum()
        first_o = g.o.first(); last_c = g.c.last()
        out[f"er_{name}"] = ((last_c - first_o).abs() / path).reindex(range(nd)).values
        out[f"ret_{name}"] = (last_c - first_o).reindex(range(nd)).values
        out[f"rng_{name}"] = (g.h.max() - g.l.min()).reindex(range(nd)).values
    # 後半天 15 分鐘變異數比（> 1 趨勢、< 1 均值回歸）
    B = D[D.mi >= CUT].copy()
    B["r1"] = B.groupby("d").c.diff()
    B["blk"] = (B.mi // 15).astype(int)
    r15 = B.groupby(["d", "blk"]).r1.sum()
    v15 = (r15 ** 2).groupby(level=0).mean()
    v1 = (B.r1 ** 2).groupby(B.d).mean()
    out["vr_B"] = (v15 / (15 * v1)).reindex(range(nd)).values
    out["cr_day"] = (out["ret_day"] / out["rng_day"])         # 收盤位置：|C−O|/(H−L)
    return pd.DataFrame(out)


def main():
    m = load("data/XAUUSD_M1_2023_2026.csv")
    day_id, minute, t0, days = trading_clock(m)
    nd = len(days)
    sc = day_scale(m, day_id)
    Y = day_targets(m, day_id, minute, nd)
    Y["year"] = days.year
    Y["abs_cr"] = Y.cr_day.abs()
    keep = np.arange(nd) >= 1                     # 第 1 天沒有前一天的尺度
    keep &= (Y.rng_A > 0).values & np.isfinite(Y.er_B.values)
    q = Y.er_day.quantile([1 / 3, 2 / 3]).values
    lab = np.where(Y.er_day >= q[1], 1, np.where(Y.er_day <= q[0], 0, -1))   # 1 = 單邊、0 = 來回

    print("=" * 100)
    print("A. 前半天（01:00~13:00）Hawkes ρ vs 當天型態   — 860 個交易日，2023/6 ~ 2026/10")
    print("   er = |收−開| / Σ|Δ收盤|（1 = 一路單邊、0 = 原地來回）；vr_B = 後半天 15 分鐘變異數比")
    print("=" * 100)
    base_auc = auc(Y.er_A.values, lab)
    print(f"免費對照：前半天 ER 對 後半天 ER 的 Spearman = {spearman(Y.er_A.values, Y.er_B.values)[0]:+.3f}；"
          f"分類單邊/來回日 AUC = {base_auc:.3f}")
    print(f"          前半天 ER 對 後半天 vr 的 Spearman = {spearman(Y.er_A.values, Y.vr_B.values)[0]:+.3f}")
    rows = []
    for kind in KINDS:
        T, S, B = make_events(m, day_id, t0, sc, kind)
        T = T.astype(np.float64); ed = day_id[B]
        # 前半天反向比例（最簡單的對照指標）
        same_day = np.r_[False, ed[1:] == ed[:-1]]
        rev = np.r_[False, S[1:] != S[:-1]] & same_day
        inA = (T - ed * 1380.0) < CUT
        nA = np.bincount(ed[inA & same_day], minlength=nd).astype(float)
        revA = np.bincount(ed[inA & rev], minlength=nd) / np.where(nA > 0, nA, np.nan)
        for beta in BETAS:
            R = batch_em_days(T, S, ed, nd, 0.0, CUT, beta, 80)
            rho = np.log(R[:, 2] / R[:, 1])                 # log ρ，避免極端值
            rho[~keep] = np.nan
            r_c, n = spearman(rho, Y.er_A.values)
            r_b, _ = spearman(rho, Y.er_B.values)
            r_v, _ = spearman(rho, Y.vr_B.values)
            p_b = partial_spearman(rho, Y.er_B.values, [Y.er_A.values, revA])
            p_v = partial_spearman(rho, Y.vr_B.values, [Y.er_A.values, revA])
            a = auc(-rho, lab)
            yr = {y: spearman(rho[(Y.year == y).values], Y.er_B.values[(Y.year == y).values])[0]
                  for y in (2023, 2024, 2025, 2026)}
            rows.append(dict(kind=kind, beta=beta, n=n, ev_A=np.nanmedian(nA[keep]),
                             rho_med=np.exp(np.nanmedian(rho)), same_A=r_c, rev_vs_erA=spearman(revA, Y.er_A.values)[0],
                             pred_erB=r_b, pred_vrB=r_v, partial_erB=p_b, partial_vrB=p_v, auc=a,
                             **{f"erB_{y}": v for y, v in yr.items()}))
    D = pd.DataFrame(rows)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print("\n每一列 = 一種事件 × β。數字是 Spearman（|r| > 0.07 約為 5% 顯著，n≈860）")
    print("  same_A    ：log ρ 對 前半天自己的 ER（應為負）     rev_vs_erA：前半天反向比例 對 前半天 ER（對照）")
    print("  pred_erB  ：log ρ 對 後半天 ER（預測要為負）        pred_vrB  ：log ρ 對 後半天變異數比（預測要為負）")
    print("  partial_* ：扣掉前半天 ER 與反向比例後的偏相關       auc       ：−ρ 分辨全天單邊日(1) vs 來回日(0)，0.5 = 沒用")
    print(D.round(3).to_string(index=False))
    D.to_csv("part22_day_type.csv", index=False)

    # A2. 遞迴（online）ρ：帶著前幾天的記憶，在 13:00 那一刻的值
    print("\nA2. 遞迴 ρ 在 13:00 的值（含前幾天記憶）對 後半天 ER / vr；括號內為對 前半天 ER")
    at13 = np.nonzero(minute == CUT)[0]
    d13 = day_id[at13]
    rows2 = []
    for kind in ["step_1", "step_2", "step_4"]:
        T, S, B = make_events(m, day_id, t0, sc, kind)
        T = T.astype(np.float64); ed = day_id[B]
        for beta in [0.05, 0.1]:
            for hl in [120.0, 480.0, 1440.0]:
                rho = online_em(T, S, ed, beta, hl)[0]
                v = np.full(nd, np.nan)
                v[d13] = to_bars(np.log(rho), B, len(m))[at13]
                v[~keep] = np.nan
                rows2.append(dict(kind=kind, beta=beta, hl=int(hl),
                                  vs_erA=spearman(v, Y.er_A.values)[0],
                                  pred_erB=spearman(v, Y.er_B.values)[0],
                                  pred_vrB=spearman(v, Y.vr_B.values)[0],
                                  partial_erB=partial_spearman(v, Y.er_B.values, [Y.er_A.values]),
                                  auc=auc(-v, lab)))
    print(pd.DataFrame(rows2).round(3).to_string(index=False))
    return D


if __name__ == "__main__":
    main()
