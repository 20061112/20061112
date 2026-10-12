"""第五十九部分：把第五十八部分有效的兩項合在一起 —— 口數依停損距離 × 組合單日虧損上限（改成跟波動等比例）。

第五十八部分的固定美元上限（135~162 美元）在 2026 年波動大時會比較常觸發；這裡把上限改成「R 單位」：
  當天上限 = k × 1R 美元，1R 美元 = 前 20 個交易日兩策略每筆「停損距離 × 口數」的平均（只用過去）
  k = 訓練期（2023/8~2024/12）「組合日損益 ÷ 當天 1R 美元」的 2% / 5% 分位
口數：固定 / ∝ 1/√停損 / ∝ 1/停損（平均口數不變：IB 1 份、M15 2 份）
另列「放大到跟基準同樣總獲利」時的 MDD（比值不受口數倍數影響）。
"""
import numpy as np
import pandas as pd
import part58_robust_opt as R8

Q = R8.Q
TRAIN_END = Q.TRAIN_END
lines = []
Q.lines = lines


def p(s=""):
    print(s, flush=True)
    lines.append(s)


def r_unit(items):
    rows = pd.concat([pd.Series(x.risk.values * np.broadcast_to(q, len(x)), index=x.進場時間.dt.normalize()) for x, q in items])
    d = rows.groupby(level=0).mean().sort_index()
    return d.rolling(20, min_periods=5).mean().shift(1)


def daily_pnl(items):
    return pd.concat([(x.net * q).groupby(x.進場時間.dt.normalize()).sum() for x, q in items], axis=1).fillna(0).sum(1).sort_index()


def cap_var(items, capd):
    rows = []
    for k, (x, q) in enumerate(items):
        qq = np.broadcast_to(q, len(x))
        for n_, (a, b, v) in enumerate(zip(x.進場時間, x.出場時間, x.net.values * qq)):
            rows.append((a, b, v, k, n_))
    rows.sort(key=lambda r: (r[0], r[3]))
    keep = [np.zeros(len(x), bool) for x, q in items]
    cur, real, X = None, [], np.inf
    for a, b, v, k, n_ in rows:
        d = a.normalize()
        if d != cur:
            cur, real = d, []
            X = capd.get(d, np.nan)
            X = np.inf if not np.isfinite(X) else X
        if sum(vv for (bb, vv) in real if bb <= a) <= -X:
            continue
        keep[k][n_] = True; real.append((b, v))
    return [(x[m].reset_index(drop=True), (np.broadcast_to(q, len(x))[m] if np.ndim(q) else q)) for (x, q), m in zip(items, keep)]


p("第五十九部分：口數依停損距離 × 組合單日虧損上限（R 單位）；基準 = IB(2R 保本) 1 份 + M15 2 份")
for slip in (0.0, 0.3):
    I, M = R8.ib(slip), R8.m15(slip)
    p(f"\n=== 滑價 {slip} ===")
    base_tot = None
    for lab, pw in (("固定口數", 0.0), ("∝1/√停損", 0.5), ("∝1/停損", 1.0)):
        qi = (1 / I.risk.values) ** pw; qi = qi / qi.mean()
        qm = (1 / M.risk.values) ** pw; qm = 2 * qm / qm.mean()
        items = [(I, qi), (M, qm)]
        ru = r_unit(items)
        d0 = daily_pnl(items)
        z = (d0 / ru.reindex(d0.index)).dropna()
        zt = z[z.index < TRAIN_END]
        variants = [("無上限", items)]
        for qq in (0.02, 0.05):
            k = -zt.quantile(qq)
            variants.append((f"上限 {k:.1f}R（訓練 {qq:.0%} 分位）", cap_var(items, (k * ru).to_dict())))
        for vl, it in variants:
            d = R8.show(f"{lab}｜{vl}", it)
            tot = d.sum()
            if base_tot is None:
                base_tot = tot
            eq = Q.minute_eq([Q.mpart(x, q) for x, q in it])
            mm = (np.maximum.accumulate(eq) - eq).max()
            p(f"      逐年比值 " + " / ".join(f"{v:.1f}" for v in R8.ratio_by_year(d)) +
              f"｜放大 {base_tot / tot:.2f} 倍到同樣總獲利 {base_tot:,.0f} 時 MDD 含浮虧 ≈ {mm * base_tot / tot:,.0f}")
open("results_part59.txt", "w").write("\n".join(lines) + "\n")
