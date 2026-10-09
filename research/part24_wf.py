"""第二十四部分 B：優化結果的檢查 —— 只刪極端組的版本、滾動前推（walk-forward）、成本敏感度。

滾動前推：從 3 月起每個月月初，只用「之前全部事件」重做第二十四部分的整套流程：
  1. 參數：訓練期切前後兩半，兩半 PF 都 > 1.05、筆數 ≥ 60 → 取「總損益 / 最大回撤」最大；
  2. 因子：同一套刪除規則（兩半都是三組中最差、訓練期平均 < 0）；
  然後原封不動用在下一個月。RFI 的分位也只用訓練期的 K 棒。
（RFI-U 的 8 個量測本身是用 1~5 月挑的，3~5 月的前推仍有這一層偏誤；6~10 月是乾淨的。）
"""
import numpy as np
import pandas as pd
from part22_lib import CUT
from part24_lib import load_all, event_frame, add_outcomes, apply_cap, summarize, fmt
from part24_opt import MR_SPECS, MOM_SPECS

out = open("results_part24.txt", "a")
def P(s=""): print(s, flush=True); out.write(s + "\n"); out.flush()

FACT = ["vwap_side", "day_der", "htf", "day_used", "vel", "acc", "er_ratio", "er20", "er10", "leg_bars", "bbw_pct",
        "sig_ratio", "atr_ratio", "dist", "wick_out", "vol_rel", "beyond_pd"]


def choose_params(EV, rfi_train, train_end, legs=("mr", "mom")):
    best = {}
    for leg in legs:
        cand = []
        for (d0, rf), E in EV.items():
            tr = E[E.t < train_end]
            if len(tr) < 50:
                continue
            mid = tr.t.iloc[len(tr) // 2]
            for q in (0.25, 1 / 3, 0.4):
                lo, hi = np.nanquantile(rfi_train, [q, 1 - q])
                m = (tr.rfi >= hi) if leg == "mr" else (tr.rfi <= lo)
                for name in (MR_SPECS if leg == "mr" else MOM_SPECS):
                    s = tr[m]; p = s[f"pnl_{name}"]
                    a, b = p[s.t < mid], p[s.t >= mid]
                    if len(s) < 60 or len(a) == 0 or len(b) == 0:
                        continue
                    pf = lambda x: x[x > 0].sum() / max(-x[x <= 0].sum(), 1e-9)
                    if pf(a) <= 1.05 or pf(b) <= 1.05:
                        continue
                    st = summarize(p.to_numpy(), s.t.to_numpy())
                    cand.append((st["ratio"], d0, rf, q, name, lo, hi))
        if cand:
            best[leg] = max(cand)
    return best


def factor_drops(S, col, extremes_only=False):
    tr = S
    mid = tr.t.iloc[len(tr) // 2]
    h1, h2 = tr.t < mid, tr.t >= mid
    drops = []
    for f in FACT + ["sess"]:
        if f == "sess":
            g = tr[f]; e = None
        else:
            e = np.nanquantile(tr[f], [1 / 3, 2 / 3]); g = pd.Series(np.digitize(tr[f], e), tr.index).where(tr[f].notna())
        m1, m2 = tr[h1].groupby(g[h1])[col].mean(), tr[h2].groupby(g[h2])[col].mean()
        if len(m1) < 2 or len(m2) < 2:
            continue
        w1, w2 = m1.idxmin(), m2.idxmin()
        alls = tr.groupby(g)[col].mean()
        if w1 == w2 and alls[w1] < 0 and not (extremes_only and e is not None and w1 == 1):
            drops.append((f, w1, e))
    return drops


def apply_drops(S, drops):
    keep = pd.Series(True, S.index)
    for f, grp, e in drops:
        g = S[f] if e is None else pd.Series(np.digitize(S[f], e), S.index)
        keep &= g != grp
    return keep


if __name__ == "__main__":
    d, A, M, X = load_all()
    EV = {}
    for d0 in (1.5, 2.0, 2.5):
        for rf in (0.5, 0.75):
            EV[(d0, rf)] = add_outcomes(d, event_frame(d, A, M, X, d0, rf), {**MR_SPECS, **MOM_SPECS})
    rfi_bars = X.rfi.dropna()

    P("\n" + "=" * 120)
    P("第二十四部分 B：檢查")
    # 1. 只刪極端組（不刪中間三分位）
    P("\n1. 同一套 IS 選參數，因子刪除只允許刪『最高或最低三分位』與時段（中間組被刪在經濟上沒有解釋）")
    best = choose_params(EV, rfi_bars[rfi_bars.index < CUT], CUT)
    for ex_only in (False, True):
        T = []
        for leg, (_, d0, rf, q, name, lo, hi) in best.items():
            E = EV[(d0, rf)]
            S = E[(E.rfi >= hi) if leg == "mr" else (E.rfi <= lo)].copy()
            col = f"pnl_{name}"
            drops = factor_drops(S[S.t < CUT], col, extremes_only=ex_only)
            S = S[apply_drops(S, drops)].assign(pnl=lambda z: z[col], ex=lambda z: z[f"exit_{name}"], leg=leg)
            T.append(S)
            P(f"  {'只刪極端' if ex_only else '原規則'} {leg:3s} 參數 d0 {d0} 武裝 {rf} q {q:.2f} {name}；刪除 "
              + ("、".join(f"{f}={g}" for f, g, _ in drops) or "無"))
        T = pd.concat(T)
        for per in ("IS", "OOS"):
            s = T[(T.t < CUT) if per == "IS" else (T.t >= CUT)]
            P(f"    合併 {per:3s} {fmt(summarize(s.pnl, s.t))}")

    # 2. 滾動前推
    P("\n2. 滾動前推（每月月初只用之前的資料重選參數、RFI 分位與因子刪除，套用到該月）")
    months = pd.date_range("2026-03-01", "2026-10-01", freq="MS")
    WF, log = [], []
    for k, m0 in enumerate(months):
        m1 = months[k + 1] if k + 1 < len(months) else pd.Timestamp("2026-11-01")
        best = choose_params(EV, rfi_bars[rfi_bars.index < m0], m0)
        desc = []
        for leg, (_, d0, rf, q, name, lo, hi) in best.items():
            E = EV[(d0, rf)]
            S = E[(E.rfi >= hi) if leg == "mr" else (E.rfi <= lo)].copy()
            col = f"pnl_{name}"
            drops = factor_drops(S[S.t < m0], col)
            S = S[(S.t >= m0) & (S.t < m1)]
            S = S[apply_drops(S, drops)].assign(pnl=lambda z: z[col], ex=lambda z: z[f"exit_{name}"], leg=leg)
            WF.append(S)
            desc.append(f"{leg} d0 {d0}/武裝 {rf}/q {q:.2f}/{name}/刪 {len(drops)}")
        mm = pd.concat([w for w in WF if len(w) and w.t.iloc[0] >= m0]) if WF else pd.DataFrame()
        sub = mm[(mm.t >= m0) & (mm.t < m1)] if len(mm) else mm
        P(f"  {m0:%Y-%m}  {len(sub):3d} 筆 總 {sub.pnl.sum() if len(sub) else 0:+6.0f}   " + " | ".join(desc))
    W = pd.concat(WF).sort_values("t")
    W.to_csv("part24_walkforward_trades.csv", index=False)
    for lab, sel in (("3~5 月（RFI 量測挑選期內）", W.t < CUT), ("6~10 月", W.t >= CUT), ("3~10 月全部", W.t >= "2026-03-01")):
        s = W[sel]
        P(f"  {lab:22s} {fmt(summarize(s.pnl, s.t))}")
    for leg in ("mr", "mom"):
        s = W[(W.leg == leg) & (W.t >= CUT)]
        P(f"  6~10 月 {('回歸' if leg == 'mr' else '動能')}  {fmt(summarize(s.pnl, s.t))}")

    # 3. 成本敏感度（第 1 節原規則版本）
    P("\n3. 成本敏感度（6~10 月，IS 選出的版本 vs 滾動前推）：每筆再多扣 x 美元")
    best = choose_params(EV, rfi_bars[rfi_bars.index < CUT], CUT)
    T = []
    for leg, (_, d0, rf, q, name, lo, hi) in best.items():
        E = EV[(d0, rf)]; S = E[(E.rfi >= hi) if leg == "mr" else (E.rfi <= lo)].copy(); col = f"pnl_{name}"
        S = S[apply_drops(S, factor_drops(S[S.t < CUT], col))].assign(pnl=lambda z: z[col]); T.append(S)
    T = pd.concat(T); T = T[T.t >= CUT]; Wo = W[W.t >= CUT]
    for x in (0, 0.2, 0.5, 1.0):
        a, b = summarize(T.pnl - x, T.t), summarize(Wo.pnl - x, Wo.t)
        P(f"  +{x:.1f} 美元：IS 選定版 每筆 {a['avg']:+.2f} PF {a['pf']:.2f} | 滾動前推 每筆 {b['avg']:+.2f} PF {b['pf']:.2f}")

    # 4. 滾動前推中重複出現的選擇（穩定 = 可信）
    P("\n4. 滾動前推 8 個月中，每個選擇出現的次數")
    from collections import Counter
    cnt = {"mr": Counter(), "mom": Counter()}; pc = {"mr": Counter(), "mom": Counter()}
    for m0 in months:
        best = choose_params(EV, rfi_bars[rfi_bars.index < m0], m0)
        for leg, (_, d0, rf, q, name, lo, hi) in best.items():
            pc[leg][f"d0={d0}"] += 1; pc[leg][f"q={q:.2f}"] += 1; pc[leg][name.split("_H")[0].replace("mr_T", "目標EMA")] += 1
            pc[leg]["H=" + name.split("_H")[1].split("_")[0]] += 1; pc[leg]["停損=" + name.split("_s")[1]] += 1
            E = EV[(d0, rf)]; S = E[(E.rfi >= hi) if leg == "mr" else (E.rfi <= lo)]
            for f, g, _ in factor_drops(S[S.t < m0], f"pnl_{name}"):
                cnt[leg][f"{f}={g}"] += 1
    for leg in ("mr", "mom"):
        P(f"  {'回歸' if leg == 'mr' else '動能'} 參數：" + "  ".join(f"{k}×{v}" for k, v in pc[leg].most_common()))
        P(f"  {'':4s} 刪除：" + "  ".join(f"{k}×{v}" for k, v in cnt[leg].most_common()))
