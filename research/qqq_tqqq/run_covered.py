"""Covered call（原報告的可選項）重測，並與雙動能組合。

規則（同原報告）：只在持有 1 倍 QQQ 的期間賣 call；0.2 delta、約 45 天（31 個交易日）；放到期不 roll；
策略換到 TQQQ（或 T-bill）前以 BS 價格買回。這裡用「QQQ + 賣出 call」的組合價格序列實作：
每 31 個交易日開一個新循環，策略持有 QQQ 期間就持有這個組合，換出時以當時 BS 價值買回（= 賣掉組合）。
定價：Black-Scholes；call 隱含波動 = VXN − 偏斜（0 / 2 / 4 / 6 點；2001/2 前用 20 日實現波動 × 1.2）；
賣出時只收到 95% 權利金、買回時多付 5%（買賣價差）。
輸出 results_covered.txt、results_covered.csv。
"""
import math
import numpy as np
import pandas as pd
import engine as E
from run_tests import q, OK0, c, r, w_rule
from run_dualmom import dm_off
from run_qld import run3
from run_options import ncdf, bs, IV, S, RF

out = []
P = lambda s: (print(s, flush=True), out.append(s))
CYCLE = 31
SPREAD_FRAC = 0.05
Z20 = -0.8416212335729143          # Φ⁻¹(0.2)


def strike_for_delta(s, sig, t, rr, d1=Z20):
    return s * math.exp(-d1 * sig * math.sqrt(t) + (rr + sig ** 2 / 2) * t)


def covered_series(skew):
    """回傳（QQQ + 賣 call 組合的日報酬, 被履約 / 買回時的損失紀錄）。"""
    rq = q.r.values; n = len(q)
    ov = np.zeros(n); starts = np.arange(0, n, CYCLE)
    info = []
    for a in starts:
        e = min(a + CYCLE, n - 1)
        T0 = (e - a) / 252
        sig0 = max(IV[a] - skew / 100, 0.05)
        K = strike_for_delta(S[a], sig0, T0, RF[a])
        prem = bs("call", S[a], K, T0, sig0, RF[a])
        ov[a] += prem * SPREAD_FRAC * -1 / S[a]          # 賣出只收 95%
        prev = prem
        for j in range(a + 1, e + 1):
            sig = max(IV[j] - skew / 100, 0.05)
            val = bs("call", S[j], K, max((e - j) / 252, 0), sig, RF[j])
            ov[j] -= (val - prev) / S[a]                   # 空頭 call：價值上升 = 虧
            prev = val
        info.append(dict(start=q.index[a], K_pct=K / S[a] - 1, prem_pct=prem / S[a], payoff_pct=max(S[e] - K, 0) / S[a]))
    return pd.Series(rq + ov, index=q.index), pd.DataFrame(info)


def codes(w, off):
    w = np.asarray(w); off = np.asarray(off, bool)
    return np.where(w > 0, 3, np.where(off, 2, 1))


if __name__ == "__main__":
    q["q_plain"] = q.close; q["t_plain"] = q.tq_syn; q["bill"] = (1 + q.rf).cumprod()
    base_w = w_rule(OK0)
    no_off = np.zeros(len(q), bool)
    dm = dm_off(c, q.rf, 200, 252).values
    base_lev = codes(base_w, no_off); dm_lev = codes(base_w, dm)
    rows = []
    base_roll = None


    def evaluate(label, lev, cols):
        global base_roll
        f = E.stats(run3(q, lev, cols=cols))
        s10 = E.stats(run3(q, lev, "2010-02-12", cols=cols))
        roll = pd.DataFrame([E.stats(run3(q, lev, f"{y}-01-01", f"{y + 10}-01-01", cols=cols)) for y in range(2000, 2017)])
        if base_roll is None:
            base_roll = roll
        d = roll.irr.values - base_roll.irr.values
        rets = {k: q[col].pct_change().fillna(0).values for k, col in zip((1, 2, 3), cols)}
        held = np.r_[1, lev[:-1]]
        ls = pd.Series([rets[h][i] for i, h in enumerate(held)], index=q.index)
        eq = (1 + ls).cumprod(); mdd = (eq / eq.cummax() - 1).min()
        yr = (1 + ls).groupby(ls.index.year).prod() - 1
        row = dict(label=label, irr99=f["irr"], irr10=s10["irr"], d10=np.median(d), win10=int((d > 0).sum()),
                   worst=f["worst"], worst10=roll.worst.min(), mdd=mdd, sharpe=ls.mean() / ls.std() * np.sqrt(252), worst_year=yr.min())
        rows.append(row)
        P(f"{label:34s} IRR {f['irr']:.3f}｜2010 起 {s10['irr']:.3f}｜10 年 IRR 差中位 {np.median(d):+.4f}（{(d > 0).sum()}/17）｜"
          f"最慘 {f['worst']:.2f} 10年段最慘 {roll.worst.min():.2f}｜MDD {mdd:.2f} Sharpe {row['sharpe']:.2f}")


    P("原策略 / 雙動能，各自加 covered call（偏斜 = 賣出的 call 隱含波動比 VXN 低幾點）")
    evaluate("原策略", base_lev, ("q_plain", "bill", "t_plain"))
    for sk in [0, 2, 4, 6]:
        rs, info = covered_series(sk)
        q[f"q_cc{sk}"] = (1 + rs).cumprod()
        if sk in (2, 4):
            P(f"  偏斜 {sk}：平均履約價 現價 +{info.K_pct.mean():.1%}、平均權利金 {info.prem_pct.mean():.2%}/次、"
              f"被漲穿的循環 {(info.payoff_pct > 0).mean():.0%}、被漲穿時平均損失 {info.payoff_pct[info.payoff_pct > 0].mean():.2%}")
        evaluate(f"原策略 + covered call（偏斜 {sk}）", base_lev, (f"q_cc{sk}", "bill", "t_plain"))
    evaluate("雙動能", dm_lev, ("q_plain", "bill", "t_plain"))
    for sk in [2, 4]:
        evaluate(f"雙動能 + covered call（偏斜 {sk}）", dm_lev, (f"q_cc{sk}", "bill", "t_plain"))

    # 實際 vs 隱含：VXN 與之後 31 日實現波動
    fut = q.r[::-1].rolling(CYCLE).std()[::-1].shift(-1) * np.sqrt(252) * 100
    gap = (q.vxn - fut).dropna()
    P(f"\n參考：VXN − 之後 {CYCLE} 個交易日實現波動 平均 {gap.mean():+.1f} 點、中位 {gap.median():+.1f} 點、VXN 較高的日子 {(gap > 0).mean():.0%}")
    pd.DataFrame(rows).to_csv("results_covered.csv", index=False, float_format="%.4f")
    open("results_covered.txt", "w").write("\n".join(out) + "\n")
