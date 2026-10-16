"""Covered call 的管理方式、天期、delta。

管理方式（股價漲過履約價時）：
  HOLD     放到期（原報告）
  BUYBACK  股價 ≥ 履約價就買回，這一期不再賣，等原本的到期日後才開下一期
  ROLL     股價 ≥ 履約價就買回，立刻賣一張新的（同 delta、同天期，往上往後 roll）
天期（交易日）：10 / 21 / 31 / 42 / 63（約 2 週 / 1 個月 / 45 天 / 2 個月 / 3 個月）
delta：0.10 / 0.15 / 0.20 / 0.30 / 0.40
偏斜（賣出的 call 隱含波動比 VXN 低幾點）：2 / 3 / 4
其他同 run_covered.py：只在持有 1 倍 QQQ 時賣、換到 TQQQ 前買回、BS 定價、買賣價差 5%（賣出少收、買回多付）。
損益以「每股一張」計算：每天的選擇權損益 ÷ 前一天股價，加到 QQQ 日報酬上。
輸出 results_covered2.txt、results_covered2.csv。
"""
import math
import itertools
import numpy as np
import pandas as pd
import engine as E
from run_tests import q, OK0, w_rule
from run_qld import run3
from run_options import bs, IV, S, RF
from run_covered import codes

out = []
P = lambda s: (print(s, flush=True), out.append(s))
SP = 0.05
NI = {0.10: -1.2815515655446004, 0.15: -1.0364333894937898, 0.20: -0.8416212335729143,
      0.30: -0.5244005127080407, 0.40: -0.2533471031357997}


def strike(s, sig, t, rr, delta):
    return s * math.exp(-NI[delta] * sig * math.sqrt(t) + (rr + sig ** 2 / 2) * t)


def cc_series(dte, delta, skew, mode):
    n = len(q); rq = q.r.values; ov = np.zeros(n)
    sk = skew / 100
    i = 0; nsold = nbreach = 0; prem_sum = 0.0
    while i < n - 1:
        e = min(i + dte, n - 1)
        sig = max(IV[i] - sk, 0.05); T = (e - i) / 252
        K = strike(S[i], sig, T, RF[i], delta)
        prev = bs("call", S[i], K, T, sig, RF[i])
        ov[i] -= prev * SP / S[i]; prem_sum += prev / S[i]; nsold += 1
        j = i + 1; nxt = e
        while j <= e:
            sj = max(IV[j] - sk, 0.05)
            val = bs("call", S[j], K, max((e - j) / 252, 0), sj, RF[j])
            ov[j] -= (val - prev) / S[j - 1]
            prev = val
            if mode != "HOLD" and j < e and S[j] >= K:
                ov[j] -= val * SP / S[j]               # 買回多付價差
                nbreach += 1
                if mode == "ROLL":
                    nxt = j                            # 立刻賣新的
                else:
                    nxt = e                            # 等到原到期日
                    # 買回後到到期日之間沒有空頭 call
                break
            j += 1
        i = nxt if nxt > i else i + 1
    yrs = (q.index[-1] - q.index[0]).days / 365.25
    return pd.Series(rq + ov, index=q.index), dict(sold_per_year=nsold / yrs, breach_rate=nbreach / max(nsold, 1),
                                                 avg_prem=prem_sum / max(nsold, 1))


q["q_plain"] = q.close; q["t_plain"] = q.tq_syn; q["bill"] = (1 + q.rf).cumprod()
lev = codes(w_rule(OK0), np.zeros(len(q), bool))
base = E.stats(run3(q, lev, cols=("q_plain", "bill", "t_plain")))
broll = pd.DataFrame([E.stats(run3(q, lev, f"{y}-01-01", f"{y + 10}-01-01", cols=("q_plain", "bill", "t_plain"))) for y in range(2000, 2017)])
P(f"原策略（不賣 call）：IRR {base['irr']:.4f}、最慘 {base['worst']:.2f}")


def run_one(dte, delta, skew, mode):
    rs, info = cc_series(dte, delta, skew, mode)
    q["q_cc"] = (1 + rs).cumprod()
    cols = ("q_cc", "bill", "t_plain")
    f = E.stats(run3(q, lev, cols=cols))
    roll = pd.DataFrame([E.stats(run3(q, lev, f"{y}-01-01", f"{y + 10}-01-01", cols=cols)) for y in range(2000, 2017)])
    d = roll.irr.values - broll.irr.values
    return dict(dte=dte, delta=delta, skew=skew, mode=mode, irr99=f["irr"], d_irr=f["irr"] - base["irr"], d10=np.median(d),
                win10=int((d > 0).sum()), worst=f["worst"], **info)


rows = []
P("\n1. 管理方式（0.20 delta、31 天）")
for skew, mode in itertools.product([2, 3, 4], ["HOLD", "BUYBACK", "ROLL"]):
    o = run_one(31, 0.20, skew, mode); rows.append(o)
    P(f"  偏斜 {skew} {mode:8s} IRR 差 {o['d_irr']:+.4f} 10 年中位差 {o['d10']:+.4f}（{o['win10']}/17）最慘 {o['worst']:.2f}｜"
      f"每年賣 {o['sold_per_year']:.1f} 張、被漲穿 {o['breach_rate']:.0%}、平均權利金 {o['avg_prem']:.2%}")

P("\n2. 天期 × delta（HOLD；表內 = 1999 起 IRR 相對原策略的差，每年 %）")
for skew in [2, 3, 4]:
    grid = {}
    for dte, delta in itertools.product([10, 21, 31, 42, 63], [0.10, 0.15, 0.20, 0.30, 0.40]):
        o = run_one(dte, delta, skew, "HOLD"); rows.append(o)
        grid[(dte, delta)] = o["d_irr"] * 100
    G = pd.Series(grid).unstack()
    G.index = [f"{x} 天" for x in G.index]; G.columns = [f"Δ{c:.2f}" for c in G.columns]
    P(f"  偏斜 {skew}")
    P(G.to_string(float_format=lambda x: f"{x:+.2f}"))

R = pd.DataFrame(rows)
R.to_csv("results_covered2.csv", index=False, float_format="%.5f")
open("results_covered2.txt", "w").write("\n".join(out) + "\n")
