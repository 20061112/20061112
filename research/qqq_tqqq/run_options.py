"""雙向保險：依持倉狀態買選擇權。

痛點 1（往下）：長空頭時的 1 倍 QQQ（最大回撤來源）、持 TQQQ 時急跌。
痛點 2（往上）：雙動能躲進 T-bill 後錯過 V 型反彈（2003、2009、2023）。
做法：把每個持倉換成「持倉 + 選擇權」的組合，策略在組合之間切換（切換時以 BS 估值賣掉舊的選擇權）。
  QQQ+put    1 倍 QQQ 加 QQQ put（數量 = 部位 / S，履約 90% 或 95%）
  TQQQ+put   TQQQ 加 QQQ put（數量 = 3 × 部位 / S，履約 90%）
  BILL+call  T-bill 加 QQQ call（數量 = 部位 / S，履約 100% 或 105%）—— 由買賣權平價，約等於有保護的 QQQ
定價：Black-Scholes；隱含波動 = VXN（2001/2 前用 20 日實現波動 × 1.2）+ 偏斜：90% put +4、95% put +2、ATM 0、105% call −1（點）；
每次買進再加 5% 權利金當買賣價差。到期 1 個月或 3 個月，每月 / 每季第一個交易日換約。
輸出 results_options.txt、results_options.csv。
"""
import math
import numpy as np
import pandas as pd
import engine as E
from run_tests import q, OK0, ABOVE, c, r, w_rule
from run_dualmom import dm_off
from run_qld import run3

out = []
P = lambda s: (print(s, flush=True), out.append(s))
MARKUP = 0.05
SKEW = {0.90: 4, 0.95: 2, 1.00: 0, 1.05: -1}
rv20 = r.rolling(20).std() * np.sqrt(252)
IV = (q.vxn / 100).fillna(rv20 * 1.2).bfill().values
S = c.values; RF = (q.rf * 252).values


def ncdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs(kind, s, k, t, sig, rr):
    if t <= 1e-6:
        return max(k - s, 0.0) if kind == "put" else max(s - k, 0.0)
    d1 = (math.log(s / k) + (rr + sig ** 2 / 2) * t) / (sig * math.sqrt(t)); d2 = d1 - sig * math.sqrt(t)
    if kind == "put":
        return k * math.exp(-rr * t) * ncdf(-d2) - s * ncdf(-d1)
    return s * ncdf(d1) - k * math.exp(-rr * t) * ncdf(d2)


def overlay(base_ret, kind, mny, mult, months):
    """回傳（組合日報酬, 每年權利金支出占部位比例）。換約日：每 months 個月的第一個交易日。
    選擇權損益以換約時部位為分母（月內部位漂移忽略）；換約時付 MARKUP 的價差。"""
    mon = q.index.to_period("M")
    ms = np.r_[True, mon[1:] != mon[:-1]]
    mi = np.cumsum(ms) - 1
    roll = ms & (mi % months == 0)
    rs = np.flatnonzero(roll); ends = np.r_[rs[1:], len(q)]
    ov = np.zeros(len(q)); spend = np.zeros(len(q))
    sk = SKEW[mny] / 100
    for a, e in zip(rs, ends):
        K = mny * S[a]; T0 = (e - a) / 252
        prev = bs(kind, S[a], K, T0, IV[a] + sk, RF[a])
        spend[a] = mult * prev * (1 + MARKUP) / S[a]
        ov[a] -= mult * prev * MARKUP / S[a]
        for j in range(a + 1, min(e + 1, len(q))):
            val = bs(kind, S[j], K, max((e - j) / 252, 0), IV[j] + sk, RF[j])
            ov[j] += mult * (val - prev) / S[a]
            prev = val
    yrs = (q.index[-1] - q.index[0]).days / 365.25
    return pd.Series(base_ret.values + ov, index=q.index), spend


rq = q.r; r3 = q.tq_syn.pct_change().fillna(0); rb = q.rf
q["bill"] = (1 + rb).cumprod()


def series(ret):
    return (1 + ret).cumprod()


def codes(w, off):
    """1 = 1 倍部位（QQQ 類）、2 = T-bill 類、3 = TQQQ 類。"""
    w = np.asarray(w); off = np.asarray(off, bool)
    return np.where(w > 0, 3, np.where(off, 2, 1))


base_w = w_rule(OK0)
no_off = np.zeros(len(q), bool)
dm = dm_off(c, q.rf, 200, 252).values


def evaluate(label, lev, cols, spend_parts=()):
    f = E.stats(run3(q, lev, cols=cols))
    roll = pd.DataFrame([E.stats(run3(q, lev, f"{y}-01-01", f"{y + 10}-01-01", cols=cols)) for y in range(2000, 2017)])
    rets = {k: q[col].pct_change().fillna(0).values for k, col in zip((1, 2, 3), cols)}
    held = np.r_[1, lev[:-1]]
    ls = pd.Series([rets[h][i] for i, h in enumerate(held)], index=q.index)
    eq = (1 + ls).cumprod(); mdd = (eq / eq.cummax() - 1).min()
    yr = (1 + ls).groupby(ls.index.year).prod() - 1
    sh = ls.mean() / ls.std() * np.sqrt(252)
    cost = 0.0
    for code, sp in spend_parts:
        cost += sp[held == code].sum()
    cost /= (q.index[-1] - q.index[0]).days / 365.25
    return dict(label=label, irr99=f["irr"], worst=f["worst"], worst10=roll.worst.min(), irr10y_med=roll.irr.median(),
                mdd=mdd, sharpe=sh, worst_year=yr.min(), y2002=yr[2002], y2009=yr[2009], y2022=yr[2022], y2023=yr[2023],
                prem=cost)


rows = []


def show(row):
    rows.append(row)
    P(f"{row['label']:46s} IRR {row['irr99']:.3f} 10年中位 {row['irr10y_med']:.3f}｜最慘 {row['worst']:.2f} 10年段最慘 {row['worst10']:.2f}｜"
      f"MDD {row['mdd']:.2f} Sharpe {row['sharpe']:.2f}｜2002 {row['y2002']:+.0%} 2009 {row['y2009']:+.0%} 2022 {row['y2022']:+.0%} "
      f"2023 {row['y2023']:+.0%}｜權利金 {row['prem']:.1%}/年")


q["q_plain"] = q.close
q["t_plain"] = q.tq_syn
P("持倉狀態：1 倍 QQQ / T-bill / TQQQ；權利金 = 每年花在選擇權上的錢占當時部位的比例（只算持有該組合的期間）")
show(evaluate("原策略", codes(base_w, no_off), ("q_plain", "bill", "t_plain")))
show(evaluate("雙動能（T-bill，無選擇權）", codes(base_w, dm), ("q_plain", "bill", "t_plain")))

# 單向：往下保險
for mny, mo in [(0.90, 1), (0.95, 1), (0.90, 3)]:
    hq, sq = overlay(rq, "put", mny, 1, mo); q["q_put"] = series(hq)
    show(evaluate(f"QQQ 1 倍時買 {mny:.0%} put（{mo} 個月）", codes(base_w, no_off), ("q_put", "bill", "t_plain"), [(1, sq)]))
ht, st = overlay(r3, "put", 0.90, 3, 1); q["t_put"] = series(ht)
show(evaluate("TQQQ 時買 90% put（1 個月）", codes(base_w, no_off), ("q_plain", "bill", "t_put"), [(3, st)]))
hq, sq = overlay(rq, "put", 0.90, 1, 3); q["q_put"] = series(hq)
show(evaluate("QQQ 與 TQQQ 都買 90% put", codes(base_w, no_off), ("q_put", "bill", "t_put"), [(1, sq), (3, st)]))

# 單向：往上保險（雙動能躲 T-bill 時買 call）
for mny, mo in [(1.00, 1), (1.05, 1), (1.00, 3), (1.05, 3)]:
    hb, sb = overlay(rb, "call", mny, 1, mo); q["b_call"] = series(hb)
    show(evaluate(f"雙動能 + T-bill 時買 {mny:.0%} call（{mo} 個月）", codes(base_w, dm), ("q_plain", "b_call", "t_plain"), [(2, sb)]))

# 雙向
hb, sb = overlay(rb, "call", 1.05, 1, 3); q["b_call"] = series(hb)
hq, sq = overlay(rq, "put", 0.90, 1, 3); q["q_put"] = series(hq)
show(evaluate("雙向：雙動能 + T-bill 買 call + QQQ 買 put（3 個月）", codes(base_w, dm), ("q_put", "b_call", "t_plain"), [(1, sq), (2, sb)]))
show(evaluate("雙向：上面 + TQQQ 也買 put", codes(base_w, dm), ("q_put", "b_call", "t_put"), [(1, sq), (2, sb), (3, st)]))
# 不用雙動能，始終同時持有 put + call（跨式 / 勒式）在 1 倍部位上
hs1, ss1 = overlay(rq, "put", 0.90, 1, 3); hs2, ss2 = overlay(pd.Series(0.0, index=q.index), "call", 1.05, 1, 3)
q["q_strangle"] = series(hs1 + hs2)
show(evaluate("勒式：QQQ 1 倍時同時買 90% put + 105% call", codes(base_w, no_off), ("q_strangle", "bill", "t_plain"), [(1, ss1 + ss2)]))

pd.DataFrame(rows).to_csv("results_options.csv", index=False, float_format="%.4f")
open("results_options.txt", "w").write("\n".join(out) + "\n")
