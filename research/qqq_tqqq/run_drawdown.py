"""降低最大回撤：機構常用的做法，對症兩種回撤來源。

回撤拆解（一次投入）：
  A 長空頭裡的 1 倍 QQQ：2000~2003（QQQ 段 −72%）、2022（−33%）
  B 開槓桿時急跌：2015~16（TQQQ 段 −51%）、2020/9（−41%）、2025/4（−28%）、2000 年初
A 類對策（跌破趨勢時的 1 倍部位怎麼放）
  TBILL     MA200 下全轉 T-bill
  TBILL50   MA200 下一半轉 T-bill
  DUALMOM   雙動能（Antonacci）：MA200 下 且 12 個月報酬 < T-bill → 轉 T-bill
  VT1X      1 倍部位做波動目標：MA200 下時 QQQ 比例 = min(1, σ* / σ̂)，其餘 T-bill
B 類對策（開槓桿時）
  BOLL      布林過度偏離：持 TQQQ 時 20 日布林 z > k → 換回 QQQ（下週重新判斷）
  EXT       離 MA200 太遠：收盤 / MA200 − 1 > k → 不開槓桿
  DDCTL     回撤控制（CPPI / TIPP 類）：策略自身淨值回撤 > D → 不開槓桿，回到 D/2 以內才解除
  PUT       保護性賣權：持 TQQQ 時每月買 1 個月期、履約價 90% / 85% 的 QQQ put，數量覆蓋 3 倍曝險；
            以 Black-Scholes 定價，隱含波動 = VXN + 偏斜（3 / 6 點；VXN 之前用 20 日實現波動 × 1.2）
輸出 results_drawdown.txt、results_drawdown.csv。
"""
import math
import numpy as np
import pandas as pd
import engine as E
import indicators as ind
from run_tests import q, OK0, Z, ABOVE, c, r, w_rule
from run_sigma import har_rel_daily

out = []
P = lambda s: (print(s, flush=True), out.append(s))
r3 = q.tq_syn.pct_change().fillna(0)


def lump(w, f=None, r3s=None):
    """一次投入、不定投的日報酬（看持有部位本身的最大回撤）。f = 1 倍部位轉 T-bill 比例（已對齊持有日）。"""
    w = np.nan_to_num(np.asarray(w, float))
    f = np.zeros(len(w)) if f is None else np.nan_to_num(np.asarray(f, float))
    r3v = (r3 if r3s is None else r3s).values
    wp = np.r_[0.0, w[:-1]]; fp = np.r_[0.0, f[:-1]]
    rs = (1 - wp) * ((1 - fp) * q.r.values + fp * q.rf.values) + wp * r3v - np.r_[0, np.abs(np.diff(w))] * E.COST
    return pd.Series(rs, index=q.index)


def mdd(rs):
    eq = (1 + rs).cumprod()
    return (eq / eq.cummax() - 1).min()


base_w = w_rule(OK0)
base_roll = E.rolling_windows(q, base_w)
rows = []


def add(group, label, w, off=None, tq_col="tq_syn", r3s=None):
    kw = dict(off=off, tq_col=tq_col)
    f = E.stats(E.run(q, w, **kw)); s10 = E.stats(E.run(q, w, start="2010-02-12", **kw))
    roll = E.rolling_windows(q, w, **kw)
    d = roll.irr.values - base_roll.irr.values
    ls = lump(w, off, r3s)
    yr = (1 + ls).groupby(ls.index.year).prod() - 1
    sh = ls.mean() / ls.std() * np.sqrt(252)
    row = dict(group=group, label=label, irr99=f["irr"], irr10=s10["irr"], worst=f["worst"],
               worst10_med=roll.worst.median(), worst10_min=roll.worst.min(), mdd_lump=mdd(ls),
               worst_year=yr.min(), d10=np.median(d), win10=int((d > 0).sum()), sharpe=sh, sw=f["sw_per_year"])
    rows.append(row)
    P(f"{group:7s} {label:32s} irr99 {f['irr']:.3f} irr10 {s10['irr']:.3f} dIRR10y {np.median(d):+.3f} | 最慘 {f['worst']:.2f} "
      f"10年段最慘 中位 {roll.worst.median():.2f} 最差 {roll.worst.min():.2f} | 一次投入 MDD {mdd(ls):.2f} 最差年 {yr.min():+.0%} "
      f"Sharpe {sh:.2f} | 換倉 {f['sw_per_year']:.1f}")
    return row


add("BASE", "原策略", base_w)

# ---------------- A 類
below = (~ABOVE).shift(1, fill_value=False)
add("A", "TBILL：MA200 下全轉 T-bill", base_w, off=below.values)
add("A", "TBILL50：MA200 下轉一半", base_w, off=below.values * 0.5)
mom12 = c / c.shift(252) - 1
rf12 = q.rf.rolling(252).sum()
dm = ((~ABOVE) & (mom12 < rf12)).shift(1, fill_value=False)
add("A", "DUALMOM：MA200 下且 12 月動能 < T-bill", base_w, off=dm.values)
hv = ind.har_vol(q, 20)
for tgt in [0.20, 0.25, 0.30]:
    frac = (1 - (tgt / hv).clip(upper=1)).where(~ABOVE, 0.0).fillna(0.0).shift(1).fillna(0).values
    add("A", f"VT1X：MA200 下 1 倍部位目標波動 {tgt:.0%}", base_w, off=frac)

# ---------------- B 類
bz20 = (c - c.rolling(20).mean()) / c.rolling(20).std()
for k in [2.0, 2.5]:
    add("B", f"BOLL：20 日布林 z > {k} 不持 TQQQ", w_rule(OK0 & ~(bz20 > k)))
ext = c / c.rolling(200).mean() - 1
for k in [0.20, 0.30]:
    add("B", f"EXT：離 MA200 > {k:.0%} 不開槓桿", w_rule(OK0 & ~(ext > k)))


def dd_control(D):
    """依策略自身淨值（一次投入）回撤決定是否允許槓桿，逐日遞迴。"""
    ok = OK0.shift(1, fill_value=False).values
    ws = q.week_start.values; rq = q.r.values; r3v = r3.values
    w = np.zeros(len(q)); eq = 1.0; peak = 1.0; lock = False; cur = 0.0
    for i in range(len(q)):
        if i > 0:
            eq *= 1 + (cur * r3v[i] + (1 - cur) * rq[i])
            peak = max(peak, eq)
        dd = eq / peak - 1
        if dd < -D:
            lock = True
        elif dd > -D / 2:
            lock = False
        if cur > 0 and (not ok[i] or lock):
            cur = 0.0
        elif cur == 0 and ws[i] and ok[i] and not lock:
            cur = 1.0
        w[i] = cur
    return w


for D in [0.20, 0.30, 0.40]:
    add("B", f"DDCTL：淨值回撤 > {D:.0%} 停槓桿", dd_control(D))


# 保護性賣權
def ncdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_put(S, K, T, sig, rr):
    if T <= 1e-6:
        return max(K - S, 0.0)
    d1 = (math.log(S / K) + (rr + sig ** 2 / 2) * T) / (sig * math.sqrt(T)); d2 = d1 - sig * math.sqrt(T)
    return K * math.exp(-rr * T) * ncdf(-d2) - S * ncdf(-d1)


def hedged_tqqq(mny, skew):
    """持有 TQQQ 時，每月第一個交易日買 1 個月 put（數量 = 3 × TQQQ 部位 / S），每日以 BS 估值；回傳 put 疊加後的 TQQQ 日報酬。
    put 損益以「月初 TQQQ 部位」為分母（月內部位漂移忽略）。"""
    S = c.values; rfa = (q.rf * 252).values
    rv = r.rolling(20).std() * np.sqrt(252)
    iv = (q.vxn / 100).fillna(rv * 1.2).values + skew / 100
    mstart = q.index.to_period("M").to_series(index=q.index)
    mstart = mstart.ne(mstart.shift(1)).values
    nxt = np.r_[np.flatnonzero(mstart)[1:], len(q)]
    starts = np.flatnonzero(mstart)
    ov = np.zeros(len(q))
    for s_i, e_i in zip(starts, nxt):
        K = mny * S[s_i]; T0 = (e_i - s_i) / 252
        prev = bs_put(S[s_i], K, T0, iv[s_i], rfa[s_i])
        for j in range(s_i + 1, min(e_i + 1, len(q))):
            T = max((e_i - j) / 252, 0)
            val = bs_put(S[j], K, T, iv[j] if np.isfinite(iv[j]) else iv[s_i], rfa[j])
            ov[j] += 3 * (val - prev) / S[s_i]
            prev = val
    # 月初部位 1 元 TQQQ 的 put 疊加：把當天 put 損益換算成占 TQQQ 部位的報酬（忽略月內 TQQQ 漂移）
    return pd.Series(r3.values + ov, index=q.index)


for mny in [0.90, 0.85]:
    for skew in [3, 6]:
        rh = hedged_tqqq(mny, skew)
        col = f"tq_put_{mny}_{skew}"
        q[col] = (1 + rh).cumprod()
        add("B", f"PUT：持 TQQQ 時買 {mny:.0%} put、偏斜 +{skew}", base_w, tq_col=col, r3s=rh)

# ---------------- 組合：A + B 各取最溫和的一個
add("A+B", "TBILL50 + DDCTL 30%", dd_control(0.30), off=below.values * 0.5)
add("A+B", "DUALMOM + BOLL 2.5", w_rule(OK0 & ~(bz20 > 2.5)), off=dm.values)

R = pd.DataFrame(rows)
R.to_csv("results_drawdown.csv", index=False, float_format="%.4f")
open("results_drawdown.txt", "w").write("\n".join(out) + "\n")
