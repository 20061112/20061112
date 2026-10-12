"""第五十二部分：使用者之前專案的 Entropy × Volatility（分支 claude/optimistic-meitner-rgzrz5 的 entropy_vol.py、volfc.py）
套用到 M15 最終版（+ 兩個防護，RR 3；0.02 口；點差 + 0.3 滑價）。

(1) 熵 / 路徑特徵當狀態因子（全部只用進場前的資料）：
    磚塊流（磚 = 0.05 × 日 ATR10，最近 60 塊）：H_dir 方向熵、H_trans 轉移熵、MI 互資訊、p_cont 延續率、drift 淨方向（× 交易方向）
    固定時間流（M1）：PE 排列熵（60 根）、J60 跳躍占比、ER60 效率比、sER60（× 方向）、V_seas 去季節化相對波動、V_roll 相對 24 小時波動
    方法同第五十一部分：訓練期五分位 → 驗證期同方向、PF 比兩段 ≥ 1.25 → 天區塊自助。
(2) 波動預測 σ̂（har_rel，未來 4 小時 = 16 根 M15，逐月 walk-forward）取代 ATR15 當停損尺度：
    SIG：停損 = k·σ̂（k 用訓練期交易校準，使中位停損 = 原本 2·ATR15 的中位數），RR 3，整條流程重跑（上限 8 → 影子 → 防護）
    VT ：停損不變（2·ATR15），口數 ∝ 中位 σ̂ / σ̂（0.5~2 倍），讓每筆「預期波動風險」一致
"""
import numpy as np
import pandas as pd
import part49_safeguards as P
import entropy_vol as EV
from volfc import har_rel

rng = np.random.default_rng(5)
W, IDX = P.W, P.IDX
lines = []


def p(s=""):
    print(s, flush=True)
    lines.append(s)


def pf(r):
    gp, gl = r[r > 0].sum(), -r[r < 0].sum()
    return gp / gl if gl > 0 else np.nan


def prep(x):
    x = x.copy()
    x["date"] = x.進場時間.dt.normalize()
    x["train"] = x.進場時間 < "2025-01-01"
    return x.reset_index(drop=True)


def guards(x):
    return P.daily_stop(x[~(x.volrel > 2.2).values].reset_index(drop=True), 15)


T = prep(guards(P.run(3)))

# ---------------- (1) 熵特徵 ----------------
d = W.B.d.copy()
dl = W.B.daily
atr10 = (dl.H - dl.L).rolling(10).mean().shift(1)
atr_by_day = {pd.Timestamp(k): v for k, v in atr10.items()}
print("建立磚塊…", flush=True)
Bk = EV.build_bricks(d, atr_by_day)
BF = EV.brick_features(Bk, 60)
print("時間流特徵…", flush=True)
TF = EV.time_features(d)
i = T.i.values
dirv = T.dir.values
tf = TF.iloc[np.maximum(i - 1, 0)].reset_index(drop=True)
bt = BF.t.values
kb = np.searchsorted(bt, T.進場時間.values, side="left") - 1
bf = BF.iloc[np.maximum(kb, 0)].reset_index(drop=True)
F = pd.DataFrame({
    "H_dir 方向熵": bf.H_dir.values, "H_trans 轉移熵": bf.H_trans.values, "MI 互資訊": bf.MI.values,
    "p_cont 延續率": bf.p_cont.values, "drift 淨方向(dir)": bf.drift.values * dirv,
    "PE 排列熵60": tf.PE.values, "J60 跳躍占比": tf.J60.values, "ER60": tf.ER60.values,
    "sER60(dir)": tf.sER60.values * dirv, "V_seas": tf.V_seas.values, "V_roll": tf.V_roll.values})
F.loc[kb < 0, :] = np.nan

tr_m = T.train.values
R = T.R.values
days = T.date.values
p("第五十二部分：使用者的 Entropy × Volatility 套用到 M15 最終版（+ 防護，RR 3）")
p(f"\n=== (1) 熵 / 路徑特徵：五分位 PF（訓練切點）；★ = 好壞五分位同方向、PF 比兩段 ≥ 1.25 ===  整體 PF 訓練 {pf(R[tr_m]):.2f} 驗證 {pf(R[~tr_m]):.2f}")
ud = np.unique(days[~tr_m]); didx = {dd: np.where(days == dd)[0] for dd in np.unique(days)}
for f in F.columns:
    v = F[f].values.astype(float)
    vt = v[tr_m & np.isfinite(v)]
    qs = np.unique(np.quantile(vt, [0.2, 0.4, 0.6, 0.8]))
    q = np.digitize(v, qs); q[~np.isfinite(v)] = -1
    nb = len(qs) + 1
    pt = [pf(R[tr_m & (q == b_)]) for b_ in range(nb)]
    pv = [pf(R[~tr_m & (q == b_)]) for b_ in range(nb)]
    hi_b, lo_b = (nb - 1, 0) if pt[-1] > pt[0] else (0, nb - 1)
    rt, rv = pt[hi_b] / pt[lo_b], pv[hi_b] / pv[lo_b]
    ok = rt >= 1.25 and rv >= 1.25
    extra = ""
    if ok:
        diffs = []
        for _ in range(1000):
            ix = np.concatenate([didx[dd] for dd in rng.choice(ud, len(ud), replace=True)])
            a_, b_ = pf(R[ix][q[ix] == hi_b]), pf(R[ix][q[ix] == lo_b])
            if np.isfinite(a_) and np.isfinite(b_):
                diffs.append(a_ - b_)
        extra = f"  驗證 PF 差 90% 區間 [{np.quantile(diffs, .05):+.2f}, {np.quantile(diffs, .95):+.2f}]"
    p(f"{'★' if ok else ' '} {f:16s} 訓練 " + " ".join(f"{x:4.2f}" for x in pt) + " | 驗證 " + " ".join(f"{x:4.2f}" for x in pv)
      + f" | 好/壞比 {rt:4.2f} / {rv:4.2f}（{'大' if hi_b == nb - 1 else '小'}較好）{extra}")

# ---------------- (2) σ̂ 取代 ATR15 ----------------
print("波動預測 σ̂…", flush=True)
vf, step = har_rel("data/XAUUSD_M15_2023_2026.csv", 16)
sig = vf.sig
# 進場所在 M15 的前一根已收完的 K 棒
bar_open = lambda ts: ts.floor("15min")
ev = P.EV
ev_t = pd.DatetimeIndex([IDX[e[0]] for e in ev])
prev_bar = bar_open(ev_t) - pd.Timedelta(minutes=15)
sig_ev = sig.reindex(prev_bar).to_numpy()
atr15_ev = np.array([e[3] for e in ev])
trm_ev = ev_t < "2025-01-01"
ok_ev = np.isfinite(sig_ev) & (sig_ev > 0)
k_sig = np.median(2 * atr15_ev[trm_ev & ok_ev]) / np.median(sig_ev[trm_ev & ok_ev])
p(f"\n=== (2) 波動預測 σ̂（未來 4 小時）取代 ATR15 ===")
p(f"  σ̂ 可用事件 {ok_ev.mean():.0%}；校準 k = {k_sig:.2f}（停損 = k·σ̂，訓練期中位停損與 2·ATR15 相同）；"
  f"σ̂ 與 ATR15 的 log 相關 {np.corrcoef(np.log(sig_ev[ok_ev]), np.log(atr15_ev[ok_ev]))[0, 1]:.2f}")


def run_with_risk(risk_fn):
    rows = []
    for n_, (i_, L, side, a_, vr) in enumerate(ev):
        risk = risk_fn(n_, a_)
        if not np.isfinite(risk) or risk <= 0:
            continue
        r = P.G.sim_detail(i_, L, side, L - side * risk, L + side * 3 * risk)
        if r is None:
            continue
        j, px_, why = r
        net = (px_ - L) * side - P.SP[i_] - P.SLIP
        rows.append((IDX[i_], IDX[j], i_, j, side, L, net, net / risk, risk, vr, sig_ev[n_]))
    x = pd.DataFrame(rows, columns=["進場時間", "出場時間", "i", "j", "dir", "entry", "net", "R", "risk", "volrel", "sig"])
    x = P.cap_total(x.sort_values("進場時間").reset_index(drop=True), 8).reset_index(drop=True)
    ex = x.sort_values("出場時間"); cum = ex.R.cumsum().values
    ma = pd.Series(cum).rolling(50, min_periods=50).mean().values; et = ex.出場時間.values
    keep = [not (k >= 50 and cum[k] < ma[k]) for k in (np.searchsorted(et, tt, side="right") - 1 for tt in x.進場時間.values)]
    return prep(guards(x[np.array(keep)].reset_index(drop=True)))


def report(x, lab, w=None):
    w = np.ones(len(x)) if w is None else w
    pnl = x.net.values * 2 * w
    xx = x.copy(); xx["net"] = x.net * w
    mm = P.minute_mdd(xx)
    yrs = (x.進場時間.max() - x.進場時間.min()).days / 365
    trm = x.train.values

    def rat(m):
        s_ = pd.Series(pnl[m]).groupby(x.date.values[m]).sum(); e = s_.cumsum(); return pnl[m].sum() / (e.cummax() - e).max()
    # 風險穩定度：每筆「實際停損美元 × 口數」的 60 天變異係數
    risk_usd = pd.Series(x.risk.values * 2 * w, index=x.進場時間).rolling("60D").mean()
    yr = " / ".join(f"{pnl[(x.進場時間.dt.year == y).values].sum():+,.0f}" for y in (2023, 2024, 2025, 2026))
    p(f"  {lab:34s} 筆 {len(x):5d} 每筆R {x.R.mean():+.3f} 停損率 {(x.R < -0.9).mean():.0%} | 總 {pnl.sum():+8,.0f} MDD含浮虧 {mm:6,.0f} "
      f"年獲利/MDD {pnl.sum() / yrs / mm:4.2f} 訓練 {rat(trm):3.1f} 驗證 {rat(~trm):3.1f} | 每筆風險$ CV {risk_usd.std() / risk_usd.mean():.2f} | {yr}")


base = T.copy()
base["sig"] = pd.Series(sig_ev, index=ev_t).reindex(base.進場時間).values if False else np.nan
report(base, "原本（2·ATR15）")
sig_series = pd.Series(sig_ev, index=range(len(ev)))
T_sig = run_with_risk(lambda n_, a_: k_sig * sig_ev[n_] if ok_ev[n_] else np.nan)
report(T_sig, f"SIG：停損 = {k_sig:.2f}·σ̂")
for m in (0.75, 1.25):
    report(run_with_risk(lambda n_, a_, m=m: m * k_sig * sig_ev[n_] if ok_ev[n_] else np.nan), f"SIG ×{m}（停損寬度 {m} 倍）")
# VT：原停損、口數依 σ̂
sb = pd.Series(sig_ev, index=ev_t)
sb = sb[~sb.index.duplicated()]
s_at = sb.reindex(T.進場時間).to_numpy()
med = np.nanmedian(s_at[T.train.values])
wv = np.clip(med / s_at, 0.5, 2.0); wv[~np.isfinite(wv)] = 1.0
report(T, "VT：原停損，口數 ∝ 中位σ̂/σ̂", wv)
wa = np.clip(np.nanmedian(T.risk[T.train]) / T.risk.values, 0.5, 2.0)
report(T, "對照：原停損，口數 ∝ 中位停損/停損（固定美元風險）", wa)
open("results_part52.txt", "w").write("\n".join(lines) + "\n")
