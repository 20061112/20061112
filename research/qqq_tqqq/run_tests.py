"""把各分支指標接到 QQQ↔TQQQ 策略上，逐一比較。

四種接法：
  AND      在原本「MA200 之上且 ATR z < 0.5」之外再加一個條件（進場、出場都看）
  ENTRY    只在進場時多看一個條件，出場仍照原規則
  VOL      取代 ATR z（保留 MA200）
  TREND    取代 MA200（保留 ATR z）
每個指標跑一個事先定好的小網格（大多 3×3），看「整片網格」有沒有勝過原策略，而不是挑最好的那格。
輸出 results_grid.csv、results_summary.txt。
"""
import itertools
import sys
import numpy as np
import pandas as pd
from engine import load, base_signals, weights_from_rules, run, stats, rolling_windows, calendar_returns, atr_z
import indicators as ind

q = load()
OK0, Z = base_signals(q)
ABOVE = q.close > q.close.rolling(200).mean()
ZOK = Z < 0.5
r, c = q.r, q.close


def w_rule(ok, exit_ok=None):
    if exit_ok is None:
        exit_ok = ok
    return weights_from_rules(q, ok, ~exit_ok)


def w_cont(target):
    """連續目標權重：往上只在週初調、往下每天調（與原策略「上車慢、下車快」一致）。"""
    t = pd.Series(target, index=q.index).shift(1).fillna(0).clip(0, 1).values
    ws = q.week_start.values
    w = np.zeros(len(t)); cur = 0.0
    for i in range(len(t)):
        if t[i] < cur:
            cur = t[i]
        elif ws[i]:
            cur = t[i]
        w[i] = cur
    return w


def evaluate(w, base_roll=None, base_cal=None, band=0.1):
    full = stats(run(q, w, rebalance_band=band))
    s10 = stats(run(q, w, start="2010-02-12", rebalance_band=band))
    roll = rolling_windows(q, w, rebalance_band=band)
    cal, daily = calendar_returns(q, w)
    eq = (1 + daily).cumprod()
    mdd = (eq / eq.cummax() - 1).min()
    sh = daily.mean() / daily.std() * np.sqrt(252)
    out = dict(irr99=full["irr"], irr10=s10["irr"], worst=full["worst"], sw=full["sw_per_year"],
               lev=1 + 2 * np.nanmean(w), g10_mean=roll.gain.mean(), g10_min=roll.gain.min(),
               irr10y_med=roll.irr.median(), sharpe=sh, mdd_lump=mdd)
    if base_roll is not None:
        d = roll.irr.values - base_roll.irr.values
        out["win10"] = int((d > 0).sum()); out["n10"] = len(d); out["d_irr10y_med"] = np.median(d)
        out["d_irr10y_min"] = d.min()
        dc = cal.values - base_cal.values
        out["yrs_better"] = int((dc > 0).sum()); out["worst_year_vs_base"] = dc.min()
        out["worst_year"] = cal.min()
    return out, roll, cal


def T(x):
    """把 NaN 條件視為通過（例如 VXN 在 2001/2 前沒有資料）。"""
    return x.fillna(True).astype(bool) if x.dtype != bool else x


def variants():
    V = []
    # 1. ER
    for n, a in itertools.product([20, 60, 120], [0.0, 0.1, 0.2]):
        e = ind.er(c, n)
        V.append(("ER", f"n={n},>{a}", "AND", T(e > a)))
        V.append(("ER", f"n={n},>{a}", "TREND", T(e > a)))
    # 2. 排列熵（越低越有序）
    for win, th in itertools.product([20, 60, 120], [-0.5, 0.0, 0.5]):
        pz = ind.rz(ind.perm_entropy(c, 3, win))
        V.append(("PermEntropy", f"win={win},z<{th}", "AND", T(pz < th)))
    # 3. 三狀態方向熵 / 轉移互資訊
    for W in [20, 60, 120]:
        HD, I1 = ind.dir_entropy(r, W)
        hz = ind.rz(HD)
        for th in [-0.5, 0.0, 0.5]:
            V.append(("DirEntropy", f"W={W},z<{th}", "AND", T(hz < th)))
        for th in [0.0, 0.02, 0.05]:
            V.append(("TransMI", f"W={W},I1<{th}", "AND", T(I1 < th)))
    # 4. 跳躍占比（低 = 平滑的波動）
    for win, th in itertools.product([20, 60, 120], [0.0, 0.5, 1.0]):
        jz = ind.rz(ind.jump_ratio(r, win))
        V.append(("JumpRatio", f"win={win},z<{th}", "AND", T(jz < th)))
    # 5. HAR 波動預測：取代 ATR z
    hv = ind.har_vol(q)
    hvz = ind.rz(np.log(hv))
    for th in [0.0, 0.5, 1.0]:
        V.append(("HAR", f"z<{th}", "VOL", T(hvz < th)))
        V.append(("HAR", f"z<{th}", "AND", T(hvz < th)))
    for th in [0.20, 0.25, 0.30]:
        V.append(("HAR", f"σ<{th:.0%}", "VOL", T(hv < th)))
        V.append(("HAR", f"σ<{th:.0%}", "AND", T(hv < th)))
    # 6. VXN
    lv = np.log(q.vxn)
    vz = ind.rz(lv)
    rv20 = r.rolling(20).std() * np.sqrt(252) * 100
    for th in [0.0, 0.5, 1.0]:
        V.append(("VXN", f"z<{th}", "VOL", T(vz < th)))
        V.append(("VXN", f"z<{th}", "AND", T(vz < th)))
    for th in [20, 25, 30]:
        V.append(("VXN", f"<{th}", "VOL", T(q.vxn < th)))
        V.append(("VXN", f"<{th}", "AND", T(q.vxn < th)))
    for th in [0, 2, 4]:
        V.append(("VRP", f"VXN−RV20>{th}", "AND", T((q.vxn - rv20) > th)))
    # VXN 5 日變化：恐慌升溫就先下車
    for th in [0.10, 0.15, 0.20]:
        V.append(("VXNjump", f"5d<+{th:.0%}", "AND", T(lv.diff(5) < np.log(1 + th))))
    # 7. CUSUM
    for k, h in itertools.product([0.25, 0.5], [4.0, 6.0, 8.0]):
        cs = ind.cusum_state(r, k, h)
        V.append(("CUSUM", f"k={k},h={h}", "AND", cs > 0))
        V.append(("CUSUM", f"k={k},h={h}", "TREND", cs > 0))
    # 8. 變異數比
    for qn, win in itertools.product([5, 10], [120, 250]):
        vr = ind.var_ratio(r, qn, win)
        for th in [0.9, 1.0, 1.1]:
            V.append(("VarRatio", f"q={qn},w={win},>{th}", "AND", T(vr > th)))
    # 9. Hawkes 下跌衝擊強度
    for tau, th in itertools.product([5, 10, 20], [0.3, 0.6, 1.0]):
        hk = ind.hawkes(r, 2.0, tau)
        V.append(("Hawkes", f"τ={tau},λ<{th}", "AND", hk < th))
    # 10. OU 偏離（太偏離平衡點不開槓桿）
    for L in [120, 250]:
        oz = ind.ou_z(c, L)
        for th in [1.5, 2.0, 3.0]:
            V.append(("OU", f"L={L},z<{th}", "AND", T(oz < th)))
    # 11. TD setup 9（完成後 N 天內不開槓桿）
    for n in [5, 10, 20]:
        td = ind.td_setup(c)
        V.append(("TD9", f"{n}d", "AND", td > n))
        V.append(("TD9", f"{n}d", "ENTRY", td > n))
    # 12. 時間序列動能
    for n in [126, 189, 252]:
        m = ind.tsmom(c, n)
        V.append(("TSMOM", f"{n}d>0", "TREND", T(m > 0)))
        V.append(("TSMOM", f"{n}d>0", "AND", T(m > 0)))
    return V


def build_w(role, cond):
    if role == "AND":
        return w_rule(OK0 & cond)
    if role == "ENTRY":
        return w_rule(OK0 & cond, OK0)
    if role == "VOL":
        return w_rule(ABOVE & cond)
    if role == "TREND":
        return w_rule(cond & ZOK)
    raise ValueError(role)


if __name__ == "__main__":
    w0 = w_rule(OK0)
    b, broll, bcal = evaluate(w0)
    b, broll, bcal = evaluate(w0, broll, bcal)
    rows = [dict(ind="BASE", param="MA200 & ATRz<0.5", role="-") | b]
    for name, p, role, cond in variants():
        w = build_w(role, cond)
        o, _, _ = evaluate(w, broll, bcal)
        rows.append(dict(ind=name, param=p, role=role) | o)
        print(f"{name:12s} {role:5s} {p:22s} irr99 {o['irr99']:.3f} irr10 {o['irr10']:.3f} "
              f"win10 {o['win10']}/{o['n10']} dIRR10y {o['d_irr10y_med']:+.3f} worst {o['worst']:.2f} "
              f"g10min {o['g10_min']:.2f} sh {o['sharpe']:.2f} mdd {o['mdd_lump']:.2f}", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv("results_grid.csv", index=False, float_format="%.4f")
    # 每個「指標 × 接法」整片網格的彙總
    G = R[R.ind != "BASE"].groupby(["ind", "role"]).agg(
        n=("irr99", "size"), irr99_med=("irr99", "median"), irr10_med=("irr10", "median"),
        share_irr99_up=("irr99", lambda s: (s > b["irr99"]).mean()),
        win10_med=("win10", "median"), d_irr10y_med=("d_irr10y_med", "median"),
        worst_med=("worst", "median"), g10min_med=("g10_min", "median"),
        sharpe_med=("sharpe", "median"), mdd_med=("mdd_lump", "median"),
        worst_year_med=("worst_year", "median")).reset_index()
    G = G.sort_values("d_irr10y_med", ascending=False)
    with open("results_summary.txt", "w") as f:
        f.write("BASE: " + ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in b.items()) + "\n\n")
        f.write(G.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(open("results_summary.txt").read())
