"""第五十九部分：版本 A 的「大時框 ATR」（日線 ATR10）改成波動預測 σ̂，結果會怎樣？

版本 A 用 ATR10（前 10 個交易日 true range 平均）做三件事，都是「尺度」：
  1. 突破緩衝：M5 收盤 > IB 高 + 0.05 × ATR
  2. F1 小 IB 篩選：IB / ATR < 門檻（且順前日方向）就不做
  3. F2 反向尾巴：tail / ATR > 門檻就不做
（停損 = IB 另一側，與 ATR 無關；門檻每月用過去資料的分位數重算，所以尺度常數不影響門檻本身。）
替換版本（其他規則、門檻算法完全不動）：
  ATR10   原版
  SIG_D   當天開盤（01:00）預測「今天一整天」的 σ̂（har_rel，h = 276 根 M5），× c 換成日 true range 單位。
          c = 之前所有日子的 平均日 TR / 平均 σ̂_day（逐日 expanding），等於一個「往前看的日線 ATR」
  SIG_R   每個 IB 結束時預測「從現在到收盤」剩下時間的 σ̂（= 持倉期間的預期波動），× 同一個 c。
          IB 越晚、剩下時間越短，σ̂ 越小；紐約數據前會預先放大
σ̂ 一律由 data/XAUUSD_M5_2025_2026.csv 計算（har_rel 暖機 2 個月，2025/7 起有樣本外預測；c 再用 20 天）。
版本 A 跑兩份資料：2026 原始 M1（樣本內）、M5 2025~2026 展開的偽 M1（2025/8~12 為真正樣本外，同第五十三部分）。
執行：python3 part59_A_sigma.py
"""
import os
import pickle
import subprocess
import sys
import numpy as np
import pandas as pd

PKL = "data/pseudo/part59_sigma.pkl"
RES = "results_part59.txt"
M5 = "data/XAUUSD_M5_2025_2026.csv"
DAY_H = 1


# ------------------------------------------------------------------ 主程序：先算好 σ̂
def prepare():
    from volfc import har_rel
    from part55_vol_verify import prep, tod_factor
    bd, _ = har_rel(M5, 276, warm=2)              # 一整天（01:00~23:55 = 276 根）
    b48, _ = har_rel(M5, 48, warm=2)
    level = (b48.sig2 / b48.futf)                 # 每根「單位時段因子」的變異數水準
    raw, step = prep(M5)
    f, key = tod_factor(raw, step)
    fr = pd.DataFrame(dict(f=f, key=key), index=raw.index)
    fr["day"] = (fr.index - pd.Timedelta(hours=DAY_H)).normalize()
    prof = {d: g.groupby("key").f.last().reindex(range(288)).fillna(1.0).to_numpy() for d, g in fr.groupby("day")}
    days = sorted(prof)
    prof_prev = {d: prof[days[i - 1]] for i, d in enumerate(days) if i > 0}   # 用前一天各時段的因子（事前可知）
    # 每天開盤前最後一根的 σ̂_day（美元）
    sday = (np.sqrt(bd.sig2) * bd.close)
    sday.index = sday.index + pd.Timedelta("5min")             # 該根收盤時間
    sig_day = {}
    for d in days:
        st = d + pd.Timedelta(hours=DAY_H)
        s = sday[:st]
        if len(s) and (st - s.index[-1]) < pd.Timedelta("3D") and np.isfinite(s.iloc[-1]):
            sig_day[d] = float(s.iloc[-1])
    # 換算常數 c：用 M5 自己的日 true range（之前日子的 平均 TR / 平均 σ̂_day，expanding，至少 20 天）
    dd = raw.copy()
    dd["day"] = (dd.index - pd.Timedelta(hours=DAY_H)).normalize()
    g = dd.groupby("day").agg(high=("high", "max"), low=("low", "min"), close=("close", "last"))
    tr = np.maximum(g.high, g.close.shift()) - np.minimum(g.low, g.close.shift())
    sd = pd.Series(sig_day).reindex(g.index)
    okd = sd.notna() & tr.notna()
    c_day = (tr.where(okd).shift(1).expanding(20).mean() / sd.where(okd).shift(1).expanding(20).mean()).to_dict()
    lvl = level.copy()
    lvl.index = lvl.index + pd.Timedelta("5min")
    close = b48.close.copy()
    close.index = close.index + pd.Timedelta("5min")
    os.makedirs("data/pseudo", exist_ok=True)
    with open(PKL, "wb") as fh:
        pickle.dump(dict(sig_day=sig_day, c_day=c_day, level=lvl, close=close, prof_prev=prof_prev), fh)


# ------------------------------------------------------------------ worker
def worker(variant, label):
    import part14_deep as P14
    import part17_balance as P17
    from part15_letters import features as basic_features
    from part16_adv_letters import adv
    from final_system import simulate
    from final_A import walk_forward, CFG
    S = pickle.load(open(PKL, "rb"))
    D, DAYS, ARR = P14.D, P14.DAYS, P14.ARR
    sd = pd.Series({d: S["sig_day"].get(d, np.nan) for d in D.index})
    c = pd.Series({d: S["c_day"].get(d, np.nan) for d in D.index})       # 在主程序用 M5 日 TR 估計（只用之前日子）
    lvl, close, prof_prev = S["level"], S["close"], S["prof_prev"]

    def atr_for(di, day, s0):
        if variant == "ATR10":
            return D.atr10.iloc[di]
        cc = c.iloc[di]
        if not np.isfinite(cc):
            return np.nan
        if variant == "SIG_D":
            return cc * sd.iloc[di]
        # SIG_R：IB 結束時刻 → 收盤（23:55 那根結束）剩下時段的預期波動
        te = day + pd.Timedelta(minutes=s0 + 60)
        if day not in prof_prev:
            return np.nan
        L = lvl[:te]
        if not len(L) or (te - L.index[-1]) > pd.Timedelta("30min") or not np.isfinite(L.iloc[-1]):
            return np.nan
        k0 = (s0 + 60) // 5
        rem = prof_prev[day][k0:288].sum()
        return cc * np.sqrt(L.iloc[-1] * rem) * close[:te].iloc[-1]

    confirm, buf, hours, step, monday = CFG["confirm"], CFG["buf"], CFG["hours"], CFG["step"], CFG["monday"]
    ib_min, win = 60, 3
    out = []
    for di, day in enumerate(DAYS):
        if di == 0 or (day.dayofweek == 0 and not monday):
            continue
        if not np.isfinite(D.atr10.iloc[di]):
            continue
        prev = D.iloc[di - 1]; pdir = np.sign(prev.close - prev.open)
        t, O, H, L, C, Sp = ARR[day]
        mins = (t - day).total_seconds().values // 60
        for s0 in range(hours[0] * 60, hours[1] * 60 + 60, step):
            atr = atr_for(di, day, s0)
            if not np.isfinite(atr) or atr <= 0:
                continue
            m = (mins >= s0) & (mins < s0 + ib_min)
            if m.sum() < ib_min * 0.8:
                continue
            ibh, ibl = H[m].max(), L[m].min()
            up_l, dn_l = ibh + buf * atr, ibl - buf * atr
            w = np.where((mins >= s0 + ib_min) & (mins < s0 + ib_min + win * 60))[0]
            w = w[(mins[w] + 1) % 5 == 0]
            up = w[C[w] > up_l]; dn = w[C[w] < dn_l]
            if len(up) and (not len(dn) or up[0] < dn[0]): side, i = 1, up[0]
            elif len(dn): side, i = -1, dn[0]
            else: continue
            lvl_px, i_ex = C[i], min(i + 1, len(C) - 1)
            out.append(dict(day=day, h=s0 / 60, i=i_ex, side=side, lvl=lvl_px, ibh=ibh, ibl=ibl, rng=ibh - ibl, atr=atr,
                            ib_atr=(ibh - ibl) / atr, aligned=bool(side == pdir), t_in=t[i], spread=Sp[i],
                            atr10=D.atr10.iloc[di]))
    s = pd.DataFrame(out)
    s["tail"] = [basic_features(r, 15)["tail"] for r in s.itertuples()]
    s["ext_letter"] = [adv(r).get("ext_letter", np.nan) for r in s.itertuples()]
    X = simulate(s)
    T, used = walk_forward(X)
    T.to_csv(f"data/pseudo/part59_{label}_{variant}.csv", index=False)
    print(f"[{label} {variant}] 原始訊號 {len(X)}、通過篩選 {len(T)}；尺度中位數 {s.atr.median():.1f}（ATR10 中位 {s.atr10.median():.1f}）")


def summary(T, lab, days_all):
    if not len(T):
        return f"  {lab:24s} 無交易"
    p = T.pnl; w, l = p[p > 0], p[p <= 0]
    eq = T.sort_values("exit_time").pnl.cumsum(); dd = (eq - eq.cummax()).min()
    days = [d for d in days_all if T.day.min() <= d <= T.day.max() and d.dayofweek != 0]
    daily = pd.Series(0.0, index=days).add(T.groupby("day").pnl.sum(), fill_value=0)
    Rd = pd.Series(0.0, index=days).add(T.groupby("day").R.sum(), fill_value=0)
    return (f"  {lab:24s} {len(T):4d}筆 勝率 {len(w) / len(p):5.1%} 每筆 {p.mean():+6.2f}（{T.R.mean():+.3f}R） PF {w.sum() / max(-l.sum(), 1e-9):.2f} "
            f"總 {p.sum():+7.0f} 回撤 {dd:+6.0f} 日Sharpe {daily.mean() / daily.std() * np.sqrt(250):+.2f} | R 日Sharpe {Rd.mean() / Rd.std() * np.sqrt(250):+.2f} "
            f"總R {T.R.sum():+6.1f}")


if __name__ == "__main__":
    if len(sys.argv) > 3 and sys.argv[1] == "--worker":
        worker(sys.argv[2], sys.argv[3]); raise SystemExit
    from part53_A_oos import write_tab, expand, m1_to_m5
    from load import load
    os.makedirs("data/pseudo", exist_ok=True)
    p2 = "data/pseudo/pseudo_from_m5_2025_2026.csv"
    if not os.path.exists(p2):
        write_tab(expand(load(M5)), p2)
    prepare()
    jobs = []
    for label, path in (("M1_2026", None), ("M5_2025_2026", p2)):
        for v in ("ATR10", "SIG_D", "SIG_R"):
            env = dict(os.environ)
            if path:
                env["XAU_M1_PATH"] = path
            jobs.append((label, v, subprocess.Popen([sys.executable, __file__, "--worker", v, label], env=env,
                                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)))
    logs = []
    for label, v, pr_ in jobs:
        o, e = pr_.communicate()
        if pr_.returncode:
            print(e); raise SystemExit(1)
        logs.append(o.strip())
    out = open(RES, "w")
    def P(s=""): print(s); out.write(s + "\n")
    P("第五十九部分：版本 A 的日線 ATR10 → 波動預測 σ̂（SIG_D = 今天整天的預測；SIG_R = IB 結束到收盤的預測）")
    P("其他規則與門檻算法不變（每月用過去資料的分位數重算）；單位 美元/盎司；每筆扣點差")
    for s in logs:
        P(s)
    for label in ("M1_2026", "M5_2025_2026"):
        P(f"\n=== {label} ===")
        Ts = {v: pd.read_csv(f"data/pseudo/part59_{label}_{v}.csv", parse_dates=["day", "t_in", "entry_time", "exit_time"]) for v in ("ATR10", "SIG_D", "SIG_R")}
        alld = sorted(set(pd.concat([t.day for t in Ts.values()])))
        allbd = list(pd.bdate_range(min(alld), max(alld)))
        start = max(t.day.min() for t in Ts.values())          # 三個版本共同的期間
        for v, T in Ts.items():
            T = T[T.day >= start]
            P(summary(T, f"{v} 全部", allbd))
            for y, g in T.groupby(T.day.dt.year):
                P(summary(g, f"{v} {y}", allbd))
        # 交易重疊度
        k = {v: set(zip(T.t_in.astype(str), T.side)) for v, T in Ts.items()}
        P(f"  與原版相同的交易：SIG_D {len(k['SIG_D'] & k['ATR10'])}/{len(k['SIG_D'])}、SIG_R {len(k['SIG_R'] & k['ATR10'])}/{len(k['SIG_R'])}（原版 {len(k['ATR10'])}）")
    out.close()
