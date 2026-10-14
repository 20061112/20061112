"""第三十一部分：H1 六合一 —— 最大停損與其他風險出場規則。
輸出：python3 part31_risk.py > results_part31_risk.txt

進場同第二十八 / 三十部分（不加 SMA20 過濾）。停利測 2R（原本）與 3R（第三十部分研究期選出）。
單筆層級：
  C  最大停損：停損距離上限 k ATR（k = 1.5 / 2 / 2.5 / 3 / 4）
       C-縮  停損拉近到 k ATR（R 變小、部位變大）
       C-略  停損距離 > k ATR 就不做
  TS 時間停損：進場後 N 根（N = 4 / 8 / 12）收盤仍虧損就出場
帳戶層級（套在六合一的進場順序上，被擋掉的單不會讓出名額給後面的訊號）：
  DL 當日已實現虧損 <= −0.5 / −1 組合 R → 當天不再進場
  ML 當月已實現虧損 <= −1.5 / −2 組合 R → 當月不再進場
挑選：只看研究期（2023~24）的「組合 R / 最大回撤」，驗證期照搬。
"""
import warnings
import numpy as np
warnings.filterwarnings("ignore", category=RuntimeWarning)
import pandas as pd
from part22_divergence import get_frames, nonoverlap, SPLIT
from part30_exits import entries, ex_fixed, summarize, line, HEAD, SIX, W


def ex_timestop(Y, ei, ep, stop, tp, N, hold=32):
    o, h, l, c, spr = Y["o"], Y["h"], Y["l"], Y["c"], Y["spr"]
    n = len(c)
    R = ep - stop
    tgt = ep + tp * R
    for j in range(ei, min(ei + hold + 1, n)):
        if l[j] <= stop:
            xp = stop if j == ei else min(stop, o[j])
            return j, (xp - ep - spr[ei]) / R, "stop"
        if j > ei and h[j] >= tgt:
            return j, (tgt - ep - spr[ei]) / R, "tp"
        if j == ei + N and c[j] < ep:
            return j, (c[j] - ep - spr[ei]) / R, "tstop"
    j = min(ei + hold, n - 1)
    return j, (c[j] - ep - spr[ei]) / R, "time"


def run(idx, rows, tp=2.0, cap=None, cap_mode="tight", tstop=None):
    out = []
    for r in rows:
        Y, ei, ep, stop = r["Y"], r["ei"], r["ep"], r["stop"]
        A = Y["atr"][ei - 1]
        if cap is not None and ep - stop > cap * A:
            if cap_mode == "skip":
                continue
            stop = ep - cap * A
        if tstop:
            xi, pr, why = ex_timestop(Y, ei, ep, stop, tp, tstop)
        else:
            xi, pr, why = ex_fixed(Y, ei, ep, stop, tp, 32)
        out.append(dict(cond=r["cond"], side=r["side"], entry_time=idx[ei], exit_time=idx[xi], pnl_R=pr, exit=why,
                        R_atr=(ep - stop) / A))
    T = pd.DataFrame(out)
    return pd.concat([nonoverlap(T[T.cond == k]) for k in SIX]).sort_values("entry_time").reset_index(drop=True)


def account_limit(T, day_lim=None, mon_lim=None):
    """依進場順序，當日 / 當月已實現（出場時間早於本筆進場）的組合虧損超過上限就不再進場。"""
    keep, acc = [], []
    for i, r in T.iterrows():
        t = r.entry_time
        ok = True
        if acc:
            A = T.loc[acc]
            done = A[A.exit_time <= t]
            if day_lim is not None:
                d = done[done.exit_time.dt.normalize() == t.normalize()]
                ok &= (d.pnl_R * W).sum() > -day_lim
            if mon_lim is not None:
                m = done[done.exit_time.dt.to_period("M") == t.to_period("M")]
                ok &= (m.pnl_R * W).sum() > -mon_lim
        if ok:
            keep.append(i)
            acc.append(i)
    return T.loc[keep]


def main():
    idx, rows = entries(get_frames()["H1"])
    res = []
    print("H1 六合一，最大停損 / 風險出場規則。欄位同第三十部分（組合 R、回撤、逐年、空單、2026/7 都是組合 R）")
    for tp in (2.0, 3.0):
        print(f"\n{'=' * 120}\n## 停利 {tp:.0f}R")
        print(HEAD)
        base = run(idx, rows, tp)
        s = summarize(base); res.append((f"{tp:.0f}R 基準", s)); print(line("基準", s))
        print(f"  （基準停損距離：中位數 {base.R_atr.median():.2f} ATR，> 2 ATR 的比例 {(base.R_atr > 2).mean():.0%}、> 3 ATR {(base.R_atr > 3).mean():.0%}、> 4 ATR {(base.R_atr > 4).mean():.0%}）")
        for mode, mn in (("tight", "縮"), ("skip", "略")):
            for k in (1.5, 2.0, 2.5, 3.0, 4.0):
                s = summarize(run(idx, rows, tp, k, mode))
                res.append((f"{tp:.0f}R C-{mn}{k}", s)); print(line(f"C-{mn} {k} ATR", s))
        for N in (4, 8, 12):
            s = summarize(run(idx, rows, tp, tstop=N))
            res.append((f"{tp:.0f}R TS{N}", s)); print(line(f"TS {N} 根仍虧", s))
        for dl in (0.5, 1.0):
            s = summarize(account_limit(base, day_lim=dl))
            res.append((f"{tp:.0f}R DL{dl}", s)); print(line(f"DL 日虧 {dl}R 停", s))
        for ml in (1.5, 2.0):
            s = summarize(account_limit(base, mon_lim=ml))
            res.append((f"{tp:.0f}R ML{ml}", s)); print(line(f"ML 月虧 {ml}R 停", s))

    # 依研究期挑選
    print(f"\n{'=' * 120}\n## 只用研究期挑選（研究期 組合R / 回撤）")
    sel = []
    for name, s in res:
        T = None
        sel.append((name, s))
    # 研究期的組合 R / 回撤需要另外算
    tab = []
    for tp in (2.0, 3.0):
        variants = {f"{tp:.0f}R 基準": run(idx, rows, tp)}
        for mode, mn in (("tight", "縮"), ("skip", "略")):
            for k in (1.5, 2.0, 2.5, 3.0, 4.0):
                variants[f"{tp:.0f}R C-{mn}{k}"] = run(idx, rows, tp, k, mode)
        for N in (4, 8, 12):
            variants[f"{tp:.0f}R TS{N}"] = run(idx, rows, tp, tstop=N)
        b = variants[f"{tp:.0f}R 基準"]
        for dl in (0.5, 1.0):
            variants[f"{tp:.0f}R DL{dl}"] = account_limit(b, day_lim=dl)
        for ml in (1.5, 2.0):
            variants[f"{tp:.0f}R ML{ml}"] = account_limit(b, mon_lim=ml)
        for name, T in variants.items():
            for per, m in (("a", T.entry_time < SPLIT), ("b", T.entry_time >= SPLIT)):
                x = T[m]
                c = (x.pnl_R * W).groupby(x.exit_time.dt.normalize()).sum().sort_index().cumsum()
                dd = (c.cummax() - c).max()
                tab.append(dict(name=name, per=per, tot=c.iloc[-1], dd=dd, ratio=c.iloc[-1] / dd))
    S = pd.DataFrame(tab).pivot(index="name", columns="per", values=["tot", "dd", "ratio"])
    S.columns = [f"{a}_{b}" for a, b in S.columns]
    S = S.sort_values("ratio_a", ascending=False)
    print(f"  {'規則':14s} {'研究 R':>7s} {'回撤':>5s} {'R/撤':>5s} | {'驗證 R':>7s} {'回撤':>5s} {'R/撤':>5s}")
    for nm, r in S.iterrows():
        print(f"  {nm:14s} {r.tot_a:+7.1f} {r.dd_a:5.1f} {r.ratio_a:5.1f} | {r.tot_b:+7.1f} {r.dd_b:5.1f} {r.ratio_b:5.1f}")
    ok = np.isfinite(S.ratio_a) & np.isfinite(S.ratio_b)
    print(f"\n  研究期 R/撤 與 驗證期 R/撤 的相關係數 {S.ratio_a[ok].corr(S.ratio_b[ok]):+.2f}（{ok.sum()} 組）")


if __name__ == "__main__":
    main()
