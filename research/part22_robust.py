"""第二十二部分 F：β 框架的穩健性。

1. 隨機對照：每一段內，把前瞻報酬 f 以「整天」為單位循環平移（保留日內結構與自相關，切斷和特徵的對應），
   重算「Δβ 三段同向且最弱段 ≥ 0.02」的量測數，200 次。
2. 跨週期：同一個物理時間的均值（≈ 100 分鐘 EMA：M1 EMA100 / M5 EMA20 / M15 EMA7；
   ≈ 4 小時：M1 EMA250 / M5 EMA50 / M15 EMA17），前瞻 = 60 分鐘。
3. 不同前瞻長度（M5：6 / 12 / 24 / 48 根）。
"""
import numpy as np
import pandas as pd
from part22_lib import load_tf, means, atr, ema
from part22_beta import frame, state_beta, FE
from part22_restore import NAMES

out = open("results_part22.txt", "a")
def P(s=""): print(s, flush=True); out.write(s + "\n"); out.flush()
rng = np.random.default_rng(11)
KEY = ["dist", "sig_ratio", "bbw_chg", "bbw_pct", "er10", "er20", "er_ratio", "er_leg", "atr_ratio", "vel", "acc",
       "osc_c", "htf", "day_used", "day_der", "vwap_side"]


def dbetas(X, feats):
    r = {}
    for f in feats:
        t = state_beta(X, f)
        r[f] = [t[p][-1] - t[p][0] for p in ("IS1", "IS2", "OOS")]
    return r


def n_pass(db, thr=0.02):
    return sum(1 for v in db.values() if (np.sign(v[0]) == np.sign(v[1]) == np.sign(v[2])) and min(map(abs, v)) >= thr)


def null_test(mn, n_perm=200):
    d = load_tf("M5"); A = atr(d, 14); M = means(d)
    X = frame(d, A, M, mn)
    real = n_pass(dbetas(X, FE))
    days = X.index.normalize()
    cnt = []
    for _ in range(n_perm):
        Z = X.copy()
        for p in ("IS1", "IS2", "OOS"):
            m = (Z.per == p).to_numpy()
            ud = np.unique(days[m])
            k = rng.integers(5, len(ud) - 5)
            newday = pd.Series(np.roll(ud, k), index=ud).reindex(days[m]).to_numpy()
            # 每一天的 f 換成 k 天後那一天同一時刻的 f（找不到 → NaN）
            src = pd.Series(Z.f.to_numpy()[m], index=Z.index[m])
            newidx = Z.index[m] - days[m] + pd.DatetimeIndex(newday)
            Z.loc[m, "f"] = src.reindex(newidx).to_numpy()
        cnt.append(n_pass(dbetas(Z, FE)))
    cnt = np.array(cnt)
    line = (f"  {mn:10s} 實際 {real:2d} 個 / {len(FE)} | 隨機平均 {cnt.mean():.2f}、95% 分位 {np.quantile(cnt, .95):.0f}、"
            f"99% 分位 {np.quantile(cnt, .99):.0f}、≥ 實際的比例 {np.mean(cnt >= real):.3f}")
    open(f"part22_null_{mn}.txt", "w").write(line + "\n")


NULL_MEANS = ("ema20", "ema50", "kama", "rvwap20", "linreg20")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2 and sys.argv[1] == "null":       # 平行跑：python3 part22_robust.py null ema20
        null_test(sys.argv[2]); sys.exit()
    P("\n" + "=" * 120)
    P("第二十二部分 F：β 框架穩健性")
    d = load_tf("M5"); A = atr(d, 14); M = means(d)
    P("\n1. 隨機對照（前瞻報酬以整天為單位在段內循環平移，200 次；由 `python3 part22_robust.py null <均值>` 平行產生）")
    for mn in NULL_MEANS:
        P(open(f"part22_null_{mn}.txt").read().rstrip())
    P("\n2. 跨週期（前瞻 60 分鐘）：Δβ = β(Q5) − β(Q1)，IS1 / IS2 / OOS")
    specs = [("M1", 100, 60), ("M5", 20, 12), ("M15", 7, 4), ("M1", 250, 60), ("M5", 50, 12), ("M15", 17, 4)]
    res = {}
    cache = {"M5": (d, A)}
    for tf, n, H in specs:
        if tf not in cache:
            dd = load_tf(tf); cache[tf] = (dd, atr(dd, 14))
        dd, AA = cache[tf]
        MM = pd.DataFrame({"x": ema(dd.close, n)})
        res[(tf, n)] = dbetas(frame(dd, AA, MM, "x", H=H), KEY)
    P(f"  {'':12s}" + "".join(f"{tf + ' EMA' + str(n):>22s}" for tf, n, _ in specs))
    for f in KEY:
        P(f"  {NAMES[f]:12s}" + "".join("   " + " ".join(f"{v:+.2f}" for v in res[(tf, n)][f]) for tf, n, _ in specs))
    P("  六個設定中 Δβ 三段同向的次數（同向且最弱段 ≥ 0.02 的次數）、方向：")
    for f in KEY:
        sg = [np.sign(res[k][f][0]) for k in res if np.sign(res[k][f][0]) == np.sign(res[k][f][1]) == np.sign(res[k][f][2])]
        strong = sum(1 for k in res if np.sign(res[k][f][0]) == np.sign(res[k][f][1]) == np.sign(res[k][f][2]) and min(map(abs, res[k][f])) >= .02)
        P(f"    {NAMES[f]:12s} 同向 {len(sg)}/6（強 {strong}）  方向 {'+' if sum(sg) > 0 else '−' if sum(sg) < 0 else '?'}{int(abs(sum(sg)))}")

    P("\n3. M5 EMA20 / EMA50 不同前瞻長度：Δβ 最弱段（帶號，三段不同向記 0）")
    for mn in ("ema20", "ema50"):
        P(f"  {mn}: " + "".join(f"{'H=' + str(H):>10s}" for H in (6, 12, 24, 48)))
        rr = {H: dbetas(frame(d, A, M, mn, H=H), KEY) for H in (6, 12, 24, 48)}
        for f in KEY:
            cells = []
            for H in (6, 12, 24, 48):
                v = rr[H][f]
                cells.append(np.sign(v[0]) * min(map(abs, v)) if np.sign(v[0]) == np.sign(v[1]) == np.sign(v[2]) else 0.0)
            P(f"    {NAMES[f]:12s}" + "".join(f"{c:+10.3f}" for c in cells))
