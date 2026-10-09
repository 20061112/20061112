"""第十六部分 B：tail + ext_letter（反方向極值是最近才做出來的 → 不做）+ va_ovl（與前日價值重疊高 → 不做）。
門檻都取 1~5 月分位數：tail 80%、ext_letter 80%、va_ovl 67%。"""
import numpy as np, pandas as pd
from part14_deep import both, SPLIT
out = open("results_part16.txt", "a")
def P(s=""): print(s); out.write(s + "\n"); out.flush()
rng = np.random.default_rng(3)


def beat(X, keep):
    r = []
    for lab, m in [("IS", X.day < SPLIT), ("OOS", X.day >= SPLIT)]:
        x, k = X[m], keep[m]; real = x.pnl[k].mean()
        sims = np.array([x.pnl.values[rng.choice(len(x), int(k.sum()), replace=False)].mean() for _ in range(2000)])
        r.append(f"{lab} 勝過隨機 {(sims < real).mean() * 100:.1f}%")
    return "；".join(r)


P("\n" + "=" * 120 + "\n第十六部分 B：進階字母篩選疊加")
for f in ("touch", "close5"):
    X = pd.read_csv(f"part16_trades_{f}.csv", parse_dates=["day", "t_in", "t_out"])
    IS = X[X.day < SPLIT]
    T = X["tail"] <= IS["tail"].quantile(0.8)
    E = X.ext_letter <= IS.ext_letter.quantile(0.8)
    V = X.va_ovl <= IS.va_ovl.quantile(0.67)
    P(f"\n[{f}]  門檻：tail ≤ {IS['tail'].quantile(.8):.3f}ATR、ext_letter ≤ {IS.ext_letter.quantile(.8):.2f}、va_ovl ≤ {IS.va_ovl.quantile(.67):.2f}")
    for lab, m in [("基準", X.pnl == X.pnl), ("tail", T), ("ext_letter", E), ("va_ovl", V), ("tail + ext_letter", T & E),
                   ("tail + va_ovl", T & V), ("tail + ext_letter + va_ovl", T & E & V)]:
        P(f"  {lab:28s} {both(X[m])}")
        if lab != "基準":
            P(f"  {'':28s} 被刪掉的：{both(X[~m])}  ｜ {beat(X, m)}")
out.close()
