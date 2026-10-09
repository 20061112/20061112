"""第十五部分 C：把字母指標（tail、lvl_poc）加到 ②+，並與 M5 收盤 + 緩衝確認合併。門檻都取 1~5 月分位數。"""
import numpy as np, pandas as pd
from part14_deep import exits, both, SPLIT
from part15_ib_params import signals
from part15_letters import features

out = open("results_part15.txt", "a")
def P(x=""): print(x); out.write(x + "\n"); out.flush()
rng_ = np.random.default_rng(1)


def add(X, p=15):
    F = pd.DataFrame([features(r, p) for r in X.itertuples()], index=X.index)
    return pd.concat([X, F], axis=1)


def rand_test(X, keep):
    res = []
    for lab, m in [("IS", X.day < SPLIT), ("OOS", X.day >= SPLIT)]:
        x, k = X[m], keep[m]
        real = x.pnl[k].mean()
        sims = np.array([x.pnl.values[rng_.choice(len(x), int(k.sum()), replace=False)].mean() for _ in range(2000)])
        res.append(f"{lab} 勝過隨機 {(sims < real).mean() * 100:.1f}%")
    return "；".join(res)


P("\n" + "=" * 110 + "\n第十五部分 C：組合（字母長度 15 分；門檻 = 1~5 月分位數）")
for lab, kw in [("②+ 觸價", dict()), ("②+ M5收盤+0.05ATR緩衝", dict(confirm="close5", buf=0.05))]:
    s, _ = signals(60, 60, **kw)
    X = add(exits(s).reset_index(drop=True))
    IS = X[X.day < SPLIT]
    t80 = IS["tail"].quantile(0.8); lp40 = IS.lvl_poc.quantile(0.4); lp20 = IS.lvl_poc.quantile(0.2)
    rules = {
        "不篩": X.pnl == X.pnl,
        f"tail ≤ {t80:.3f}ATR（去尾巴最長 20%）": X["tail"] <= t80,
        f"lvl_poc ≥ {lp20:.3f}ATR（去最靠近 POC 20%）": X.lvl_poc >= lp20,
        f"lvl_poc ≥ {lp40:.3f}ATR（去最靠近 POC 40%）": X.lvl_poc >= lp40,
        "tail + lvl_poc(20%)": (X["tail"] <= t80) & (X.lvl_poc >= lp20),
        "tail + lvl_poc(40%)": (X["tail"] <= t80) & (X.lvl_poc >= lp40),
    }
    P(f"\n[{lab}]")
    for name, m in rules.items():
        P(f"  {name:34s} {both(X[m])}")
        if name != "不篩":
            P(f"  {'':34s} 被刪掉的：{both(X[~m])}")
            P(f"  {'':34s} {rand_test(X, m)}")
    X.to_csv(f"part15_combo_{'close5' if kw else 'touch'}.csv", index=False)
out.close()
