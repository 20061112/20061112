"""第二十四部分補充：箱體邊緣「守住」的效果是不是量測方式造成的？輸出 results_part24_touch.txt。
同一批 M15 5 根箱體（2023~2026），比較三種定義下「先突破上緣的比例 − 隨機漫步基準」（按現價在箱體中的位置分組）。"""
import numpy as np
import pandas as pd
from load import load


def test(c, boxhi, boxlo, cross_hi, cross_lo, L=5, F=48, step=3):
    rows = []
    for t0 in range(200, len(c) - F - 1, step):
        hi, lo = boxhi[t0 - L + 1:t0 + 1].max(), boxlo[t0 - L + 1:t0 + 1].min()
        c0 = c[t0]
        if not (hi > lo and lo < c0 < hi):
            continue
        fu = np.flatnonzero(cross_hi[t0 + 1:t0 + 1 + F] > hi)
        fd = np.flatnonzero(cross_lo[t0 + 1:t0 + 1 + F] < lo)
        a = fu[0] if len(fu) else 10 ** 6
        b = fd[0] if len(fd) else 10 ** 6
        if a == b:
            continue
        rows.append((float(a < b), (c0 - lo) / (hi - lo)))
    X = pd.DataFrame(rows, columns=["up", "exp"])
    X["ex"] = X.up - X.exp
    return X.groupby(pd.cut(X.exp, [0, .2, .4, .6, .8, 1]), observed=True).ex.mean()


if __name__ == "__main__":
    m = load("data/XAUUSD_M15_2023_2026.csv")
    h, l, c = m.high.to_numpy(), m.low.to_numpy(), m.close.to_numpy()
    lines = ["# 箱體邊緣效果 vs 量測方式（M15、5 根箱體；數字 = 先突破上緣比例 − 隨機漫步基準）",
             "位置分組: (0,0.2] (0.2,0.4] (0.4,0.6] (0.6,0.8] (0.8,1]"]
    for name, args in (("影線箱體、收盤穿越（第二十二部分安慰劑）", (h, l, c, c)),
                       ("收盤箱體、收盤穿越", (c, c, c, c)),
                       ("影線箱體、盤中觸及（= 實際停損/停利的成交方式）", (h, l, h, l))):
        r = test(c, *args)
        lines.append(f"{name:28s} " + "  ".join(f"{v:+.3f}" for v in r))
    print("\n".join(lines))
    open("results_part24_touch.txt", "w").write("\n".join(lines) + "\n")
