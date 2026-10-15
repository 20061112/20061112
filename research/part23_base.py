"""第二十三部分對照：同樣 1S/2R，不篩選、障礙低（≤30%）、障礙中間（30~70%）vs 障礙高（≥90%），range c1 h1。輸出 results_part23_base.txt"""
import numpy as np
import pandas as pd
import part23_trade as T

T.lines.clear()
for tf in T.TFS:
    E = pd.read_pickle(f"part23_events_M{tf}.pkl")
    E["year"] = E.time_close.dt.year
    E["half"] = np.where(E.time_close < "2025-01-01", "A", "B")
    A = (E.half == "A").to_numpy()
    x = E["range_c1.0_h1_sh"].to_numpy(float)
    q = lambda p: np.nanquantile(x[A], p)
    T.P(f"\nM{tf}（停損 1S、停利 2R）")
    T.summ(T.simulate(E, np.ones(len(E), bool), b=1.0), "全部邊緣事件")
    T.summ(T.simulate(E, x <= q(0.3), b=1.0), "障礙 ≤ 30%")
    T.summ(T.simulate(E, (x > q(0.3)) & (x < q(0.7)), b=1.0), "障礙 30~70%")
    T.summ(T.simulate(E, x >= q(0.9), b=1.0), "障礙 ≥ 90%")
open("results_part23_base.txt", "w").write("\n".join(T.lines) + "\n")
