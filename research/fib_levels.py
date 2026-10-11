"""哪個 Fib 位比較好 + 小週期（M1/M5）是否還有效果

每個 Fib 位當成獨立策略：價格回撤到該位才開始等確認（較淺的回撤不理）。
止損 = 下一個 Fib 位（以回撤最深碰到的位為準），停利 1.272 延伸。
"""
import sys
import time
import pandas as pd
from fib_pullback import Data, stats
from fib_confirm import Conf, CONF_TF, run_leg

D = Data()
pd.set_option("display.width", 250); pd.set_option("display.float_format", "{:.3f}".format)
pd.set_option("display.max_rows", 500)
tfs = sys.argv[1].split(",") if len(sys.argv) > 1 else ["M1", "M5", "M15", "H1"]
rows, allt = [], []
for tf in tfs:
    t0 = time.time()
    legs = D.legs(tf, 3)
    C = Conf(D, CONF_TF[tf])
    for filt in ["none", "hhhl+ema"]:
        for mode in ["A", "B"]:
            for arm in [0.382, 0.5, 0.618, 0.786]:
                tr = pd.DataFrame([r for lg in legs if (r := run_leg(D, C, lg, filt, "TP127", mode, arm))])
                tr = tr.sort_values("t")
                tr["TF"], tr["趨勢"], tr["確認"], tr["等待位"] = tf, filt, mode, arm
                allt.append(tr)
                rows.append(dict(TF=tf, 趨勢=filt, 確認=mode, 等待位=arm, **stats(tr),
                                 多R=tr.loc[tr.s == 1, "R"].mean(), 空R=tr.loc[tr.s == -1, "R"].mean(),
                                 平均風險美元=tr["風險"].mean(), 點差佔風險=(tr["點差"] / tr["風險"]).mean()))
    print(f"{tf}: {len(legs)} 段行情, {time.time()-t0:.0f}s", flush=True)
res = pd.DataFrame(rows)
print(res.to_string(index=False))
pd.concat(allt).to_csv("fib_levels_trades.csv", index=False)
