"""fib_pullback 的拆解：H1 + EMA 過濾、1.272 停利，依多空與年度拆開看"""
import pandas as pd
from fib_pullback import Data, run_leg

D = Data()
legs = D.legs("H1", 3)
pd.set_option("display.float_format", "{:.3f}".format); pd.set_option("display.width", 200)
for filt in ["ema", "hhhl+ema", "none"]:
    tr = pd.DataFrame([x for lg in legs for x in run_leg(D, lg, filt, "TP127")])
    tr["年"] = pd.to_datetime(tr["t"]).dt.year
    tr["方向"] = tr["s"].map({1: "多", -1: "空"})
    print(f"\n=== H1 {filt} TP127：平均R（筆數）===")
    print(tr.pivot_table(index="lv", columns="方向", values="R", aggfunc=["mean", "count"]))
    print(tr.pivot_table(index="lv", columns="年", values="R", aggfunc="mean"))
