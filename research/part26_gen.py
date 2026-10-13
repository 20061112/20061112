"""第二十六部分（產生訊號）：版本 A 與複雜度階梯，在 2023/6 ~ 2026/10 完整資料上重新產生全部訊號與逐筆出場。
規則、參數一律沿用（part17_balance.gen / final_system.simulate），只把資料換成 data/XAUUSD_M1_2023_2026.csv。
（該檔與 data/XAUUSD_M1_2026.csv 在重疊期間逐分鐘完全相同。）輸出 part26_signals_*.pkl。
"""
import sys
import tpo

_orig = tpo.load_m1
tpo.load_m1 = lambda path="data/XAUUSD_M1_2023_2026.csv": _orig(path)

from part17_balance import gen   # noqa: E402  （必須在換資料之後才 import）

CFGS = {
    "L0": ("touch", 0.0, True),     # IB 觸價、做星期一
    "L1": ("close5", 0.05, True),   # + M5 收盤確認
    "A": ("close5", 0.05, False),   # + 不做星期一（L2~L5 與版本 A 都用這組訊號，只差篩選）
}
for k in sys.argv[1:] or CFGS:
    cf, buf, mon = CFGS[k]
    X = gen(cf, buf, (2, 10), 30, mon)
    X.to_pickle(f"part26_signals_{k}.pkl")
    print(k, len(X), X.day.min(), X.day.max(), flush=True)
