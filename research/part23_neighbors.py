"""第二十三部分 D 的第一名（M15 倫敦：反向且實體 1.5~2 倍 → 反做、持有 6 根）鄰近參數檢查。"""
import pandas as pd
from load import load
from part23_intraday import build, trade, line, SPLIT

m1 = load("data/XAUUSD_M1_2023_2026.csv")
for name, rule, mins in (("M15", "15min", 15), ("M30", "30min", 30)):
    f = build(m1, rule, mins)
    print(f"\n## {name}  反向（這根與前一根不同色）→ 反這根做（= 回到前一根方向），倫敦，扣點差，不重疊")
    for lo, hi in ((1, 1.5), (1.25, 2), (1.5, 2), (1.5, 3), (2, 3), (1.5, 99)):
        for h in (3, 6):
            c = (f.body_raw >= lo) & (f.body_raw < hi) & ~f.same & (f.sess == "倫敦")
            T = trade(f, c, -1, h, mins)
            print(f"  倍數 {lo}-{hi:<4} 持有{h}: 發現 {line(T[T.index < SPLIT])[:62]} | 驗證 {line(T[T.index >= SPLIT])[:62]}")
    for s in ("亞洲", "紐約"):
        c = (f.body_raw >= 1.5) & (f.body_raw < 2) & ~f.same & (f.sess == s)
        T = trade(f, c, -1, 6, mins)
        print(f"  {s} 倍數 1.5-2 持有6: 發現 {line(T[T.index < SPLIT])[:62]} | 驗證 {line(T[T.index >= SPLIT])[:62]}")
