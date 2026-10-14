"""第二十六部分：有沒有能看出方向的東西？兩個最常被文獻支持的候選，用 2023~2026 檢查。
  A. 時段季節性：每個 broker 小時的平均報酬（黃金常見「亞洲/夜間漲、美盤跌」的說法）
  B. 日線時間序列動能：過去 k 天報酬的方向 → 之後 1 / 5 天報酬
都扣掉同年平均漂移再看（黃金 2023~2026 大漲，不扣的話「做多」什麼都會贏）。
"""
import numpy as np
import pandas as pd
from load import load
from part22_stalin_rank import tstat
from part25_rv import D

pd.set_option("display.width", 220)
m = load("data/XAUUSD_M15_2023_2026.csv")
r = np.log(m.close).diff() * 100
r = r[r.index.to_series().diff() <= pd.Timedelta(hours=1)]      # 去掉週末 / 缺資料跳空
H = pd.DataFrame({"r": r, "hour": r.index.hour, "year": r.index.year})
H = H.groupby([H.index.floor("h"), "hour", "year"]).r.sum().reset_index()   # 每個小時的報酬
H["r_dm"] = H.r - H.groupby("year").r.transform("mean")

print("=== A. 每個 broker 小時的平均報酬（%，扣同年平均）===")
tab = H.pivot_table(index="hour", columns="year", values="r_dm", aggfunc="mean")
tab["全部"] = H.groupby("hour").r_dm.mean()
tab["t"] = H.groupby("hour").r_dm.apply(tstat)
tab["同號年數"] = [(np.sign(tab.loc[h, [2023, 2024, 2025, 2026]]) == np.sign(tab.loc[h, "全部"])).sum() for h in tab.index]
print(tab.round(4).to_string())
tr, te = H[H.year <= 2024], H[H.year >= 2025]
sgn = np.sign(tr.groupby("hour").r_dm.mean())
pnl_te = te.r_dm * te.hour.map(sgn)
print(f"\n訓練 2023~24 決定每小時做多/做空 → 測試 2025~26 每小時平均 {pnl_te.mean():+.4f}%（t {tstat(pnl_te):+.2f}）；"
      f"同期每小時點差約 {(m.spread * 0.01 / m.close * 100).median():.4f}%")
# 只挑訓練期 |t| > 2 的小時
tt = tr.groupby("hour").r_dm.apply(tstat)
pick = tt[tt.abs() > 2].index
p2 = te[te.hour.isin(pick)]
p2v = p2.r_dm * p2.hour.map(sgn)
print(f"只用訓練期 |t|>2 的小時 {list(pick)} → 測試 {p2v.mean():+.4f}%（t {tstat(p2v):+.2f}，{len(p2v)} 筆）")

print("\n=== B. 日線時間序列動能（過去 k 天報酬方向 → 之後報酬，%，扣同年平均）===")
d = D[["ret"]].copy(); d["year"] = d.index.year
d["ret_dm"] = d.ret - d.groupby("year").ret.transform("mean")
rows = []
for k in [1, 5, 20, 60, 120]:
    past = d.ret.rolling(k).sum()
    for h in [1, 5]:
        fwd = d.ret_dm.rolling(h).sum().shift(-h)
        s = (np.sign(past) * fwd).dropna().iloc[::h]
        rows.append({"k": k, "h": h, "平均": s.mean(), "t": tstat(s),
                     **s.groupby(s.index.year).mean().round(3).to_dict()})
print(pd.DataFrame(rows).round(3).to_string(index=False))
print("\n（參考）不扣漂移的每年平均日報酬 %：", d.groupby("year").ret.mean().round(3).to_dict())
