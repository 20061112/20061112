"""第三十部分：A+B 過濾的 M15 突破，限制「最高同時持倉筆數」（多空合計）。

第二十九部分的「同方向一次 1 筆」每天只剩約 3 筆；這裡改成總持倉上限 N = 1 / 2 / 3 / 5 / 8 / 10 / 15 / 20 / 不限，
新訊號出現時若已持有 N 筆就略過。停損 2 ATR15，RR 3 / 4 / 收盤；另列停損 3 ATR15、RR 3。
每單 1 盎司，已扣點差；訓練 2023/6~2024/12、驗證 2025/1~2026/10。需要先跑 part28_m15_filters.py。
"""
import numpy as np
import pandas as pd
import part29_rr_grid as G


def cap_total(df, n):
    if n is None:
        return df
    keep, open_ = [], []
    for r in df.itertuples():
        open_ = [x for x in open_ if x > r.進場時間]
        if len(open_) < n:
            open_.append(r.出場時間)
            keep.append(r.Index)
    return df.loc[keep]


def max_conc(df):
    ev = pd.concat([pd.Series(-1, df.出場時間), pd.Series(1, df.進場時間)]).sort_index(kind="stable")  # 同一分鐘先出後進
    return int(ev.cumsum().max())


evs = [e for e, k in zip(G.ev, G.mA & G.mB) if k]
lines = ["第三十部分：A+B 過濾的 M15 突破 —— 限制最高同時持倉筆數（多空合計，每單 1 盎司）", ""]
hdr = (f"{'上限':>4s} {'筆/天':>6s} {'實際最多':>6s} {'訓練R':>7s} {'驗證R':>7s} {'全期R':>7s} {'每筆$':>7s} {'PF':>5s} "
       f"{'日t':>6s} {'總$':>8s} {'回撤$':>7s} {'總/回撤':>7s} | 逐年 R 2023 2024 2025 2026")
for s, t in ((2.0, 3.0), (2.0, 4.0), (2.0, None), (3.0, 3.0)):
    T = G.simulate(evs, s, t).sort_values("進場時間").reset_index(drop=True)
    lines.append(f"=== 停損 {s:g} ATR15，RR {t:g} ===" if t else f"=== 停損 {s:g} ATR15，抱到收盤 ===")
    lines.append(hdr)
    for n in (1, 2, 3, 5, 8, 10, 15, 20, None):
        x = cap_total(T, n)
        a, b, c = G.st(x[x.train]), G.st(x[~x.train]), G.st(x)
        yr = " ".join(f"{x.R[x.進場時間.dt.year == y].mean():+.3f}" for y in (2023, 2024, 2025, 2026))
        lines.append(f"{str(n) if n else '不限':>4s} {c['n']:6.1f} {max_conc(x):6d} {a['R']:+7.3f} {b['R']:+7.3f} {c['R']:+7.3f} "
                     f"{c['usd']:+7.2f} {c['pf']:5.2f} {c['t']:+6.2f} {c['tot']:+8,.0f} {c['dd']:7,.0f} {c['tot'] / c['dd']:7.1f} | {yr}")
    lines.append("")
print("\n".join(lines))
open("results_part30.txt", "w").write("\n".join(lines) + "\n")
