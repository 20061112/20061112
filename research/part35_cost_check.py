"""第三十五部分：用實際滑價（每筆來回多付 $0.3 / $0.6）重新檢查目前所有「看起來能賺」的候選。

滑價以固定美元加在每筆來回上（模型裡點差在買進時付一次，所以直接加到點差欄位 = 每筆來回多付一次）：
  $0.3 / 來回：使用者估計的滑價；$0.6 / 來回：進場、出場各 $0.3 的保守版。
候選：
  A  亞洲 IB 突破底倉（第二十八部分 BASE：M1 收盤突破 IB ± 0.05 ATR、IB 另一側停損、23:50 平）
  B  亞洲 IB 區內底倉（第三十部分 ZB：IB 高上抱多 / IB 低下抱空、回 IB 一格（0.1 ATR）出、可重進）
  C  H1 通道順勢網格（第三十四部分：KC / BB、K4 / K10、往內 / 回中線）
單位：1 盎司（0.01 手）美元；另列 ÷ ATR 的 t。樣本內 2023~24 / 樣本外 2025~26。
"""
import numpy as np, pandas as pd
from multiprocessing import Pool
import part28_ib_grid as P28
import part29_va_grid as V
import part33_band_grid as B
from part28_ib_grid import t

V.ib_va = lambda ohlc, atr: (ohlc[:, 1].max(), ohlc[:, 2].min())
DAYS0 = [(k, a, m, o, s.copy()) for k, a, m, o, s in P28.DAYS]
SP0 = B.SP.copy()
SLIPS = (0.0, 0.3, 0.6)
BANDS = [("1h", "KC", 4, "inward"), ("1h", "KC", 4, "mean"), ("1h", "KC", 10, "inward"), ("1h", "KC", 10, "mean"),
         ("1h", "BB", 10, "inward"), ("1h", "BB", 4, "inward")]


def job(args):
    name, slip = args
    P28.DAYS[:] = [(k, a, m, o, s + slip) for k, a, m, o, s in DAYS0]
    B.SP = SP0 + slip
    if name == "A":
        _, df = P28.job((60, 0.1, True, None)); return args, df.assign(pnl=df.BASE)[["day", "atr", "pnl"]]
    if name == "B":
        _, df = V.job(("S2", 60, 0.1, "-", None)); return args, df.assign(pnl=df.ZB)[["day", "atr", "pnl"]]
    c = BANDS[int(name[1:])]
    mid, u = B.bands(B.OHLC0, c[0], c[1])
    r = B.sim(B.OHLC0, mid, u, c[2], c[3], -1)
    return args, r.rename(columns={"week": "day"})[["day", "atr", "pnl", "trades"]]


if __name__ == "__main__":
    names = ["A", "B"] + [f"C{i}" for i in range(len(BANDS))]
    with Pool(4) as pool:
        res = dict(pool.map(job, [(n, s) for n in names for s in SLIPS], chunksize=1))
    label = {"A": "A 亞洲 IB 突破底倉（每天）", "B": "B 亞洲 IB 區內底倉（每天）"}
    for i, c in enumerate(BANDS):
        label[f"C{i}"] = f"C H1 {c[1]} {'K4' if c[2] == 4 else 'K10'} {'往內' if c[3] == 'inward' else '回中線'}（每週）"
    out = open("results_part35.txt", "w")
    def P(s=""): print(s); out.write(s + "\n")
    P("第三十五部分：每筆來回多付 $0.3 / $0.6 滑價後的候選檢查（1 盎司 = 0.01 手，美元）")
    rows = []
    for n in names:
        P(f"\n{label[n]}")
        P(f"  {'滑價':>5s} | {'2023':>7s} {'2024':>7s} {'2025':>7s} {'2026':>7s} | {'合計':>7s} | {'t 樣本內':>8s} {'t 樣本外':>8s} | {'PF(段)':>6s} {'最大回撤':>8s}")
        for s in SLIPS:
            df = res[(n, s)]; y = df.day.dt.year
            yr = [df.pnl[y == k].sum() for k in range(2023, 2027)]
            a = df.pnl / df.atr; IS = (y < 2025).values
            eq = df.pnl.cumsum(); dd = (eq - eq.cummax()).min()
            pf = df.pnl[df.pnl > 0].sum() / -df.pnl[df.pnl < 0].sum()
            P(f"  ${s:4.1f} | " + " ".join(f"{v:+7.0f}" for v in yr) + f" | {sum(yr):+7.0f} | {t(a[IS]):+8.2f} {t(a[~IS]):+8.2f} | {pf:6.2f} {dd:+8.0f}")
            rows.append(dict(cand=label[n], slip=s, **{str(k): v for k, v in zip(range(2023, 2027), yr)}, total=sum(yr),
                             t_IS=t(a[IS]), t_OOS=t(a[~IS]), pf=pf, maxdd=dd))
    pd.DataFrame(rows).to_csv("part35_summary.csv", index=False, float_format="%.3f")
    out.close()
