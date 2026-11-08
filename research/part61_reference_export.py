"""第六十一部分：重現用參考資料（給另一個環境逐步對照）。
輸出到 repro/：
  checkpoints.txt                 每一步的數量與關鍵數值（資料、K 棒、事件、每層筆數、熵門檻、績效）
  m15_events_ABS.csv              A+B+S 篩選後的事件與指標（進場分鐘、點位、ATR15、日線值、熵變化）
  m15_trades_slip0.csv / _slip03  M15 最終版實際下單（含 R、出場原因）
  ib_trades_slip0.csv / _slip03   IB 核心 + 2R 保本
  daily_pnl.csv                   每日損益（IB 1 份、M15 2 份；兩種成本）
"""
import os
import numpy as np
import pandas as pd
import part58_robust_opt as R8

Q = R8.Q
P, G, W, B, CI = Q.P, Q.G, Q.W, Q.B, Q.CI
IDX = Q.IDX
OUT = "repro"
os.makedirs(OUT, exist_ok=True)
L = []


def p(s=""):
    print(s, flush=True)
    L.append(s)


# ---------- 資料 ----------
d = W.B.d
p("== 1. 資料（M15 策略用）==")
p(f"M1 根數 {len(d)}（保留 M1 ≥ 600 根的交易日）、交易日 {d.day.nunique()}、第一根 {d.index[0]}、最後一根 {d.index[-1]}")
dl = W.B.daily
p(f"日線列數 {len(dl)}；前 3 天 O/H/L/C：")
for k, r in dl.head(3).iterrows():
    p(f"  {k.date()} {r.O:.2f} {r.H:.2f} {r.L:.2f} {r.C:.2f}")
ds = P.ds
for day in ("2023-08-24", "2024-06-03", "2025-03-03", "2026-06-01"):
    r = ds.loc[pd.Timestamp(day)]
    p(f"  日線指標 {day}: 前收 {r.pc:.2f} SMA50(前日) {r.ma:.3f} SMA50(11天前) {r.ma10:.3f} ATR10日(前日, H−L 平均) {r.atr:.3f} 今開 {r.o:.2f} ATR比 {r.volrel:.4f}")

# ---------- M15 K 棒 ----------
pid = B.period_ids(15)
g = pd.DataFrame({"pid": pid, "t": IDX, "H": W.H, "L": W.Lo, "C": W.C, "O": W.O1})
bars = g.groupby("pid").agg(t=("t", "first"), O=("O", "first"), H=("H", "max"), L=("L", "min"), C=("C", "last"))
p(f"\n== 2. M15 K 棒（每日 01:00 起切 15 分；跨日連續編號）== 共 {len(bars)} 根")
for k in (0, 1, 1000, 50000):
    r = bars.iloc[k]
    p(f"  第 {k} 根 {r.t} O {r.O:.2f} H {r.H:.2f} L {r.L:.2f} C {r.C:.2f}")

# ---------- 事件 ----------
E = Q.EVS[15]
p(f"\n== 3. 事件 == M15 原始突破事件（2023-08-24 起、日線指標齊全）{len(E)}；A {E.A.sum()}、B {E.B.sum()}、S {E.S.sum()}；A+B+S {(E.A & E.B & E.S).sum()}")
p(f"   L0（未篩）事件總數（part56 L0 對照，含 2023-08-24 前）{len(G.ev)}")
EA = E[E.A & E.B & E.S].copy()
EA["進場時間"] = IDX[EA.i.values]
EA["日"] = W.dayv[EA.i.values]
for c_ in ("pc", "ma", "ma10", "atr", "o"):
    EA["日線_" + c_] = ds.loc[EA.日].get(c_).values
EA[["進場時間", "side", "L", "a", "日線_pc", "日線_ma", "日線_ma10", "日線_atr", "日線_o", "volrel", "dent"]].rename(
    columns={"side": "方向", "L": "點位", "a": "ATR15", "volrel": "ATR比", "dent": "熵變化"}).round(5).to_csv(
    f"{OUT}/m15_events_ABS.csv", index=False, encoding="utf-8-sig")
p(f"熵門檻 CUT（訓練期 80% 分位，只扣點差的流程算出）= {Q.CUT[15]:.6f}")

# ---------- 每層筆數 ----------
p("\n== 4. 每層筆數（只扣點差 / 扣 0.3）==")
for slip in (0.0, 0.3):
    acc, row = [], []
    for f in ["A", "B", "S", "CAP", "EQ", "VOL", "ENT", "DS"]:
        acc = acc + [f]
        x = Q.pipeline(15, acc, slip=slip)
        row.append(f"+{f} {len(x)}")
    p(f"  滑價 {slip}: " + "，".join(row))


# ---------- 交易明細與績效 ----------
def stats(lab, items):
    daily = pd.concat([(x.net * q).groupby(x.進場時間.dt.normalize()).sum() for x, q in items], axis=1).fillna(0).sum(1).sort_index()
    eq = Q.minute_eq([Q.mpart(x, q) for x, q in items])
    mm = (np.maximum.accumulate(eq) - eq).max()
    e = daily.cumsum(); md = (e.cummax() - e).max()
    yrs = (daily.index.max() - daily.index.min()).days / 365
    yr = " / ".join(f"{daily[daily.index.year == y].sum():+,.2f}" for y in (2023, 2024, 2025, 2026))
    p(f"  {lab:28s} 總 {daily.sum():+,.2f} | MDD 逐分鐘含浮虧 {mm:,.2f} | 日結 MDD {md:,.2f} | 年獲利/MDD {daily.sum() / yrs / mm:.3f} | 逐年 {yr}")
    return daily


p("\n== 5. 最終結果（份 = 0.01 口 = 1 盎司；IB 從 2023-08-24 起，與 M15 對齊）==")
dd = {}
for slip, tag in ((0.0, "slip0"), (0.3, "slip03")):
    M = R8.m15(slip)
    I = R8.ib(slip)
    p(f"-- 滑價 {slip} --  M15 {len(M)} 筆（勝率 {(M.net > 0).mean():.4f}，每筆 {M.net.mean():.4f} $/oz）；IB {len(I)} 筆（勝率 {(I.net > 0).mean():.4f}，每筆 {I.net.mean():.4f} $/oz）")
    M.assign(方向=M.dir).rename(columns={"entry": "進場價", "net": "淨損益_每盎司", "risk": "停損距離"})[
        ["進場時間", "出場時間", "方向", "進場價", "停損距離", "淨損益_每盎司", "R", "volrel", "dent"]].round(5).to_csv(
        f"{OUT}/m15_trades_{tag}.csv", index=False, encoding="utf-8-sig")
    I.rename(columns={"side": "方向", "lvl": "進場價", "net": "淨損益_每盎司", "risk": "停損距離", "s0": "IB起點分鐘"})[
        ["進場時間", "出場時間", "IB起點分鐘", "方向", "ibh", "ibl", "atr", "peff", "進場價", "停損距離", "淨損益_每盎司", "R", "why"]].round(5).to_csv(
        f"{OUT}/ib_trades_{tag}.csv", index=False, encoding="utf-8-sig")
    dd[f"M15_2份_{tag}"] = stats(f"M15 2 份", [(M, 2)])
    dd[f"IB_1份_{tag}"] = stats(f"IB(2R保本) 1 份", [(I, 1)])
    stats("IB 1 + M15 2", [(I, 1), (M, 2)])
    qi = 1 / I.risk.values; qi = qi / qi.mean()
    qm = 1 / M.risk.values; qm = 2 * qm / qm.mean()
    stats("固定風險（∝1/停損）", [(I, qi), (M, qm)])
pd.DataFrame(dd).fillna(0).round(4).to_csv(f"{OUT}/daily_pnl.csv", encoding="utf-8-sig")
open(f"{OUT}/checkpoints.txt", "w").write("\n".join(L) + "\n")
