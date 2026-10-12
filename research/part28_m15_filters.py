"""第二十八部分：M15 前高/前低突破（停損 2 ATR15、停利 3 倍）—— 過濾低品質交易。

交易：part27_m15_trades_s2_t3.csv（67,937 筆，每天 79 筆）。
做法：只用 **2023/6 ~ 2024/12（訓練）** 看每個特徵的五分位 / 分組，挑出方向清楚的條件；
     **2025/1 ~ 2026/10（驗證）** 只拿來確認，不調整門檻。單位以 R（已扣點差）為主，並列美元/盎司。
特徵（全部在進場當下已知）：
  hour        進場小時（broker）
  trend50     順日線趨勢：做多且前日收盤在 50 日均線上 / 做空且在下方
  ema_al      M15 收盤相對 EMA50(M15) 與方向一致
  day_move    今天開盤到進場價的漲跌（ATRd，乘上方向；+ = 順今天方向）
  day_er      今天到進場為止的效率比（|現價−開盤| / 逐分鐘路徑長）
  day_pos     進場價在今天到目前為止區間的位置（乘方向；1 = 突破今天新高/新低）
  prev_rng    前一根 M15 波幅 / ATR15（小 = 盤整後突破）
  prev_col    前一根 M15 顏色與方向一致
  atr_ratio   ATR15 / 日 ATR（M15 相對波動）
  touch_min   在這根 M15 的第幾分鐘碰到
  nth         今天同方向第幾筆
  h4_al       進場價在前一根 H4 高點之上（做多）/ 低點之下（做空）
"""
import numpy as np
import pandas as pd
import part24_prev_bar_breakout as B

OUT = "results_part28.txt"
d = B.d
H, Lo, C, O1 = B.H, B.Lo, B.C, d.open.values
IDX = d.index
T = pd.read_csv("part27_m15_trades_s2_t3.csv", parse_dates=["進場時間", "出場時間"])
T["i"] = IDX.get_indexer(T.進場時間)
assert (T.i >= 0).all()
T["dir"] = np.where(T.方向 == "多", 1, -1)
T["usd"] = T.淨利_每盎司
T["day"] = B.dayv[T.i.values]

# ---- M15 / H4 K 棒 ----
def bars(P):
    pid = B.period_ids(P)
    g = pd.DataFrame({"pid": pid, "H": H, "L": Lo, "O": O1, "C": C})
    a = g.groupby("pid").agg(H=("H", "max"), L=("L", "min"), O=("O", "first"), C=("C", "last"))
    return pid, a

pid15, b15 = bars(15)
atr15 = (b15.H - b15.L).rolling(20).mean().shift(1)
ema50 = b15.C.ewm(span=50, adjust=False).mean().shift(1)  # 前一根為止
prev = b15.shift(1)
k = pid15[T.i.values]
T["prev_rng"] = ((prev.H - prev.L) / atr15).values[k]
T["prev_col"] = (np.sign(prev.C - prev.O).values[k] == T.dir).astype(int)
T["ema_al"] = (np.sign(prev.C - ema50).values[k] == T.dir).astype(int)
pid4, b4 = bars(240)
p4 = b4.shift(1)
k4 = pid4[T.i.values]
T["h4_al"] = np.where(T.dir > 0, T.進場價 > p4.H.values[k4], T.進場價 < p4.L.values[k4]).astype(int)

# ---- 日內 ----
daily = B.daily
ATRd = (daily.H - daily.L).rolling(10).mean().shift(1)
first = pd.Series(np.arange(len(d))).groupby(B.dayv).transform("first").values
cum_path = np.concatenate([[0], np.cumsum(np.abs(np.diff(C)))])
hi_sofar = pd.Series(H).groupby(B.dayv).cummax().values
lo_sofar = pd.Series(Lo).groupby(B.dayv).cummin().values
i = T.i.values
f0 = first[i]
dopen = O1[f0]
atrd = T.day.map(ATRd).values
T["atr_ratio"] = atr15.values[k] / atrd
T["day_move"] = (T.進場價 - dopen) * T.dir / atrd
path_len = cum_path[i] - cum_path[f0] + 1e-9
T["day_er"] = np.abs(C[np.maximum(i - 1, f0)] - dopen) / path_len
hs, ls = hi_sofar[np.maximum(i - 1, f0)], lo_sofar[np.maximum(i - 1, f0)]
pos = (T.進場價 - ls) / np.maximum(hs - ls, 1e-9)
T["day_pos"] = np.where(T.dir > 0, pos, 1 - pos)
T["touch_min"] = IDX[i].minute % 15
T["hour"] = IDX[i].hour
T["nth"] = T.groupby(["day", "dir"]).cumcount() + 1
T["trend50"] = np.where(T.前日在50日均線上.isna(), np.nan, ((T.dir > 0) == (T.前日在50日均線上 == True)).astype(float))
T["train"] = T.進場時間 < "2025-01-01"
lines = []


def p(s=""):
    print(s)
    lines.append(s)


def cell(x):
    return f"{x.R.mean():+.3f}R/{x.usd.mean():+.2f}$ ({len(x)})"


p("第二十八部分：M15 突破（停損 2 ATR15、停利 3 倍）過濾低品質交易")
p("格式：每筆 R / 每筆美元(每盎司) (筆數)；訓練 = 2023/6~2024/12，驗證 = 2025/1~2026/10\n")
tr, te = T[T.train], T[~T.train]
p(f"全部：訓練 {cell(tr)}  驗證 {cell(te)}\n")

p("=== 1. 連續特徵：訓練期五分位（Q1 小 → Q5 大），驗證期用訓練期的切點 ===")
for f in ["day_move", "day_er", "day_pos", "prev_rng", "atr_ratio", "touch_min", "nth"]:
    qs = np.unique(np.nanquantile(tr[f], [0.2, 0.4, 0.6, 0.8]))
    bins = [-np.inf, *qs, np.inf]
    a = tr.groupby(pd.cut(tr[f], bins), observed=True).R.mean()
    b = te.groupby(pd.cut(te[f], bins), observed=True).R.mean()
    p(f"  {f:10s} 切點 {np.round(qs, 2)}")
    p(f"    訓練 " + "  ".join(f"{v:+.3f}" for v in a.values))
    p(f"    驗證 " + "  ".join(f"{v:+.3f}" for v in b.values))

p("\n=== 2. 分組特徵 ===")
for f, lab in [("trend50", "順 50 日均線"), ("ema_al", "順 M15 EMA50"), ("prev_col", "前一根同色"), ("h4_al", "已突破前一根 H4")]:
    for v in (1, 0):
        p(f"  {lab}={'是' if v else '否'}：訓練 {cell(tr[tr[f] == v])}  驗證 {cell(te[te[f] == v])}")
p("  時段：")
for a_, b_ in [(1, 4), (4, 8), (8, 11), (11, 15), (15, 18), (18, 22)]:
    m1, m2 = tr.hour.between(a_, b_ - 1), te.hour.between(a_, b_ - 1)
    p(f"    {a_:02d}-{b_:02d} 點：訓練 {cell(tr[m1])}  驗證 {cell(te[m2])}")
for dname, dv in (("多", 1), ("空", -1)):
    p(f"  方向 {dname}：訓練 {cell(tr[tr.dir == dv])}  驗證 {cell(te[te.dir == dv])}")
T.to_pickle("part28_features.pkl")  # 給 part28_combo.py 用（不進版控）
open(OUT, "w").write("\n".join(lines) + "\n")
