"""第三十二部分：M15 前高/前低突破 —— 最終版（A + B + 50MA 斜率 + 權益濾網，停損 2 ATR15，RR 3，上限 8）。

輸出：
  final_m15_signals.csv   全部影子訊號（含沒下單的），每筆的進場條件數值、權益濾網狀態、是否實際下單
  final_m15_trades.csv    實際下單的交易
  results_final_m15.txt   統計
  final_m15_equity.png    權益曲線
需要先跑 part28_m15_filters.py（part28_features.pkl）。
"""
import numpy as np
import pandas as pd
import part29_rr_grid as G
from part30_max_positions import cap_total

W, IDX = G.W, G.IDX
H, Lo, C, O1, SP, dayv, day_end = W.H, W.Lo, W.C, W.O1, W.SP, W.dayv, W.day_end
S_ATR, RR, NMAX, LOOK = 2.0, 3.0, 8, 50

# ---- M15 K 棒（為了在明細裡列出前一根高低點與 ATR15）----
pid = W.B.period_ids(15)
g = pd.DataFrame({"pid": pid, "H": H, "L": Lo})
b15 = g.groupby("pid").agg(H=("H", "max"), L=("L", "min"))
atr15 = (b15.H - b15.L).rolling(20).mean().shift(1).values
prevH, prevL = b15.H.shift(1).values, b15.L.shift(1).values

# ---- 日線狀態（前一日為止）----
dl = W.B.daily
sma50 = dl.C.rolling(50).mean()
ATRd = (dl.H - dl.L).rolling(10).mean().shift(1)
dstate = pd.DataFrame({"前日收盤": dl.C.shift(1), "前日SMA50": sma50.shift(1), "SMA50_10日前": sma50.shift(11),
                       "ATR日": ATRd, "今日開盤": dl.O})

rows = []
for i, L, side, a in G.ev:
    day = dayv[i]
    s = dstate.loc[day]
    if not np.isfinite(s.前日SMA50) or not np.isfinite(s.SMA50_10日前) or not np.isfinite(s.ATR日):
        continue
    condA = (s.前日收盤 - s.前日SMA50) * side > 0
    slope = (s.前日SMA50 - s.SMA50_10日前) * side > 0
    day_move = (L - s.今日開盤) * side / s.ATR日
    condB = day_move > -0.17
    if not (condA and slope and condB):
        continue
    risk = S_ATR * a
    stop, tgt = L - side * risk, L + side * RR * risk
    r = G.sim_detail(i, L, side, stop, tgt)
    if r is None:
        continue
    j, px, why = r
    net = (px - L) * side - SP[i]
    k = pid[i]
    rows.append({"進場時間": IDX[i], "方向": "多" if side > 0 else "空", "dir": side,
                 "前一根M15高": prevH[k], "前一根M15低": prevL[k], "進場價(=點位)": round(L, 2), "ATR15": round(a, 3),
                 "停損價": round(stop, 2), "停利價": round(tgt, 2), "停損距離": round(risk, 2),
                 "前日收盤": round(s.前日收盤, 2), "前日SMA50": round(s.前日SMA50, 2), "SMA50_10日斜率": round(s.前日SMA50 - s.SMA50_10日前, 2),
                 "今日開盤": round(s.今日開盤, 2), "ATR日": round(s.ATR日, 2), "今日漲跌(ATR日,順方向)": round(day_move, 3),
                 "出場時間": IDX[j], "出場價": round(px, 2), "出場原因": why, "持有分鐘": int((IDX[j] - IDX[i]).total_seconds() // 60),
                 "點差": round(SP[i], 2), "淨利_每盎司": round(net, 2), "R": round(net / risk, 3)})

S = pd.DataFrame(rows).sort_values("進場時間").reset_index(drop=True)
S = cap_total(S, NMAX).reset_index(drop=True)  # 影子訊號也遵守上限 8

# ---- 權益濾網（影子交易）----
ex = S.sort_values("出場時間")
cum = ex.R.cumsum().values
ma = pd.Series(cum).rolling(LOOK, min_periods=LOOK).mean().values
ex_t = ex.出場時間.values
st_cum, st_ma, take = [], [], []
for r in S.itertuples():
    kk = np.searchsorted(ex_t, np.datetime64(r.進場時間), side="right") - 1
    c0 = cum[kk] if kk >= 0 else 0.0
    m0 = ma[kk] if kk >= LOOK else np.nan
    st_cum.append(round(c0, 2)); st_ma.append(round(m0, 2) if np.isfinite(m0) else np.nan)
    take.append(not (kk >= LOOK and c0 < m0))
S["影子累積R"] = st_cum
S[f"影子累積R的{LOOK}筆均線"] = st_ma
S["實際下單"] = take
S.drop(columns=["dir"]).to_csv("final_m15_signals.csv", index=False, encoding="utf-8-sig")
T = S[S.實際下單].copy()
T.drop(columns=["dir", "實際下單"]).to_csv("final_m15_trades.csv", index=False, encoding="utf-8-sig")

# ---- 統計 ----
lines = []


def p(s=""):
    print(s)
    lines.append(s)


def streak(x):
    best = cur = 0
    for v in x:
        cur = cur + 1 if v < 0 else 0
        best = max(best, cur)
    return best


def blk(x, lab):
    d = x.groupby(x.進場時間.dt.date).R.sum()
    e = d.cumsum()
    du = x.groupby(x.進場時間.dt.date).淨利_每盎司.sum()
    eu = du.cumsum()
    w, l = x[x.R > 0], x[x.R <= 0]
    p(f"--- {lab} ---")
    p(f"  筆數 {len(x)}（有交易的日子 {x.進場時間.dt.date.nunique()} 天，平均每天 {len(x) / x.進場時間.dt.date.nunique():.1f} 筆）")
    p(f"  勝率 {(x.R > 0).mean():.1%}；平均賺 {w.R.mean():+.2f}R / 賠 {l.R.mean():+.2f}R；PF {w.R.sum() / -l.R.sum():.2f}；每筆 {x.R.mean():+.3f}R")
    p(f"  每盎司：每筆 {x.淨利_每盎司.mean():+.2f} 美元，平均賺 {w.淨利_每盎司.mean():+.2f} / 賠 {l.淨利_每盎司.mean():+.2f}，總計 {x.淨利_每盎司.sum():+,.0f}")
    p(f"  總 {x.R.sum():+.0f}R，MDD {(e.cummax() - e).max():.0f}R（每單 1 盎司：{(eu.cummax() - eu).max():,.0f} 美元），最長連虧 {streak(x.R.values)} 筆")
    p(f"  出場：停利 {(x.出場原因 == '停利').mean():.0%}、停損 {(x.出場原因 == '停損').mean():.0%}、收盤 {(x.出場原因 == '收盤').mean():.0%}；"
      f"持有時間中位數 {x.持有分鐘.median():.0f} 分（停利單 {x[x.出場原因 == '停利'].持有分鐘.median():.0f} 分、停損單 {x[x.出場原因 == '停損'].持有分鐘.median():.0f} 分）")
    p(f"  停損距離中位數 {x.停損距離.median():.2f} 美元/盎司；點差平均 {x.點差.mean():.2f}（佔 R {(x.點差 / x.停損距離).mean():.3f}）")


p("最終版 M15 突破：A + B + 50MA 斜率 + 權益濾網，停損 2 ATR15，RR 3，上限 8（XAUUSD 2023/6 ~ 2026/10）\n")
p(f"影子訊號 {len(S)} 筆，實際下單 {len(T)} 筆（{len(T) / len(S):.0%}），權益濾網停手時間佔 {1 - S.實際下單.mean():.0%} 的訊號\n")
blk(S, "影子訊號（全部下單時）")
blk(T, "實際下單")
p("\n逐年（實際下單）：")
for y, x in T.groupby(T.進場時間.dt.year):
    d = x.groupby(x.進場時間.dt.date).R.sum(); e = d.cumsum()
    p(f"  {y}: {len(x):5d} 筆  勝率 {(x.R > 0).mean():.0%}  每筆 {x.R.mean():+.3f}R / {x.淨利_每盎司.mean():+.2f}$  總 {x.R.sum():+6.0f}R / {x.淨利_每盎司.sum():+7,.0f}$"
      f"  MDD {(e.cummax() - e).max():.0f}R  多單 {(x.dir > 0).mean():.0%}  停損距離中位 {x.停損距離.median():.1f}")
p("\n多 / 空：")
for dname, x in T.groupby("方向"):
    p(f"  {dname}: {len(x)} 筆 勝率 {(x.R > 0).mean():.0%} 每筆 {x.R.mean():+.3f}R / {x.淨利_每盎司.mean():+.2f}$")
p("\n進場時段（broker 時間 = 紐約 + 7）：")
for a, b in [(1, 4), (4, 8), (8, 11), (11, 15), (15, 18), (18, 22)]:
    x = T[T.進場時間.dt.hour.between(a, b - 1)]
    p(f"  {a:02d}-{b:02d} 點: {len(x):5d} 筆 ({len(x) / len(T):.0%}) 每筆 {x.R.mean():+.3f}R")
m = T.groupby(T.進場時間.dt.to_period("M")).agg(筆數=("R", "size"), R=("R", "sum"), 美元=("淨利_每盎司", "sum"))
p(f"\n逐月（實際下單，正報酬 {(m.R > 0).sum()}/{len(m)} 個月）：")
for k, v in m.iterrows():
    p(f"  {k}: {int(v.筆數):4d} 筆  {v.R:+6.1f}R  {v.美元:+7,.0f}$")
p("\n部位大小換算（固定風險）：口數 = 每筆風險金額 ÷ (停損距離 × 100)；例：帳戶 10,000、每筆 0.1% = 10 美元、停損 7.5 → 0.013 手")
open("results_final_m15.txt", "w").write("\n".join(lines) + "\n")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True, gridspec_kw=dict(height_ratios=[2, 1]))
    for x, lab in ((S, "all shadow signals"), (T, "traded (equity filter on)")):
        d = x.groupby(x.進場時間.dt.date).R.sum().cumsum()
        ax[0].plot(pd.DatetimeIndex(d.index), d.values, label=lab)
    ax[0].set_ylabel("cum R (1R per trade)"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[0].set_title("M15 prev-bar breakout: A+B+MA50 slope+equity filter, stop 2 ATR15, RR 3, max 8")
    ax[1].plot(dl.index, dl.C.values, color="goldenrod"); ax[1].set_ylabel("XAUUSD"); ax[1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig("final_m15_equity.png", dpi=110)
except ImportError:
    pass
