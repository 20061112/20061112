"""第三十八部分：用因子預測「連續虧損段」。

樣本：最終版實際下單（final_m15_signals.csv 的 實際下單=True，7,492 筆），每筆 R。
標記：依出場時間排序，屬於「連續虧損 ≥ 10 筆」那一段的單 = 連虧單（約佔 x%）。
因子：part34 的 27 個靜態因子 + 這裡新增的「動態」因子（進場當下已知）：
  浮虧持倉數      進場時，已持有的單中目前浮虧的筆數（用進場那分鐘的收盤價）
  浮虧佔比        浮虧持倉數 / 已持倉數
  今日已停損數    今天在進場前已經停損出場的單數（影子訊號流）
  最近10筆勝率    進場前最後 10 筆已出場影子單的勝率
  距上次停損分鐘  上一筆影子單停損出場到現在的分鐘數
  1小時逆向幅度   過去 60 分鐘內，價格往交易反方向的最大走幅 / ATR15
評估：每個因子在訓練（2023~2024）/驗證（2025~2026）的 AUC（區分連虧單）、五分位的連虧比例與平均 R；
再把「訓練期挑出的危險區」當成「不下單」規則，看 年化 R / MDD 是否改善。
"""
import numpy as np
import pandas as pd
import part24_prev_bar_breakout as B

C, H, Lo = B.C, B.H, B.Lo
IDX = B.d.index
S = pd.read_csv("final_m15_signals.csv", parse_dates=["進場時間", "出場時間"])
X = pd.read_pickle("part34_X.pkl")
S["dir"] = np.where(S.方向 == "多", 1, -1)
S["i"] = IDX.get_indexer(S.進場時間)
S["train"] = S.進場時間 < "2025-01-01"
ent, ext = S.進場時間.values, S.出場時間.values
Rv, dirv, entry = S.R.values, S.dir.values, S["進場價(=點位)"].values
ii = S.i.values
atr15 = S.ATR15.values
day = S.進場時間.dt.normalize().values

# ---------- 動態因子（以影子訊號流計算；進場當下已知）----------
order_exit = np.argsort(ext, kind="stable")
ext_sorted = ext[order_exit]
win_sorted = (Rv[order_exit] > 0).astype(float)
stop_sorted = (S.出場原因.values[order_exit] == "停損")
float_n, float_frac, stops_today, wr10, since_stop, adverse = [], [], [], [], [], []
for k in range(len(S)):
    t = ent[k]
    op = np.where((ent < t) & (ext > t))[0]
    px = C[ii[k]]
    if len(op):
        fl = ((px - entry[op]) * dirv[op] < 0).sum()
        float_n.append(fl); float_frac.append(fl / len(op))
    else:
        float_n.append(0); float_frac.append(np.nan)
    m = np.searchsorted(ext_sorted, t, side="right")  # 已出場的筆數
    done = order_exit[:m]
    stops_today.append(int(((ext[done] >= day[k]) & (S.出場原因.values[done] == "停損")).sum()) if m else 0)
    wr10.append(win_sorted[max(0, m - 10):m].mean() if m >= 10 else np.nan)
    st_idx = np.where(stop_sorted[:m])[0]
    since_stop.append((t - ext_sorted[st_idx[-1]]) / np.timedelta64(1, "m") if len(st_idx) else np.nan)
    lo, hi = max(ii[k] - 60, 0), ii[k]
    adverse.append(((C[ii[k]] - Lo[lo:hi].min()) if dirv[k] < 0 else (H[lo:hi].max() - C[ii[k]])) / atr15[k] if hi > lo else np.nan)
X["浮虧持倉數"] = float_n
X["浮虧佔比"] = float_frac
X["今日已停損數"] = stops_today
X["最近10筆勝率"] = wr10
X["距上次停損分鐘"] = since_stop
X["1小時逆向幅度/ATR15"] = adverse

T = X[S.實際下單.values].copy()
T["進場時間"] = S.進場時間.values[S.實際下單.values]
T["出場時間"] = S.出場時間.values[S.實際下單.values]
T = T.reset_index(drop=True)
# 連虧段標記（實際下單，依出場順序）
o = T.sort_values("出場時間")
neg = (o.R < 0).values
run_id = np.cumsum(np.r_[1, neg[1:] != neg[:-1]])
run_len = pd.Series(run_id).map(pd.Series(run_id[neg]).value_counts()).fillna(0).values
lab = np.zeros(len(o), bool)
lab[neg & (run_len >= 10)] = True
T["連虧"] = pd.Series(lab, index=o.index).reindex(T.index).values

lines = []


def p(s=""):
    print(s)
    lines.append(s)


def auc(y, x):
    m = np.isfinite(x)
    y, x = y[m], x[m]
    r = pd.Series(x).rank().values
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan


tr, te = T[T.train], T[~T.train]
p("第三十八部分：用因子預測連續虧損段（最終版實際下單 7,492 筆）")
p(f"連虧單（屬於連虧 ≥ 10 筆的段落）：全部 {T.連虧.mean():.0%}（訓練 {tr.連虧.mean():.0%}、驗證 {te.連虧.mean():.0%}）")
p("AUC > 0.5 = 數值越大越容易落在連虧段；|AUC−0.5| 兩段都 ≥ 0.05 且同向 → ★\n")
FEATS = [c for c in T.columns if c not in ("R", "train", "進場時間", "出場時間", "連虧", "hour", "dow", "突破前日高低(dir)")]
rows = []
for f in FEATS:
    a1 = auc(tr.連虧.values, tr[f].values.astype(float))
    a2 = auc(te.連虧.values, te[f].values.astype(float))
    ok = np.sign(a1 - .5) == np.sign(a2 - .5) and min(abs(a1 - .5), abs(a2 - .5)) >= 0.05
    rows.append((f, a1, a2, ok))
for f, a1, a2, ok in sorted(rows, key=lambda r: -abs(r[1] - .5) - abs(r[2] - .5)):
    p(f"  {'★' if ok else ' '} {f:24s} AUC 訓練 {a1:.3f}  驗證 {a2:.3f}")

p("\n=== ★ 因子的五分位（訓練切點）：連虧比例 / 平均 R ===")
good = [r for r in rows if r[3]]
for f, a1, a2, ok in good:
    v = tr[f].astype(float).dropna()
    qs = np.unique(np.nanquantile(v, [0.2, 0.4, 0.6, 0.8]))
    bins = [-np.inf, *qs, np.inf]
    for nm, x in (("訓練", tr), ("驗證", te)):
        g = x.groupby(pd.cut(x[f].astype(float), bins), observed=True).agg(c=("連虧", "mean"), r=("R", "mean"))
        p(f"  {f:20s} {nm} " + "  ".join(f"{c:.0%}/{r:+.2f}" for c, r in zip(g.c, g.r)) + f"   切點 {np.round(qs, 2)}")


# ---------- 規則測試 ----------
def ratio(x):
    d = x.groupby(pd.to_datetime(x.進場時間).dt.date).R.sum(); e = d.cumsum(); m = (e.cummax() - e).max()
    return x.R.sum(), m


def show(mask, lab):
    x = T[mask]
    out = []
    for part in (x[x.train], x[~x.train], x):
        tot, m = ratio(part)
        out.append(f"{tot:+6.0f}/{m:4.0f}={tot / m:4.1f}")
    yrs = (pd.to_datetime(x.進場時間).max() - pd.to_datetime(x.進場時間).min()).days / 365
    tot, m = ratio(x)
    rm = T[~mask]
    p(f"  {lab:34s} 留 {mask.mean():4.0%} 每筆 {x.R.mean():+.3f} 排除的每筆 {rm.R.mean() if len(rm) else np.nan:+.3f} | 訓練 {out[0]} | 驗證 {out[1]} | 年化/MDD {tot / yrs / m:4.2f}")


p("\n=== 不下單規則（危險區 = 訓練期最差的五分之一）：總R/MDD=比值 ===")
show(np.ones(len(T), bool), "最終版")
for f, a1, a2, ok in good:
    v = tr[f].astype(float)
    cut = np.nanquantile(v, 0.8 if a1 > .5 else 0.2)
    m = ~((T[f].astype(float) > cut) if a1 > .5 else (T[f].astype(float) < cut))
    show(m.values, f"不做：{f} {'>' if a1 > .5 else '<'} {cut:.2f}")
for k in (2, 3, 4):
    show((T["浮虧持倉數"] < k).values, f"不做：浮虧持倉數 ≥ {k}")
for k in (2, 3, 5):
    show((T["今日已停損數"] < k).values, f"不做：今日已停損 ≥ {k}")
open("results_part38.txt", "w").write("\n".join(lines) + "\n")

p("\n=== 唯一兩段都改善的：提高權益濾網門檻（影子累積 R 要比 50 筆均線高出 X R 才下單）— 敏感度 ===")
for x in (0, 2, 3, 5, 8, 10, 15):
    show((T["權益濾網(累積R−均線)"] >= x).values, f"權益濾網門檻 +{x}R")
open("results_part38.txt", "w").write("\n".join(lines) + "\n")
