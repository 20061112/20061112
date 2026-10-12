"""第三十四部分：最終版 M15 突破的「狀態因子」—— 什麼狀態下容易賺 / 容易虧。

樣本：final_m15_signals.csv 全部影子訊號（A + B + 50MA 斜率，上限 8；14,050 筆），每筆 R（停損 2 ATR15、RR 3）。
每個因子只用進場當下已知的資料；連續因子用「訓練期（2023~2024）」的五分位切點，驗證期（2025~2026）沿用同一切點。
判定：Q5 − Q1（或分組差）在訓練與驗證同號、且兩段都 ≥ 0.08R → 「一致」。
帶 dir 的因子已乘上方向（+ = 順著交易方向）。
"""
import numpy as np
import pandas as pd
import part24_prev_bar_breakout as B

d = B.d
H, Lo, C, O1 = B.H, B.Lo, B.C, d.open.values
TV = d.tickvol.values.astype(float)
IDX = d.index
S = pd.read_csv("final_m15_signals.csv", parse_dates=["進場時間", "出場時間"])
S["dir"] = np.where(S.方向 == "多", 1, -1)
S["i"] = IDX.get_indexer(S.進場時間)
S["train"] = S.進場時間 < "2025-01-01"
i = S.i.values
dirv = S.dir.values
px = S["進場價(=點位)"].values

# ---------- M15 ----------
pid = B.period_ids(15)
g = pd.DataFrame({"pid": pid, "H": H, "L": Lo, "O": O1, "C": C, "V": TV})
b = g.groupby("pid").agg(H=("H", "max"), L=("L", "min"), O=("O", "first"), C=("C", "last"), V=("V", "sum"))
rng = b.H - b.L
sma20, sd20 = b.C.rolling(20).mean(), b.C.rolling(20).std()
bbw = (4 * sd20 / sma20)
f15 = pd.DataFrame({
    "sma20": sma20, "sd20": sd20,
    "atr20": rng.rolling(20).mean(), "atr5": rng.rolling(5).mean(), "atr100": rng.rolling(100).mean(),
    "bbw_pct": bbw.rolling(500, min_periods=100).rank(pct=True),
    "vol_ratio": b.V / b.V.rolling(20).mean(),
    "prng": rng, "pbody": (b.C - b.O),
    "ret4": b.C - b.C.shift(4), "ret16": b.C - b.C.shift(16),
}).shift(1)  # 全部用前一根（已收完）的 M15
k = pid[i]
F = f15.iloc[k].reset_index(drop=True)

# ---------- 日線（前一日為止）----------
dl = B.daily
dC = dl.C
atrd = (dl.H - dl.L).rolling(10).mean()
dd = pd.DataFrame({
    "dsma20": dC.rolling(20).mean(), "dsd20": dC.rolling(20).std(), "dsma50": dC.rolling(50).mean(),
    "atrd": atrd, "atrd_chg": (dl.H - dl.L).rolling(5).mean() / (dl.H - dl.L).rolling(20).mean(),
    "atrd_pct": atrd.rolling(250, min_periods=60).rank(pct=True),
    "er10": (dC - dC.shift(10)).abs() / dC.diff().abs().rolling(10).sum(),
    "slope": dC.rolling(50).mean() - dC.rolling(50).mean().shift(10),
    "pret": dC - dl.O, "prng": dl.H - dl.L, "pC": dC, "pH": dl.H, "pL": dl.L,
}).shift(1)
dd["O"] = dl.O
day = B.dayv[i]
D = dd.loc[day].reset_index(drop=True)

# ---------- 日內 ----------
first = pd.Series(np.arange(len(d))).groupby(B.dayv).transform("first").values
f0 = first[i]
cum_path = np.concatenate([[0], np.cumsum(np.abs(np.diff(C)))])
hi_sofar = pd.Series(H).groupby(B.dayv).cummax().values
lo_sofar = pd.Series(Lo).groupby(B.dayv).cummin().values
im1 = np.maximum(i - 1, f0)

X = pd.DataFrame(index=S.index)
X["R"] = S.R.values
X["train"] = S.train.values
# 時間
X["hour"] = S.進場時間.dt.hour
X["dow"] = S.進場時間.dt.dayofweek
# sigma / 偏離均值
X["z20_m15(dir)"] = (px - F.sma20.values) / F.sd20.values * dirv
X["dz20_日(dir)"] = (px - D.dsma20.values) / D.dsd20.values * dirv
X["離50日均線_ATR日(dir)"] = (px - D.dsma50.values) / D.atrd.values * dirv
X["離前日收_ATR日(dir)"] = (px - D.pC.values) / D.atrd.values * dirv
# ATR / 波動
X["ATR15_5/20(短期波動變化)"] = F.atr5.values / F.atr20.values
X["ATR15_20/100"] = F.atr20.values / F.atr100.values
X["ATR15/ATR日"] = F.atr20.values / D.atrd.values
X["日ATR_5/20"] = D.atrd_chg.values
X["日ATR百分位(250日)"] = D.atrd_pct.values
X["BB寬度百分位(M15)"] = F.bbw_pct.values
# 趨勢強度
X["日ER10"] = D.er10.values
X["50MA斜率/ATR日"] = np.abs(D.slope.values) / D.atrd.values
X["前日漲跌/ATR日(dir)"] = D.pret.values / D.atrd.values * dirv
X["前日波幅/ATR日"] = D.prng.values / D.atrd.values
# 日內
X["今日漲跌/ATR日(dir)"] = (px - D.O.values) / D.atrd.values * dirv
X["今日ER"] = np.abs(C[im1] - O1[f0]) / (cum_path[i] - cum_path[f0] + 1e-9)
pos = (px - lo_sofar[im1]) / np.maximum(hi_sofar[im1] - lo_sofar[im1], 1e-9)
X["今日區間位置(dir)"] = np.where(dirv > 0, pos, 1 - pos)
X["今日波幅/ATR日"] = (hi_sofar[im1] - lo_sofar[im1]) / D.atrd.values
X["突破前日高低(dir)"] = np.where(dirv > 0, px > D.pH.values, px < D.pL.values).astype(int)
# 前一根 M15 / 動能 / 量
X["前根波幅/ATR15"] = F.prng.values / F.atr20.values
X["前根實體/ATR15(dir)"] = F.pbody.values / F.atr20.values * dirv
X["1小時動能/ATR15(dir)"] = F.ret4.values / F.atr20.values * dirv
X["4小時動能/ATR15(dir)"] = F.ret16.values / F.atr20.values * dirv
X["前根量/20根均量"] = F.vol_ratio.values
X["點差/停損"] = S.點差.values / S.停損距離.values
# 部位 / 訊號
X["當日第幾個訊號"] = S.groupby(S.進場時間.dt.date).cumcount().values + 1
ent, ext = S.進場時間.values, S.出場時間.values
X["進場時已持倉數"] = [int(((ent < t) & (ext > t)).sum()) for t in ent]
X["權益濾網(累積R−均線)"] = (S["影子累積R"] - S["影子累積R的50筆均線"]).values

lines = []


def p(s=""):
    print(s)
    lines.append(s)


tr, te = X[X.train], X[~X.train]
p("第三十四部分：最終版 M15 突破 —— 狀態因子（14,050 筆影子訊號，每筆 R）")
p(f"全部：訓練 {tr.R.mean():+.3f}R（{len(tr)}）  驗證 {te.R.mean():+.3f}R（{len(te)}）")
p("每列：五分位 Q1（小）→ Q5（大）的平均 R；訓練切點沿用到驗證\n")
summary = []
CONT = [c for c in X.columns if c not in ("R", "train", "hour", "dow", "突破前日高低(dir)")]
for f in CONT:
    v = tr[f].replace([np.inf, -np.inf], np.nan).dropna()
    qs = np.unique(np.nanquantile(v, [0.2, 0.4, 0.6, 0.8]))
    bins = [-np.inf, *qs, np.inf]
    a = tr.groupby(pd.cut(tr[f], bins), observed=True).R.agg(["mean", "size"])
    b_ = te.groupby(pd.cut(te[f], bins), observed=True).R.agg(["mean", "size"])
    da = a["mean"].iloc[-1] - a["mean"].iloc[0]
    db = b_["mean"].iloc[-1] - b_["mean"].iloc[0]
    ok = np.sign(da) == np.sign(db) and min(abs(da), abs(db)) >= 0.08
    summary.append((f, da, db, ok, a, b_, qs))
    p(f"{'★' if ok else ' '} {f:24s} 切點 {np.round(qs, 3)}")
    p(f"     訓練 " + " ".join(f"{m:+.3f}" for m in a["mean"]) + f"   Q5−Q1 {da:+.3f}")
    p(f"     驗證 " + " ".join(f"{m:+.3f}" for m in b_["mean"]) + f"   Q5−Q1 {db:+.3f}")

p("\n=== 分組因子 ===")
for a0, b0, nm in [(1, 4, "01-04 亞洲早"), (4, 8, "04-08 亞洲"), (8, 11, "08-11 倫敦開"), (11, 15, "11-15 倫敦"),
                   (15, 18, "15-18 紐約開"), (18, 22, "18-22 紐約")]:
        col = "hour"
        m1, m2 = tr[col].between(a0, b0 - 1), te[col].between(a0, b0 - 1)
        p(f"  {nm:12s} 訓練 {tr.R[m1].mean():+.3f}（{m1.sum()}）  驗證 {te.R[m2].mean():+.3f}（{m2.sum()}）")
for k_, nm in enumerate("一二三四五"):
    p(f"  星期{nm}       訓練 {tr.R[tr.dow == k_].mean():+.3f}（{(tr.dow == k_).sum()}）  驗證 {te.R[te.dow == k_].mean():+.3f}（{(te.dow == k_).sum()}）")
for v, nm in ((1, "已突破前日高(多)/低(空)"), (0, "還在前日區間內")):
    p(f"  {nm:12s} 訓練 {tr.R[tr['突破前日高低(dir)'] == v].mean():+.3f}  驗證 {te.R[te['突破前日高低(dir)'] == v].mean():+.3f}")

p("\n=== 摘要：訓練與驗證方向一致（|Q5−Q1| ≥ 0.08R）的因子 ===")
for f, da, db, ok, a, b_, qs in sorted(summary, key=lambda x: -abs(x[1] + x[2])):
    if ok:
        best = "大" if da > 0 else "小"
        p(f"  {f:24s} 訓練 Q5−Q1 {da:+.3f}  驗證 {db:+.3f}  → 數值越{best}越好")
p(f"\n（共 {len(summary)} 個連續因子；純屬巧合時，約有 {len(summary) * 0.5 * 0.3:.0f} 個會兩段同號且夠大，請把結果當成線索）")
X.to_pickle("part34_X.pkl")
open("results_part34.txt", "w").write("\n".join(lines) + "\n")
