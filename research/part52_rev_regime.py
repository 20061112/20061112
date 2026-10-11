"""第五十二部分：反向單（候選 E）與定案版的相關性；波動率 / 混亂度與兩者的配合。

（資料處理沿用第五十部分）
第五十部分：M1 2023/6 ~ 2026/10 全期（新上傳 2023/6~2024/9，與既有 M1 合併；重疊 28,681 根完全相同）。

2023/6 ~ 2024/8 是 M1 第一次看到的期間（M15 顯示這段是區間盤、策略不賺）。
A. M1 定案版（OW + IB 寬度 + 2R 保本 + 分倉 3/2/1 + 最多 5 筆）逐期
B. 對照：無寬度篩選、全部 0.01 手、不保本
C. 候選 F（只加碼在賺錢部位）、G（前 20 天日內單邊比例中位 < 0.4 暫停）
D. 依突破快慢分組（驗證分倉）
"""
import numpy as np
import pandas as pd
from oos_engine import signals, simulate
from part27_ow import data, pf, prev_profiles

out = open("results_part52.txt", "w")
def P(s=""): print(s); out.write(s + "\n"); out.flush()
PER = [("23H2", "2023-06-01", "2023-12-31"), ("24H1", "2024-01-01", "2024-06-30"), ("24H2", "2024-07-01", "2024-12-31"),
       ("25H1", "2025-01-01", "2025-06-30"), ("25H2", "2025-07-01", "2025-12-31"), ("26H1", "2026-01-01", "2026-06-30"),
       ("26H2", "2026-07-01", "2026-12-31"), ("新 23/6~24/8", "2023-06-01", "2024-08-31"), ("全部", "2023-06-01", "2026-12-31")]
days, D, ARR = data("data/XAUUSD_M1_2023_2026.csv", 1)
PV = prev_profiles(days, ARR)


def manage(r, be=True):
    t, O, H, L, C, S, mins = ARR[r.day]
    s, e = r.side, r.lvl
    st = r.ibl if s == 1 else r.ibh
    R = s * (e - st); best = e; done = False
    for k in range(r.i, len(C)):
        if (L[k] <= st) if s == 1 else (H[k] >= st):
            return (min(O[k], st) if s == 1 else max(O[k], st)), t[k] + pd.Timedelta(minutes=1)
        best = max(best, H[k]) if s == 1 else min(best, L[k])
        if be and not done and s * (best - e) >= 2 * R:
            done = True; st = e
    return C[-1], t[-1] + pd.Timedelta(minutes=1)


def cap(T, n=5):
    keep, open_ = [], []
    for r in T.sort_values("t_in").itertuples():
        open_ = [x for x in open_ if x > r.t_in]
        if len(open_) < n:
            keep.append(r.Index); open_.append(r.exit_time)
    return T.loc[keep]


W = {}
for d in days:
    t, O, H, L, C, Sp, mins = ARR[d]
    row = {}
    for h in np.arange(2, 11, 0.5):
        m = (mins >= h * 60) & (mins < h * 60 + 60)
        if m.sum() >= 48:
            row[h] = H[m].max() - L[m].min()
    W[d] = row
REL = pd.DataFrame(W).T.sort_index().rolling(20, min_periods=10).median().shift(1)
X = simulate(signals(days, D, ARR, 1, monday=True), ARR, 1)
X["ow"] = [(r.day in PV) and ((r.ibl > PV[r.day][1]) if r.side == 1 else (r.ibh < PV[r.day][2])) for r in X.itertuples()]
X["w_atr"] = (X.ibh - X.ibl) / X.atr
X["w_rel"] = [(r.ibh - r.ibl) / REL.at[r.day, r.h] if (r.day in REL.index and r.h in REL.columns and REL.at[r.day, r.h] > 0) else np.nan for r in X.itertuples()]
X["wok"] = (X.w_atr <= 0.4) & (X.w_rel >= 0.6)
TR = ((D.close - D.open).abs() / (D.high - D.low)).rolling(20).median().shift(1)


def build(mask, be=True):
    T = X[mask].sort_values(["t_in", "h"]).drop_duplicates(["t_in", "side"]).copy()
    ex = [manage(r, be) for r in T.itertuples()]
    T["exit_px"] = [a for a, b in ex]; T["exit_time"] = [b for a, b in ex]
    T["pnl"] = T.side * (T.exit_px - T.lvl) - T.spread; T["R"] = T.pnl / T.risk
    T["delay"] = (T.t_in - T.day - pd.to_timedelta(T.h + 1, unit="h")).dt.total_seconds() / 3600
    T["lots"] = np.where(T.delay <= 0.5, 3.0, np.where(T.delay <= 1, 2.0, 1.0))
    return T.sort_values("t_in").reset_index(drop=True)


alldays = [d for d in days if d >= pd.Timestamp("2023-06-02")]


def line(name, T, w=None):
    C = cap(T)
    lots = C.lots if w is None else w(C)
    C = C.assign(v=C.pnl * lots)
    cells = []
    for n, a, b in PER:
        x = C[(C.day >= a) & (C.day <= b)]
        if not len(x):
            continue
        xs = x.sort_values("exit_time"); eq = xs.v.cumsum(); dd = (eq - eq.cummax()).min()
        dl = x.groupby("day").v.sum().reindex([d for d in alldays if pd.Timestamp(a) <= d <= pd.Timestamp(b)], fill_value=0)
        cells.append(f"{n} {x.R.mean():+.2f}R/{x.v.sum():+,.0f}$/DD{dd:,.0f}/S{dl.mean() / dl.std() * np.sqrt(250):.1f}")
    P(f"  {name:26s} " + "  ".join(cells))
    return C



q = np.array([(1 if r.ibl > PV[r.day][1] else (-1 if r.ibh < PV[r.day][2] else 0)) if r.day in PV else 0 for r in X.itertuples()])
X["qual"] = q
fin = build(X.ow & X.wok); fin["kind"] = "順"
rev = build((X.qual != 0) & (X.side == -X.qual) & X.wok); rev["kind"] = "反"
for T in (fin, rev):
    T["usd1"] = T.pnl                      # 0.01 手
P("第五十二部分（M1 2023/6~2026/10）")

# ---------------- A. 相關性 ----------------
P("\nA. 反向單 vs 定案（都不設上限、0.01 手；日 / 週 / 月損益相關）")
idx = pd.DatetimeIndex(alldays)
df = pd.DataFrame({"順": fin.groupby("day").usd1.sum(), "反": rev.groupby("day").usd1.sum()}).reindex(idx).fillna(0)
dR = pd.DataFrame({"順": fin.groupby("day").R.sum(), "反": rev.groupby("day").R.sum()}).reindex(idx).fillna(0)
for lab, f in [("日", None), ("週", "W"), ("月", "ME")]:
    a = df if f is None else df.resample(f).sum()
    b = dR if f is None else dR.resample(f).sum()
    P(f"  {lab}：美元相關 {a['順'].corr(a['反']):+.2f}   R 相關 {b['順'].corr(b['反']):+.2f}")
both = (df["順"] != 0) & (df["反"] != 0)
P(f"  有交易的日子：只有順 {((df['順'] != 0) & (df['反'] == 0)).sum()} 天、只有反 {((df['反'] != 0) & (df['順'] == 0)).sum()} 天、兩者同一天 {both.sum()} 天")
P(f"  同一天兩者都做時的日相關：{df[both]['順'].corr(df[both]['反']):+.2f}")
P("  定案虧錢的日子，反向單當天平均：" + f"{df[df['順'] < 0]['反'].mean():+.1f} 美元；定案賺錢的日子：{df[df['順'] > 0]['反'].mean():+.1f}")
m = df.resample("ME").sum()
P(f"  月份：定案虧的 {(m['順'] < 0).sum()} 個月中，反向單賺的有 {((m['順'] < 0) & (m['反'] > 0)).sum()} 個月；兩者同時虧 {((m['順'] < 0) & (m['反'] < 0)).sum()} 個月（共 {len(m)} 個月）")
rc = df["順"].rolling(60).corr(df["反"])
P("  60 日滾動相關（各半年平均）：" + "  ".join(f"{n} {rc[(rc.index >= a) & (rc.index <= b)].mean():+.2f}" for n, a, b in PER[:7]))

# ---------------- B. 波動率 / 混亂度 ----------------
TRd = np.maximum(D.high - D.low, np.maximum((D.high - D.close.shift()).abs(), (D.low - D.close.shift()).abs()))
feat = pd.DataFrame(index=D.index)
feat["波動 ATR10÷價格%"] = (D.atr10 / D.close.shift() * 100)
feat["波動擴張 ATR5÷ATR20"] = (TRd.rolling(5).mean() / TRd.rolling(20).mean()).shift(1)
dc = D.close.diff().abs()
feat["趨勢效率 ER20"] = ((D.close - D.close.shift(20)).abs() / dc.rolling(20).sum()).shift(1)
feat["日內單邊 20日中位"] = ((D.close - D.open).abs() / (D.high - D.low)).rolling(20).median().shift(1)
feat["震盪指數 CHOP14"] = (100 * np.log10(TRd.rolling(14).sum() / (D.high.rolling(14).max() - D.low.rolling(14).min())) / np.log10(14)).shift(1)


def intraday(T):
    er, rg = [], []
    for r in T.itertuples():
        t, O, H, L, C, S, mins = ARR[r.day]
        c = C[:r.i:15]                                   # 進場前每 15 分取一個收盤
        er.append(abs(c[-1] - c[0]) / max(np.abs(np.diff(c)).sum(), 1e-9) if len(c) > 2 else np.nan)
        rg.append((H[:r.i].max() - L[:r.i].min()) / r.atr)
    return er, rg


for T in (fin, rev):
    for c in feat.columns:
        T[c] = T.day.map(feat[c])
    T["盤中效率（進場前 15分 ER）"], T["盤中已走區間÷ATR"] = intraday(T)
cols = list(feat.columns) + ["盤中效率（進場前 15分 ER）", "盤中已走區間÷ATR"]
P("\nB. 依波動 / 混亂度三等分（門檻 = 兩種單合併後的三分位；每格 = 每筆R / 筆數；右側：各半年哪一邊較好）")
for c in cols:
    allv = pd.concat([fin[c], rev[c]]).dropna()
    q1, q2 = allv.quantile([1 / 3, 2 / 3])
    P(f"  [{c}]  門檻 {q1:.3f} / {q2:.3f}")
    for lab, lo, hi in [("低", -np.inf, q1), ("中", q1, q2), ("高", q2, np.inf)]:
        f = fin[(fin[c] > lo) & (fin[c] <= hi)]; r = rev[(rev[c] > lo) & (rev[c] <= hi)]
        per = "  ".join(f"{n} {x.R.mean():+.2f}/{y.R.mean():+.2f}" for n, a, b in PER[:7]
                        for x, y in [(f[(f.day >= a) & (f.day <= b)], r[(r.day >= a) & (r.day <= b)])] if len(x) and len(y))
        P(f"    {lab}  順 {f.R.mean():+.3f}R/{len(f):4d}  反 {r.R.mean():+.3f}R/{len(r):4d}  ｜ 各期 順/反：{per}")

# ---------------- C. 配合規則 ----------------
P("\nC. 配合規則（分倉 3/2/1、合併後最多 5 筆；每格 = 每筆R / 美元 / 回撤 / Sharpe）")
line("只有定案", fin)
line("定案 + 反向（全部）", pd.concat([fin, rev]).sort_values(["t_in", "h"]).drop_duplicates(["t_in", "side"]).reset_index(drop=True))
line("定案 + 反向（反向固定 0.01）", pd.concat([fin, rev.assign(lots=1.0)]).sort_values(["t_in", "h"]).drop_duplicates(["t_in", "side"]).reset_index(drop=True))
for c, th, side in [("日內單邊 20日中位", None, "low"), ("趨勢效率 ER20", None, "low"), ("震盪指數 CHOP14", None, "high"), ("波動 ATR10÷價格%", None, "high")]:
    allv = pd.concat([fin[c], rev[c]]).dropna(); med = allv.median()
    sel = rev[c] <= med if side == "low" else rev[c] > med
    lab = f"定案 + 反向（只在 {c} {'≤' if side == 'low' else '>'} 中位 {med:.2f}）"
    line(lab, pd.concat([fin, rev[sel]]).sort_values(["t_in", "h"]).drop_duplicates(["t_in", "side"]).reset_index(drop=True))
out.close()


# ---------------- D. 依 B 的發現做的配合（門檻 = 全樣本三分位，屬樣本內，需看各期是否一致） ----------------
out = open("results_part52.txt", "a")
P("\nD. 依 B 的發現做的配合")
def q(c, p):
    return pd.concat([fin[c], rev[c]]).dropna().quantile(p)
ex, er_, tr_, ch_ = "波動擴張 ATR5÷ATR20", "盤中效率（進場前 15分 ER）", "日內單邊 20日中位", "震盪指數 CHOP14"
mix = lambda a, b: pd.concat([a, b]).sort_values(["t_in", "h"]).drop_duplicates(["t_in", "side"]).reset_index(drop=True)
line("定案：波動擴張高 ×2 手", fin.assign(lots=np.where(fin[ex] > q(ex, 2 / 3), fin.lots * 2, fin.lots)))
line("定案：不做盤中效率低（<1/3）", fin[~(fin[er_] < q(er_, 1 / 3))])
r1 = rev[rev[tr_] > q(tr_, 0.5)]
r2 = rev[rev[er_] <= q(er_, 2 / 3)]
r3 = rev[rev[ch_] <= q(ch_, 0.5)]
line("反向：只在 日內單邊20日中位 > 中位", r1)
line("反向：只在 盤中效率 ≤ 2/3 分位", r2)
line("反向：只在 CHOP14 ≤ 中位", r3)
line("定案 + 反向（日內單邊 > 中位）", mix(fin, r1))
line("定案 + 反向（盤中效率 ≤ 2/3）", mix(fin, r2))
line("定案 + 反向（CHOP ≤ 中位）", mix(fin, r3))
line("定案 + 反向（日內單邊 > 中位，反向 0.01）", mix(fin, r1.assign(lots=1.0)))
out.close()
