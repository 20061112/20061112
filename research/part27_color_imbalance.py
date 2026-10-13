"""第二十七部分：最近 N 根 K 棒的紅綠（陰陽）不平衡 → 之後怎麼走？
例：N = 20，正常應該約 10 陽 / 10 陰；若出現 15/5 或 5/15，之後是延續（順多數色）還是反轉？
- 時框 M1（只當對照）/ M5 / M15 / M30 / H1；N = 10 / 20 / 30 / 50；之後 1 / 3 / 6 / 12 / 24 根。
- 不平衡 d = 陽線數 − 陰線數（十字線不算）。「順多數」報酬 = sign(d) × (C[t+h] − C[t]) / ATR20。
- 發現期 2023/6~2024/12、驗證期 2025/1~2026/10；多空分開（扣掉黃金多頭漂移）。
- 控制組：不平衡是不是只是「N 根漲跌幅」的另一種說法？在同樣的 N 根漲跌幅分組內，比較顏色多數 vs 少數。
- 交易測試：N=20，陽 ≥ 15（或陰 ≥ 15）→ 順勢 / 逆勢，持有 h 根，扣點差，不重疊，逐年。
"""
import numpy as np
import pandas as pd
from math import comb
from load import load

SPLIT = pd.Timestamp("2025-01-01")
TFS = [("M1", "1min", 1), ("M5", "5min", 5), ("M15", "15min", 15), ("M30", "30min", 30), ("H1", "1h", 60)]
AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "tickvol": "sum", "spread": "mean"}
NS = (10, 20, 30, 50)
HS = (1, 3, 6, 12, 24)
out = open("results_part27.txt", "w")


def P(s=""):
    print(s)
    out.write(s + "\n")
    out.flush()


def pct(x):
    return f"{100 * x:5.1f}%"


def build(m1, rule, mins):
    df = m1 if rule == "1min" else m1.resample(rule, label="left", closed="left").agg(AGG).dropna()
    f = pd.DataFrame(index=df.index)
    d = np.sign(df.close - df.open)
    atr = (df.high - df.low).rolling(20).mean()
    tm = df.index.to_series()
    for N in NS:
        span = (tm - tm.shift(N - 1)).dt.total_seconds() / 60
        ok = span <= (N - 1) * mins * 1.5 + 65                 # 視窗不跨週末
        f[f"g{N}"] = (d > 0).astype(int).rolling(N).sum().where(ok)
        f[f"r{N}"] = (d < 0).astype(int).rolling(N).sum().where(ok)
        f[f"mv{N}"] = ((df.close - df.close.shift(N)) / atr).where(ok)   # N 根漲跌幅（ATR）
    for h in HS:
        span = (tm.shift(-h) - tm).dt.total_seconds() / 60
        f[f"f{h}"] = ((df.close.shift(-h) - df.close) / atr).where(span <= h * mins + 65)
    f["nxt"] = np.sign(df.close.shift(-1) - df.open.shift(-1))
    f["cost"] = df.spread * 0.01 / atr
    f["per"] = np.where(f.index < SPLIT, 0, 1)
    f["year"] = f.index.year
    return f.dropna(subset=["f1"])


def binom_tail(N, k):
    """N 根中「多數色 ≥ k」的機率（p = 0.5，雙邊）。"""
    return 2 * sum(comb(N, j) for j in range(k, N + 1)) / 2 ** N


m1 = load("data/XAUUSD_M1_2023_2026.csv")
F = {tf: build(m1, r, m) for tf, r, m in TFS}

P("第二十七部分：最近 N 根紅綠不平衡 → 之後的走勢（資料 2023/6 ~ 2026/10）")
P("報酬單位 ATR20；「順多數」= 往多數顏色的方向；已扣掉同期間所有 K 棒的平均（漂移）")

P("\n# 1. N = 20 依「陽線數」分組（十字線不算，所以陽 + 陰 ≤ 20）")
P("   頻率 = 實際出現比例；丟銅板 = 若每根 50/50 獨立，理論比例")
for tf in [t for t, *_ in TFS]:
    f = F[tf]
    g = f.g20
    bins = [-1, 4, 5, 6, 7, 12, 13, 14, 15, 20]
    labs = ["≤4", "5", "6", "7", "8~12", "13", "14", "15", "≥16"]
    theo = {"≤4": sum(comb(20, j) for j in range(0, 5)), "5": comb(20, 5), "6": comb(20, 6), "7": comb(20, 7),
            "8~12": sum(comb(20, j) for j in range(8, 13)), "13": comb(20, 13), "14": comb(20, 14), "15": comb(20, 15),
            "≥16": sum(comb(20, j) for j in range(16, 21))}
    q = pd.cut(g, bins, labels=labs)
    base = {h: f.groupby("per")[f"f{h}"].transform("mean") for h in (1, 6, 24)}
    P(f"\n## {tf}（n={g.notna().sum():,}）")
    P(f"  {'陽線數':>6s} {'頻率':>7s} {'丟銅板':>7s} | {'下一根陽':>8s} | {'之後1根':>8s} {'6根':>8s} {'24根':>8s}（往上，ATR，扣漂移）| 24 根 發現 / 驗證")
    for lab in labs:
        m = q == lab
        s = f[m]
        x = {h: (s[f"f{h}"] - base[h][m]).mean() for h in (1, 6, 24)}
        a = (s.f24 - base[24][m])[s.per == 0].mean()
        b = (s.f24 - base[24][m])[s.per == 1].mean()
        P(f"  {lab:>6s} {pct(m.sum() / g.notna().sum())} {pct(theo[lab] / 2 ** 20)} | {pct((s.nxt > 0).mean() / max((s.nxt != 0).mean(), 1e-9))} | "
          f"{x[1]:+8.3f} {x[6]:+8.3f} {x[24]:+8.3f} | {a:+.3f} / {b:+.3f}")

P("\n\n# 2. 不同 N：多數色比例達到「丟銅板只有約 2% 機率」的極端時，順多數方向的報酬（扣漂移，ATR）")
P("   極端門檻：N=10 → ≥9；N=20 → ≥15；N=30 → ≥21；N=50 → ≥33（雙邊理論機率約 2~4%）")
TH = {10: 9, 20: 15, 30: 21, 50: 33}
for tf in [t for t, *_ in TFS]:
    f = F[tf]
    P(f"\n## {tf}")
    for N in NS:
        k = TH[N]
        up, dn = f[f"g{N}"] >= k, f[f"r{N}"] >= k
        cells = []
        for h in (1, 6, 24):
            b = f.groupby("per")[f"f{h}"].transform("mean")
            fol = pd.concat([(f[f"f{h}"] - b)[up], -(f[f"f{h}"] - b)[dn]])
            per = pd.concat([f.per[up], f.per[dn]])
            cells.append(f"{h:2d}根 {fol.mean():+.3f}（發 {fol[per == 0].mean():+.3f} / 驗 {fol[per == 1].mean():+.3f}）")
        n_ = int(up.sum() + dn.sum())
        P(f"  N={N:2d} 多數≥{k:2d}: 頻率 {pct(n_ / f[f'g{N}'].notna().sum())}（銅板 {pct(binom_tail(N, k))}）n={n_:6d} | " + " | ".join(cells))

P("\n\n# 3. 控制組：顏色不平衡有沒有「漲跌幅以外」的資訊？（N = 20，之後 12 根，往上報酬，扣漂移）")
P("   把 20 根漲跌幅（ATR）分 5 組，每組內再依陽線數分 少（≤8）/ 中（9~11）/ 多（≥12）")
for tf in ("M5", "M15", "M30", "H1"):
    f = F[tf].dropna(subset=["mv20", "f12"]).copy()
    f["x"] = f.f12 - f.groupby("per").f12.transform("mean")
    f["mq"] = pd.qcut(f.mv20, 5, labels=["大跌", "小跌", "平", "小漲", "大漲"])
    f["gq"] = pd.cut(f.g20, [-1, 8, 11, 20], labels=["陽少", "陽中", "陽多"])
    t = f.pivot_table(index="mq", columns="gq", values="x", aggfunc="mean", observed=True)
    n = f.pivot_table(index="mq", columns="gq", values="x", aggfunc="size", observed=True)
    P(f"\n## {tf}（整體：20 根漲跌幅 → 之後 12 根 相關 {f.mv20.corr(f.x):+.3f}；陽線數 → 相關 {f.g20.corr(f.x):+.3f}）")
    for mq in t.index:
        P(f"  {mq}: " + "  ".join(f"{c} {t.loc[mq, c]:+.3f} (n={int(n.loc[mq, c])})" for c in t.columns))
    # 迴歸：x ~ mv20 + g20（標準化），看 g20 係數
    X = np.column_stack([np.ones(len(f)), (f.mv20 - f.mv20.mean()) / f.mv20.std(), (f.g20 - f.g20.mean()) / f.g20.std()])
    for per in (0, 1):
        m = (f.per == per).to_numpy()
        beta, *_ = np.linalg.lstsq(X[m], f.x.to_numpy()[m], rcond=None)
        P(f"    迴歸（{'發現' if per == 0 else '驗證'}）：漲跌幅係數 {beta[1]:+.4f}  陽線數係數 {beta[2]:+.4f}（每 1 個標準差，ATR）")

P("\n\n# 4. 交易測試：N = 20，多數色 ≥ 15 → 順勢 / 逆勢，持有 h 根，收盤進場，扣點差，不重疊（單位 ATR）")


def trades(f, side_fn, h, mins):
    sig = f[(f.g20 >= 15) | (f.r20 >= 15)].dropna(subset=[f"f{h}"])
    rows, last = [], pd.Timestamp(0)
    for t, r in sig.iterrows():
        if t < last:
            continue
        s = side_fn(1 if r.g20 >= 15 else -1)
        rows.append((t, s, s * r[f"f{h}"] - r.cost))
        last = t + pd.Timedelta(minutes=h * mins)
    return pd.DataFrame(rows, columns=["t", "side", "R"]).set_index("t")


for tf, rule, mins in TFS:
    f = F[tf]
    P(f"\n## {tf}")
    for lab, fn in (("順多數", lambda s: s), ("逆多數", lambda s: -s)):
        for h in (6, 24):
            T = trades(f, fn, h, mins)
            yr = T.groupby(T.index.year).R.mean()
            lo, sh = T[T.side == 1].R, T[T.side == -1].R
            P(f"  {lab} 持有{h:2d}: n={len(T):5d} 每筆 {T.R.mean():+.3f} | 做多 {lo.mean():+.3f} (n={len(lo)}) 做空 {sh.mean():+.3f} (n={len(sh)}) | 逐年 "
              + " ".join(f"{y}:{v:+.3f}" for y, v in yr.items()))


P("\n\n# 5. 第 3 段看到的「顏色和漲跌幅背離」：20 根下跌但陽線多數 → 做多；20 根上漲但陰線多數 → 做空")
P("   （= 漲跌幅是少數幾根大 K 做出來的，之後往多數顏色走？）持有 h 根，扣點差，不重疊，單位 ATR")


def div_trades(f, h, mins, k=12):
    up = (f.g20 >= k) & (f.mv20 < 0)
    dn = (f.r20 >= k) & (f.mv20 > 0)
    sig = f[up | dn].dropna(subset=[f"f{h}"])
    rows, last = [], pd.Timestamp(0)
    for t, r in sig.iterrows():
        if t < last:
            continue
        s = 1 if (r.g20 >= k and r.mv20 < 0) else -1
        rows.append((t, s, s * r[f"f{h}"] - r.cost, s * r[f"f{h}"]))
        last = t + pd.Timedelta(minutes=h * mins)
    return pd.DataFrame(rows, columns=["t", "side", "R", "gross"]).set_index("t")


for tf, rule, mins in TFS[1:]:
    f = F[tf]
    for h in (6, 12, 24):
        T = div_trades(f, h, mins)
        yr = T.groupby(T.index.year).R.mean()
        lo, sh = T[T.side == 1].R, T[T.side == -1].R
        P(f"  {tf:3s} 持有{h:2d}: n={len(T):5d} 未扣點差 {T.gross.mean():+.3f} 扣後 {T.R.mean():+.3f} | 做多 {lo.mean():+.3f} 做空 {sh.mean():+.3f} | 逐年 "
          + " ".join(f"{y}:{v:+.3f}" for y, v in yr.items()))
out.close()
