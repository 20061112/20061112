"""第二十四部分報表：讀 part24_patterns.csv，多空合併（依筆數加權），輸出 results_part24.txt。
另外做「影線比例連續分組」→ 下一根報酬。"""
import numpy as np
import pandas as pd
from load import load
from part24_candles import TFS, AGG, SPLIT

R = pd.read_csv("part24_patterns.csv")


def wavg(g, col, ncol):
    w = g[ncol].fillna(0)
    v = g[col]
    m = v.notna() & (w > 0)
    return (v[m] * w[m]).sum() / w[m].sum() if m.any() else np.nan


def combine(g):
    out = {"n": g.n.sum(), "每天": g.per_day.sum(), "下一根同向": wavg(g, "win1", "n")}
    for q in (0, 1):
        for h in (1, 3, 6, 12):
            out[f"x{h}_{q}"] = wavg(g, f"x{h}_{q}", f"n_{q}")
            out[f"r{h}_{q}"] = wavg(g, f"r{h}_{q}", f"n_{q}")
        for tg in (1, 2):
            out[f"R{tg}_{q}"] = wavg(g, f"R{tg}_{q}", f"Rn{tg}_{q}")
            out[f"B{tg}_{q}"] = wavg(g, f"Rb{tg}_{q}", f"Rn{tg}_{q}")
            out[f"Rn{tg}_{q}"] = g[f"Rn{tg}_{q}"].sum()
    out["多 2R 發/驗"] = f"{g[g.side == 1][f'R2_0'].iloc[0]:+.2f}/{g[g.side == 1][f'R2_1'].iloc[0]:+.2f}"
    out["空 2R 發/驗"] = f"{g[g.side == -1][f'R2_0'].iloc[0]:+.2f}/{g[g.side == -1][f'R2_1'].iloc[0]:+.2f}"
    return pd.Series(out)


C = R.groupby(["tf", "family", "pattern"], sort=False).apply(combine, include_groups=False).reset_index()
C["e2_0"] = C.R2_0 - C.B2_0
C["e2_1"] = C.R2_1 - C.B2_1
C["e1_0"] = C.R1_0 - C.B1_0
C["e1_1"] = C.R1_1 - C.B1_1

print("資料 2023/6 ~ 2026/10；發現期 2023/6~2024/12、驗證期 2025/1~2026/10；多空合併（看跌版 = 鏡像定義）")
print("欄位：")
print("  每天     = 平均每個交易日出現次數（多 + 空）")
print("  同向     = 下一根 K 與型態方向同色的比例（50% = 沒用）")
print("  收盤進場 = 型態收盤進場、持有 3 / 12 根，未扣成本、扣掉同方向漂移的超額報酬（ATR），發現 / 驗證")
print("  突破 2R  = 下一根突破型態高(低)點進場、停損另一端、目標 2R、最多 12 根，扣點差的每筆 R；")
print("             「超額」= 型態 − 所有 K 棒用同樣方式進場的基準（發現 / 驗證）")
for tf, *_ in TFS:
    d = C[C.tf == tf]
    print(f"\n\n## {tf}（突破 2R 基準：發現 {d.B2_0.mean():+.3f}R、驗證 {d.B2_1.mean():+.3f}R；1R 基準 {d.B1_0.mean():+.3f} / {d.B1_1.mean():+.3f}）")
    print(f"{'型態':38s} {'每天':>5s} {'同向':>6s} | {'收盤進場 持有3':>15s} | {'持有12':>15s} | {'突破2R 每筆R':>15s} | {'2R 超額':>15s} | {'1R 超額':>15s} | 多2R 發/驗  空2R 發/驗")
    for _, r in d.iterrows():
        print(f"{r.pattern:38s} {r['每天']:5.2f} {100 * r['下一根同向']:5.1f}% | {r.x3_0:+.3f} / {r.x3_1:+.3f} | {r.x12_0:+.3f} / {r.x12_1:+.3f} | "
              f"{r.R2_0:+.3f} / {r.R2_1:+.3f} | {r.e2_0:+.3f} / {r.e2_1:+.3f} | {r.e1_0:+.3f} / {r.e1_1:+.3f} | {r['多 2R 發/驗']}  {r['空 2R 發/驗']}")

print("\n\n# 全局檢查：發現期的好壞，驗證期還在嗎？（每個 型態 × 時框 一格，多空合併）")
for col, lab in (("x3", "收盤進場持有3 超額"), ("x12", "收盤進場持有12 超額"), ("e2", "突破2R 超額"), ("e1", "突破1R 超額")):
    a, b = C[f"{col}_0"], C[f"{col}_1"]
    m = a.notna() & b.notna()
    corr = a[m].corr(b[m])
    top = C[m].sort_values(f"{col}_0", ascending=False).head(20)
    bot = C[m].sort_values(f"{col}_0").head(20)
    print(f"  {lab:18s}: 發現/驗證相關 {corr:+.2f}；同號 {100 * (np.sign(a[m]) == np.sign(b[m])).mean():.0f}%；"
          f"發現期前 20 名在驗證期平均 {top[f'{col}_1'].mean():+.3f}（全部平均 {b[m].mean():+.3f}），後 20 名 {bot[f'{col}_1'].mean():+.3f}")

print("\n\n# 「實際能做」的：突破 2R 扣點差後，發現期 > 0 且 驗證期 > 0，且兩段筆數各 ≥ 100")
ok = C[(C.R2_0 > 0) & (C.R2_1 > 0) & (C.Rn2_0 >= 100) & (C.Rn2_1 >= 100)].sort_values("R2_1", ascending=False)
for _, r in ok.iterrows():
    print(f"  {r.tf:3s} {r.pattern:38s} 筆數 {int(r.Rn2_0)}/{int(r.Rn2_1)}  每筆 {r.R2_0:+.3f}R / {r.R2_1:+.3f}R  （基準 {r.B2_0:+.3f} / {r.B2_1:+.3f}）  多 {r['多 2R 發/驗']}  空 {r['空 2R 發/驗']}")
print(f"  共 {len(ok)} / {len(C)} 格")
ok1 = C[(C.R1_0 > 0) & (C.R1_1 > 0) & (C.Rn1_0 >= 100) & (C.Rn1_1 >= 100)].sort_values("R1_1", ascending=False)
print("\n  同上，目標 1R：")
for _, r in ok1.iterrows():
    print(f"  {r.tf:3s} {r.pattern:38s} 筆數 {int(r.Rn1_0)}/{int(r.Rn1_1)}  每筆 {r.R1_0:+.3f}R / {r.R1_1:+.3f}R  （基準 {r.B1_0:+.3f} / {r.B1_1:+.3f}）")
print(f"  共 {len(ok1)} / {len(C)} 格")

print("\n\n# 型態家族平均（各家族所有定義 × 時框平均）")
F = C.groupby("family")[["同向" if False else "下一根同向", "x3_0", "x3_1", "x12_0", "x12_1", "e2_0", "e2_1", "R2_0", "R2_1"]].mean()
print(F.round(3).to_string())

# ---------------- 影線比例連續分組 ----------------
print("\n\n# 影線比例（連續）→ 下一根 / 持有 3 根報酬，「看漲方向」= 下影長、上影短")
print("  不平衡 = (下影 − 上影) / 全長，+1 = 只有下影（錘子），−1 = 只有上影（流星）；報酬 = 往上的報酬（ATR），未扣成本")
m1 = load("data/XAUUSD_M1_2023_2026.csv")
for tf, rule, mins in TFS:
    df = m1.resample(rule, label="left", closed="left").agg(AGG).dropna()
    o, h, l, c = df.open, df.high, df.low, df.close
    rng = h - l
    atr = rng.rolling(20).mean()
    imb = ((np.minimum(o, c) - l) - (h - np.maximum(o, c))) / rng
    br = (c - o).abs() / rng
    r1 = (c.shift(-1) - c) / atr
    r3 = (c.shift(-3) - c) / atr
    d = pd.DataFrame({"imb": imb, "br": br, "r1": r1, "r3": r3, "per": np.where(df.index < SPLIT, "發現", "驗證"), "big": rng >= atr.shift()}).dropna()
    d["r1"] -= d.groupby("per").r1.transform("mean")
    d["r3"] -= d.groupby("per").r3.transform("mean")
    q = pd.cut(d.imb, [-1.01, -.6, -.3, -.1, .1, .3, .6, 1.01], labels=["≤−0.6", "−0.6~−0.3", "−0.3~−0.1", "±0.1", "0.1~0.3", "0.3~0.6", "≥0.6"])
    print(f"\n## {tf}")
    for lab in q.cat.categories:
        s = d[q == lab]
        a, b = s[s.per == "發現"], s[s.per == "驗證"]
        sb = s[s.big]
        print(f"  不平衡 {lab:>10s}: n={len(s):6d} 下一根 {s.r1.mean():+.3f}  3根 {s.r3.mean():+.3f} | 發現 {a.r3.mean():+.3f} / 驗證 {b.r3.mean():+.3f} | 只看全長≥1ATR: 3根 {sb.r3.mean():+.3f} (n={len(sb)})")
    qb = pd.cut(d.br, [-.01, .1, .3, .5, .7, .9, 1.01], labels=["≤10%", "10-30%", "30-50%", "50-70%", "70-90%", ">90%"])
    sign = np.sign(df.close - df.open).reindex(d.index)
    print("  實體比例 → 順這根方向的 3 根報酬（漂移調整前）：" + "  ".join(
        f"{lab}:{(((c.shift(-3) - c) / atr).reindex(d.index) * sign)[qb == lab].mean():+.3f}" for lab in qb.cat.categories))
