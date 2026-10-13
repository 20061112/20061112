"""第二十五部分報表 → results_part25.txt"""
import numpy as np
import pandas as pd

D = pd.read_csv("part25_exits_raw.csv.gz")
D["rr"] = D.rr.astype(str)
K = ["tf", "pattern", "family", "stop", "rr", "hold", "per"]
G = D.groupby(K, sort=False)[["n", "sumR", "sumR2", "sumA", "win"]].sum().reset_index()   # 多空合併
G["R"] = G.sumR / G.n
G["A"] = G.sumA / G.n
G["t"] = G.R / np.sqrt((G.sumR2 / G.n - G.R ** 2) / G.n)
W = G.pivot_table(index=["tf", "pattern", "family", "stop", "rr", "hold"], columns="per", values=["R", "A", "n", "t"]).reset_index()
W.columns = [a if b == "" else f"{a}{b}" for a, b in W.columns]
base = W[W.family == "基準"].set_index(["tf", "stop", "rr", "hold"])[["R0", "R1", "A0", "A1"]]
W = W.join(base, on=["tf", "stop", "rr", "hold"], rsuffix="_b")
W["ex0"], W["ex1"] = W.R0 - W.R0_b, W.R1 - W.R1_b
P = W[W.family != "基準"]
TFS = ["M1", "M5", "M15", "M30"]
STOPS = ["型態端點", "0.5", "1.0", "1.5", "2.0", "3.0"]
RRS = ["0.5", "1.0", "1.5", "2.0", "3.0", "無"]

print("第二十五部分：型態 × 出場參數（停損方式 × 盈虧比 × 持有）在 M1 / M5 / M15（M30 對照）")
print("每筆 R 已扣點差；多空合併；發現 2023/6~2024/12、驗證 2025/1~2026/10\n")

print("# 1. 沒有進場優勢時（隨機 K 棒），出場參數能做出什麼？每筆 R：發現 / 驗證（持有 24 根）")
for tf in TFS:
    print(f"\n## {tf}")
    print(f"  {'停損':8s} " + " ".join(f"{'RR ' + r:>15s}" for r in RRS))
    for s in STOPS:
        cells = []
        for r in RRS:
            x = base.loc[(tf, s, r, 24)] if (tf, s, r, 24) in base.index else None
            cells.append(f"{x.R0:+.3f}/{x.R1:+.3f}" if x is not None else "")
        print(f"  {s:8s} " + " ".join(f"{c:>15s}" for c in cells))
print("\n  → 隨機進場下，每一種 停損 × 盈虧比 都 < 0（虧的大約就是點差 ÷ 停損距離）；停損越大，點差占比越小，虧得越少，但不會變正。")
print("    盈虧比只改變「勝率 × 賺賠比」的分配，不改變期望值。")

print("\n\n# 2. 型態有沒有比隨機好？每個 型態 × 時框，在 72 種出場（6 停損 × 6 盈虧比 × 2 持有）中")
print("     兩段都勝過隨機的比例、超額 R 中位數（發現 / 驗證）")
for tf in TFS:
    d = P[(P.tf == tf) & (P.n0 >= 100) & (P.n1 >= 100)]
    s = d.groupby("pattern", sort=False).agg(both=("ex0", lambda x: np.nan), ex0=("ex0", "median"), ex1=("ex1", "median"), n=("n0", "max"))
    s["both"] = d.assign(b=(d.ex0 > 0) & (d.ex1 > 0)).groupby("pattern", sort=False).b.mean()
    s = s.sort_values("ex0", ascending=False)
    print(f"\n## {tf}：整體兩段都勝過隨機 {100 * ((d.ex0 > 0) & (d.ex1 > 0)).mean():.0f}%（全部出場組合 × 型態）；發現/驗證超額相關 {d.ex0.corr(d.ex1):+.2f}")
    for pn, r in s.head(8).iterrows():
        print(f"    發現前段 {pn:36s} 超額中位 {r.ex0:+.3f} / {r.ex1:+.3f}  兩段都勝 {100 * r.both:3.0f}%")
    for pn, r in s.tail(3).iterrows():
        print(f"    發現後段 {pn:36s} 超額中位 {r.ex0:+.3f} / {r.ex1:+.3f}  兩段都勝 {100 * r.both:3.0f}%")

print("\n\n# 3. 「用發現期挑最佳出場」會不會延續？每個 型態 × 時框 在發現期挑 72 種出場中最好的一個，看驗證期")
for tf in TFS:
    d = P[(P.tf == tf) & (P.n0 >= 100) & (P.n1 >= 100)]
    best = d.loc[d.groupby("pattern").R0.idxmax()]
    dflt = d[(d.stop == "型態端點") & (d.rr == "2.0") & (d.hold == 24)].set_index("pattern")
    print(f"\n## {tf}：{len(best)} 個型態")
    print(f"  挑到的最佳出場：發現期平均 {best.R0.mean():+.3f}R → 驗證期 {best.R1.mean():+.3f}R；驗證期 > 0 的 {(best.R1 > 0).sum()} / {len(best)}")
    print(f"     扣掉「隨機進場用同一組出場」後的超額：發現 {best.ex0.mean():+.3f}R → 驗證 {best.ex1.mean():+.3f}R；驗證超額 > 0 的 {(best.ex1 > 0).sum()} / {len(best)}")
    da = d[d.stop != "型態端點"]
    bx = da.loc[da.groupby("pattern").ex0.idxmax()]
    print(f"  改用「超額」挑最佳出場（只限 ATR 停損，基準才可比）：發現超額 {bx.ex0.mean():+.3f}R → 驗證超額 {bx.ex1.mean():+.3f}R；驗證 > 0 的 {(bx.ex1 > 0).sum()} / {len(bx)}")
    print(f"  固定預設（型態端點、2R、24 根）：發現 {dflt.R0.mean():+.3f}R → 驗證 {dflt.R1.mean():+.3f}R")
    print(f"  挑到的停損分布：{best.stop.value_counts().to_dict()}  盈虧比：{best.rr.value_counts().to_dict()}")
    top = best.sort_values("R0", ascending=False).head(5)
    for _, r in top.iterrows():
        print(f"    {r.pattern:36s} 停損 {r.stop:>5s} RR {r.rr:>3s} 持有 {r.hold:2d}: 發現 {r.R0:+.3f}R (t{r.t0:+.1f}, n={int(r.n0)}) → 驗證 {r.R1:+.3f}R (t{r.t1:+.1f}, n={int(r.n1)})")

print("\n  注意：「型態端點」停損的隨機基準用的是隨機 K 棒自己的低點，和型態的停損距離不同，超額只在 ATR 停損下可比。")
print("\n\n# 4. 事先指定：第二十四部分的候選「晨星 / 夜星 A（經典）」，完整出場表（持有 24 根），每筆 R 發現 / 驗證，括號 = 驗證期 − 隨機")
for tf in TFS:
    d = P[(P.tf == tf) & (P.pattern.str.startswith("晨星A")) & (P.hold == 24)].set_index(["stop", "rr"])
    n0, n1 = int(d.n0.max()), int(d.n1.max())
    print(f"\n## {tf}（筆數 {n0} / {n1}）")
    print(f"  {'停損':8s} " + " ".join(f"{'RR ' + r:>22s}" for r in RRS))
    for s in STOPS:
        cells = []
        for r in RRS:
            x = d.loc[(s, r)]
            cells.append(f"{x.R0:+.2f}/{x.R1:+.2f}({x.ex1:+.2f})")
        print(f"  {s:8s} " + " ".join(f"{c:>22s}" for c in cells))

print("\n\n# 5. 全部 型態 × 時框 × 出場 中，兩段都 > 0 且驗證 t ≥ 2 的（n 各 ≥ 200）")
q = P[(P.R0 > 0) & (P.R1 > 0) & (P.t1 >= 2) & (P.n0 >= 200) & (P.n1 >= 200)].sort_values("t1", ascending=False)
print(f"  共 {len(q)} / {len(P)} 格（格子之間高度相關，不是獨立檢定；只列出供參考）")
for _, r in q.head(25).iterrows():
    print(f"    {r.tf:3s} {r.pattern:36s} 停損 {r.stop:>5s} RR {r.rr:>3s} 持有 {r.hold:2d}: 發現 {r.R0:+.3f} (t{r.t0:+.1f}) 驗證 {r.R1:+.3f} (t{r.t1:+.1f}) n={int(r.n0)}/{int(r.n1)}")
