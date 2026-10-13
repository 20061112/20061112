"""第二十八部分報表 → results_part28.txt、part28_heatmap.png"""
import numpy as np
import pandas as pd
from math import comb, ceil

G = pd.read_csv("part28_grid.csv").drop_duplicates(["tf", "N", "k", "h"])   # 不同比例算出同一個 k 的只留一格
out = open("results_part28.txt", "w")


def P(s=""):
    print(s)
    out.write(s + "\n")


NS = sorted(G.N.unique())
FR = sorted(G.fr.unique())
TFS = ["M1", "M5", "M15", "M30", "H1"]
P("第二十八部分：紅綠不平衡網格（切片 N × 多數色比例 × 持有 h × 時框）")
P("對稱值 = (陽多數做多 + 陰多數做空) / 2，已扣漂移，單位 ATR；> 0 延續、< 0 反轉。發現 2023/6~2024/12 / 驗證 2025/1~2026/10")
P(f"網格：N {[int(x) for x in NS]}，比例 {[float(x) for x in FR]}，h {[int(x) for x in sorted(G.h.unique())]}；每邊每段 < 50 筆的格子不算")

P("\n# 1. 出現頻率（M15）：多數色 ≥ 比例 的 K 棒占比 / 丟銅板理論值")
m = G[(G.tf == "M15") & (G.h == 1)]
P("   N   " + " ".join(f"{f:>13.0%}" for f in FR))
for N in NS:
    cells = []
    for f in FR:
        r = m[(m.N == N) & (m.fr == f)]
        k = ceil(f * N - 1e-9)
        th = 2 * sum(comb(N, j) for j in range(k, N + 1)) / 2 ** N if k > N / 2 else np.nan
        cells.append(f"{100 * r.freq.iloc[0]:5.2f}/{100 * th:5.2f}%" if len(r) else " " * 13)
    P(f"  {N:3d} " + " ".join(cells))

P("\n# 2. 全局一致性：每個時框所有格子，發現期 vs 驗證期的對稱值")
for tf in TFS:
    d = G[(G.tf == tf)].dropna(subset=["sym0", "sym1"])
    parts = []
    for h in sorted(d.h.unique()):
        e = d[d.h == h]
        parts.append(f"h{h}: 相關 {e.sym0.corr(e.sym1):+.2f} 同號 {100 * (np.sign(e.sym0) == np.sign(e.sym1)).mean():3.0f}%")
    P(f"  {tf:3s}（{len(d)} 格）延續格比例 發現 {100 * (d.sym0 > 0).mean():3.0f}% / 驗證 {100 * (d.sym1 > 0).mean():3.0f}% | " + " | ".join(parts))

for h in (6, 24):
    P(f"\n# 3. 熱圖（h = {h}）：對稱值 發現/驗證（ATR；+ 延續、− 反轉；空白 = 樣本不足）")
    for tf in TFS:
        d = G[(G.tf == tf) & (G.h == h)]
        cst = d[["cost0", "cost1"]].mean().round(3).tolist()
        P(f"\n## {tf}（平均點差 發現 {cst[0]:.3f} / 驗證 {cst[1]:.3f} ATR）")
        P("   N   " + " ".join(f"{f:>12.0%}" for f in FR))
        for N in NS:
            cells = []
            for f in FR:
                r = d[(d.N == N) & (d.fr == f)]
                if len(r) and np.isfinite(r.sym0.iloc[0]) and np.isfinite(r.sym1.iloc[0]):
                    cells.append(f"{r.sym0.iloc[0]:+.2f}/{r.sym1.iloc[0]:+.2f}")
                else:
                    cells.append("")
            P(f"  {N:3d} " + " ".join(f"{c:>12s}" for c in cells))

P("\n\n# 4. 候選格：同時滿足")
P("   (a) 發現、驗證的對稱值同號；(b) 兩段裡多邊、空邊也都同號（不是靠做多 beta）；(c) 兩段 |對稱值| 都 > 點差")
G["dir"] = np.sign(G.sym0)
ok = G.dropna(subset=["sym0", "sym1", "L0", "S0", "L1", "S1"])
c = ok[(np.sign(ok.sym1) == ok.dir) & (np.sign(ok.L0) == ok.dir) & (np.sign(ok.S0) == ok.dir) & (np.sign(ok.L1) == ok.dir)
       & (np.sign(ok.S1) == ok.dir) & (ok.sym0.abs() > ok.cost0) & (ok.sym1.abs() > ok.cost1)]
P(f"   共 {len(c)} / {len(ok)} 格")
for _, r in c.sort_values(["tf", "h", "N"]).iterrows():
    P(f"   {r.tf:3s} N={r.N:3d} ≥{r.fr:.0%}（{r.k}根） h={r.h:2d} {'延續' if r.dir > 0 else '反轉'} | 發現 多 {r.L0:+.3f} 空 {r.S0:+.3f} (n {r.nL0}/{r.nS0}, 日t {r.t0:+.1f}) "
      f"| 驗證 多 {r.L1:+.3f} 空 {r.S1:+.3f} (n {r.nL1}/{r.nS1}, 日t {r.t1:+.1f}) | 點差 {r.cost0:.3f}/{r.cost1:.3f}")

P("\n\n# 5. 只用發現期挑：每個時框 |對稱值| − 點差 最大的 10 格（方向照發現期），驗證期只看一次")
G["edge0"] = G.sym0.abs() - G.cost0
G["v_dir"] = G.dir * G.sym1 - G.cost1
for tf in TFS[1:]:
    d = G[(G.tf == tf)].dropna(subset=["sym0", "sym1"]).sort_values("edge0", ascending=False).head(10)
    P(f"\n## {tf}：前 10 格在驗證期平均 {d.v_dir.mean():+.3f} ATR（扣點差），驗證 > 0 的 {(d.v_dir > 0).sum()} / 10")
    for _, r in d.iterrows():
        P(f"   N={r.N:3d} ≥{r.fr:.0%} h={r.h:2d} {'延續' if r.dir > 0 else '反轉'}: 發現 {r.edge0:+.3f}（多 {r.L0:+.2f} 空 {r.S0:+.2f}, 日t {r.t0:+.1f}）"
          f" → 驗證 {r.v_dir:+.3f}（多 {r.L1:+.2f} 空 {r.S1:+.2f}, 日t {r.t1:+.1f}）")

P("\n\n# 6. 實際損益（不扣漂移、扣點差）：順勢 = 陽多數做多 + 陰多數做空。看多空各自的帳")
P("   每個時框 h=24 的格子中：做多那邊 > 0 的比例、做空那邊 > 0 的比例（兩段）")
for tf in TFS:
    d = G[(G.tf == tf) & (G.h == 24)]
    P(f"  {tf:3s} 順勢做多 >0：發現 {100 * (d.rawL0 > 0).mean():3.0f}% 驗證 {100 * (d.rawL1 > 0).mean():3.0f}% | 順勢做空 >0：發現 {100 * (d.rawS0 > 0).mean():3.0f}% 驗證 {100 * (d.rawS1 > 0).mean():3.0f}%")
out.close()


def heat():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tfs = ["M5", "M15", "M30", "H1"]
    fig, ax = plt.subplots(len(tfs), 4, figsize=(17, 4 * len(tfs)))
    for i, tf in enumerate(tfs):
        for j, (h, q) in enumerate([(6, 0), (6, 1), (24, 0), (24, 1)]):
            d = G[(G.tf == tf) & (G.h == h)].pivot(index="N", columns="fr", values=f"sym{q}")
            v = 0.5 if h == 24 else 0.2
            im = ax[i, j].imshow(d.values, cmap="RdBu", vmin=-v, vmax=v, aspect="auto")
            ax[i, j].set_xticks(range(len(d.columns)), [f"{c:.0%}" for c in d.columns], fontsize=7)
            ax[i, j].set_yticks(range(len(d.index)), d.index, fontsize=7)
            ax[i, j].set_title(f"{tf} hold {h}  {'2023/6-2024' if q == 0 else '2025-2026/10'}", fontsize=9)
            if j == 0:
                ax[i, j].set_ylabel("window N")
            plt.colorbar(im, ax=ax[i, j], fraction=.046)
    fig.suptitle("Color imbalance: drift-neutral follow return (ATR). Blue = continuation, red = reversal. x = majority share ≥", fontsize=11)
    plt.tight_layout()
    plt.savefig("part28_heatmap.png", dpi=95)


heat()
