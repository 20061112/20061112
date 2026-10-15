"""第二十二部分分析：ΔU/D 十分位 vs 反彈率（可被推翻的預測：應單調上升）。
輸入 part22_events.pkl（part22_kramers.py），輸出 results_part22.txt、part22_deciles.png、part22_variants.csv。
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from part22_kramers import OCCS, CS, HS, NS

E = pd.read_pickle("part22_events.pkl")
E["year"] = E.time.dt.year
E["half"] = np.where(E.time < "2025-01-01", "A", "B")       # A = 2023/6~2024，B = 2025~2026/10
OUT = ["reb05", "reb10", "rebpoc"]
lines = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    lines.append(s)


def cluster_se(y, day):
    """日群聚標準誤（平均值）。"""
    y = np.asarray(y, float); m = ~np.isnan(y)
    y, day = y[m], np.asarray(day)[m]
    r = pd.Series(y - y.mean()).groupby(day).sum()
    return np.sqrt((r ** 2).sum()) / len(y)


def deciles(x, y, q=10):
    m = np.isfinite(x) & np.isfinite(y)
    b = pd.qcut(pd.Series(x[m]).rank(method="first"), q, labels=False)
    return pd.Series(y[m]).groupby(b.values).mean()


def spearman(a, b):
    a, b = pd.Series(np.asarray(a, float)), pd.Series(np.asarray(b, float))
    return a.rank().corr(b.rank())


def mono_stats(rates):
    d = np.diff(rates.values)
    rho = spearman(rates.values, np.arange(len(rates)))
    return rho, int((d > 0).sum()), rates.iloc[-1] - rates.iloc[0]


# ---- 特徵 ----
feats = {}
temps = {"none": None}
for e in ("cc", "pk"):
    for N in NS:
        temps[f"{e}{N}"] = (E[f"Deq_{e}"] / E[f"D_{e}{N}"]).replace([np.inf], np.nan)
for occ in OCCS:
    for c in CS:
        for h in HS:
            for kind in ("sh", "cur", "harm"):
                base = E[f"{occ}_c{c}_h{h}_{kind}"]
                for tn, tr in temps.items():
                    feats[(occ, c, h, kind, tn)] = base if tr is None else base * tr
# 非物理基準
for tn, tr in temps.items():
    if tr is not None:
        feats[("base", 0, 0, "volratio", tn)] = tr
ref_poc = E["range_c1.0_h1_poc"]
for N in NS:
    feats[("base", 0, 0, "zpoc", f"cc{N}")] = (E.close - ref_poc).abs() / np.sqrt(E[f"D_cc{N}"] * 60)
feats[("base", 0, 0, "pos_edge", "none")] = np.where(E.side == 1, E.pos, 1 - E.pos)
feats[("base", 0, 0, "minute", "none")] = E["min"].astype(float)

P("=" * 100)
P("第二十二部分：位能井 + Kramers 逃逸率")
P(f"事件 {len(E)} 筆，{E.day.nunique()} 個交易日（2023/6 ~ 2026/10，M1）")
for o in OUT:
    y = E[o]
    P(f"  整體反彈率 {o}: {y.mean():.4f} ± {cluster_se(y, E.day):.4f}（鞅的虛無假設 = 0.5）  有效 {y.notna().sum()}")
for o in ("fwd30", "fwd60", "fwd120"):
    P(f"  {o} 往區間內平均 {E[o].mean():+.4f} S ± {cluster_se(E[o], E.day):.4f}")
P("  逐年 reb05：", E.groupby("year").reb05.mean().round(4).to_dict())

# ---- 掃全部變體 ----
rows = []
for key, x in feats.items():
    x = np.asarray(x, float)
    r = dict(occ=key[0], c=key[1], h=key[2], kind=key[3], temp=key[4])
    for o in OUT:
        y = E[o].to_numpy()
        rates = deciles(x, y)
        rho, ninc, spread = mono_stats(rates)
        r[f"{o}_rho"], r[f"{o}_inc"], r[f"{o}_d10_1"] = rho, ninc, spread
        for hv in ("A", "B"):
            m = (E.half == hv).to_numpy()
            ra = deciles(x[m], y[m])
            r[f"{o}_rho_{hv}"], _, r[f"{o}_d10_1_{hv}"] = mono_stats(ra)
        # 事件層級的 Spearman
        mm = np.isfinite(x) & np.isfinite(y)
        r[f"{o}_ev_rho"] = spearman(x[mm], y[mm])
    rows.append(r)
V = pd.DataFrame(rows)
V.to_csv("part22_variants.csv", index=False)

phys = V[V.occ != "base"]
P("\n" + "=" * 100)
P(f"共 {len(phys)} 個物理變體（3 停留算法 × 3 格寬 × 3 平滑 × 3 障礙定義 × 7 溫度）+ {len(V) - len(phys)} 個非物理基準")
for o in OUT:
    P(f"\n[{o}] 十分位 Spearman ρ（反彈率對十分位序）分佈：")
    P(f"  中位 {phys[o + '_rho'].median():+.3f}，ρ>0.8 的比例 {(phys[o + '_rho'] > 0.8).mean():.1%}，"
      f"ρ<0 的比例 {(phys[o + '_rho'] < 0).mean():.1%}，9 個相鄰差全部為正（嚴格單調）的變體 {(phys[o + '_inc'] == 9).sum()} 個")
    P(f"  D10−D1 中位 {phys[o + '_d10_1'].median():+.4f}；前半 A 與後半 B 的 ρ 同號且 >0.5 的變體 "
      f"{((phys[o + '_rho_A'] > 0.5) & (phys[o + '_rho_B'] > 0.5)).sum()} 個")

P("\n依『障礙定義 × 溫度』分組（中位數，跨停留算法/格寬/平滑共 27 個）：")
g = phys.groupby(["kind", "temp"])[["reb05_rho", "reb05_d10_1", "reb10_rho", "rebpoc_rho", "rebpoc_d10_1", "reb05_ev_rho"]].median()
P(g.round(3).to_string())
P("\n依停留算法 × 格寬 × 平滑（K_temp 定義 kind=sh，temp=cc30）：")
P(phys[(phys.kind == "sh") & (phys.temp == "cc30")][["occ", "c", "h", "reb05_rho", "reb05_d10_1", "rebpoc_rho", "rebpoc_d10_1", "reb05_rho_A", "reb05_rho_B"]].round(3).to_string(index=False))
P("\n非物理基準：")
P(V[V.occ == "base"][["kind", "temp", "reb05_rho", "reb05_d10_1", "reb10_rho", "rebpoc_rho", "rebpoc_d10_1", "reb05_ev_rho"]].round(3).to_string(index=False))

# ---- 只用前半選變體，後半驗證 ----
P("\n" + "=" * 100)
P("前半（2023/6~2024）選 reb05 十分位 ρ 最高的 10 個物理變體，看後半（2025~2026/10）：")
top = phys.sort_values("reb05_rho_A", ascending=False).head(10)
P(top[["occ", "c", "h", "kind", "temp", "reb05_rho_A", "reb05_d10_1_A", "reb05_rho_B", "reb05_d10_1_B", "rebpoc_rho_A", "rebpoc_rho_B"]].round(3).to_string(index=False))
P(f"  所有物理變體前後半 reb05 ρ 的相關：{phys.reb05_rho_A.corr(phys.reb05_rho_B):+.3f}")

# ---- 代表變體逐十分位詳表 ----
REPS = [("range", 1.0, 1, "sh", "none", "K_shape（字面定義，D 消掉）"),
        ("range", 1.0, 1, "sh", "cc30", "K_temp = D_eq/D_30 · ln(pmax/pedge)"),
        ("range", 1.0, 1, "harm", "cc30", "K_harm（諧振子井）"),
        ("close", 1.0, 1, "sh", "pk30", "close 停留 + Parkinson"),
        ("base", 0, 0, "volratio", "cc30", "只有溫度比 D_eq/D_30")]
P("\n" + "=" * 100)
fig, axes = plt.subplots(1, len(REPS), figsize=(4.2 * len(REPS), 3.8), sharey=True)
for ax, (occ, c, h, kind, tn, label) in zip(axes, REPS):
    x = np.asarray(feats[(occ, c, h, kind, tn)], float)
    P(f"\n{label}  [{occ} c={c} h={h} {kind} {tn}]")
    tab = {}
    for o in OUT:
        tab[o] = deciles(x, E[o].to_numpy())
    for hv in ("A", "B"):
        m = (E.half == hv).to_numpy()
        tab[f"reb05_{hv}"] = deciles(x[m], E.reb05.to_numpy()[m])
    tab["fwd60"] = deciles(x, E.fwd60.to_numpy())
    m = np.isfinite(x)
    tab["K_med"] = pd.Series(x[m]).groupby(pd.qcut(pd.Series(x[m]).rank(method="first"), 10, labels=False).values).median()
    T = pd.DataFrame(tab); T.index = [f"D{i + 1}" for i in range(10)]
    P(T.round(3).to_string())
    for o, col in (("reb05", "C0"), ("reb10", "C1"), ("rebpoc", "C2")):
        ax.plot(range(1, 11), T[o].values, "o-", color=col, label=o)
    ax.plot(range(1, 11), T["reb05_A"].values, ":", color="C0", alpha=.6, label="reb05 2023-24")
    ax.plot(range(1, 11), T["reb05_B"].values, "--", color="C0", alpha=.6, label="reb05 2025-26")
    ax.axhline(0.5, color="k", lw=.6)
    ax.set_title(f"{occ} {kind} {tn}", fontsize=9)
    ax.set_xlabel("decile of ΔU/D")
axes[0].set_ylabel("rebound rate"); axes[0].legend(fontsize=7)
plt.tight_layout(); plt.savefig("part22_deciles.png", dpi=110)

# ---- 控制混淆：時間與離邊緣距離分層後的 K_temp 效果 ----
P("\n" + "=" * 100)
x = np.asarray(feats[("range", 1.0, 1, "sh", "cc30")], float)
E["_k"] = x
E["_tod"] = pd.cut(E["min"], [0, 420, 720, 900, 1300], labels=["亞洲", "倫敦", "紐約前", "紐約後"])
E["_kq"] = E.groupby("_tod", observed=True)._k.transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False))
P("K_temp 在各時段內分五分位的 reb05（時段內分位，排除『時間越晚 ln(pmax/pedge) 越大』的機械效果）：")
P(E.pivot_table(index="_tod", columns="_kq", values="reb05", aggfunc="mean", observed=True).round(3).to_string())
E["_new"] = E.newext
P("\n當根創新高/低（newext=1）vs 沒有：")
P(E.groupby(["_new", pd.qcut(E._k.rank(method="first"), 5, labels=False)]).reb05.mean().unstack().round(3).to_string())

open("results_part22.txt", "w").write("\n".join(lines) + "\n")
