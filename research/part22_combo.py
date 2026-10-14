"""第二十二部分：三個因子的交互作用與 AMT 組合（事前依理論登記，2023~24 發現 / 2025~26 驗證）。

輸入：part22_levels.csv（價位觸碰事件）、part22_breakout.csv（OR 突破事件）、part22_sessions.pkl。
單位：ATR10；r60 = 60 分、rS = 抱到 session 結束；已扣點差（突破事件）/ 未扣（價位事件，點差 ≈ 0.003 ATR）。
"""
import pickle
import numpy as np
import pandas as pd
from part22_core import tstat, SESS_ORDER

E = pd.read_csv("part22_levels.csv")
B = pd.read_csv("part22_breakout.csv")
_, S = pickle.load(open("part22_sessions.pkl", "rb"))
S["day"] = S.day.astype(str)
YRS = (2023, 2024, 2025, 2026)
out = []


def bounce(g, col="b10"):
    x = g[col]
    x = x[x != 0]
    return (x > 0).mean() * 100 if len(x) else np.nan


# ---------------------------------------------------------------- E1 AMT 價位 × OR 位置
out += ["=" * 110, "E1. AMT 價位 × 碰到時相對 OR 的位置（OR 形成中 / 價位在 OR 內 / 價位在 OR 外 = 已突破 OR）",
        "    數字 = 真價位反彈% − 假價位反彈%（b = 0.10 ATR）；括號 = 真價位筆數", "=" * 110]
A = E[~E.level.isin(["ORH", "ORL", "ORM"])]
groups = {"前日價值 pPOC/pVAH/pVAL": ["pPOC", "pVAH", "pVAL"], "前日高低 PDH/PDL": ["PDH", "PDL"],
          "前一 session 高低 pSH/pSL": ["pSH", "pSL"]}
for gname, lvls in groups.items():
    out.append(f"\n  {gname}")
    for s in SESS_ORDER + ["ALL"]:
        cells = []
        for pos in ("during_or", "inside_or", "outside_or"):
            g = A[A.level.isin(lvls) & (A.orpos == pos) & ((A.sess == s) | (s == "ALL"))]
            re, pl = g[g.kind == "real"], g[g.kind == "placebo"]
            dif = bounce(re) - bounce(pl)
            yr = [bounce(re[re.year == y]) - bounce(pl[pl.year == y]) for y in YRS]
            cells.append(f"{pos:<10} {dif:+5.1f} ({len(re):4d}) [{sum(np.array(yr) > 0)}/4 年正]")
        out.append(f"    {s:<5} " + " | ".join(cells))

# ---------------------------------------------------------------- E2 OR 價位 × 開盤位置（平衡 / 失衡）× 與 AMT 價位重疊
out += ["", "=" * 110, "E2. OR 價位（區間內反應）× 開盤在前日價值區內（平衡）/ 外（失衡）× OR 邊是否與 AMT 價位重疊（±0.1 ATR）",
        "    數字 = 反彈%（真 OR）；安慰劑整體約 50%", "=" * 110]
O = E[E.level.isin(["ORH", "ORL"]) & (E.kind == "real")].drop(columns=["pvah", "pval"]).copy()
O["day"] = O.day.astype(str)
O = O.merge(S[["day", "sess", "orh", "orl", "ppoc", "pvah", "pval", "pdh", "pdl", "psh", "psl"]], on=["day", "sess"])
O["lev"] = np.where(O.level == "ORH", O.orh, O.orl)
amt = O[["ppoc", "pvah", "pval", "pdh", "pdl", "psh", "psl"]].values
O["confl"] = (np.abs(amt - O.lev.values[:, None]) <= 0.1 * O.atr.values[:, None]).any(axis=1)
for s in SESS_ORDER + ["ALL"]:
    g = O if s == "ALL" else O[O.sess == s]
    c = [f"{lab} {bounce(gg):5.1f}% ({len(gg):3d})" for lab, gg in
         (("平衡(VA內開)", g[g.open_loc == "inside"]), ("失衡(VA外開)", g[g.open_loc != "inside"]),
          ("與AMT重疊", g[g.confl]), ("不重疊", g[~g.confl]))]
    out.append(f"    {s:<5} " + " | ".join(c))

# ---------------------------------------------------------------- F 事前登記的 AMT 組合
out += ["", "=" * 110, "F. 事前依 AMT 理論登記的組合（交易方向已標明；2023~24 = 發現、2025~26 = 驗證）",
        "   r60 / rS 單位 ATR；年 = 2023 / 2024 / 2025 / 2026 的 rS 平均", "=" * 110]
R = B[B.kind == "real"].copy()
lev = E[E.kind == "real"].copy()


def report(name, x, col="rS"):
    """x：DataFrame，含 year、r60、rS（已轉成交易方向）。"""
    a, b = x[x.year <= 2024], x[x.year >= 2025]
    yr = " ".join(f"{x[x.year == y][col].mean():+.3f}" for y in YRS)
    return (f"  {name:<52} n={len(x):5d}  r60 {x.r60.mean():+.3f}  rS {x[col].mean():+.3f} (t {tstat(x[col]):+4.1f})  "
            f"發現 {a[col].mean():+.3f} / 驗證 {b[col].mean():+.3f} (t {tstat(b[col]):+4.1f})  年 {yr}")


def flip(x):
    y = x.copy()
    y["r60"], y["rS"] = -x.r60, -x.rS
    return y


rows = [
    ("基準：所有真 OR 突破，順突破", R),
    ("F1 initiative：突破時已在前日 VA 外（順突破）", R[R.beyond_va]),
    ("F1b F1 且不是往價值方向（rel ≠ toward）", R[R.beyond_va & (R.rel != "toward")]),
    ("F2 F1 + 同時突破前一 session 高/低", R[R.beyond_va & R.beyond_ps]),
    ("F2b 突破 OR 同時突破前一 session 高/低", R[R.beyond_ps]),
    ("F3 responsive：VA 外開盤、往價值方向突破 → 做反向", flip(R[R.rel == "toward"])),
    ("F3b 仍在前日 VA 內的突破 → 做反向", flip(R[~R.beyond_va])),
]
for name, x in rows:
    out.append(report(name, x))
out.append("")
for s in SESS_ORDER:
    out.append(report(f"F1 {s}", R[R.beyond_va & (R.sess == s)]))
    out.append(report(f"F2 {s}", R[R.beyond_va & R.beyond_ps & (R.sess == s)]))

# 價位事件轉成「延續」交易（bounce 的反向）
out.append("")
cont = flip(lev[lev.level.isin(["PDH", "PDL", "pSH", "pSL"])])
out.append(report("F4 前日 / 前 session 高低第一次被碰 → 順勢穿越", cont))
out.append(report("F4b 同上，且價位在 OR 外（OR 已突破）", cont[cont.orpos == "outside_or"]))
out.append(report("F4c 同上，價位在 OR 內 / OR 形成中", cont[cont.orpos != "outside_or"]))
for s in SESS_ORDER:
    out.append(report(f"F4 {s}", cont[cont.sess == s]))
out.append("")
bal = lev[lev.level.isin(["ORH", "ORL"]) & (lev.open_loc == "inside")]
out.append(report("F5 平衡日（VA 內開盤）OR 邊第一次被碰 → 回區間內", bal))
out.append(report("F5b 失衡日（VA 外開盤）OR 邊第一次被碰 → 回區間內", lev[lev.level.isin(["ORH", "ORL"]) & (lev.open_loc != "inside")]))
out.append(report("F6 ASIA 前日 POC 第一次被碰 → 反彈", lev[(lev.level == "pPOC") & (lev.sess == "ASIA")]))
out.append(report("F6b 全 session 前日 POC → 反彈", lev[lev.level == "pPOC"]))

txt = "\n".join(out)
print(txt)
open("results_part22_combo.txt", "w").write(txt + "\n")
