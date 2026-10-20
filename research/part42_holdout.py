"""第四十二部分 C：研究期定案的 IVB 改良版，在保留期 2025/10 ~ 2026/10 只跑一次（參數不再調整）。
→ results_part42_holdout.txt、part42_holdout_trades.csv"""
import numpy as np
import pandas as pd
import part42_improve as imp
from part41_fabio import load_symbol, stats, fmt

# 研究期定案（見 README 第四十二部分）；X0 / N0 = 原版（只改成穿價成交）
FROZEN = {
    "XAUUSD": [
        ("X0 原版 vah/poc 2R", dict(entry_lv="vah", stop_lv="poc"), ("rr", 2.0), None),
        ("X1 移動停損 1.5R/1.5R", dict(entry_lv="vah", stop_lv="poc"), ("trail", 1.5, 1.5), None),
        ("X2 X1 + 17:15 後才突破", dict(entry_lv="vah", stop_lv="poc"), ("trail", 1.5, 1.5), lambda g: g.brk_m >= 17 * 60 + 15),
        ("X3 X2 + 區間不小 (z>-0.5)", dict(entry_lv="vah", stop_lv="poc"), ("trail", 1.5, 1.5),
         lambda g: (g.brk_m >= 17 * 60 + 15) & (g.or_z > -0.5)),
    ],
    "NAS100": [
        ("N0 原版 vah/poc 2R", dict(entry_lv="vah", stop_lv="poc"), ("rr", 2.0), None),
        ("N1 edge/val 2R+1R保本", dict(entry_lv="edge", stop_lv="val"), ("be", 2.0, 1.0), None),
        ("N2 N1 + 開在前日VA外", dict(entry_lv="edge", stop_lv="val"), ("be", 2.0, 1.0), lambda g: g.pva != 0),
        ("N3 N1 + 順倫敦方向", dict(entry_lv="edge", stop_lv="val"), ("be", 2.0, 1.0), lambda g: g.ldn_dir == 1),
    ],
}
STRESS = {"XAUUSD": 0.3, "NAS100": 3.0}

out, allrows = [], []
for sym, cands in FROZEN.items():
    days = load_symbol(sym)
    out.append(f"\n===== {sym} =====")
    for name, kw, ex, flt in cands:
        res = {}
        for label, th in (("穿價", imp.THROUGH[sym]), ("壓力", STRESS[sym])):
            base_th = imp.THROUGH[sym]
            imp.THROUGH[sym] = th
            g = imp.with_exit(imp.run_ivb(days, sym, **kw), ex)
            imp.THROUGH[sym] = base_th
            if flt is not None:
                g = g[flt(g)]
            res[label] = g
        g = res["穿價"]
        rs, ho = g[g.day < imp.RESEARCH_END], g[g.day >= imp.RESEARCH_END]
        hs = res["壓力"][res["壓力"].day >= imp.RESEARCH_END]
        out.append(f"\n[{name}]")
        out.append("  研究期  " + fmt(stats(rs.R)))
        out.append("  保留期  " + fmt(stats(ho.R)))
        out.append(f"  保留期（穿價 {STRESS[sym]} 才成交）" + fmt(stats(hs.R)))
        out.append("  保留期 + 每筆多扣 1 倍點差 " + fmt(stats(ho.R - ho.cost / ho.R_pts)))
        mo = ho.groupby(ho.day.dt.to_period("M")).R.sum()
        out.append("  保留期逐月：" + ", ".join(f"{k.month}月 {v:+.1f}" for k, v in mo.items()))
        allrows.append(g.assign(sym=sym, cand=name, period=np.where(g.day < imp.RESEARCH_END, "research", "holdout")))
txt = "\n".join(out)
print(txt)
open("results_part42_holdout.txt", "w").write(txt + "\n")
pd.concat(allrows).drop(columns=["i"]).to_csv("part42_holdout_trades.csv", index=False)
