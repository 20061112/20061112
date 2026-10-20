"""第四十一部分補充：IVB 限價回踩版的參數鄰域、成本壓力、限價穿價成交。→ results_part41_ivb.txt"""
import numpy as np
import pandas as pd
from part41_fabio import load_symbol, ivb, stats, fmt

out = []
for sym in ("XAUUSD", "NAS100"):
    days = load_symbol(sym)
    out.append(f"\n===== {sym}  IVB-limit 參數鄰域（淨 R） =====")
    grid = []
    for or_len in (15, 30, 60):
        for rr in (1.5, 2.0, 2.5, 3.0):
            for buf in (0.0, 0.02, 0.1):
                tr = [t for day, d in days.items() for t in ivb(day, d, "limit", rr=rr, or_len=or_len, buf_f=buf)]
                g = pd.DataFrame(tr)
                g["y"] = pd.to_datetime(g.day).dt.year
                yr = g.groupby("y").R.mean()
                s = stats(g.R)
                grid.append(dict(or_len=or_len, rr=rr, buf=buf, n=s["n"], avg=s["avg"], t=s["t"], pf=s["pf"],
                                 yrs_pos=f"{(yr > 0).sum()}/{len(yr)}", long=g[g.side > 0].R.mean(), short=g[g.side < 0].R.mean()))
    gd = pd.DataFrame(grid)
    out.append(gd.round(3).to_string(index=False))
    out.append(f"為正的參數組：{(gd.avg > 0).mean():.0%}，中位 {gd.avg.median():+.3f}R")
    # 基準參數下的成本壓力：每筆再加 0.5 / 1 個中位點差的滑價
    g = pd.DataFrame([t for day, d in days.items() for t in ivb(day, d, "limit")])
    med_sp = g.cost_R.median()
    for k in (0.5, 1.0, 2.0):
        out.append(f"每筆額外扣 {k} 倍點差：" + fmt(stats(g.R - k * g.cost_R)))
    out.append(f"中位停損距離 {g.R_pts.median():.2f}，中位成本 {med_sp:.3f}R")
    for th in ((0.1, 0.3) if sym == "XAUUSD" else (1.0, 3.0)):
        gt = pd.DataFrame([t for day, d in days.items() for t in ivb(day, d, "limit", through=th)])
        out.append(f"限價要穿價 {th} 才成交：" + fmt(stats(gt.R)))
    out.append("逐半年：" + ", ".join(f"{k}: {v:+.3f}" for k, v in g.groupby(pd.to_datetime(g.day).dt.to_period("Q").map(lambda p: f'{p.year}H{1 + (p.quarter > 2)}')).R.mean().items()))
open("results_part41_ivb.txt", "w").write("\n".join(out) + "\n")
print("\n".join(out))
