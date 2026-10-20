"""第四十二部分 A：第四十一部分逐筆交易的細部統計（MFE/MAE、持有時間、進場時段、連虧、月份）。
→ results_part42_detail.txt、part42_trades_detail.csv"""
import numpy as np
import pandas as pd
from part41_fabio import load_symbol, NY_CLOSE

t = pd.read_csv("part41_trades.csv", parse_dates=["day"])
rows = []
for sym, g in t.groupby("sym"):
    days = load_symbol(sym)
    for _, r in g.iterrows():
        d = days[r.day]
        M, H, L = d["M"], d["H"], d["L"]
        i, k = np.searchsorted(M, r.t_in), np.searchsorted(M, r.t_out)
        R, s = r.R_pts, r.side
        seg = slice(i + 1 if k > i else i, k + 1)   # 同一根就停損的單，只看那一根
        fav = (H[seg].max() - r.entry) if s > 0 else (r.entry - L[seg].min())
        adv = (r.entry - L[seg].min()) if s > 0 else (H[seg].max() - r.entry)
        # 出場後到紐約收盤，原方向還能走多遠（看停利是不是太早）
        post = slice(k + 1, np.searchsorted(M, NY_CLOSE))
        if post.stop > post.start:
            after = ((H[post].max() - r.entry) if s > 0 else (r.entry - L[post].min())) / R
        else:
            after = np.nan
        rows.append(dict(r, mfe=fav / R, mae=adv / R, after_max=after))
df = pd.DataFrame(rows)
df["hold"] = df.t_out - df.t_in
df["hour"] = df.t_in // 60
df["month"] = df.day.dt.to_period("M")
df.to_csv("part42_trades_detail.csv", index=False)


def streak(r):
    best = cur = 0
    for x in r:
        cur = cur + 1 if x < 0 else 0
        best = max(best, cur)
    return best


out = []
for (sym, strat), g in df.groupby(["sym", "strat"]):
    if strat.endswith("-2.5R") or strat.endswith("-to-21"):
        continue
    g = g.sort_values(["day", "t_in"])
    w, l = g[g.R > 0], g[g.R <= 0]
    out.append(f"\n=== {sym} {strat}  n={len(g)}  淨 {g.R.mean():+.3f}R ===")
    out.append(f"  平均賺 {w.R.mean():+.2f}R / 平均虧 {l.R.mean():+.2f}R；中位持有 {g.hold.median():.0f} 分；最長連虧 {streak(g.R)} 筆；"
               f"中位停損距離 {g.R_pts.median():.2f}")
    out.append(f"  虧損單中，曾經浮盈 ≥0.5R：{(l.mfe >= 0.5).mean():.0%}，≥1R：{(l.mfe >= 1).mean():.0%}"
               f"（→ 1R 保本規則能救的比例）")
    tg = g[g.exit == "target"]
    if len(tg):
        out.append(f"  停利單出場後到紐約收盤，原方向最遠中位 {tg.after_max.median():.2f}R（停利 {tg.R_gross.median():.1f}R）；"
                   f"其中 ≥ 3R 的 {(tg.after_max >= 3).mean():.0%}")
    out.append(f"  MFE 中位 {g.mfe.median():.2f}R、MAE 中位 {g.mae.median():.2f}R")
    hr = g.groupby("hour").R.agg(["size", "mean"])
    out.append("  進場時段（broker 時）：" + ", ".join(f"{h}點 {int(a)}筆 {m:+.2f}" for h, (a, m) in hr.iterrows()))
    mo = g.groupby("month").R.sum()
    out.append(f"  月份：{(mo > 0).sum()}/{len(mo)} 個月為正，最好 {mo.max():+.1f}R、最差 {mo.min():+.1f}R")
txt = "\n".join(out)
print(txt)
open("results_part42_detail.txt", "w").write(txt + "\n")
