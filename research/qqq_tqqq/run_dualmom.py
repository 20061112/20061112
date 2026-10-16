"""雙動能 T-bill 避險的穩健性：
  1. 參數鄰域：動能回看 6 / 9 / 12 個月 × 趨勢 MA150 / 200 / 250 × 比較基準（T-bill / 0）
  2. 逐次空頭：哪幾段在 T-bill、各段省了多少
  3. 區塊拼貼蒙地卡羅（與前面同法）
輸出 results_dualmom.txt、results_dualmom.csv。
"""
import itertools
import numpy as np
import pandas as pd
import engine as E
from run_tests import q, OK0, Z, c, w_rule
from run_drawdown import lump, mdd

out = []
P = lambda s: (print(s, flush=True), out.append(s))
base_w = w_rule(OK0)
b = E.stats(E.run(q, base_w)); broll = E.rolling_windows(q, base_w)
P(f"BASE irr99 {b['irr']:.3f} 最慘 {b['worst']:.2f} 10 年段最慘最差 {broll.worst.min():.2f}")


def dm_off(cc, rfs, ma, look, vs_rf=True):
    below = cc < cc.rolling(ma).mean()
    mom = cc / cc.shift(look) - 1
    hurdle = rfs.rolling(look).sum() if vs_rf else 0.0
    return (below & (mom < hurdle)).shift(1, fill_value=False)


rows = []
P("\n1. 參數鄰域（槓桿規則不變，只改「何時把 1 倍部位轉 T-bill」）")
for ma, look, vs in itertools.product([150, 200, 250], [126, 189, 252], [True, False]):
    off = dm_off(c, q.rf, ma, look, vs)
    f = E.stats(E.run(q, base_w, off=off.values)); roll = E.rolling_windows(q, base_w, off=off.values)
    d = roll.irr.values - broll.irr.values
    ls = lump(base_w, off.values)
    rows.append(dict(ma=ma, look=look, hurdle="T-bill" if vs else "0", irr99=f["irr"], worst=f["worst"],
                     worst10_min=roll.worst.min(), worst10_med=roll.worst.median(), d10=np.median(d), mdd=mdd(ls),
                     off_frac=off.mean(), sw=f["sw_per_year"]))
    P(f"  MA{ma} 動能{look:3d}d 門檻 {'T-bill' if vs else '0':6s} irr99 {f['irr']:.3f} dIRR10y {np.median(d):+.3f} 最慘 {f['worst']:.2f} "
      f"10年段最慘 最差 {roll.worst.min():.2f} 中位 {roll.worst.median():.2f} MDD {mdd(ls):.2f} T-bill 時間 {off.mean():.0%} 換倉 {f['sw_per_year']:.1f}")
R = pd.DataFrame(rows)
R.to_csv("results_dualmom.csv", index=False, float_format="%.4f")
P(f"  18 組：IRR {R.irr99.min():.3f} ~ {R.irr99.max():.3f}（原 {b['irr']:.3f}）；最慘 {R.worst.min():.2f} ~ {R.worst.max():.2f}（原 {b['worst']:.2f}）；"
  f"10 年段最慘最差 {R.worst10_min.min():.2f} ~ {R.worst10_min.max():.2f}（原 {broll.worst.min():.2f}）")

P("\n2. 逐次空頭（MA200、12 月、門檻 T-bill）：QQQ 跌段的一次投入報酬")
off = dm_off(c, q.rf, 200, 252)
lb = lump(base_w); ld = lump(base_w, off.values)
for a, z_ in [("2000-03-27", "2002-10-09"), ("2007-10-31", "2009-03-09"), ("2011-04-29", "2011-10-03"), ("2015-07-20", "2016-02-11"),
              ("2018-08-29", "2018-12-24"), ("2020-02-19", "2020-03-23"), ("2021-11-19", "2022-12-28"), ("2024-12-16", "2025-04-08")]:
    s = slice(a, z_)
    P(f"  {a} ~ {z_}：QQQ {(1 + q.r[s]).prod() - 1:+.0%}｜原策略 {(1 + lb[s]).prod() - 1:+.0%}｜雙動能 {(1 + ld[s]).prod() - 1:+.0%}｜T-bill 時間 {off[s].mean():.0%}")
P("  反面：轉 T-bill 後錯過的反彈（各年度）")
yb = (1 + lb).groupby(lb.index.year).prod() - 1; yd = (1 + ld).groupby(ld.index.year).prod() - 1
diff = (yd - yb)
P("  " + "、".join(f"{y} {v:+.0%}" for y, v in diff[diff.abs() > 0.02].items()))

P("\n3. 區塊拼貼蒙地卡羅（300 條 10 年路徑）")
import run_ideas_mc as MC
res = []
for k in range(300):
    d = MC.path(); cc = d.close; z = E.atr_z(d)
    w = E.weights_from_rules(d, (cc > cc.rolling(200).mean()) & (z < 0.5), ~((cc > cc.rolling(200).mean()) & (z < 0.5)))
    s0 = d.index[MC.WARM]; row = {}
    st = E.stats(E.run(d, w, start=s0, z=z)); row["b"] = st["irr"]; row["bw"] = st["worst"]
    for look in [126, 252]:
        o = dm_off(cc, d.rf, 200, look).values
        st = E.stats(E.run(d, w, start=s0, z=z, off=o)); row[f"d{look}"] = st["irr"]; row[f"d{look}w"] = st["worst"]
    o = (cc < cc.rolling(200).mean()).shift(1, fill_value=False).values
    st = E.stats(E.run(d, w, start=s0, z=z, off=o)); row["t"] = st["irr"]; row["tw"] = st["worst"]
    res.append(row)
M = pd.DataFrame(res)
P(f"  原策略：IRR 中位 {M.b.median():.3f}；最慘 中位 {M.bw.median():.2f}、10% {M.bw.quantile(.1):.2f}、5% {M.bw.quantile(.05):.2f}")
for key, nm in [("d252", "雙動能 12 月"), ("d126", "雙動能 6 月"), ("t", "MA200 下全轉 T-bill")]:
    dd = M[key] - M.b; dw = M[key + "w"] - M.bw
    P(f"  {nm:16s} IRR 差 中位 {dd.median():+.4f} 平均 {dd.mean():+.4f} 勝率 {(dd > 0).mean():.0%}｜最慘 中位 {M[key + 'w'].median():.2f}、"
      f"10% {M[key + 'w'].quantile(.1):.2f}、5% {M[key + 'w'].quantile(.05):.2f}｜最慘改善 中位 {dw.median():+.3f}、改善的路徑比例 {(dw > 0.005).mean():.0%}")
open("results_dualmom.txt", "w").write("\n".join(out) + "\n")
