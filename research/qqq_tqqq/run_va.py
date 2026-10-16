"""速度 v / 加速度 a 的 EWMA z-score（M1DIP 與 kline-angle-velocity 分支的日線版）。

  v_n = EMA(日報酬, span n) / ATR%20                （M1DIP：EMA(Δ, 20) / ATR60）
  a_n = v_n − v_n(k 天前)，k = max(n/4, 3)         （角度分支：速度的變化）
  z(x) = (x − EWM 平均) / EWM 標準差，半衰期 hl = 63 / 252 天
四種接法（其他規則不變）：
  TREND  取代 MA200：v_n > 0（或 z_v > 0）且 ATR z < 0.5
  EXIT   持 TQQQ 時 z_a < −k 或 z_v < −k → 換回 QQQ（減速就下車）
  EARLY  提早上車：MA200 之上 或 z_v > k（MA200 下的強勢起漲）
  PANIC  恐慌補倉：z_v < −k（急跌）時把現金買 QQQ，取代 ATR z > 2
有超出雜訊（10 年 IRR 差 > ±3%，或最慘改善 ≥ 5 點）的做蒙地卡羅。
輸出 results_va.txt、results_va.csv。
"""
import itertools
import numpy as np
import pandas as pd
import engine as E
from run_tests import q, OK0, Z, ABOVE, c, r, w_rule

out = []
P = lambda s: (print(s, flush=True), out.append(s))


def atrp(d, n=20):
    pc = d.close.shift(1)
    tr = np.maximum(d.high - d.low, np.maximum((d.high - pc).abs(), (d.low - pc).abs()))
    return tr.rolling(n).mean() / d.close


def va(d, n):
    v = d.close.pct_change().ewm(span=n, adjust=False).mean() / atrp(d)
    a = v - v.shift(max(n // 4, 3))
    return v, a


def ewz(x, hl):
    m = x.ewm(halflife=hl, min_periods=hl).mean(); s = x.ewm(halflife=hl, min_periods=hl).std()
    return (x - m) / s


base_w = w_rule(OK0)
b = E.stats(E.run(q, base_w)); broll = E.rolling_windows(q, base_w)
rows = []


def add(group, label, w, zpanic=None):
    kw = {} if zpanic is None else dict(z=zpanic)
    f = E.stats(E.run(q, w, **kw)); s10 = E.stats(E.run(q, w, start="2010-02-12", **kw))
    roll = E.rolling_windows(q, w, **kw)
    d = roll.irr.values - broll.irr.values
    rows.append(dict(group=group, label=label, irr99=f["irr"], irr10=s10["irr"], worst=f["worst"], worst10_min=roll.worst.min(),
                     d10=np.median(d), win10=int((d > 0).sum()), sw=f["sw_per_year"]))
    P(f"{group:6s} {label:34s} irr99 {f['irr']:.3f} irr10 {s10['irr']:.3f} dIRR10y {np.median(d):+.3f} win10 {(d > 0).sum():2d}/17 "
      f"最慘 {f['worst']:.2f} 10年段最慘 {roll.worst.min():.2f} 換倉 {f['sw_per_year']:.1f}")


P(f"BASE   原策略{'':29s} irr99 {b['irr']:.3f} 最慘 {b['worst']:.2f} 10年段最慘 {broll.worst.min():.2f} 換倉 {b['sw_per_year']:.1f}")
P("雜訊基準（只改星期幾 / 延遲）：10 年 IRR 差 −2.0% ~ +3.3%、最慘 −0.51 ~ −0.57")
for n, hl in itertools.product([20, 50, 100], [63, 252]):
    v, a = va(q, n); zv = ewz(v, hl); za = ewz(a, hl)
    tag = f"n={n},hl={hl}"
    if hl == 252:
        add("TREND", f"v>0 取代 MA200 {tag}", w_rule((v > 0) & (Z < 0.5)))
    add("TREND", f"z_v>0 取代 MA200 {tag}", w_rule((zv > 0).fillna(False) & (Z < 0.5)))
    for k in [1.5, 2.0]:
        add("EXIT", f"z_a<−{k} 下車 {tag}", w_rule(OK0 & ~(za < -k).fillna(False)))
        add("EXIT", f"z_v<−{k} 下車 {tag}", w_rule(OK0 & ~(zv < -k).fillna(False)))
    for k in [1.0, 1.5]:
        add("EARLY", f"MA200 或 z_v>{k} {tag}", w_rule((ABOVE | (zv > k).fillna(False)) & (Z < 0.5)))
    for k in [2.0, 2.5]:
        trig = (zv < -k).fillna(False)
        zp = pd.Series(np.where(trig, 3.0, 0.0), index=q.index)
        add("PANIC", f"z_v<−{k} 補倉 {tag}", base_w, zpanic=zp)

R = pd.DataFrame(rows)
R.to_csv("results_va.csv", index=False, float_format="%.4f")
P("\n各接法整片網格彙總")
G = R.groupby("group").agg(n=("irr99", "size"), irr99_med=("irr99", "median"), irr99_min=("irr99", "min"), irr99_max=("irr99", "max"),
                           d10_med=("d10", "median"), d10_max=("d10", "max"), worst_med=("worst", "median"),
                           worst_best=("worst", "max"), worst_worst=("worst", "min"))
P(G.to_string(float_format=lambda x: f"{x:.3f}"))

# 超出雜訊的候選 → 蒙地卡羅
cand = R[(R.d10 > 0.03) | (R.worst > b["worst"] + 0.05)]
P(f"\n超出雜訊的組合：{len(cand)} 組")
P(cand[["group", "label", "irr99", "d10", "worst"]].to_string(index=False, float_format=lambda x: f"{x:.3f}") if len(cand) else "  （無）")
if len(cand):
    import run_ideas_mc as MC
    specs = []
    for _, row in cand.iterrows():
        n = int(row.label.split("n=")[1].split(",")[0]); hl = int(row.label.split("hl=")[1])
        specs.append((row.group, row.label, n, hl))
    res = []
    for k_ in range(300):
        d = MC.path(); z = E.atr_z(d); cc = d.close; above = cc > cc.rolling(200).mean()
        ok0 = above & (z < 0.5); s0 = d.index[MC.WARM]
        rec = {}
        wb = E.weights_from_rules(d, ok0, ~ok0)
        st = E.stats(E.run(d, wb, start=s0, z=z)); rec["BASE"] = st["irr"]; rec["BASE|w"] = st["worst"]
        for g, lab, n, hl in specs:
            v, a = va(d, n); zv = ewz(v, hl); za = ewz(a, hl)
            zz = z
            if g == "TREND":
                ok = ((v > 0) if lab.startswith("v>0") else (zv > 0).fillna(False)) & (z < 0.5)
            elif g == "EXIT":
                kk = float(lab.split("<−")[1].split(" ")[0])
                ok = ok0 & ~(((za if "z_a" in lab else zv) < -kk).fillna(False))
            elif g == "EARLY":
                kk = float(lab.split(">")[1].split(" ")[0])
                ok = (above | (zv > kk).fillna(False)) & (z < 0.5)
            else:
                kk = float(lab.split("<−")[1].split(" ")[0]); ok = ok0
                zz = pd.Series(np.where((zv < -kk).fillna(False), 3.0, 0.0), index=d.index)
            w = E.weights_from_rules(d, ok, ~ok)
            st = E.stats(E.run(d, w, start=s0, z=zz)); rec[lab] = st["irr"]; rec[lab + "|w"] = st["worst"]
        res.append(rec)
    M = pd.DataFrame(res)
    P("\n區塊拼貼蒙地卡羅（300 條 10 年路徑）")
    for g, lab, n, hl in specs:
        dd = M[lab] - M.BASE; dw = M[lab + "|w"] - M["BASE|w"]
        P(f"  {lab:34s} IRR 差 中位 {dd.median():+.4f} 勝率 {(dd > 0).mean():.0%}｜最慘 差 中位 {dw.median():+.3f} 5% {dw.quantile(.05):+.3f}")
open("results_va.txt", "w").write("\n".join(out) + "\n")
