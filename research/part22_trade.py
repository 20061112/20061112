"""第二十二部分：把 Kramers 規則實際當交易跑（M1 逐根，扣點差）+ 十分位單調性的置換檢定。

交易：事件 K 棒收盤反向進場（往區間內），停利 = 井底（該變體的 POC），停損 = 區間邊緣外 b·S，
最長 240 分鐘或交易日結束；同一根同時碰到停損停利算停損；同一時間只持有一筆。
門檻只用前半（2023/6~2024）的分位數決定，後半（2025~2026/10）驗證。
"""
import numpy as np
import pandas as pd
from part22_kramers import load_all, HORIZON
from part22_analyze import E, feats, deciles, mono_stats, cluster_se, OUT

lines = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    lines.append(s)


d = load_all()
DAYS = {k: g for k, g in d.groupby("day")}
ARR = {k: (g.high.to_numpy(), g.low.to_numpy(), g.close.to_numpy()) for k, g in DAYS.items()}


def simulate(sel, poc, b=0.5):
    """sel：布林遮罩（事件）；poc：每筆停利價。回傳逐筆 R。"""
    out = []
    busy_until = {}
    idx = np.nonzero(sel)[0]
    for i in idx:
        ev = E.iloc[i]
        day, t, s = ev.day, int(ev.t_idx), int(ev.side)
        if busy_until.get(day, -1) >= t:
            continue
        hh, ll, cc = ARR[day]
        entry = ev.close
        dirn = -s                                   # 往區間內
        stop = ev.edge + s * b * ev.S
        tgt = poc[i]
        risk = abs(entry - stop)
        if dirn * (tgt - entry) <= 0 or risk <= 0:
            continue
        end = min(len(cc), t + 1 + HORIZON)
        res, j_exit = None, end - 1
        for j in range(t + 1, end):
            hit_stop = (hh[j] >= stop) if dirn == -1 else (ll[j] <= stop)
            hit_tgt = (ll[j] <= tgt) if dirn == -1 else (hh[j] >= tgt)
            if hit_stop:
                res, j_exit = -risk, j; break
            if hit_tgt:
                res, j_exit = abs(tgt - entry), j; break
        if res is None:
            res = dirn * (cc[end - 1] - entry)
        busy_until[day] = j_exit
        out.append(dict(time=ev.time, day=day, half=ev.half, year=ev.year,
                        R=(res - ev.spread) / risk, rr=abs(tgt - entry) / risk))
    return pd.DataFrame(out)


def summ(T, label):
    if len(T) == 0:
        P(f"  {label:<44} 無交易"); return
    for hv in ("A", "B"):
        x = T[T.half == hv]
        if len(x) < 5:
            continue
        se = cluster_se(x.R, x.day)
        pf = x.R[x.R > 0].sum() / -x.R[x.R < 0].sum()
        P(f"  {label:<44} {hv}  n={len(x):5d}  勝率 {(x.R > 0).mean():.1%}  期望 {x.R.mean():+.3f}R  t {x.R.mean() / se:+.2f}  "
          f"PF {pf:.2f}  累計 {x.R.sum():+7.1f}R  平均 RR {x.rr.median():.2f}")
    P(f"  {'':<44} 逐年 " + "  ".join(f"{y}:{v:+.3f}" for y, v in T.groupby("year").R.mean().items()))


# ---- 置換檢定：十分位 ρ 與 D10−D1 在『結果與 K 無關』下的分佈 ----
P("=" * 100)
P("置換檢定：事件層級打亂結果 500 次（忽略日內相關，p 值偏樂觀；日群聚標準誤約為獨立假設的 1.2 倍）")
rng = np.random.default_rng(0)
CHECK = [("range", 1.0, 1, "sh", "none"), ("range", 1.0, 1, "sh", "cc30"), ("range", 1.0, 1, "harm", "cc30"),
         ("tickvol", 2.0, 3, "harm", "pk60"), ("base", 0, 0, "volratio", "pk60"), ("base", 0, 0, "volratio", "cc30")]
for key in CHECK:
    x = np.asarray(feats[key], float)
    for o in ("reb05", "rebpoc"):
        y = E[o].to_numpy()
        m = np.isfinite(x) & np.isfinite(y)
        xr, yr = x[m], y[m]
        rho0, _, d0 = mono_stats(deciles(xr, yr))
        null_rho, null_d = [], []
        for _ in range(500):
            yp = rng.permutation(yr)
            r_, _, d_ = mono_stats(deciles(xr, yp))
            null_rho.append(r_); null_d.append(d_)
        null_rho, null_d = np.array(null_rho), np.array(null_d)
        P(f"  {str(key):<44} {o:<7} ρ={rho0:+.3f}（p 上尾 {np.mean(null_rho >= rho0):.3f}，下尾 {np.mean(null_rho <= rho0):.3f}）"
          f"  D10−D1={d0:+.4f}（p 上尾 {np.mean(null_d >= d0):.3f}，下尾 {np.mean(null_d <= d0):.3f}）")
P("  註：567 個變體一起看，單一變體 p<0.05 預期約有 28 個純屬運氣。")

# ---- 交易 ----
P("\n" + "=" * 100)
P("交易回測：停利 = 井底 POC，停損 = 邊緣外 0.5S；門檻 = 前半分位數")
A = (E.half == "A").to_numpy()
P("\n[基準] 所有邊緣事件都反向做")
ref_poc = E["range_c1.0_h1_poc"].to_numpy()
summ(simulate(np.ones(len(E), bool), ref_poc), "全部事件（range POC）")

RULES = [(("range", 1.0, 1, "sh", "cc30"), "K_temp range c1 h1 cc30"),
         (("range", 1.0, 1, "harm", "cc30"), "K_harm range c1 h1 cc30"),
         (("tickvol", 2.0, 3, "harm", "pk60"), "K_harm tickvol c2 h3 pk60（前半最佳）"),
         (("range", 1.0, 1, "sh", "none"), "K_shape 字面定義")]
for key, label in RULES:
    x = np.asarray(feats[key], float)
    poc = E[f"{key[0]}_c{key[1]}_h{key[2]}_poc"].to_numpy()
    P(f"\n[{label}] ΔU/D 夠大才反向做")
    for q in (0.5, 0.7, 0.9):
        thr = np.nanquantile(x[A], q)
        summ(simulate(x >= thr, poc), f"K ≥ 前半 {int(q * 100)} 百分位 ({thr:.2f})")

P("\n[反向假說] 字面 K_shape 小（邊緣已被停留很久）才反向做；K_shape 大 → 順勢")
x = np.asarray(feats[("range", 1.0, 1, "sh", "none")], float)
for q in (0.1, 0.3):
    thr = np.nanquantile(x[A], q)
    summ(simulate(x <= thr, ref_poc), f"K_shape ≤ 前半 {int(q * 100)} 百分位 ({thr:.2f})")

P("\n[停損寬度敏感度] K_temp ≥ 前半 70 百分位")
key = ("range", 1.0, 1, "sh", "cc30")
x = np.asarray(feats[key], float)
thr = np.nanquantile(x[A], 0.7)
for b in (0.25, 1.0):
    summ(simulate(x >= thr, ref_poc, b=b), f"停損 = 邊緣外 {b}S")

open("results_part22_trade.txt", "w").write("\n".join(lines) + "\n")
