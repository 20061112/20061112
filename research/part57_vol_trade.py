"""第五十七部分（應用 E）：把波動本身當交易對象 —— 波動預測能不能變成方向優勢？

說明：真正「交易波動」需要選擇權（隱含 vs 預測波動、變異數風險溢酬），本資料只有現貨 XAUUSD，
所以這裡測的是「波動狀態 → 價格行為」的兩個預先登記假設：
  E1 波動擴張突破（類跨式）：預測擴張比 X = σ̂_h / 過去 h 根實現波動。每 h 根於收盤掛 ±kσ̂ 的突破單（OCO，k = 0.5），
     先觸發的方向進場，停損 = 進場價反向 1σ̂，第 h 根收盤出場。假設：模型預期擴張（X 高）時，突破後延續較好。
  E2 波動驚奇：S = log(過去 h 根實現波動 / h 根前的預測 σ̂)。d = 過去 h 根報酬方向。
     看未來 h 根往 d 的報酬（σ̂ 單位）；交易 = 下一根開盤往 d（延續）進場，停損 1σ̂，h 根後出場。
     假設：實際波動遠大於預期（資訊衝擊）時方向會延續；遠小於預期時無方向。
分組：X、S 五分位邊界只用之前月份（expanding）估計。非重疊（每 h 根一個決策點）。
資料：M15 2023/4 ~ 2026/10（h = 16 = 4 小時、h = 4 = 1 小時）；M5 2025/8 ~ 2026/10（h = 12 = 1 小時）。
R = 損益 / σ̂（1R = 停損距離），已扣點差。
"""
import numpy as np
import pandas as pd
from volfc import har_rel

OUT = "results_part57.txt"


def wf_quint(x, month):
    """逐月 expanding 五分位（只用之前月份）。"""
    q = np.full(len(x), -1)
    ms = sorted(set(month))
    for m in ms[1:]:
        tr = (month < m) & np.isfinite(x)
        te = (month == m) & np.isfinite(x)
        if tr.sum() < 200:
            continue
        e = np.quantile(x[tr], [.2, .4, .6, .8])
        q[te] = np.searchsorted(e, x[te])
    return q


def tstat(v):
    v = v[np.isfinite(v)]
    return v.mean() / v.std(ddof=1) * np.sqrt(len(v)) if len(v) > 2 else np.nan


def e1(bars, h, k=0.5):
    o, hi, lo, c = (bars[x].to_numpy() for x in ("open", "high", "low", "close"))
    sig = bars.sig.to_numpy()
    spr = bars.spread.to_numpy() * 0.01
    g = bars.gapid.to_numpy()
    n = len(c)
    res = []
    for t in range(0, n - h - 1, h):
        s = sig[t]
        if not np.isfinite(s) or g[t + h] != g[t]:
            continue
        up, dn = c[t] + k * s, c[t] - k * s
        side, e, j0 = 0, np.nan, None
        for j in range(t + 1, t + h + 1):
            hu, hd = hi[j] >= up, lo[j] <= dn
            if hu and hd:              # 同一根兩邊都碰：取離開盤較近的一側先觸發；同根內再檢查停損（偏保守）
                side, e, j0 = (1, max(o[j], up), j) if abs(o[j] - up) < abs(o[j] - dn) else (-1, min(o[j], dn), j)
                break
            if hu:
                side, e, j0 = 1, max(o[j], up), j
                break
            if hd:
                side, e, j0 = -1, min(o[j], dn), j
                break
        if side == 0:
            res.append((t, 0, np.nan))
            continue
        stop = e - side * s
        px = c[t + h]
        for j in range(j0, t + h + 1):
            if (side > 0 and lo[j] <= stop) or (side < 0 and hi[j] >= stop):
                # 進場那一根以停損價成交；之後的 K 棒若開盤已跳過停損，以開盤價成交
                px = stop if j == j0 else (min(o[j], stop) if side > 0 else max(o[j], stop))
                break
        res.append((t, side, ((px - e) * side - spr[t]) / s))
    return pd.DataFrame(res, columns=["i", "side", "R"])


def e2(bars, h):
    o, hi, lo, c = (bars[x].to_numpy() for x in ("open", "high", "low", "close"))
    sig = bars.sig.to_numpy()
    rvp = bars.rvpast.to_numpy()
    spr = bars.spread.to_numpy() * 0.01
    g = bars.gapid.to_numpy()
    lc = np.log(c)
    n = len(c)
    rows = []
    for t in range(h, n - h - 1, h):
        s, s0 = sig[t], sig[t - h]
        if not (np.isfinite(s) and np.isfinite(s0)) or g[t + h] != g[t] or g[t - h] != g[t]:
            continue
        d = np.sign(lc[t] - lc[t - h])
        if d == 0:
            continue
        S = np.log(rvp[t] / s0)
        z = (c[t + h] - c[t]) / s * d
        e = o[t + 1]
        stop = e - d * s
        px = c[t + h]
        for j in range(t + 1, t + h + 1):
            if (d > 0 and lo[j] <= stop) or (d < 0 and hi[j] >= stop):
                px = min(o[j], stop) if d > 0 else max(o[j], stop)
                break
        R = ((px - e) * d - spr[t + 1]) / s
        mv = abs(lc[t] - lc[t - h]) * c[t] / s0           # 過去 h 根位移 / 預測 σ̂
        rows.append((t, S, d, z, R, mv))
    return pd.DataFrame(rows, columns=["i", "S", "d", "z", "R", "mv"])


def report(pr, df, bars, key, title, val="R", extra=None):
    month = bars.index.to_period("M").to_numpy()[df.i]
    years = bars.index.year.to_numpy()[df.i]
    x = df[key].to_numpy() if key else None
    q = wf_quint(x, month) if key else np.zeros(len(df), int)
    pr(f"  {title}")
    pr(f"    {'組':4s} {'n':>5s} {'avg':>7s} {'t':>6s} {'win':>5s} " + " ".join(f"{y:>7d}" for y in sorted(set(years))))
    for g in range(5 if key else 1):
        m = (q == g) & np.isfinite(df[val].to_numpy())
        v = df[val].to_numpy()[m]
        if len(v) < 5:
            continue
        ys = " ".join(f"{df[val].to_numpy()[m & (years == y)].mean():+7.3f}" if (m & (years == y)).sum() > 5 else f"{'—':>7s}" for y in sorted(set(years)))
        pr(f"    Q{g + 1 if key else '全'}  {len(v):5d} {v.mean():+7.3f} {tstat(v):+6.2f} {np.mean(v > 0):5.0%} {ys}")
    if key:
        m = (q >= 0) & np.isfinite(df[val].to_numpy())
        v = df[val].to_numpy()
        hi, lo_ = v[m & (q == 4)], v[m & (q == 0)]
        pr(f"    Q5 − Q1 = {hi.mean() - lo_.mean():+.3f}（t {(hi.mean() - lo_.mean()) / np.sqrt(hi.var(ddof=1) / len(hi) + lo_.var(ddof=1) / len(lo_)):+.2f}）")


def main():
    out = open(OUT, "w")

    def pr(*a):
        print(*a, file=out, flush=True)
    pr("第五十七部分（應用 E）：波動狀態能否帶來方向優勢（E1 擴張突破、E2 波動驚奇）")
    pr("欄位：avg = 每筆平均 R（已扣點差）、t = t 值、逐年平均 R；分組 = 逐月 expanding 五分位（只用過去）")
    sets = [("M15", "data/XAUUSD_M15_2023_2026.csv", 16), ("M15", "data/XAUUSD_M15_2023_2026.csv", 4),
            ("M5", "data/XAUUSD_M5_2025_2026.csv", 12)]
    for tf, path, h in sets:
        bars, step = har_rel(path, h)
        pr(f"\n================ {tf}，h = {h} 根（{h * step} 分鐘）================")
        r1 = e1(bars, h)
        sig = bars.sig.to_numpy()
        rv = bars.rvpast.to_numpy()
        r1["X"] = np.log(sig[r1.i] / rv[r1.i])
        trig = r1[r1.side != 0].reset_index(drop=True)
        pr(f"E1 擴張突破：決策點 {len(r1)}，觸發 {len(trig)}（{len(trig) / len(r1):.0%}）")
        report(pr, trig, bars, None, "全部觸發")
        report(pr, trig, bars, "X", "依預測擴張比 X = log(σ̂ / 過去實現) 五分位（Q5 = 預期最大擴張）")
        r2 = e2(bars, h)
        pr(f"E2 波動驚奇：決策點 {len(r2)}")
        report(pr, r2, bars, None, "全部（往過去 h 根方向延續）")
        report(pr, r2, bars, "S", "依驚奇 S = log(實現 / 預測) 五分位：未來 z（σ̂ 單位，往過去方向，未扣成本）", val="z")
        report(pr, r2, bars, "S", "依驚奇 S 五分位：延續交易 R（停損 1σ̂，扣點差）")
        report(pr, r2, bars, "mv", "對照：依過去位移 / σ̂ 五分位（純動能，不看驚奇）：未來 z", val="z")
    out.close()
    print(open(OUT).read())


if __name__ == "__main__":
    main()
