"""Part 24：(1) 滾動 POC（前 N 根）被碰到有沒有反應？ (2) FVG 做逆勢（碰到就做穿越）的完整數字。

(1) 滾動 POC
  - 每根偵測週期 K 棒 j，用前 N 根（j-N ~ j-1）的 M1 收盤 + tick volume，$0.5 一格，取量最大的價位 = poc_j
    （只用 j 開盤前已收完的資料）。
  - 價格收盤離 POC 至少 AWAY×ATR 之後，第一次回碰當下的 poc_j（±0.1 ATR）算一次觸碰；碰到後重新等待離開。
  - 從上方回碰 → 反彈 = 做多 / 穿越 = 做空；下方鏡像。
  - 交給 part22 的 run()：區域在 j-1 成立、j 被碰到，在 M1 上成交與出場，扣點差；同一批觸碰也做 rev（穿越）。
  - 對照組：同一時間、同寬度、隨機平移 0.5~3 ATR 的假價位。
(2) FVG 逆勢 = part22 run(rev=True)：在 FVG 近端以停損單做穿越方向，停損 1 ATR，停利 1 / 2 ATR。
"""
import numpy as np
import pandas as pd
import part22_smc as P
from load import load

pd.set_option("display.width", 250)
AWAY = 1.0


def rolling_poc(m1, df, nbars):
    mc = m1.close.to_numpy(float)
    mv = m1.tickvol.to_numpy(float)
    start = m1.index.searchsorted(df.index)
    N = len(df)
    poc = np.full(N, np.nan)
    for j in range(nbars, N):
        a, b = start[j - nbars], start[j]
        if b - a < 10:
            continue
        px = np.round(mc[a:b] * 2).astype(np.int64)
        base = px.min()
        poc[j] = (base + np.argmax(np.bincount(px - base, weights=mv[a:b]))) / 2
    return poc


def poc_events(df, A, poc):
    h, l, c = (df[k].to_numpy(float) for k in ("high", "low", "close"))
    zones = []
    armed = 0                 # +1：曾在上方遠離；-1：曾在下方遠離
    for j in range(1, len(df) - 1):
        p, a = poc[j], A[j - 1]
        if np.isnan(p) or not a > 0:
            continue
        w = 0.1 * a
        if armed == 1 and l[j] <= p + w and c[j - 1] > p + w:
            zones.append(("rPOC", 1, p - w, p + w, j - 1))
            armed = 0
        elif armed == -1 and h[j] >= p - w and c[j - 1] < p - w:
            zones.append(("rPOC", -1, p - w, p + w, j - 1))
            armed = 0
        if c[j] >= poc[j] + AWAY * A[j]:
            armed = 1
        elif c[j] <= poc[j] - AWAY * A[j]:
            armed = -1
    return zones


def paired(zones, pl, args):
    out = []
    for zs, flag in ((zones, 0), (pl, 1)):
        f = pd.DataFrame(P.run(zs, *args, flag))
        r = pd.DataFrame(P.run(zs, *args, flag, rev=True))
        m = f.merge(r[["zid", "b1", "r1", "r2"]], on="zid", suffixes=("_fade", "_cont"))
        out.append(m)
    return pd.concat(out, ignore_index=True)


def agg(g):
    r, p = g[g.placebo == 0].dropna(subset=["b1_fade", "b1_cont"]), g[g.placebo == 1].dropna(subset=["b1_fade", "b1_cont"])
    sk_r, sk_p = r.b1_fade - r.b1_cont, p.b1_fade - p.b1_cont
    se = np.sqrt(sk_r.var() / len(r) + sk_p.var() / len(p))
    return pd.Series({"N": len(r),
                      "反彈b1": r.b1_fade.mean(), "穿越b1": r.b1_cont.mean(),
                      "反彈b1_pl": p.b1_fade.mean(), "穿越b1_pl": p.b1_cont.mean(),
                      "淨方向差": sk_r.mean() - sk_p.mean(), "t": (sk_r.mean() - sk_p.mean()) / se,
                      "反彈2R": r.r2_fade.mean(), "穿越1R": r.r1_cont.mean(), "穿越2R": r.r2_cont.mean(),
                      "穿越2R_pl": p.r2_cont.mean()})


def main():
    m1 = load("data/XAUUSD_M1_2023_2026.csv")
    m1a = tuple(m1[k].to_numpy(float) for k in ("open", "high", "low", "close")) + \
        (m1.spread.to_numpy(float) * 0.01,)
    res_poc, res_fvg = [], []
    for rule, tf_min in (("15min", 15), ("1h", 60)):
        df = P.resample(m1, rule)
        zones, A, tr = P.detect(df)
        P.A_glob = A
        c = df.close.to_numpy(float)
        m1_start = m1.index.searchsorted(df.index)
        args = (df, A, tr, m1a, m1_start, tf_min)
        for nb in (20, 50, 100):
            z = [q + (n,) for n, q in enumerate(poc_events(df, A, rolling_poc(m1, df, nb)))]
            pl = [p[:5] + (10**8 + n,) for n, p in enumerate(P.make_placebos(z, c))]
            e = paired(z, pl, args)
            e["tf"], e["nbars"] = rule, nb
            res_poc.append(e)
        fz = [q + (n,) for n, q in enumerate(z for z in zones if z[0] == "FVG")]
        pl = [p[:5] + (10**8 + n,) for n, p in enumerate(P.make_placebos(fz, c))]
        e = paired(fz, pl, args)
        e["tf"] = rule
        res_fvg.append(e)

    poc = pd.concat(res_poc, ignore_index=True)
    fvg = pd.concat(res_fvg, ignore_index=True)
    print("b1 = 先往該方向 1 ATR 的機率；淨方向差 = (反彈−穿越)真實 − (反彈−穿越)對照，>0 代表價位偏向擋住；")
    print("反彈2R：停損在區域外 0.1 ATR；穿越 1R/2R：停損 1 ATR。R 皆已扣點差。_pl = 隨機價位對照。")
    print("\n===== (1) 滾動 POC =====")
    print(poc.groupby(["tf", "nbars"]).apply(agg, include_groups=False).round(3).to_string())
    print("\n逐年：")
    print(poc.groupby(["tf", "nbars", "year"]).apply(agg, include_groups=False)
          [["N", "反彈b1", "穿越b1", "淨方向差", "反彈2R", "穿越2R"]].round(3).unstack().to_string())

    print("\n===== (2) FVG 逆勢（穿越）=====")
    print(fvg.groupby("tf").apply(agg, include_groups=False).round(3).to_string())
    fvg["wbin"] = pd.cut(fvg.width, [0, 0.25, 0.5, 1, 99])
    print("\n依結構方向（aligned=1：FVG 與最後一次 BOS/CHoCH 同向）：")
    print(fvg.groupby(["tf", "aligned"]).apply(agg, include_groups=False).round(3).to_string())
    print("\n依 FVG 大小（ATR）：")
    print(fvg.groupby(["tf", "wbin"], observed=True).apply(agg, include_groups=False).round(3).to_string())
    print("\n逐年：")
    print(fvg.groupby(["tf", "year"]).apply(agg, include_groups=False)
          [["N", "反彈b1", "穿越b1", "淨方向差", "穿越1R", "穿越2R"]].round(3).to_string())


if __name__ == "__main__":
    main()
