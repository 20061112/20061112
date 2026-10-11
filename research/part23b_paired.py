"""Part 23b：同一批觸碰同時算「反彈」與「穿越」，看淨方向偏差。

對每個觸碰事件（真實與隨機對照都一樣）：
  fade = 先往反彈方向 1 ATR 的機率（part22 的 b1）
  cont = 先往穿越方向 1 ATR 的機率（part23 的 b1）
  兩者都沒贏 = 同一分鐘兩邊都到 / 成交那一分鐘就掃過 1 ATR（多半是消息行情）
  skew = fade − cont：> 0 代表價位真的偏向「擋住」，< 0 代表偏向「被穿過去」。
若真實價位只是比隨機價位少一些「兩邊都沒贏」的情況，fade 與 cont 會一起變好，skew 不會變。
"""
import numpy as np
import pandas as pd
import part22_smc as P
from load import load

pd.set_option("display.width", 250)


def main():
    m1 = load("data/XAUUSD_M1_2023_2026.csv")
    m1a = tuple(m1[k].to_numpy(float) for k in ("open", "high", "low", "close")) + \
        (m1.spread.to_numpy(float) * 0.01,)
    out = []
    for rule, tf_min in (("15min", 15), ("1h", 60)):
        df = P.resample(m1, rule)
        zones, A, tr = P.detect(df)
        zones += P.poc_zones(m1, df, A)
        P.A_glob = A
        c = df.close.to_numpy(float)
        m1_start = m1.index.searchsorted(df.index)
        args = (df, A, tr, m1a, m1_start, tf_min)
        zr = [z + (n,) for n, z in enumerate(zones)]
        pl = [p[:5] + (10**8 + n,) for n, p in enumerate(P.make_placebos(zones, c))]
        for zs, flag in ((zr, 0), (pl, 1)):
            f = pd.DataFrame(P.run(zs, *args, flag))[["zid", "kind", "placebo", "b1", "year"]]
            r = pd.DataFrame(P.run(zs, *args, flag, rev=True))[["zid", "b1"]]
            m = f.merge(r, on="zid", suffixes=("_fade", "_cont"))
            m["tf"] = rule
            out.append(m)
    ev = pd.concat(out, ignore_index=True).dropna()
    ev["skew"] = ev.b1_fade - ev.b1_cont
    ev["none"] = ((ev.b1_fade == 0) & (ev.b1_cont == 0)).astype(float)

    def agg(g):
        r, p = g[g.placebo == 0], g[g.placebo == 1]
        se = np.sqrt(r["skew"].var() / len(r) + p["skew"].var() / len(p))
        return pd.Series({"N": len(r), "fade": r.b1_fade.mean(), "cont": r.b1_cont.mean(),
                          "都沒贏": r.none.mean(), "skew": r["skew"].mean(),
                          "fade_pl": p.b1_fade.mean(), "cont_pl": p.b1_cont.mean(),
                          "都沒贏_pl": p.none.mean(), "skew_pl": p["skew"].mean(),
                          "skew差": r["skew"].mean() - p["skew"].mean(),
                          "t": (r["skew"].mean() - p["skew"].mean()) / se})
    for tf in ("15min", "1h"):
        print(f"\n===== {tf}：同一批觸碰，反彈 vs 穿越 =====")
        print(ev[ev.tf == tf].groupby("kind").apply(agg, include_groups=False).round(3).to_string())
    print("\n----- 逐年 skew差（真實 − 對照）-----")
    s = ev.groupby(["tf", "kind", "year"]).apply(agg, include_groups=False)["skew差"].unstack()
    print(s.round(3).to_string())


if __name__ == "__main__":
    main()
