"""第二十五部分 (2)：以 ER 回檔順勢為主體的延伸條件（A/B/C 三組合併看，出場固定 tp2）

1. 大週期同向：H1 / H4 已收完 K 棒的 ER20 與 d 同向（>0 / >0.2）
2. 時段（broker 時間）：亞洲 01–08、倫敦 09–14、紐約 15–23
3. 星期
4. 波動度：ATR14 / ATR100（訊號 K 棒）三分位
5. 回檔深度：(前 M 根順勢極值 − 收盤) / ATR14 三分位
6. 長 ER 強度：d·ER長 三分位
7. 進場方式：下一根開盤（原版） vs 突破訊號 K 棒高點（做多）/ 低點（做空）才進，3 根內沒突破取消
三分位門檻用前半段（1/2~5/31）的訊號決定，後半段直接套用（避免用到未來）。
→ results_part25_ext.txt
"""
import numpy as np
import pandas as pd
from part23_ma_cycle import load_m1, SPLIT
from part24_er_cascade import ser, atr
from part25_exits import CFGS, prep, entries, trade, metrics


def tf_er_done(m1, B, rule, n=20):
    """較大週期已收完 K 棒的 ER(n)，對齊到 B 的 K 棒收盤時刻。"""
    hi = m1.resample(rule, label="right", closed="left").agg({"close": "last"}).dropna()
    e = pd.Series(ser(hi.close.to_numpy(float), n), index=hi.index)   # 標籤 = 收完的時間
    step = B["idx"][1] - B["idx"][0]
    close_time = B["idx"] + step
    return e.reindex(close_time, method="ffill").to_numpy()


def stop_entry_trade(B, k, s, L, M, tp=2, wait=3, buf=0.1):
    o, h, l, c, A = B["o"], B["h"], B["l"], B["c"], B["A"]
    n = len(c)
    trig = h[k] if s == 1 else l[k]
    ext = l[k - M + 1:k + 1].min() if s == 1 else h[k - M + 1:k + 1].max()
    stop = ext - s * buf * A[k]
    for e in range(k + 1, min(k + 1 + wait, n)):
        if (l[e] <= stop) if s == 1 else (h[e] >= stop):
            return np.nan, e          # 先碰停損 → 取消
        if (h[e] >= trig) if s == 1 else (l[e] <= trig):
            ep = max(trig, o[e]) if s == 1 else min(trig, o[e])
            R = s * (ep - stop)
            if not R > 0:
                return np.nan, e
            tgt = ep + s * tp * R
            if (h[e] >= tgt) if s == 1 else (l[e] <= tgt):   # 進場那根：只在收盤越過目標時算到價
                if s * (c[e] - tgt) >= 0:
                    return (tp * R - B["spr"][e]) / R, e
            for j in range(e + 1, min(e + 2 * L, n)):
                if (l[j] <= stop) if s == 1 else (h[j] >= stop):
                    xp = min(stop, o[j]) if s == 1 else max(stop, o[j])
                    return (s * (xp - ep) - B["spr"][e]) / R, j
                if (h[j] >= tgt) if s == 1 else (l[j] <= tgt):
                    return (tp * R - B["spr"][e]) / R, j
            j = min(e + 2 * L, n) - 1
            return (s * (c[j] - ep) - B["spr"][e]) / R, j
    return np.nan, k


def main():
    m1 = load_m1()
    days = sorted(set(m1.index.date))
    cache = {}
    rows = []
    for name, (tf, L, M, S, a, cool) in CFGS.items():
        B = cache.setdefault(tf, prep(m1, tf))
        if "h1" not in B:
            B["h1"], B["h4"] = tf_er_done(m1, B, "1h"), tf_er_done(m1, B, "4h")
            B["A100"] = atr(B["h"], B["l"], B["c"], 100)
        t, side = entries(B, L, M, S, a, cool)
        for k, s in zip(t, side):
            R, j = trade(B, k, s, L, M, ("tp", 2))
            R2, j2 = stop_entry_trade(B, k, s, L, M)
            ext = B["h"][k - M + 1:k + 1].max() if s == 1 else B["l"][k - M + 1:k + 1].min()
            ts = B["idx"][k]
            rows.append(dict(leg=name, time=ts, xtime=B["idx"][j], side=s, R=R, bars=j - k,
                             R_stopentry=R2, xtime2=B["idx"][j2],
                             h1=s * B["h1"][k], h4=s * B["h4"][k],
                             hour=(ts + (B["idx"][1] - B["idx"][0])).hour, dow=ts.dayofweek,
                             vol=B["A"][k] / B["A100"][k], depth=s * (ext - B["c"][k]) / B["A"][k],
                             strength=s * ser(B["c"][k - L:k + 1], L)[-1]))
    D = pd.DataFrame(rows).dropna(subset=["R"])
    D["half"] = np.where(D.time < SPLIT, "前半", "後半")
    out = []
    P = out.append
    P("基準（A+B+C、tp2）：" + str(metrics(D, days)))

    def bucket(col, cuts_or_fn, labels=None):
        if callable(cuts_or_fn):
            g = cuts_or_fn(D)
        else:
            q = D.loc[D.half == "前半", col].quantile(cuts_or_fn).to_numpy()
            g = pd.cut(D[col], [-np.inf, *q, np.inf], labels=labels)
        r = D.groupby(g, observed=True).apply(lambda x: pd.Series(dict(
            n=len(x), avgR=x.R.mean(), PF=x.R[x.R > 0].sum() / -x.R[x.R < 0].sum(),
            前半=x.R[x.half == "前半"].mean(), 後半=x.R[x.half == "後半"].mean(),
            A=x.R[x.leg.str[0] == "A"].mean(), B=x.R[x.leg.str[0] == "B"].mean(),
            C=x.R[x.leg.str[0] == "C"].mean())), include_groups=False)
        return r.round(3).to_string()

    P("\n=== 1. H1 ER20 方向（d·ER）===")
    P(bucket("h1", lambda d: pd.cut(d.h1, [-1, -0.2, 0, 0.2, 1], labels=["逆 <-0.2", "逆 0~-0.2", "順 0~0.2", "順 >0.2"])))
    P("\n=== 1b. H4 ER20 方向 ===")
    P(bucket("h4", lambda d: pd.cut(d.h4, [-1, -0.2, 0, 0.2, 1], labels=["逆 <-0.2", "逆 0~-0.2", "順 0~0.2", "順 >0.2"])))
    P("\n=== 2. 時段（訊號 K 棒收盤時間，broker）===")
    P(bucket("hour", lambda d: pd.cut(d.hour, [-1, 8, 14, 24], labels=["亞洲 0-8", "倫敦 9-14", "紐約 15-23"])))
    P("\n=== 3. 星期 ===")
    P(bucket("dow", lambda d: d.dow.map({0: "一", 1: "二", 2: "三", 3: "四", 4: "五"})))
    P("\n=== 4. 波動度 ATR14/ATR100（前半三分位）===")
    P(bucket("vol", [1 / 3, 2 / 3], ["低", "中", "高"]))
    P("\n=== 5. 回檔深度（ATR，前半三分位）===")
    P(bucket("depth", [1 / 3, 2 / 3], ["淺", "中", "深"]))
    P("\n=== 6. 長 ER 強度（前半三分位）===")
    P(bucket("strength", [1 / 3, 2 / 3], ["弱", "中", "強"]))

    P("\n=== 7. 進場方式：下一根開盤 vs 突破訊號 K 棒高/低點（3 根內）===")
    E2 = D.dropna(subset=["R_stopentry"]).assign(R=lambda x: x.R_stopentry, xtime=lambda x: x.xtime2)
    P("下一根開盤：" + str(metrics(D, days)))
    P(f"突破進場  ：成交 {len(E2)}/{len(D)} " + str(metrics(E2, days)))
    for leg in CFGS:
        P(f"  {leg}: 開盤 {D[D.leg == leg].R.mean():+.3f}（{(D.leg == leg).sum()}）"
          f" / 突破 {E2[E2.leg == leg].R.mean():+.3f}（{(E2.leg == leg).sum()}）")

    P("\n=== 8. 組合濾網（只用兩半都改善的條件）===")
    for label, m in [("H1 順向 (>0)", D.h1 > 0), ("H4 順向 (>0)", D.h4 > 0), ("H1 且 H4 順向", (D.h1 > 0) & (D.h4 > 0))]:
        P(f"{label:14s}：" + str(metrics(D[m], days)) + f"  ← 剔除的 {(~m).sum()} 筆 avgR {D[~m].R.mean():+.3f}")
    D.to_csv("part25_ext_trades.csv", index=False)
    open("results_part25_ext.txt", "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
