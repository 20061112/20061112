"""第二十五部分：順日線趨勢 1-2-3（突破 H2 進場、停損 L1 下方、L1 為 50 根極值）
  A. M15 代表參數（碎形 3、等幅測量停利）的詳細交易數據
  B. 同一套規則套到 M5 / M1（型態仍以「該週期的根數」定義；持倉上限統一 24 小時）

其餘設定同第二十四部分：M1 逐根模擬、扣點差、同一時間只持有一筆、隨機基準 = 同日線趨勢狀態的隨機進場。
"""
import sys

import numpy as np
import pandas as pd

import part24_123_2b as P
from load import load
from part24_baseline import baseline

P.RULE.update({"M5": "5min", "M1": "1min"})
P.HOLD.update({"M5": 288, "M1": 1440})
M1 = load("data/XAUUSD_M1_2023_2026.csv")
P.M1 = M1


def run(d, swing, tps=("1", "2", "mm")):
    sigs = [s for s in P.signals_123(d, P.get_pivots(d, swing)) if s["ext50"] and s["side"] == s["trend"]]
    out = {tp: [] for tp in tps}
    for s in sigs:
        t, side, A = s["t"], s["side"], d.atr[s["t"]]
        if t + 1 >= d.n or not A > 0:
            continue
        f = P.fill_stop(d, t + 1, 20, side, s["p2"], s["p3"])
        if f is None:
            continue
        a, px = f
        stop = s["p1"] - side * P.BUF * A
        for tp in tps:
            r = P.trade(d, t, a, side, px, stop, tp, P.HOLD[d.tf], mm=s["p2"] + (s["p2"] - s["p1"]))
            if r is not None:
                r.update(px=px, risk=side * (px - stop), spread=d.mspr[a], trend=s["trend"])
                out[tp].append(r)
    return {tp: P.no_overlap(pd.DataFrame(v)) for tp, v in out.items() if v}


def detail(tr, d):
    tr = tr.copy()
    tr["year"] = tr.time.dt.year
    p = tr.pnl_r.to_numpy()
    eq = np.r_[0, p.cumsum()]
    dd = np.maximum.accumulate(eq) - eq
    streak = max(len(x) for x in "".join("1" if v < 0 else "0" for v in p).split("0"))
    hrs = (M1.index[tr.exit_m1.to_numpy()] - M1.index[tr.fill_m1.to_numpy()]).total_seconds() / 3600
    weeks = (tr.time.max() - tr.time.min()).days / 7
    print(f"筆數 {len(p)}（約每週 {len(p) / weeks:.1f} 筆），勝率 {(p > 0).mean():.1%}，平均 {p.mean():+.3f}R，"
          f"中位數 {np.median(p):+.2f}R，PF {p[p > 0].sum() / -p[p < 0].sum():.2f}")
    print(f"總 {p.sum():+.1f}R，最大回撤 {dd.max():.1f}R，最長連虧 {streak} 筆")
    print(f"持倉時間：中位數 {np.median(hrs):.1f} 小時，75% 分位 {np.percentile(hrs, 75):.1f} 小時，"
          f"最長 {hrs.max():.1f} 小時")
    print(f"停損距離：中位數 ${tr.risk.median():.2f}（{tr.risk_atr.median():.1f} ATR），"
          f"點差佔停損 中位數 {(tr.spread / tr.risk).median():.1%}")
    print("出場原因", tr.why.value_counts(normalize=True).round(3).to_dict())
    print("\n逐年")
    g = tr.groupby("year")
    print(pd.DataFrame({"筆數": g.size(), "多": g.side.apply(lambda s: (s == 1).sum()),
                        "空": g.side.apply(lambda s: (s == -1).sum()),
                        "勝率": g.pnl_r.apply(lambda x: (x > 0).mean()), "平均R": g.pnl_r.mean(),
                        "總R": g.pnl_r.sum(), "停損中位數$": g.risk.median()}).round(3).to_string())
    m = tr.groupby(tr.time.dt.to_period("M")).pnl_r.sum()
    print(f"\n逐月：{len(m)} 個月，賺錢月份 {(m > 0).mean():.0%}，最好 {m.max():+.1f}R，最差 {m.min():+.1f}R")
    print(m.round(1).to_string())
    print("\n多 / 空")
    print(tr.groupby("side").pnl_r.agg(["size", "mean", lambda x: (x > 0).mean()]).round(3).to_string())


def main():
    print("=" * 30, "A. M15、碎形 3、等幅測量停利", "=" * 30)
    d15 = P.TF(M1, "M15")
    res = run(d15, "frac3", tps=("mm",))
    tr = res["mm"]
    tr.to_csv("part25_m15_frac3_mm_trades.csv", index=False)
    detail(tr, d15)

    print("\n" + "=" * 30, "B. 套到 M5 / M1（與 M15 對照）", "=" * 30)
    rows = []
    for tf in ("M15", "M5", "M1"):
        d = d15 if tf == "M15" else P.TF(M1, tf)
        for swing in ("frac2", "frac3", "frac5", "zz2", "zz3"):
            for tp, tr in run(d, swing).items():
                b = baseline(d, tr, reps=5)
                gross = tr.pnl_r + tr.spread / tr.risk
                r = dict(tf=tf, swing=swing, tp=tp, n=len(tr), per_week=len(tr) / ((tr.time.max() - tr.time.min()).days / 7),
                         win=(tr.pnl_r > 0).mean(), R=tr.pnl_r.mean(), gross_R=gross.mean(),
                         spread_R=(tr.spread / tr.risk).mean(), rand=b.pnl_r.mean())
                r["edge"] = r["R"] - r["rand"]
                r["edge_t"] = r["edge"] / np.sqrt(tr.pnl_r.var() / len(tr) + b.pnl_r.var() / len(b))
                for tag, m, bm in (("IS", tr.time < P.IS_END, b.time < P.IS_END),
                                   ("OOS", tr.time >= P.IS_END, b.time >= P.IS_END)):
                    r[f"{tag}_R"] = tr[m].pnl_r.mean()
                    r[f"{tag}_edge"] = tr[m].pnl_r.mean() - b[bm].pnl_r.mean()
                r["risk_$"] = tr.risk.median()
                rows.append(r)
                print(tf, swing, tp, len(tr), file=sys.stderr, flush=True)
    res = pd.DataFrame(rows)
    res.to_csv("part25_tf.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(res.round(3).to_string(index=False))
    print("\n各週期平均")
    print(res.groupby("tf")[["n", "per_week", "win", "R", "gross_R", "spread_R", "edge", "edge_t", "IS_edge",
                             "OOS_edge", "risk_$"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
