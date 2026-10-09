"""第三十四部分：把支撐壓力價位加進版本 A（IB 突破延續）。

版本 A 的 735 筆交易（final_A_trades.csv，3/3~10/7，滾動前推）不變，只加價位規則：
  價位來源（每天開盤前決定，只用過去資料）：
    kde15  M15 插針 KDE（擴張）    kde60  H1 插針 KDE（滾動 5 天）
    pdhl   前日高低                r50    50 美元整數關卡
    all    以上合併
    placebo  kde15 隨機平移 3~6 個頻寬（20 次）→ 用來判斷「價位」本身有沒有作用
  距離單位：R（= 版本 A 的停損距離，中位數約 26 美元）。
  1. 濾網：進場方向正前方最近的價位距離（front）分組，看每組的績效；
     以及「剛突破一條價位」（價位在進場價後方 0.5 R 內）是否較好。
  2. 停利：在正前方第一條距離 >= k R 的價位掛停利（k = 0.5 / 1 / 2 / 3），全出或出一半；
     與「固定 k R 停利」和 placebo 價位比較。
     用 M1 檢查：進場後到原出場時間之前，若碰到停利價就在停利價出場；原出場那根若是停損，停損優先（保守）。
  損益單位：美元/盎司（扣版本 A 的點差）。拆半：3/3~5/31 vs 6/1~10/7。
"""
import numpy as np
import pandas as pd
from load import load
from td import atr
from kde_levels import find_pins
import part30_kde as P30
from part30_kde import build_levels, shift_levels
from part33_bounce_break import resample, simple_levels

SPLIT = pd.Timestamp("2026-06-01")


def load_trades():
    t = pd.read_csv("final_A_trades.csv", encoding="utf-8-sig")
    T = pd.DataFrame({
        "day": pd.to_datetime(t["交易日"]),
        "dir": np.where(t["方向"] == "多", 1, -1),
        "t_in": pd.to_datetime(t["進場時間"]),
        "entry": t["進場價"], "risk": t["停損距離"],
        "t_out": pd.to_datetime(t["出場時間"]), "exit": t["出場價"],
        "reason": t["出場原因"], "pnl": t["損益_美元每盎司"], "spread": t["點差"],
    })
    T["half"] = np.where(T.day < SPLIT, 1, 2)
    return T


def front_behind(T, lv):
    """每筆：正前方最近價位距離、後方最近價位距離（R），沒有則 inf；以及正前方所有價位（R 排序）。"""
    fr, bh, fronts = [], [], []
    for r in T.itertuples():
        L = lv.get(r.day)
        if L is None or len(L) == 0:
            fr.append(np.inf); bh.append(np.inf); fronts.append(np.array([]))
            continue
        d = (L.price.to_numpy() - r.entry) * r.dir / r.risk
        f = np.sort(d[d > 0])
        b = -d[d <= 0]
        fr.append(f[0] if len(f) else np.inf)
        bh.append(b.min() if len(b) else np.inf)
        fronts.append(f)
    return np.array(fr), np.array(bh), fronts


def tp_pnl(T, m1, tp_R, frac=1.0):
    """tp_R：每筆的停利距離（R，nan = 不設）。frac = 停利出場比例。回傳新的損益。"""
    hi, lo = m1.high.to_numpy(), m1.low.to_numpy()
    idx = m1.index
    out = T.pnl.to_numpy(float).copy()
    for i, r in enumerate(T.itertuples()):
        k = tp_R[i]
        if not np.isfinite(k):
            continue
        tp = r.entry + r.dir * k * r.risk
        a = idx.searchsorted(r.t_in, side="right")
        b = idx.searchsorted(r.t_out, side="left")  # 不含原出場那根（停損優先）
        if r.reason == "收盤":
            b = idx.searchsorted(r.t_out, side="right")
        seg = hi[a:b] >= tp if r.dir == 1 else lo[a:b] <= tp
        if seg.any():
            new = k * r.risk - r.spread
            out[i] = frac * new + (1 - frac) * r.pnl
    return out


def stats(p, days):
    p = np.asarray(p, float)
    if len(p) == 0:
        return dict(n=0)
    eq = np.cumsum(p)
    dd = (np.maximum.accumulate(np.r_[0, eq]) - np.r_[0, eq]).max()
    return dict(n=len(p), avg=p.mean(), PF=p[p > 0].sum() / -p[p < 0].sum(), win=(p > 0).mean(),
                total=p.sum(), maxDD=dd, ret_dd=p.sum() / dd if dd > 0 else np.nan)


def main():
    T = load_trades()
    m1 = load("data/XAUUSD_M1_2026.csv")
    x5 = resample(m1, "5min")
    days = sorted(T.day.unique())
    d15, d60 = resample(m1, "15min"), resample(m1, "1h")
    P30.RNG = np.random.default_rng(4)
    lv = {
        "kde15": build_levels(find_pins(d15, atr(d15)), days, "expanding", None, 0.5),
        "kde60": build_levels(find_pins(d60, atr(d60)), days, "rolling", 5, 0.5),
        "pdhl": simple_levels(x5, "pdhl"),
        "r50": simple_levels(x5, "r50"),
    }
    lv["all"] = {d: pd.concat([lv[s].get(d, pd.DataFrame(columns=["price"])) for s in ("kde15", "kde60", "pdhl", "r50")])
                 for d in days}
    placebos = [shift_levels(lv["kde15"]) for _ in range(20)]

    lines = []
    P = lines.append
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 300)
    base = stats(T.pnl, len(days))
    P("=== 版本 A 原始 ===\n" + str({k: round(v, 3) for k, v in base.items()}))
    for h in (1, 2):
        P(f"  第 {h} 半：" + str({k: round(v, 3) for k, v in stats(T.pnl[T.half == h], 1).items()}))

    # 1. 濾網：正前方價位距離分組
    bins = [0, 0.5, 1, 2, 4, np.inf]
    labels = ["0~0.5R", "0.5~1R", "1~2R", "2~4R", ">4R/無"]
    P("\n=== 1. 正前方最近價位距離（R）分組：筆數 / 每筆平均美元 / PF（前半、後半）；placebo 為 20 次平均 ===")
    rows = []
    for src in ("kde15", "kde60", "pdhl", "r50", "all"):
        fr, bh, _ = front_behind(T, lv[src])
        T["front_" + src], T["behind_" + src] = fr, bh
        g = pd.cut(fr, bins, labels=labels, right=False)
        for lab in labels:
            m = g == lab
            s1, s2 = stats(T.pnl[m & (T.half == 1)], 1), stats(T.pnl[m & (T.half == 2)], 1)
            s = stats(T.pnl[m], 1)
            rows.append(dict(src=src, front=lab, n=s["n"], avg=s.get("avg"), PF=s.get("PF"),
                             avg_H1=s1.get("avg"), avg_H2=s2.get("avg"), PF_H1=s1.get("PF"), PF_H2=s2.get("PF")))
    pl_rows = []
    for k, pl in enumerate(placebos):
        fr, bh, _ = front_behind(T, pl)
        g = pd.cut(fr, bins, labels=labels, right=False)
        for lab in labels:
            m = g == lab
            s = stats(T.pnl[m], 1)
            pl_rows.append(dict(front=lab, n=s["n"], avg=s.get("avg"), PF=s.get("PF")))
    F = pd.DataFrame(rows)
    PL = pd.DataFrame(pl_rows).groupby("front").mean().reindex(labels)
    P(F.round(2).to_string())
    P("placebo（kde15 平移）\n" + PL.round(2).to_string())

    P("\n=== 1b. 剛突破一條價位（後方 0.5R 內有價位）vs 沒有 ===")
    for src in ("kde15", "kde60", "pdhl", "r50", "all"):
        m = T["behind_" + src] <= 0.5
        a, b = stats(T.pnl[m], 1), stats(T.pnl[~m], 1)
        P(f"{src:6s} 有：n={a['n']:4d} avg={a['avg']:7.2f} PF={a['PF']:.2f} | 無：n={b['n']:4d} avg={b['avg']:7.2f} PF={b['PF']:.2f}")

    # 1c. 濾網規則：前方 < x R 有價位就不做
    P("\n=== 1c. 濾網：正前方 x R 內有價位就不做（真實 vs placebo）===")
    frow = []
    for src in ("kde15", "kde60", "pdhl", "r50", "all"):
        for xR in (0.5, 1, 2):
            keep = T["front_" + src] >= xR
            s = stats(T.pnl[keep], len(days))
            s1, s2 = stats(T.pnl[keep & (T.half == 1)], 1), stats(T.pnl[keep & (T.half == 2)], 1)
            frow.append(dict(src=src, xR=xR, n=s["n"], avg=s["avg"], PF=s["PF"], total=s["total"], maxDD=s["maxDD"],
                             ret_dd=s["ret_dd"], PF_H1=s1["PF"], PF_H2=s2["PF"]))
    for xR in (0.5, 1, 2):
        ss = []
        for pl in placebos:
            fr, _, _ = front_behind(T, pl)
            ss.append(stats(T.pnl[fr >= xR], len(days)))
        frow.append(dict(src="placebo", xR=xR, **{k: np.mean([s[k] for s in ss]) for k in ("n", "avg", "PF", "total", "maxDD", "ret_dd")}))
    P(pd.DataFrame(frow).round(2).to_string())

    # 2. 停利
    P("\n=== 2. 停利：正前方第一條 >= k R 的價位（全出 / 出一半）vs 固定 k R vs placebo ===")
    trow = []
    for k in (0.5, 1, 2, 3, 5):
        fixed = tp_pnl(T, m1, np.full(len(T), float(k)))
        s = stats(fixed, len(days))
        trow.append(dict(rule=f"固定 {k}R", **s, PF_H1=stats(fixed[T.half == 1], 1)["PF"], PF_H2=stats(fixed[T.half == 2], 1)["PF"]))
        for src in ("kde15", "kde60", "pdhl", "r50", "all"):
            _, _, fronts = front_behind(T, lv[src])
            tpR = np.array([f[f >= k][0] if (f >= k).any() else np.nan for f in fronts])
            for frac in (1.0, 0.5):
                p = tp_pnl(T, m1, tpR, frac)
                s = stats(p, len(days))
                trow.append(dict(rule=f"{src} >= {k}R 出{int(frac * 100)}%", **s,
                                 PF_H1=stats(p[T.half == 1], 1)["PF"], PF_H2=stats(p[T.half == 2], 1)["PF"]))
        ss = []
        for pl in placebos[:10]:
            _, _, fronts = front_behind(T, pl)
            tpR = np.array([f[f >= k][0] if (f >= k).any() else np.nan for f in fronts])
            ss.append(stats(tp_pnl(T, m1, tpR, 1.0), len(days)))
        trow.append(dict(rule=f"placebo >= {k}R 出100%", **{kk: np.mean([s[kk] for s in ss]) for kk in ss[0]}))
        print("tp", k, "done", flush=True)
    TP = pd.DataFrame(trow)
    P(TP.round(2).to_string())
    TP.to_csv("part34_tp.csv", index=False, float_format="%.3f")
    pd.DataFrame(frow).to_csv("part34_filter.csv", index=False, float_format="%.3f")
    open("results_part34.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
