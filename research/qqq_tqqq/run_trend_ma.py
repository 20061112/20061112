"""趨勢濾網的種類 × 週期：只換「QQQ 站上 X」的 X，ATR z < 0.5、定投、出場規則都不動。

種類：
  SMA    簡單均線
  EMA    指數均線（span = n）
  WMA    線性加權均線
  VWMA   成交量加權均線 Σ(C·V) / ΣV
  VWAP   滾動 VWAP：Σ(典型價 (H+L+C)/3 · V) / ΣV
  KAMA   Kaufman 自適應均線（ER 週期 = n/10，快 2、慢 30；ER 來自 FINAL_STRATEGY_ER）
  CROSS  均線交叉：SMA(n/4) > SMA(n)（n = 200 時就是 50/200 黃金交叉）
週期 100~300（每 20）。歷史看整片網格；蒙地卡羅比較各種類在 n = 150 / 200 / 250 的平均。
輸出 results_trend_ma.csv、results_trend_ma.txt。
"""
import numpy as np
import pandas as pd
import engine as E
from run_tests import q, OK0, Z, w_rule, evaluate


def wma(x, n):
    wts = np.arange(1, n + 1, dtype=float)
    return x.rolling(n).apply(lambda a: np.dot(a, wts) / wts.sum(), raw=True)


def kama(c, n):
    er_n = max(int(round(n / 10)), 2)
    ch = c.diff(er_n).abs(); vol = c.diff().abs().rolling(er_n).sum()
    er = (ch / vol).fillna(0).values
    fast, slow = 2 / 3, 2 / 31
    sc = (er * (fast - slow) + slow) ** 2
    cv = c.values; out = np.full(len(cv), np.nan)
    start = n
    if start >= len(cv):
        return pd.Series(out, index=c.index)
    out[start] = cv[:start + 1].mean()
    for i in range(start + 1, len(cv)):
        out[i] = out[i - 1] + sc[i] * (cv[i] - out[i - 1])
    return pd.Series(out, index=c.index)


def line(d, kind, n):
    c, v = d.close, d.volume
    if kind == "SMA":
        return c.rolling(n).mean()
    if kind == "EMA":
        return c.ewm(span=n, adjust=False, min_periods=n).mean()
    if kind == "WMA":
        return wma(c, n)
    if kind == "VWMA":
        return (c * v).rolling(n).sum() / v.rolling(n).sum()
    if kind == "VWAP":
        tp = (d.high + d.low + c) / 3
        return (tp * v).rolling(n).sum() / v.rolling(n).sum()
    if kind == "KAMA":
        return kama(c, n)
    raise ValueError(kind)


def trend_ok(d, kind, n):
    c = d.close
    if kind == "CROSS":
        return c.rolling(max(n // 4, 2)).mean() > c.rolling(n).mean()
    return c > line(d, kind, n)


KINDS = ["SMA", "EMA", "WMA", "VWMA", "VWAP", "KAMA", "CROSS"]
NS = list(range(100, 301, 20))

if __name__ == "__main__":
    out = []
    P = lambda s: (print(s, flush=True), out.append(s))
    w0 = w_rule(OK0)
    b, broll, bcal = evaluate(w0); b, broll, bcal = evaluate(w0, broll, bcal)
    rows = []
    for kind in KINDS:
        for n in NS:
            w = w_rule(trend_ok(q, kind, n) & (Z < 0.5))
            o, _, _ = evaluate(w, broll, bcal)
            rows.append(dict(kind=kind, n=n) | o)
            P(f"{kind:5s} {n:3d} irr99 {o['irr99']:.3f} irr10 {o['irr10']:.3f} win10 {o['win10']:2d}/17 dIRR10y {o['d_irr10y_med']:+.3f} "
              f"worst {o['worst']:.2f} g10min {o['g10_min']:.2f} worstyr {o['worst_year']:.2f} sh {o['sharpe']:.2f} sw {o['sw']:.1f}")
    R = pd.DataFrame(rows)
    R.to_csv("results_trend_ma.csv", index=False, float_format="%.4f")
    P(f"\nBASE SMA200：irr99 {b['irr99']:.3f}、最慘 {b['worst']:.2f}、換倉 {b['sw']:.1f}/年")
    P("\n1999 起 IRR（列 = 種類，欄 = 週期）")
    P(R.pivot(index="kind", columns="n", values="irr99").reindex(KINDS).to_string(float_format=lambda x: f"{x:.3f}"))
    P("\n最慘相對本金")
    P(R.pivot(index="kind", columns="n", values="worst").reindex(KINDS).to_string(float_format=lambda x: f"{x:.2f}"))
    P("\n換倉次數 / 年")
    P(R.pivot(index="kind", columns="n", values="sw").reindex(KINDS).to_string(float_format=lambda x: f"{x:.1f}"))
    G = R.groupby("kind").agg(irr99_med=("irr99", "median"), irr99_min=("irr99", "min"), irr99_max=("irr99", "max"),
                              d10_med=("d_irr10y_med", "median"), worst_med=("worst", "median"), worst_min=("worst", "min"),
                              sw_med=("sw", "median")).reindex(KINDS)
    P("\n各種類整片週期（100~300）彙總")
    P(G.to_string(float_format=lambda x: f"{x:.3f}"))

    # 蒙地卡羅：各種類在 150 / 200 / 250 三個週期的平均 IRR，相對 SMA 同樣三個週期
    import run_ideas_mc as MC
    res = []
    for k in range(200):
        d = MC.path(); d["volume"] = q.volume.values[np.random.default_rng(k).integers(0, len(q), len(d))] * 0 + 1.0
        # 拼貼路徑沒有真實成交量 → VWMA / VWAP 在蒙地卡羅中退化，不納入
        z = E.atr_z(d); s0 = d.index[MC.WARM]
        row = {}
        for kind in ["SMA", "EMA", "WMA", "KAMA", "CROSS"]:
            irs = []; ws = []
            for n in [150, 200, 250]:
                st = E.stats(E.run(d, E.weights_from_rules(d, trend_ok(d, kind, n) & (z < 0.5), ~(trend_ok(d, kind, n) & (z < 0.5))), start=s0, z=z))
                irs.append(st["irr"]); ws.append(st["worst"])
                if n == 200:
                    row[f"{kind}200"] = st["irr"]; row[f"{kind}200|w"] = st["worst"]
            row[kind] = np.mean(irs); row[kind + "|w"] = np.mean(ws)
        res.append(row)
    M = pd.DataFrame(res)
    P("\n區塊拼貼蒙地卡羅（200 條 10 年路徑；VWMA / VWAP 因拼貼路徑沒有真實成交量不納入）")
    for kind in ["EMA", "WMA", "KAMA", "CROSS"]:
        for suf in ["", "200"]:
            dd = M[kind + suf] - M["SMA" + suf]; dw = M[kind + suf + "|w"] - M["SMA" + suf + "|w"]
            lab = f"{kind}{suf or '（150/200/250 平均）'}"
            P(f"  {lab:22s} − SMA：IRR 差 中位 {dd.median():+.4f} 勝率 {(dd > 0).mean():4.0%} | 最慘相對本金 差 中位 {dw.median():+.3f} 5% {dw.quantile(.05):+.3f}")
    open("results_trend_ma.txt", "w").write("\n".join(out) + "\n")
