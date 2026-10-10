"""第五十八部分：把 σ̂ 換成和 ATR 同一個時框、同一個單位，直接比較。

ATR14（某時框）= 過去 14 根 K 棒 true range 的平均 → 隱含的意思是「下一根大概會有多大的 true range」。
σ̂_h       = 未來 h 根「收盤到收盤」報酬的一個標準差（美元）。
兩者不同的地方：(1) 時框 h；(2) 單位：範圍（高 − 低）vs 標準差。隨機漫步下一根的期望範圍 ≈ √(8/π)·σ ≈ 1.6σ。
對齊方式：σ̂ 用 h = 1 根（和 ATR 同時框），再乘上換算常數 c 變成「預測的下一根 true range」（σ̂-ATR）：
  c = 之前月份「實際下一根 TR / σ̂」的平均（逐月 expanding，只用過去），理論值約 1.6。
比較對象：下一根的實際 true range（max(高, 前收) − min(低, 前收)）。
時框：M5（ATR14 M5 vs σ̂ 5 分）、M15（2023~2026）、H1（由 M15 合成；ATR14 H1 vs σ̂ 1 小時）。
"""
import numpy as np
import pandas as pd
from volfc import har_rel
from part55_vol_verify import prep

OUT = "results_part58.txt"


def next_tr(b):
    h, l, c = b.high.to_numpy(), b.low.to_numpy(), b.close.to_numpy()
    pc = np.r_[np.nan, c[:-1]]
    tr = np.maximum(h, pc) - np.minimum(l, pc)
    nxt = np.r_[tr[1:], np.nan]
    return tr, nxt


def compare(pr, name, b, sig, atr, tr_next, gapok):
    month = b.index.to_period("M").to_numpy()
    ok = np.isfinite(sig) & np.isfinite(atr) & np.isfinite(tr_next) & (tr_next > 0) & gapok
    # 逐月 expanding 換算常數 c（只用之前月份）
    c = np.full(len(sig), np.nan)
    ms = sorted(set(month[ok]))
    for m in ms[1:]:
        tr_ = ok & (month < m)
        te = ok & (month == m)
        c[te] = tr_next[tr_].mean() / sig[tr_].mean()
    satr = c * sig
    m = ok & np.isfinite(satr)
    y = tr_next[m]
    pr(f"\n=== {name}：{m.sum()} 根（{b.index[m][0].date()} ~ {b.index[m][-1].date()}）；換算常數 c 最後 = {c[m][-1]:.2f}（理論 1.60）===")
    pr(f"  中位數：實際下一根 TR {np.median(y):.2f}、ATR14 {np.median(atr[m]):.2f}、σ̂（1 根，標準差）{np.median(sig[m]):.2f}、σ̂-ATR（= c·σ̂）{np.median(satr[m]):.2f}  美元")
    pr(f"  {'預測':10s} {'平均偏誤':>8s} {'log誤差std':>10s} {'MAE':>7s} {'corr(log)':>9s} {'R²(log)':>8s}   TR/預測 依預測十分位（D1 低 → D10 高）")
    for nm, p in (("ATR14", atr[m]), ("σ̂-ATR", satr[m])):
        le = np.log(y) - np.log(p)
        r2 = 1 - np.var(le - le.mean()) / np.var(np.log(y))
        dec = pd.qcut(p, 10, labels=False, duplicates="drop")
        ratio = pd.Series(y / p).groupby(dec).mean().to_numpy()
        pr(f"  {nm:10s} {y.mean() / p.mean():8.2f} {le.std():10.3f} {np.mean(np.abs(y - p)):7.2f} {np.corrcoef(np.log(y), np.log(p))[0, 1]:9.3f} {r2:8.3f}   "
           + " ".join(f"{x:.2f}" for x in ratio))
    # 依小時
    hr = b.index.hour.to_numpy()[m]
    df = pd.DataFrame(dict(h=hr, tr=y, atr=atr[m], s=satr[m]))
    g = df.groupby("h").median()
    pr("  依小時中位數（伺服器時間）：")
    pr("    時    " + " ".join(f"{x:>5d}" for x in g.index))
    pr("    實際TR " + " ".join(f"{x:5.2f}" for x in g.tr))
    pr("    ATR14 " + " ".join(f"{x:5.2f}" for x in g.atr))
    pr("    σ̂-ATR " + " ".join(f"{x:5.2f}" for x in g.s))
    return pd.DataFrame(dict(close=b.close.to_numpy(), atr=atr, satr=satr, tr_next=tr_next), index=b.index)


def main():
    out = open(OUT, "w")

    def pr(*a):
        print(*a, file=out, flush=True)
    pr("第五十八部分：σ̂ 換成 ATR 的時框與單位（預測下一根 true range），與 ATR14 正面比較")
    pr("指標：平均偏誤 = 平均實際 / 平均預測（1 = 無偏）；log 誤差 std 越小越好；R²(log) = 預測解釋了多少實際 TR 的變化；")
    pr("      TR/預測 十分位越平 = 預測大小與實際相符（ATR 若在低檔低估、高檔高估，會呈現由大到小）")

    for path, name in (("data/XAUUSD_M5_2025_2026.csv", "M5（ATR14 M5 vs σ̂ 5 分）"),
                       ("data/XAUUSD_M15_2023_2026.csv", "M15（ATR14 M15 vs σ̂ 15 分）")):
        bars, step = har_rel(path, 1)
        tr, nxt = next_tr(bars)
        g = bars.gapid.to_numpy()
        gapok = np.r_[g[1:] == g[:-1], False]
        res = compare(pr, name, bars, bars.sig.to_numpy(), bars.atr.to_numpy(), nxt, gapok)
        if step == 5:
            d = res.loc["2026-09-15"]
            d = d[d.index.minute == 0]
            pr("  例：2026/9/15 每小時整點（下一根 5 分 K 的實際 TR、ATR14、σ̂-ATR）")
            for t, r in d.iterrows():
                pr(f"    {t:%H:%M}  實際 {r.tr_next:5.2f}  ATR14 {r.atr:5.2f}  σ̂-ATR {r.satr:5.2f}")

    # H1：由 M15 合成 H1 K 棒，ATR14(H1) vs σ̂ 1 小時（在 M15 上 h = 4，只取整點）
    m15, step = har_rel("data/XAUUSD_M15_2023_2026.csv", 4)
    b, _ = prep("data/XAUUSD_M15_2023_2026.csv")
    h1 = b.resample("1h", label="left", closed="left").agg(
        dict(open="first", high="max", low="min", close="last", gapid="last")).dropna()
    tr, nxt = next_tr(h1)
    atr_h1 = pd.Series(tr).rolling(14).mean().to_numpy()
    # 整點 t 的 H1 K 棒在 t+1h 收盤；預測要在那時做 → 用 M15 中 t+45 分那根（該小時最後一根）的 σ̂
    key = h1.index + pd.Timedelta("45min")
    sig = m15.sig.reindex(key).to_numpy()
    g = h1.gapid.to_numpy()
    gapok = np.r_[g[1:] == g[:-1], False]
    h1.index.name = "time"
    compare(pr, "H1（ATR14 H1 vs σ̂ 1 小時，由 M15 合成）", h1, sig, atr_h1, nxt, gapok)
    out.close()
    print(open(OUT).read())


if __name__ == "__main__":
    main()
