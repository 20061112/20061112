"""原報告與前面各部分都還沒測過的參數，一次只動一個（其他 = 原報告設定）。

訊號：ATR 週期、進場檢查頻率、出場確認天數、訊號延遲、每週檢查日
資金：現金比例、布林加碼門檻 / 倍數、恐慌補倉買 QQQ 或 TQQQ、定投頻率
曝險：跌破 MA200 時 1 倍部位改持 T-bill、最高槓桿 2 倍、換倉成本
輸出 results_params.csv、results_params.txt。
"""
import numpy as np
import pandas as pd
import engine as E
from run_tests import q, OK0, Z, ABOVE, c

out = []
P = lambda s: (print(s, flush=True), out.append(s))
F = 0.0095; SP = 0.0045
rf_y = q.rf * 252
q["q2_syn"] = (1 + 2 * q.r - F / 252 - (rf_y + SP) / 252).cumprod()


def ev(w, **kw):
    full = E.stats(E.run(q, w, **kw))
    s10 = E.stats(E.run(q, w, start="2010-02-12", **kw))
    roll = E.rolling_windows(q, w, **kw)
    return full, s10, roll


w0 = E.weights_from_rules(q, OK0, ~OK0)
b_full, b10, broll = ev(w0)
rows = []


def add(group, label, w, **kw):
    f, s10, roll = ev(w, **kw)
    d = roll.irr.values - broll.irr.values
    rows.append(dict(group=group, label=label, irr99=f["irr"], irr10=s10["irr"], worst=f["worst"], sw=f["sw_per_year"],
                     win10=int((d > 0).sum()), d10_med=np.median(d), d10_min=d.min(), g10_min=roll.gain.min()))
    P(f"{group:8s} {label:30s} irr99 {f['irr']:.3f} irr10 {s10['irr']:.3f} win10 {(d > 0).sum():2d}/17 dIRR10y {np.median(d):+.4f}"
      f"(min {d.min():+.4f}) worst {f['worst']:.2f} g10min {roll.gain.min():.2f} sw {f['sw_per_year']:.1f}")


def rule(ok, bad=None, mask=None):
    bad = ~ok if bad is None else bad
    return E.weights_from_rules(q, ok, bad, entry_mask=mask)


def weekday_mask(d):
    """每週第一個「星期 ≥ d」的交易日。"""
    wk = q.index.to_period("W")
    dow = q.index.dayofweek
    s = pd.Series(dow >= d, index=q.index)
    first = s & ~s.groupby(wk).cumsum().shift(1, fill_value=0).astype(bool).groupby(wk).transform(lambda x: x)
    m = pd.Series(False, index=q.index)
    for _, g in s.groupby(wk):
        hit = g[g]
        m[(hit.index[0] if len(hit) else g.index[-1])] = True
    return m.values


if __name__ == "__main__":
    P(f"BASE {'':35s} irr99 {b_full['irr']:.3f} irr10 {b10['irr']:.3f} worst {b_full['worst']:.2f} g10min {broll.gain.min():.2f} sw {b_full['sw_per_year']:.1f}")

    # ---------------- 訊號
    for n in [10, 14, 30, 40]:
        z = E.atr_z(q, n)
        add("ATR期", f"ATR{n}", rule((c > c.rolling(200).mean()) & (z < 0.5)), z=z)
    month_start = q.index.to_period("M").to_series(index=q.index)
    month_start = month_start.ne(month_start.shift(1)).values
    weekno = q.index.to_period("W").to_series(index=q.index).apply(lambda p: p.ordinal).values
    add("進場頻率", "每天", rule(OK0, mask=np.ones(len(q), bool)))
    add("進場頻率", "每 2 週", rule(OK0, mask=q.week_start.values & (weekno % 2 == 0)))
    add("進場頻率", "每月", rule(OK0, mask=month_start))
    bad = ~OK0
    add("出場確認", "連 2 天", rule(OK0, bad & bad.shift(1, fill_value=False)))
    add("出場確認", "連 3 天", rule(OK0, bad & bad.shift(1, fill_value=False) & bad.shift(2, fill_value=False)))
    add("出場確認", "只在每週檢查", rule(OK0, bad & pd.Series(q.week_start, index=q.index)))
    add("延遲", "0 天（當天收盤訊號、收盤成交）", rule(OK0.shift(-1, fill_value=False)))
    add("延遲", "多 1 天", rule(OK0.shift(1, fill_value=False)))
    add("延遲", "多 2 天", rule(OK0.shift(2, fill_value=False)))
    for d, nm in [(2, "週三"), (4, "週五")]:
        m = weekday_mask(d)
        add("檢查日", nm, rule(OK0, mask=m), dca_mask=m)

    # ---------------- 資金
    for cf in [0.0, 0.1, 0.3]:
        add("現金比例", f"{cf:.0%}", w0, cash_frac=cf)
    for th in [-0.3, -1.0, -1.5]:
        add("加碼門檻", f"布林 z ≤ {th}", w0, boost_thr=th)
    add("加碼門檻", "不加碼", w0, boost=False)
    for m in [1.5, 3.0]:
        add("加碼倍數", f"{m}x", w0, boost_mult=m)
    add("恐慌補倉", "買 TQQQ", w0, panic_asset="TQQQ")
    add("恐慌補倉", "不補倉（現金一直放著）", w0, panic=False)
    add("定投頻率", "每月", w0, dca_mask=month_start)

    # ---------------- 曝險
    off = (~ABOVE).shift(1, fill_value=False).values
    add("跌破避險", "MA200 下 1 倍部位改持 T-bill", w0, off=off)
    off2 = ((~ABOVE) & (Z > 0.5)).shift(1, fill_value=False).values
    add("跌破避險", "MA200 下且 z>0.5 才改持 T-bill", w0, off=off2)
    add("最高槓桿", "2 倍（QQQ↔QLD）", w0, tq_col="q2_syn")
    for cst in [0.002, 0.005]:
        add("換倉成本", f"{cst:.1%}", w0, cost=cst)

    R = pd.DataFrame(rows)
    R.to_csv("results_params.csv", index=False, float_format="%.4f")
    open("results_params.txt", "w").write("\n".join(out) + "\n")
