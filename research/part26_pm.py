"""第二十六部分：(1) H2 / H3 / H4 用換算參數；(2) 持倉方法：停損 ATR、盈虧比（停利）、持有時間、部分停利。
換算：s = √(週期分鐘/60)；k × s、h ÷ s、停損 ATR 倍數 ÷ s（H1 k0.25 h6 停損2 → H2 k0.354 h4.24 停損1.41）。
評估（每筆固定風險 1R）：每筆 R、t、PF、勝率、每年 R、日 Sharpe（以出場日實現損益計）、總R/回撤、最差年、去漂移。
同一根 K 棒同時碰到停損與停利一律算停損（保守）。輸出 results_part26.txt
"""
import itertools, math, warnings
import numpy as np, pandas as pd
warnings.simplefilter("ignore", RuntimeWarning)
from load import load
from cusum_lab import bars, zscore, backtest, backtest_pm, with_atr, m1_backtest, summ, line, detrend_R, M1_PATH
from changepoint import cusum_signals

out = open("results_part26.txt", "w")
def P(s=""):
    print(s); out.write(s + "\n"); out.flush()

def ev(b, T):
    d = summ(T, b)
    if d["n"] < 10:
        return None
    yrs = (b.index[-1] - b.index[0]).days / 365.25
    days = pd.date_range(b.index[0].normalize(), b.index[-1].normalize(), freq="B")
    dp = T.groupby(T.exit_time.dt.normalize()).pnl_R.sum().reindex(days, fill_value=0)
    eq = dp.cumsum(); mdd = (eq.cummax() - eq).max()
    d.update(per_year=len(T) / yrs, R_year=T.pnl_R.sum() / yrs, sharpe=dp.mean() / dp.std() * math.sqrt(252), rdd=T.pnl_R.sum() / mdd, mdd=mdd)
    return d
HDR = f"{'':26s} {'筆/年':>5s} {'每筆R':>7s} {'t':>5s} {'PF':>5s} {'勝率':>4s} {'R/年':>6s} {'Sharpe':>6s} {'R/回撤':>6s} {'回撤':>5s} {'最差年':>6s} {'去漂移':>6s} {'持有天':>5s}"
def row(lab, b, T):
    d = ev(b, T)
    if d is None:
        P(f"  {lab:26s} 筆數不足"); return None
    P(f"  {lab:26s} {d['per_year']:5.0f} {d['mean']:+7.3f} {d['t']:+5.2f} {d['pf']:5.2f} {d['win']:4.0%} {d['R_year']:+6.1f} "
      f"{d['sharpe']:6.2f} {d['rdd']:6.1f} {d['mdd']:5.1f} {d['minyr']:+6.2f} {d['adj']:+6.3f} {d['hold']:5.1f}")
    return d

MIN = {"H1": 60, "H2": 120, "H3": 180, "H4": 240}
B = {tf: bars(tf) for tf in MIN}
Z = {tf: zscore(b) for tf, b in B.items()}
def mapped(tf):
    s = math.sqrt(MIN[tf] / 60); return 0.25 * s, 6 / s, 2 / s

# ======================================================================== 1. 換算到較大週期
P("1) 換算參數到 H2 / H3 / H4（2023/1 ~ 2026/10）")
P(HDR)
SIG = {}
for tf in MIN:
    k, h, st = mapped(tf)
    SIG[tf] = cusum_signals(Z[tf], k, h)
    row(f"{tf} 換算 k{k:.3f} h{h:.2f} 停{st:.2f}", B[tf], backtest(B[tf], SIG[tf], st))
    if tf != "H1":
        row(f"{tf} 直接沿用 k0.25 h6 停2", B[tf], backtest(B[tf], cusum_signals(Z[tf], 0.25, 6), 2.0))

P("\n  H2 換算點附近網格（每筆 R / 去漂移 / 筆數；停損 1.41 ATR(H2)）")
rows = []
for k, h in itertools.product((0.25, 0.3, 0.354, 0.4, 0.5), (3.5, 4.0, 4.24, 4.5, 5.0, 5.5)):
    T = backtest(B["H2"], cusum_signals(Z["H2"], k, h), 1.414); d = summ(T, B["H2"])
    rows.append(dict(k=k, h=h, mean=d.get("mean"), adj=d.get("adj"), n=d["n"], minyr=d.get("minyr")))
G = pd.DataFrame(rows)
P("        h→ " + " ".join(f"{x:>14.2f}" for x in sorted(G.h.unique())))
for k, g in G.groupby("k"):
    P(f"  k={k:.3f} " + " ".join(f"{r.mean:+.2f}/{r.adj:+.2f}/{r.n:3d}" for r in g.itertuples()))
ok = G[G.n >= 30]
P(f"  {len(ok)} 組：原始>0 {(ok['mean'] > 0).mean():.0%}、去漂移>0 {(ok.adj > 0).mean():.0%}、每年都>0 {(ok.minyr > 0).mean():.0%}，平均 {ok['mean'].mean():+.3f} / {ok.adj.mean():+.3f}")

m1 = load(M1_PATH)
P("\n  M1 逐根（2023/6 起）")
for tf in ("H1", "H2"):
    k, h, st = mapped(tf)
    T = m1_backtest(B[tf], SIG[tf], m1, st)
    yr = T.groupby(T.entry_time.dt.year).pnl_R.mean()
    P(f"  {tf}: {line(summ(T, B[tf]))}")
    P("      逐年 " + "  ".join(f"{y}:{v:+.2f}" for y, v in yr.items()))
TH = backtest(B["H1"], SIG["H1"], 2.0); T2 = backtest(B["H2"], SIG["H2"], mapped("H2")[2])
from changepoint import daily_pnl_R
c = pd.concat([daily_pnl_R(B["H1"], TH), daily_pnl_R(B["H2"], T2)], axis=1).fillna(0)
P(f"  H1 與 H2 換算版 日損益相關 {c.corr().iloc[0, 1]:.2f}")
both = pd.concat([TH.assign(pnl_R=TH.pnl_R * .5), T2.assign(pnl_R=T2.pnl_R * .5)]).sort_values("exit_time")
P(HDR)
row("H1 1R", B["H1"], TH); row("H2 換算 1R", B["H2"], T2)
d = ev(B["H1"], both); P(f"  {'H1 0.5R + H2 0.5R':26s} 筆/年 {len(both) / ((B['H1'].index[-1] - B['H1'].index[0]).days / 365.25):.0f}  每年 {d['R_year']:+.1f}R  Sharpe {d['sharpe']:.2f}  R/回撤 {d['rdd']:.1f}  回撤 {d['mdd']:.1f}R")

# ======================================================================== 2. 持倉方法
for tf in ("H1", "H2"):
    b = B[tf]; sig = SIG[tf]; k, h, st0 = mapped(tf)
    hpb = MIN[tf] / 60                       # 每根幾小時
    P("\n" + "=" * 110)
    P(f"2) 持倉方法 — {tf}（k{k:.3f} h{h:.2f}；基準 = 停損 {st0:.2f} ATR、無停利、反手出場）")

    P(f"\n  2a) 停損 ATR 倍數（ATR14 {tf}）")
    P(HDR)
    for s_ in (0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0):
        m = s_ / (1 if tf == "H1" else math.sqrt(2))
        row(f"停損 {m:.2f} ATR（≈H1 {s_:.2f}）", b, backtest_pm(b, sig, m))
    row("不設停損（R=基準停損距離）", b, backtest_pm(b, sig, None, risk_atr=st0))

    P(f"\n  2b) ATR 長度（停損 = 同樣的 H1 等價距離）")
    P(HDR)
    for n in (14, 50, 100, int(14 * 24 / hpb)):
        lab = f"ATR{n}" + ("（≈日線 ATR14）" if n == int(14 * 24 / hpb) else "")
        row(lab, b, backtest_pm(with_atr(b, n), sig, st0))

    P(f"\n  2c) 盈虧比：固定停利（停損 {st0:.2f} ATR；反向訊號仍會先出場）")
    P(HDR)
    for tp in (1, 1.5, 2, 3, 4, 5, 6, 8, 10, 15, None):
        row(f"停利 {tp}R" if tp else "不停利（基準）", b, backtest_pm(b, sig, st0, tp_R=tp))

    P(f"\n  2d) 持有時間上限（到時間就在開盤平倉）")
    P(HDR)
    for hrs in (12, 24, 48, 72, 120, 168, 240, 360, 480, None):
        mb = None if hrs is None else max(1, int(round(hrs / hpb)))
        row(f"最長 {hrs/24:.1f} 天" if hrs else "不限（基準）", b, backtest_pm(b, sig, st0, max_bars=mb))

    P(f"\n  2e) 部分停利（先平一部分，其餘跟訊號）")
    P(HDR)
    for lvl, f in ((1, .5), (2, .5), (3, .5), (4, .5), (3, 1 / 3), (5, .5)):
        row(f"{lvl}R 平 {f:.0%}", b, backtest_pm(b, sig, st0, partial=(lvl, f)))
        row(f"{lvl}R 平 {f:.0%} + 剩下移保本", b, backtest_pm(b, sig, st0, partial=(lvl, f), be_after_partial=True))

    P(f"\n  2f) 停損 × 停利 組合（每格：每筆R / Sharpe / R回撤）")
    tps = (2, 3, 5, 8, None)
    P("        停利→ " + " ".join(f"{('不停利' if t is None else str(t) + 'R'):>17s}" for t in tps))
    for s_ in (1.0, 1.5, 2.0, 3.0, 4.0):
        m = s_ / (1 if tf == "H1" else math.sqrt(2))
        cells = []
        for tp in tps:
            d = ev(b, backtest_pm(b, sig, m, tp_R=tp))
            cells.append(f"{d['mean']:+.2f}/{d['sharpe']:.2f}/{d['rdd']:4.1f}")
        P(f"  停損{m:4.2f}ATR " + " ".join(f"{x:>17s}" for x in cells))
