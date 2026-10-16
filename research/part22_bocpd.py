"""第二十二部分：BOCPD 變點順勢（波段）與 CUSUM 重做，D1 / H4 / H1 / M15。

資料：data/XAUUSD_M15_2023_2026.csv（2023/1/3 ~ 2026/10/9），其他週期由 M15 重取樣。
切分（事先決定）：研究期 2023~2024，驗證期 2025/1 ~ 2026/10；另外做逐年前推（只用過去年份挑參數）。
可推翻的預測（事先寫下）：
  P1 變點後 N 根的報酬方向要與偵測到的新漂移一致（扣掉整段多頭的平均漂移後仍為正）。
  P2 要比「固定 20 根動能（TSMOM20）」更早抓到同一段趨勢；
  P3 若 BOCPD 只是比較吵的均線（部位與 TSMOM / 均線高度相關、迴歸後沒有截距），就放棄。
輸出：results_part22.txt、part22_grid.csv、part22_trades_*.csv
"""
import itertools, math, time
import numpy as np
import pandas as pd
from changepoint import bars, zscore, bocpd, bocpd_ng, bocpd_signals, cusum_signals, backtest, stats, fmt, daily_pnl_R, bar_pnl, detrend
from part22_bocpd_helpers import event_study, random_dir, ols

SPLIT = pd.Timestamp("2025-01-01")
TFS = ["D1", "H4", "H1", "M15"]
HOR = {"D1": [1, 5, 10, 20], "H4": [1, 6, 12, 30], "H1": [1, 4, 12, 24], "M15": [1, 4, 16, 48]}
# 代表參數：網格正中間，事先指定（不是挑最好的）
REP_B = dict(lam=50, w=10, theta=0.5, c=0.8, stop=2.0, exit="flip")
REP_C = dict(k=0.25, h=4, stop=2.0)

out = open("results_part22.txt", "w")
def P(s=""):
    print(s); out.write(s + "\n"); out.flush()


def split_stats(T):
    a = T[T.entry_time < SPLIT] if len(T) else T
    b = T[T.entry_time >= SPLIT] if len(T) else T
    return stats(a), stats(b)


def tsmom_sign(b, L=20):
    return np.sign(b.close - b.close.shift(L)).fillna(0).to_numpy()


def lead_lag(b, sig, L=20, horizon=60):
    """P2：每個訊號（方向 d，時間 t）相對 TSMOM20 是早還是晚。
    - TSMOM 在 t 已經是 d：BOCPD 比較晚，晚了 t − (TSMOM 翻成 d 的時間) 根
    - TSMOM 在 t 還不是 d：之後第一次翻成 d 的時間 − t = BOCPD 領先的根數；horizon 內都沒翻 = 「TSMOM 從未確認」"""
    s = tsmom_sign(b, L)
    n = len(s)
    rec = []
    for t in np.where(sig != 0)[0]:
        d = sig[t]
        if s[t] == d:
            j = t
            while j > 0 and s[j - 1] == d:
                j -= 1
            rec.append(("later", t - j))
        else:
            k = next((j for j in range(t + 1, min(n, t + horizon + 1)) if s[j] == d), None)
            rec.append(("earlier", k - t) if k is not None else ("never", np.nan))
    return pd.DataFrame(rec, columns=["kind", "bars"])


def flip_events(b, L=20):
    s = tsmom_sign(b, L)
    e = np.zeros(len(s), int)
    ch = (s != np.r_[0, s[:-1]]) & (s != 0)
    e[ch] = s[ch]
    e[:L + 1] = 0
    return e


def walk_forward(grid_trades, keys):
    """逐年前推：用「過去所有年份」的平均 R（至少 15 筆）挑一組，套用到下一年。"""
    years = [2024, 2025, 2026]
    pieces, picks = [], []
    for y in years:
        best, bk = -np.inf, None
        for k in keys:
            T = grid_trades[k]
            if len(T) == 0:
                continue
            tr = T[T.entry_time.dt.year < y]
            if len(tr) < 15:
                continue
            m = tr.pnl_R.mean()
            if m > best:
                best, bk = m, k
        if bk is None:
            continue
        T = grid_trades[bk]
        pieces.append(T[T.entry_time.dt.year == y]); picks.append((y, bk, best))
    return (pd.concat(pieces) if pieces else pd.DataFrame()), picks


t0 = time.time()
P("第二十二部分：BOCPD 變點順勢 + CUSUM 重做（XAUUSD，2023/1 ~ 2026/10，M15 重取樣）")
P(f"研究期 < {SPLIT.date()} ≤ 驗證期；成本 = 點差 + 融資 0.02%/日曆天（多空都扣）")
P("訊號在 K 棒收盤產生，下一根開盤進場；停損 = k × ATR14（該週期）；R = 停損距離")
P("=" * 100)

grid_rows = []
all_B, all_C, BAR = {}, {}, {}
for tf in TFS:
    b = bars(tf); BAR[tf] = b
    z = zscore(b, tf).to_numpy()
    yrs = (b.index[-1] - b.index[0]).days / 365.25
    P(f"\n## {tf}：{len(b)} 根，z 標準差 {z.std():.2f}")

    # ---------- 0) 已知變異數版（只當對照）：只會被單根大 K 棒觸發
    if tf in ("D1", "H4"):
        for lam in (25, 50, 100):
            cp, conf, _ = bocpd(z, lam, 0.3, w=5)
            sig, ep = bocpd_signals(cp, conf, 0.5, 0.8)
            ii = np.where(sig != 0)[0]
            P(f"  [對照] 已知變異數 BOCPD λ={lam} τ=0.3：cp≥0.5 的 K 棒 {np.mean(cp >= .5):.2%}，訊號 {len(ii)} 個，"
              f"訊號 K 棒 |z| 中位 {np.median(np.abs(z[ii])) if len(ii) else float('nan'):.2f}")

    # ---------- 1) BOCPD（Normal-Gamma）網格
    for lam, w in itertools.product((25, 50, 100), (5, 10)):
        cp, conf, mapr = bocpd_ng(z, lam, w=w)
        for theta, c in itertools.product((0.3, 0.5), (0.7, 0.8)):
            sig, ep = bocpd_signals(cp, conf, theta, c)
            for stop, ex in itertools.product((2.0, 3.0), ("flip", "cp")):
                T = backtest(b, sig, stop, ex, ep_start=ep)
                key = ("B", tf, lam, w, theta, c, stop, ex)
                all_B[key] = T
                a, v = split_stats(T)
                grid_rows.append(dict(method="BOCPD", tf=tf, lam=lam, w=w, theta=theta, c=c, stop=stop, exit=ex,
                                      n=len(T), per_year=len(T) / yrs, hold_bars=T.bars.median() if len(T) else np.nan,
                                      hold_days=T.days.mean() if len(T) else np.nan,
                                      mean_all=T.pnl_R.mean() if len(T) else np.nan,
                                      gross_all=T.gross_R.mean() if len(T) else np.nan,
                                      n_is=a["n"], mean_is=a["mean"], n_oos=v["n"], mean_oos=v["mean"], t_oos=v["t"],
                                      adj_all=detrend(b, T).mean() if len(T) else np.nan,
                                      adj_oos=detrend(b, T)[T.entry_time >= SPLIT].mean() if len(T) else np.nan,
                                      long_mean=T[T.side > 0].pnl_R.mean() if len(T) else np.nan,
                                      short_mean=T[T.side < 0].pnl_R.mean() if len(T) else np.nan))
                if (lam, w, theta, c, stop, ex) == tuple(REP_B.values()):
                    BAR[tf + "_repB"] = (sig, ep, cp, conf, mapr)

    # ---------- 2) CUSUM 網格（停損 + 反手）
    for k, h, stop in itertools.product((0.0, 0.25, 0.5), (2, 4, 6, 8), (2.0, 3.0)):
        sig = cusum_signals(z, k, h)
        T = backtest(b, sig, stop, "flip")
        key = ("C", tf, k, h, stop)
        all_C[key] = T
        a, v = split_stats(T)
        grid_rows.append(dict(method="CUSUM", tf=tf, k=k, h=h, stop=stop, exit="flip",
                              n=len(T), per_year=len(T) / yrs, hold_bars=T.bars.median() if len(T) else np.nan,
                              hold_days=T.days.mean() if len(T) else np.nan,
                              mean_all=T.pnl_R.mean() if len(T) else np.nan,
                              gross_all=T.gross_R.mean() if len(T) else np.nan,
                              n_is=a["n"], mean_is=a["mean"], n_oos=v["n"], mean_oos=v["mean"], t_oos=v["t"],
                              adj_all=detrend(b, T).mean() if len(T) else np.nan,
                              adj_oos=detrend(b, T)[T.entry_time >= SPLIT].mean() if len(T) else np.nan,
                              long_mean=T[T.side > 0].pnl_R.mean() if len(T) else np.nan,
                              short_mean=T[T.side < 0].pnl_R.mean() if len(T) else np.nan))
        if (k, h, stop) == tuple(REP_C.values()):
            BAR[tf + "_repC"] = sig
    P(f"  （{time.time() - t0:.0f}s）")

G = pd.DataFrame(grid_rows)
G.to_csv("part22_grid.csv", index=False)

# ======================================================================== 網格總覽
P("\n" + "=" * 100)
P("A. 網格總覽（每組參數的平均 R/筆；扣成本）")
P(f"{'方法':6s} {'週期':4s} {'組數':>4s} {'每年筆數':>8s} {'持有(根)':>8s} {'持有(天)':>8s} {'全期':>7s} {'未扣成本':>8s} "
  f"{'研究期':>7s} {'驗證期':>7s} {'驗證>0':>6s} {'多單':>7s} {'空單':>7s} {'去漂移':>7s} {'去漂移>0':>8s}")
for (m, tf), g in G.groupby(["method", "tf"], sort=False):
    g = g[g.n >= 10]
    P(f"{m:6s} {tf:4s} {len(g):4d} {g.per_year.median():8.1f} {g.hold_bars.median():8.1f} {g.hold_days.median():8.2f} "
      f"{g.mean_all.mean():+7.3f} {g.gross_all.mean():+8.3f} {g.mean_is.mean():+7.3f} {g.mean_oos.mean():+7.3f} "
      f"{(g.mean_oos > 0).mean():6.0%} {g.long_mean.mean():+7.3f} {g.short_mean.mean():+7.3f} {g.adj_all.mean():+7.3f} {(g.adj_all > 0).mean():8.0%}")

P("\nB. 研究期前 5 名（研究期至少 15 筆）在驗證期的表現")
for (m, tf), g in G.groupby(["method", "tf"], sort=False):
    g = g[g.n_is >= 15].sort_values("mean_is", ascending=False).head(5)
    P(f"  {m} {tf}: " + " | ".join(
        f"研究 {r.mean_is:+.2f}({r.n_is})→驗證 {r.mean_oos:+.2f}({r.n_oos})" for r in g.itertuples()))

P("\nC. 逐年前推（每年用過去所有年份挑平均 R 最高的一組；2024、2025、2026 串起來）")
for m, D in (("BOCPD", all_B), ("CUSUM", all_C)):
    for tf in TFS:
        keys = [k for k in D if k[1] == tf]
        W, picks = walk_forward(D, keys)
        if len(W):
            P(f"  {m} {tf}: {fmt(stats(W))}  多 {W[W.side > 0].pnl_R.mean():+.2f} / 空 {W[W.side < 0].pnl_R.mean():+.2f}"
              f" / 去漂移 {detrend(BAR[tf], W).mean():+.3f}R")
            P("      挑選：" + "；".join(f"{y}←{k[2:]}" for y, k, _ in picks))

# ======================================================================== 代表參數的深入檢查
P("\n" + "=" * 100)
P(f"D. 代表參數（事先指定，網格中間，不是最佳）：BOCPD {REP_B}；CUSUM {REP_C}")
LL_rows = []
for tf in TFS:
    b = BAR[tf]
    for lab, sig, T in (("BOCPD", BAR[tf + "_repB"][0], all_B[("B", tf) + tuple(REP_B.values())]),
                        ("CUSUM", BAR[tf + "_repC"], all_C[("C", tf) + tuple(REP_C.values())]),
                        ("TSMOM20 翻轉", flip_events(b), backtest(b, flip_events(b), REP_B["stop"], "flip"))):
        P(f"\n  ### {tf} {lab}：訊號 {int((sig != 0).sum())} 個（多 {int((sig > 0).sum())} / 空 {int((sig < 0).sum())}）")
        a, v = split_stats(T)
        P(f"    交易 全期 {fmt(stats(T))}")
        P(f"         研究 {fmt(a)}")
        P(f"         驗證 {fmt(v)}")
        if len(T) > 1:
            dt = detrend(b, T)
            yy = T.groupby(T.entry_time.dt.year).pnl_R
            P(f"    去漂移後每筆 {dt.mean():+.3f}R（t {dt.mean() / dt.std() * math.sqrt(len(dt)):+.2f}）；逐年 "
              + "  ".join(f"{y}: {v.mean():+.2f}({len(v)})" for y, v in yy) + f"；多 {T[T.side > 0].pnl_R.mean():+.2f} 空 {T[T.side < 0].pnl_R.mean():+.2f}")
        if len(T) > 5:
            real, rnd, pct, lo = random_dir(T, b)
            P(f"    方向檢定：策略 {real:+.3f}R，同時間隨機方向平均 {rnd:+.3f}R（勝過 {pct:.0%} 的隨機），同時間全做多 {lo:+.3f}R；"
              f"平均持有 {T.bars.mean():.1f} 根 / {T.days.mean():.1f} 天")
        E = event_study(b, sig, HOR[tf])
        P("    P1 事件研究（訊號後 h 根，報酬×方向÷ATR；adj = 扣掉整段平均漂移；nov = 不重疊取樣）")
        for r in E.itertuples():
            P(f"      h={r.h:3d}: n={r.n:4d} 原始 {r.raw:+.3f} adj {r.adj:+.3f} (t {r.t_adj:+.2f}) 命中 {r.hit:.2f} | "
              f"不重疊 n={r.n_nov} adj {r.adj_nov:+.3f} (t {r.t_nov:+.2f}) | 多 {r.long_adj:+.3f} 空 {r.short_adj:+.3f}")
        if lab != "TSMOM20 翻轉":
            LLd = lead_lag(b, sig)
            vc = LLd.kind.value_counts(normalize=True)
            P(f"    P2 相對 TSMOM20：訊號較早 {vc.get('earlier', 0):.0%}（領先中位 {LLd[LLd.kind == 'earlier'].bars.median():.0f} 根）"
              f"、較晚 {vc.get('later', 0):.0%}（落後中位 {LLd[LLd.kind == 'later'].bars.median():.0f} 根）"
              f"、TSMOM 60 根內從未確認 {vc.get('never', 0):.0%}")
            # 早的那些訊號，事後到底對不對？
            ii = np.where(sig != 0)[0]
            fwd_h = HOR[tf][2]
            c_, atr_ = b.close.to_numpy(), b.atr.to_numpy()
            ok = ii + fwd_h < len(c_)
            fr = (c_[ii[ok] + fwd_h] - c_[ii[ok]]) / atr_[ii[ok]] * sig[ii[ok]]
            kinds = LLd.kind.to_numpy()[ok]
            P("       各類訊號之後 " + str(fwd_h) + " 根報酬×方向÷ATR："
              + "  ".join(f"{k} {fr[kinds == k].mean():+.3f}（{(kinds == k).sum()}）" for k in ("earlier", "later", "never")))
            LL_rows.append(dict(tf=tf, method=lab, **{k: vc.get(k, 0) for k in ("earlier", "later", "never")}))

    # P3：是不是吵的均線？部位相關 + 逐根報酬迴歸（開盤→下一根開盤，訊號都在前一根收盤決定）
    rB = all_B[("B", tf) + tuple(REP_B.values())]
    yB, pB = bar_pnl(b, rB)
    o = b.open.to_numpy(float); atr_prev = b.atr.shift().to_numpy()
    rr = np.r_[(o[1:] - o[:-1]), 0.0] / atr_prev
    rr = np.nan_to_num(rr)
    pT = np.sign(b.close - b.close.shift(20)).shift().fillna(0).to_numpy()
    ema_f, ema_s = b.close.ewm(span=10).mean(), b.close.ewm(span=40).mean()
    pE = np.sign(ema_f - ema_s).shift().fillna(0).to_numpy()
    yT, yE, yL = pT * rr, pE * rr, rr
    m = pB != 0
    tt = lambda y: y.mean() / y.std() * math.sqrt(len(y))
    bb_, tt_, r2 = ols(yB, [yT, yE, yL])
    P(f"\n    P3 {tf}：BOCPD 部位與 TSMOM20 部位相關 {np.corrcoef(pB, pT)[0, 1]:+.2f}、與 EMA10/40 {np.corrcoef(pB, pE)[0, 1]:+.2f}；"
      f"持倉時方向與 TSMOM20 相同的比例 {(np.sign(pB[m]) == np.sign(pT[m])).mean():.0%}；持倉時間比例 {m.mean():.0%}、其中做多 {(pB[m] > 0).mean():.0%}")
    P(f"       BOCPD 逐根報酬 = {bb_[0]:+.4f}（t {tt_[0]:+.2f}） + {bb_[1]:+.2f}×TSMOM20（t {tt_[1]:+.2f}） + {bb_[2]:+.2f}×EMA（t {tt_[2]:+.2f}）"
      f" + {bb_[3]:+.2f}×買進持有（t {tt_[3]:+.2f}），R² {r2:.2f}")
    P(f"       逐根平均（÷ATR，未扣成本的指標 vs 扣成本的 BOCPD）：TSMOM20 {yT.mean():+.4f}（t {tt(yT):+.2f}）、EMA10/40 {yE.mean():+.4f}（t {tt(yE):+.2f}）、"
      f"買進持有 {yL.mean():+.4f}（t {tt(yL):+.2f}）、BOCPD {yB.mean():+.4f}（t {tt(yB):+.2f}）")
    rB.to_csv(f"part22_trades_bocpd_{tf}.csv", index=False)

# ======================================================================== 與日內策略的相關
P("\n" + "=" * 100)
P("E. 與日內策略的每日損益相關（代表參數，逐日盯市，R 為單位）")
try:
    L = pd.read_csv("oos_lonbrk_trades.csv", parse_dates=["entry_time"])
    lon = L.groupby(L.entry_time.dt.normalize()).pnl_R.sum()
except Exception:
    lon = None
try:
    A = pd.read_csv("final_A_trades.csv", parse_dates=["交易日"])
    fa = A.groupby("交易日")["損益_R"].sum()
except Exception:
    fa = None
for tf in TFS:
    for lab, T in (("BOCPD", all_B[("B", tf) + tuple(REP_B.values())]), ("CUSUM", all_C[("C", tf) + tuple(REP_C.values())])):
        dp = daily_pnl_R(BAR[tf], T)
        parts = []
        for nm, other in (("LONBRK 2023~26", lon), ("版本A 2026", fa)):
            if other is None:
                continue
            j = pd.concat([dp, other], axis=1, join="inner").fillna(0)
            j = j[(j.iloc[:, 1] != 0)]
            if len(j) > 20:
                parts.append(f"{nm} 相關 {np.corrcoef(j.iloc[:, 0], j.iloc[:, 1])[0, 1]:+.2f}（{len(j)} 天）")
        P(f"  {tf} {lab}: " + "；".join(parts))

P(f"\n總耗時 {time.time() - t0:.0f}s")
