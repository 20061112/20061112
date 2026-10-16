"""第二十三部分 (2)(3)：CUSUM 波段的細部參數研究 + 「不去漂移」的原始結果。
核心 = H1、k=0.25、h=6、停損 2×ATR14(H1)、反手（第二十二部分選出）。所有結果同時列「原始」與「去漂移」。
輸出：results_part23_fine.txt、part23_grid_kh.csv
"""
import itertools, math, warnings
warnings.simplefilter("ignore", RuntimeWarning)
import numpy as np, pandas as pd
from cusum_lab import bars, zscore, backtest, summ, line, detrend_R
from changepoint import cusum_signals

out = open("results_part23_fine.txt", "w")
def P(s=""):
    print(s); out.write(s + "\n"); out.flush()

hb = bars("H1"); z = zscore(hb)
KS = np.round(np.arange(0.0, 0.81, 0.1), 2); HS = np.arange(3.0, 10.01, 0.5)

# ---------------------------------------------------------------- 0. 什麼是「去漂移」
P("0) 原始 vs 去漂移")
yrs = (hb.index[-1] - hb.index[0]).days / 365.25
P(f"  期間黃金 {hb.close.iloc[0]:.0f} → {hb.close.iloc[-1]:.0f}（年化 {((hb.close.iloc[-1] / hb.close.iloc[0]) ** (1 / yrs) - 1):.0%}），"
  f"平均每天漲 {(hb.close.iloc[-1] - hb.close.iloc[0]) / ((hb.index[-1] - hb.index[0]).days):.2f} 美元")
P("  原始 = 實際會賺到的 R（扣點差、融資）。去漂移 = 原始 − 方向 × 平均每小時漲幅 × 持有小時 ÷ R，")
P("  也就是把「只要拿著多單就會賺的那部分」拿掉，看方向判斷本身有沒有價值。實際下單拿到的是「原始」。")

# ---------------------------------------------------------------- A. k × h 細網格
rows = []
for k, h in itertools.product(KS, HS):
    T = backtest(hb, cusum_signals(z, k, h), 2.0)
    d = summ(T, hb)
    rows.append(dict(k=k, h=h, n=d["n"], mean=d.get("mean"), t=d.get("t"), adj=d.get("adj"), t_adj=d.get("t_adj"),
                     minyr=d.get("minyr"), minyr_adj=d.get("minyr_adj"), long=d.get("long"), short=d.get("short"),
                     hold=d.get("hold"), dd=d.get("dd"), tot=d.get("tot")))
G = pd.DataFrame(rows); G.to_csv("part23_grid_kh.csv", index=False)

def mat(col, f="{:+.2f}"):
    M = G.pivot(index="k", columns="h", values=col)
    P("      h→ " + " ".join(f"{h:>5.1f}" for h in M.columns))
    for k, r in M.iterrows():
        P(f"  k={k:.1f} " + " ".join(f.format(v) if pd.notna(v) else "   . " for v in r.values))

P("\nA) k × h 細網格（H1，停損 2 ATR，反手；2023/1~2026/10）")
P("  [每筆平均 R — 原始]"); mat("mean")
P("  [每筆平均 R — 去漂移]"); mat("adj")
P("  [t 值 — 原始]"); mat("t", "{:+5.1f}")
P("  [最差年份的平均 R — 原始]"); mat("minyr")
P("  [交易筆數]"); mat("n", "{:5.0f}")
P("  [平均持有天數]"); mat("hold", "{:5.1f}")
ok = G[G.n >= 30]
P(f"\n  筆數 ≥ 30 的 {len(ok)} 組：原始 > 0 {(ok['mean'] > 0).mean():.0%}、去漂移 > 0 {(ok.adj > 0).mean():.0%}、"
  f"原始 t > 2 {(ok.t > 2).mean():.0%}、每年都 > 0 {(ok.minyr > 0).mean():.0%}")
for lo, hi in ((3, 4.5), (5, 7), (7.5, 10)):
    g = ok[(ok.h >= lo) & (ok.h <= hi)]
    P(f"  h {lo}~{hi}：平均 原始 {g['mean'].mean():+.3f}R / 去漂移 {g.adj.mean():+.3f}R，每年都 > 0 的比例 {(g.minyr > 0).mean():.0%}，"
      f"每年筆數 {g.n.mean() / yrs:.0f}、持有 {g.hold.mean():.1f} 天")

# 平台：3×3 鄰域平均
M = G.pivot(index="k", columns="h", values="mean").to_numpy(); Ma = G.pivot(index="k", columns="h", values="adj").to_numpy()
def smooth(M):
    S = np.full_like(M, np.nan)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            S[i, j] = np.nanmean(M[max(0, i - 1):i + 2, max(0, j - 1):j + 2])
    return S
S = smooth(M); Sa = smooth(Ma)
i, j = np.unravel_index(np.nanargmax(S), S.shape)
P(f"  3×3 鄰域平均最高的中心：k={KS[i]:.1f} h={HS[j]:.1f}（鄰域原始 {S[i, j]:+.3f}R、去漂移 {Sa[i, j]:+.3f}R）")
i2, j2 = np.unravel_index(np.nanargmax(Sa), Sa.shape)
P(f"  去漂移鄰域最高的中心：k={KS[i2]:.1f} h={HS[j2]:.1f}（鄰域原始 {S[i2, j2]:+.3f}R、去漂移 {Sa[i2, j2]:+.3f}R）")

# ---------------------------------------------------------------- B. 停損
CORE = dict(k=0.25, h=6.0)
sigc = cusum_signals(z, **CORE)
NB = [(k, h) for k in (0.2, 0.3) for h in (5.5, 6.0, 6.5)]
P("\nB) 停損倍數（核心；括號內為鄰近 6 組 k∈{0.2,0.3}×h∈{5.5,6,6.5} 的平均）")
for st in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, None):
    d = summ(backtest(hb, sigc, st), hb)
    nb = [summ(backtest(hb, cusum_signals(z, k, h), st), hb) for k, h in NB]
    lab = f"{st:.1f} ATR" if st else "不設停損（R=2ATR 計）"
    P(f"  {lab:>12s}: {line(d)}  （鄰近 原始 {np.mean([x['mean'] for x in nb]):+.3f} / 去漂移 {np.mean([x['adj'] for x in nb]):+.3f}）")

# ---------------------------------------------------------------- C. 出場方式
P("\nC) 出場方式（核心，停損 2 ATR）")
for lab, kw in (("基準：反手 + 停損", {}), ("最長 24 根", dict(max_bars=24)), ("最長 48 根", dict(max_bars=48)),
                ("最長 120 根", dict(max_bars=120)), ("最長 240 根", dict(max_bars=240)),
                ("移動停損 3 ATR", dict(trail_atr=3)), ("移動停損 4 ATR", dict(trail_atr=4)), ("移動停損 6 ATR", dict(trail_atr=6)),
                ("浮盈 1R 移保本", dict(be_R=1)), ("浮盈 2R 移保本", dict(be_R=2))):
    P(f"  {lab:14s}: {line(summ(backtest(hb, sigc, 2.0, **kw), hb))}")

# ---------------------------------------------------------------- D. 方向（不去漂移時最重要）
P("\nD) 只做多 / 只做空 / 雙向（反向訊號一律平倉；只限制開新倉方向）")
for sd in ("both", "long", "short"):
    T = backtest(hb, sigc, 2.0, side=sd)
    P(f"  {sd:5s}: {line(summ(T, hb))}")
    P("         逐年 原始 " + "  ".join(f"{y}:{v:+.2f}" for y, v in summ(T, hb)["years"].items()))
# 對照：同一段時間單純持有多單（以 2 ATR 為 R 換算的期望）
P("  對照：同樣的持有時間、全部做多 → 見第二十二部分「同時間全做多」（核心 +0.39R），雙向比它多 +0.17R/筆")

# ---------------------------------------------------------------- E. 週期
P("\nE) 週期（k∈{0.15,0.25,0.35,0.5} × h∈{4,5,6,7,8}，停損 2 ATR(該週期)）")
for tf in ("M30", "H1", "H2", "H3", "H4"):
    b = hb if tf == "H1" else bars(tf); zz = zscore(b)
    R = []
    for k, h in itertools.product((0.15, 0.25, 0.35, 0.5), (4, 5, 6, 7, 8)):
        d = summ(backtest(b, cusum_signals(zz, k, h), 2.0), b)
        if d["n"] >= 20:
            R.append(dict(k=k, h=h, **{x: d[x] for x in ("n", "mean", "adj", "minyr", "hold")}))
    R = pd.DataFrame(R)
    c = summ(backtest(b, cusum_signals(zz, 0.25, 6), 2.0), b)
    P(f"  {tf:4s}: {len(R)} 組 原始 {R['mean'].mean():+.3f}（>0 {(R['mean'] > 0).mean():.0%}）去漂移 {R.adj.mean():+.3f}（>0 {(R.adj > 0).mean():.0%}）"
      f" 每年都>0 {(R.minyr > 0).mean():.0%}｜k0.25 h6: {line(c)}")

# ---------------------------------------------------------------- F. 波動標準化
P("\nF) z 的計算方式（核心 k、h）")
for lab, kw in (("基準 時段季節性 span20 clip4", {}), ("span 10 日", dict(span=10)), ("span 40 日", dict(span=40)),
                ("clip 3", dict(clip=3)), ("clip 6", dict(clip=6)), ("不去季節性（一般 EWMA）", dict(seasonal=False))):
    P(f"  {lab:24s}: {line(summ(backtest(hb, cusum_signals(zscore(hb, **kw), **CORE), 2.0), hb))}")

# ---------------------------------------------------------------- G. 進場延遲
P("\nG) 進場延遲（訊號後晚 N 根才進場；邊際若靠第一根就消失 = 脆弱）")
for dl in (0, 1, 2, 3, 6):
    P(f"  延遲 {dl} 根: {line(summ(backtest(hb, sigc, 2.0, delay=dl), hb))}")

# ---------------------------------------------------------------- H. 挑參數的偏誤：逐年前推
P("\nH) 逐年前推（每年只用過去年份，在 A 的 135 組裡挑；2024~2026 串接）")
cache = {(k, h): backtest(hb, cusum_signals(z, k, h), 2.0) for k, h in itertools.product(KS, HS)}
def wf(score):
    parts = []; picks = []
    for y in (2024, 2025, 2026):
        sc = {}
        for kh, T in cache.items():
            tr = T[T.entry_time.dt.year < y]
            if score == "t":
                sc[kh] = tr.pnl_R.mean() / tr.pnl_R.std() * math.sqrt(len(tr)) if len(tr) >= 40 else np.nan
            else:
                sc[kh] = tr.pnl_R.mean() if len(tr) >= 20 else np.nan
        Sg = pd.Series(sc).unstack()
        if score == "plateau":
            Sg = pd.DataFrame(smooth(Sg.to_numpy()), index=Sg.index, columns=Sg.columns)
        kh = Sg.stack().idxmax(); picks.append(f"{y}←k{kh[0]:.1f}/h{kh[1]:.1f}")
        T = cache[kh]; parts.append(T[T.entry_time.dt.year == y])
    W = pd.concat(parts)
    return W, picks
for sc, lab in (("best", "單點平均R最佳（≥20筆）"), ("plateau", "3×3 平台最佳"), ("t", "t 值最佳（≥40筆）")):
    W, picks = wf(sc)
    P(f"  {lab}: {line(summ(W, hb))}")
    P("      " + "；".join(picks))
F = backtest(hb, sigc, 2.0)
F = F[F.entry_time.dt.year >= 2024]
P(f"  對照 核心 k0.25/h6（第二十二部分選的，2024~2026；注意它是看過全期才選的）: {line(summ(F, hb))}")
