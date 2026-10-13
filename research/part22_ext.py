"""第二十二部分延伸：
E1. 變大/變小主要由「時段」決定（M15、H1 依小時）
E2. 大小可預測、方向不可預測：前一根全長 / ATR → 下一根全長變大機率、下一根全長 / ATR
E3. 大實體 K 棒之後 1~10 根的順勢報酬（扣點差前後），逐年
E4. 由「小 K 之後通常變大」延伸：突破前一根高 / 低點的掛單（OCO），下一根收盤出場，扣點差
E5. 第二十二部分 B/C 中「兩段同向」的少數格子，逐年檢查
"""
import numpy as np
import pandas as pd
from load import load
from part22_bar_pairs import bars, pair_frame, pct, tstat, SPLIT

m1 = load("data/XAUUSD_M1_2023_2026.csv")
TF = {k: bars(m1, r, m) for k, r, m in [("M1", "1min", 1), ("M5", "5min", 5), ("M15", "15min", 15), ("H1", "1h", 60), ("H4", "4h", 240), ("D1", "1D", 1440)]}
F = {k: pair_frame(v) for k, v in TF.items()}


def yearly(s, years):
    s, years = np.asarray(s, float), np.asarray(years)
    return "  ".join(f"{y}:{np.nanmean(s[years == y]):+.3f}" for y in sorted(set(years)))


print("# E1. 這根全長比前一根大的機率，依小時（伺服器時間；只列明顯偏離 50% 的小時）")
for name in ("M15", "H1"):
    df = TF[name]
    ok = df.ok & (df.rng != df.rng.shift())
    big = (df.rng > df.rng.shift())[ok]
    by = big.groupby(big.index.hour).mean()
    a = big[big.index < SPLIT].groupby(big[big.index < SPLIT].index.hour).mean()
    b = big[big.index >= SPLIT].groupby(big[big.index >= SPLIT].index.hour).mean()
    print(f"\n## {name}  （整體 {pct(big.mean())}）")
    print("  小時 :  " + " ".join(f"{h:>5d}" for h in by.index))
    print("  全部 :  " + " ".join(f"{100 * v:5.0f}" for v in by))
    print("  發現 :  " + " ".join(f"{100 * v:5.0f}" for v in a))
    print("  驗證 :  " + " ".join(f"{100 * v:5.0f}" for v in b))
    if name == "M15":
        # 15:30 這根（紐約開盤）
        mm = big.index.hour * 100 + big.index.minute
        for t in (1000, 1530, 1545, 1600, 1700):
            print(f"  {t:04d} 這根比前一根大：{pct(big[mm == t].mean())}  ")
    # 排除時段效果後：以「同一個小時的過去 20 天平均全長」標準化
    rel = df.rng / df.rng.groupby(df.index.time).transform(lambda s: s.shift().rolling(20, min_periods=10).mean())
    ok2 = ok & rel.notna() & rel.shift().notna()
    bigr = (rel > rel.shift())[ok2]
    print(f"  扣掉時段效果（全長 ÷ 同時段過去 20 天平均）後，變大機率 {pct(bigr.mean())}")

print("\n\n# E2. 大小可預測：這根全長 / ATR20 → 下一根（方向不可預測，見 B/C）")
for name in ("M5", "M15", "H1", "H4", "D1"):
    f, df = F[name], TF[name]
    nr = (df.rng.shift(-1) / df.atr).reindex(f.index)
    q = pd.cut(f.rng_atr, [0, .5, .75, 1, 1.5, 2, np.inf])
    g = pd.DataFrame({"q": q, "grow": f.nxt_rng_grow > 0, "nr": nr, "cont": f.nxt_cont}).dropna().groupby("q", observed=True)
    print(f"\n## {name}")
    for k, v in g:
        print(f"  全長 {str(k):>12s}: n={len(v):6d}  下一根全長變大 {pct(v.grow.mean())}  下一根全長中位 {v.nr.median():.2f} ATR  順勢率 {pct(v.cont.mean())}")
    # 相關：|這根報酬| 與 |下一根報酬| vs 有號報酬
    r = (df.close - df.open) / df.atr.shift()
    pair = pd.DataFrame({"r": r, "rn": r.shift(-1), "ok": df.ok.shift(-1)}).dropna()
    pair = pair[pair.ok.astype(bool)]
    print(f"  相關係數：大小 corr(|r|, |r_next|) = {pair.r.abs().corr(pair.rn.abs()):+.3f}   方向 corr(r, r_next) = {pair.r.corr(pair.rn):+.3f}")

print("\n\n# E3. 大實體 K 棒（實體 ≥ k ATR）之後 h 根的順勢報酬（單位 ATR；負 = 回吐），以及扣點差後做反向的淨值")
for name, k in (("M1", 2), ("M5", 2), ("M15", 2), ("H1", 1.5)):
    df, f = TF[name], F[name]
    sel = f[(f.body_atr >= k) & f.ok_prev].index
    c = df.close
    print(f"\n## {name}  實體 ≥ {k} ATR：{len(sel)} 根")
    for h in (1, 3, 5, 10, 20):
        fut = (c.shift(-h) - c).reindex(sel) * df.dir.reindex(sel) / df.atr.reindex(sel)
        fade_net = -fut - df.cost_atr.reindex(sel) if "cost_atr" in df else -fut - (df.cost / df.atr).reindex(sel)
        yrs = sel.year
        print(f"  {h:2d} 根: 順勢 {fut.mean():+.3f} (t{tstat(fut):+5.1f}) | 反向扣點差 {fade_net.mean():+.3f} | 逐年反向淨值 {yearly(fade_net.to_numpy(), yrs.to_numpy())}")

print("\n\n# E4. 前一根高 / 低點突破掛單（OCO：先碰到哪邊做哪邊），下一根收盤出場，扣點差；單位 = 前一根全長")
print("  同一根同時碰到兩邊的，以 M1 判斷先後；判斷不了（同一分鐘）算兩邊都停在對面，保守處理成 −1 全長")
m1h, m1l = m1.high, m1.low
for name, rule in (("M5", "5min"), ("M15", "15min"), ("H1", "1h")):
    df = TF[name]
    ph, pl, prng = df.high.shift(), df.low.shift(), df.rng.shift()
    upb, dnb = df.high > ph, df.low < pl
    ok = df.ok & (prng > 0) & df.atr.shift().notna()
    # 先後：每根內第一次越過前高 / 前低的分鐘
    key = m1.index.floor(rule)
    ph_m = ph.reindex(key).to_numpy()
    pl_m = pl.reindex(key).to_numpy()
    t_up = pd.Series(np.where(m1h.to_numpy() > ph_m, np.arange(len(m1)), np.inf), index=key).groupby(level=0).min()
    t_dn = pd.Series(np.where(m1l.to_numpy() < pl_m, np.arange(len(m1)), np.inf), index=key).groupby(level=0).min()
    t_up, t_dn = t_up.reindex(df.index), t_dn.reindex(df.index)
    side = np.where(t_up < t_dn, 1, np.where(t_dn < t_up, -1, np.where(np.isfinite(t_up), 9, 0)))
    entry = np.where(side == 1, ph, pl)
    pnl = np.where(np.isin(side, (1, -1)), (df.close - entry) * side, np.where(side == 9, -prng, np.nan)) - np.where(side != 0, df.cost, 0)
    R = pd.Series(pnl / prng, index=df.index)[ok & (side != 0)]
    rel = (prng / df.atr.shift())[R.index]
    yrs = R.index.year
    print(f"\n## {name}：觸發 {len(R)} / {ok.sum()} 根（{pct(len(R) / ok.sum())}），兩邊同分鐘 {pct((side[ok.to_numpy()] == 9).mean())}")
    print(f"  全部        : 平均 {R.mean():+.3f} (t{tstat(R):+5.1f})  勝率 {pct((R > 0).mean())}  逐年 {yearly(R.to_numpy(), yrs.to_numpy())}")
    for lo, hi in ((0, .5), (.5, 1), (1, 9)):
        m = (rel >= lo) & (rel < hi)
        print(f"  前根全長 {lo}-{hi} ATR: n={m.sum():6d} 平均 {R[m].mean():+.3f} (t{tstat(R[m]):+5.1f})  逐年 {yearly(R[m].to_numpy(), yrs[m.to_numpy()].to_numpy())}")

print("\n\n# E5. B/C 中兩段同向的格子，逐年（下一根順勢報酬，ATR）")
cells = [
    ("D1 實體變大+順 → 下一根順勢", "D1", lambda f: (f.body_grow > 0) & f.same_prev),
    ("H1 連續實體放大 3 根 → 下一根順勢", "H1", lambda f: f.expand_run == 3),
    ("M1 實體 ≥ 2 ATR → 下一根回吐", "M1", lambda f: f.body_atr >= 2),
    ("M1 連續同色 ≥ 3 根 → 下一根回吐", "M1", lambda f: f.run >= 3),
    ("M15 連續同色 ≥ 8 根 → 下一根回吐", "M15", lambda f: f.run >= 8),
]
for lab, tf, cond in cells:
    f = F[tf]
    s = f[f.ok_prev & cond(f)]
    net = s.nxt_ret - s.cost_atr
    print(f"  {lab:32s}: n={len(s):6d} 順勢 {s.nxt_ret.mean():+.3f} | 逐年 {yearly(s.nxt_ret.to_numpy(), s.year.to_numpy())} | 點差/ATR {s.cost_atr.mean():.3f}")

print("\n\n# E6. D1「實體變大 + 同向」再拆多空（黃金 2023~2026 大多頭，檢查是不是只是做多 beta）")
f = F["D1"]
base = f[f.ok_prev & f.nxt_ret.notna()]
sig = base[(base.body_grow > 0) & base.same_prev]
for lab, d in (("全部日子（基準）", base), ("訊號", sig)):
    for side, nm in ((1, "陽線→做多"), (-1, "陰線→做空")):
        s = d[d.dir == side]
        print(f"  {lab:10s} {nm}: n={len(s):4d} 順勢率 {pct(s.nxt_cont.mean())} 順勢 {s.nxt_ret.mean():+.3f} ATR (t{tstat(s.nxt_ret):+4.1f}) 逐年 {yearly(s.nxt_ret, s.year)}")
# 只看「訊號 − 同方向基準」的差，排除多頭漂移
for side in (1, -1):
    a, b = sig[sig.dir == side].nxt_ret, base[base.dir == side].nxt_ret
    print(f"  {'多' if side > 0 else '空'}方 訊號比同方向基準多 {a.mean() - b.mean():+.3f} ATR")
# 鄰近定義：實體放大倍數門檻
for k in (1.0, 1.25, 1.5, 2.0):
    s = base[(base.body_atr.notna()) & base.same_prev & (TF["D1"].body / TF["D1"].body.shift()).reindex(base.index).ge(k)]
    print(f"  實體 ≥ 前一根 {k:.2f} 倍且同向: n={len(s):4d} 順勢率 {pct(s.nxt_cont.mean())} 順勢 {s.nxt_ret.mean():+.3f} 逐年 {yearly(s.nxt_ret, s.year)}")
