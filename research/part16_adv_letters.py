"""第十六部分：進階 TPO 字母指標（進場當下、當天發展中剖面，字母 15 分；全部按交易方向調整）。

預先登記的挑選規則（避免看 6~10 月挑指標）：
  每個指標用 1~5 月切五分位，找 1~5 月每筆平均最差的那一組，規則 = 刪掉那一組；
  6~10 月只拿來看這個「事先決定的刪除」有沒有效。之後再看能不能疊加在 tail 篩選之上。

指標：
  ext_age    反方向當天極值形成到現在的時間（小時）：做多看當天低點是多久前做的
  ext_letter 反方向極值是第幾個字母做的 / 目前字母數（0 = 開盤第一個字母）
  n_ext      突破方向做出「當天新極值」的字母數 − 反方向的字母數（區間延伸次數差）
  skew       TPO 分佈偏態 × 方向（負 = 做多時是 P 型，量集中在上方、尾巴向下）
  bimodal    是否雙峰分佈（兩個高峰之間有薄區）；bim_above：進場價在上方那個分佈之外
  va_mig     發展中 VA 中點相對前日 VA 中點 / ATR（價值往交易方向移動）
  va_ovl     發展中 VA 與前日 VA 重疊比例（1 = 完全重疊 = 平衡）
  last_z     最後一個完成字母的波幅相對今天各字母波幅的 z（突破前是否擴張）
  comp_z     進場價相對「前 5 天合成剖面」的 z（(價 − μ) / σ，× 方向）
  init_sgl   dPOC 到進場價之間的單印列數 / ATR（往突破方向留下的 initiative 單印）
  day_rng    今天到目前的波幅 / ATR
"""
import numpy as np
import pandas as pd
from part14_deep import ARR, D, DAYS, SPLIT, exits, both
from part15_ib_params import signals
from part15_letters import features as basic_features
from tpo import build_days, poc_va

_, PROF = build_days(__import__("tpo").load_m1())
P_LET = 15


def composite(day, n=5):
    di = DAYS.index(day)
    prev = DAYS[max(0, di - n):di]
    if len(prev) < n:
        return None
    lo = min(PROF[k][0][0] for k in prev); hi = max(PROF[k][0][-1] for k in prev)
    cnt = np.zeros(int(hi - lo) + 1)
    for k in prev:
        pr, c = PROF[k][0], PROF[k][1]
        a = int(pr[0] - lo); cnt[a:a + len(c)] += c
    px = lo + np.arange(len(cnt)) + 0.5
    w = cnt / cnt.sum(); mu = (w * px).sum(); sd = np.sqrt((w * (px - mu) ** 2).sum())
    return mu, sd


COMP = {k: composite(k) for k in DAYS}


def adv(r):
    t, O, H, L, C, S = ARR[r.day]
    mins = (t - r.day).total_seconds().values // 60
    i, side, atr = r.i, r.side, r.atr
    if i < 30:
        return {}
    Hh, Ll = H[:i], L[:i]
    per = mins[:i] // P_LET
    ks = np.unique(per)
    ph = np.array([Hh[per == k].max() for k in ks]); pl = np.array([Ll[per == k].min() for k in ks])
    lo0 = np.floor(Ll.min()); n = int(np.floor(Hh.max()) - lo0) + 1
    diff = np.zeros(n + 1)
    for a_, b_ in zip(pl, ph):
        diff[int(np.floor(a_) - lo0)] += 1; diff[int(np.floor(b_) - lo0) + 1] -= 1
    cnt = np.cumsum(diff)[:n]; px = lo0 + np.arange(n) + 0.5
    w = cnt / cnt.sum(); mu = (w * px).sum(); sd = np.sqrt((w * (px - mu) ** 2).sum()) + 1e-9
    skew = (w * ((px - mu) / sd) ** 3).sum()
    # 反方向極值的時間與字母
    j_ext = np.argmin(Ll) if side == 1 else np.argmax(Hh)
    ext_age = (i - j_ext) / 60
    ext_letter = np.searchsorted(ks, per[j_ext]) / max(len(ks) - 1, 1)
    # 區間延伸字母數
    run_h = np.maximum.accumulate(ph); run_l = np.minimum.accumulate(pl)
    up_ext = int((ph[1:] > run_h[:-1]).sum()); dn_ext = int((pl[1:] < run_l[:-1]).sum())
    n_ext = side * (up_ext - dn_ext)
    # 雙峰：平滑後找兩個高峰，中間谷底 < 50% 的較小峰
    sm = np.convolve(cnt, np.ones(3) / 3, mode="same")
    peaks = [k for k in range(1, n - 1) if sm[k] >= sm[k - 1] and sm[k] > sm[k + 1] and sm[k] >= 0.5 * sm.max()]
    bim, bim_above = 0, 0
    if len(peaks) >= 2:
        a_, b_ = peaks[0], peaks[-1]
        if sm[a_:b_ + 1].min() < 0.5 * min(sm[a_], sm[b_]) and b_ - a_ >= 3:
            bim = 1
            hi_peak = px[b_] if side == 1 else px[a_]
            bim_above = int(side * (r.lvl - hi_peak) > 0)
    # 發展中 VA vs 前日 VA
    poc, vah, val = poc_va(px - 0.5, cnt.astype(int))
    vah += 1
    di = DAYS.index(r.day); prev = D.iloc[di - 1]
    va_mig = side * ((vah + val) / 2 - (prev.vah + prev.val) / 2) / atr
    ovl = max(0, min(vah, prev.vah) - max(val, prev.val)) / max(vah - val, 1e-9)
    # 最後完成字母的波幅 z
    rngs = ph - pl
    last_z = (rngs[-1] - rngs.mean()) / (rngs.std() + 1e-9) if len(rngs) >= 4 else np.nan
    # 前 5 天合成剖面
    cz = COMP.get(r.day)
    comp_z = side * (r.lvl - cz[0]) / cz[1] if cz else np.nan
    # dPOC → 進場價之間的單印
    k_poc = int(np.floor(poc - lo0 + 0.5)); k_lvl = int(np.clip(np.floor(r.lvl - lo0), 0, n - 1))
    seg = cnt[min(k_poc, k_lvl):max(k_poc, k_lvl) + 1]
    init_sgl = (seg == 1).sum() / atr
    return dict(ext_age=ext_age, ext_letter=ext_letter, n_ext=n_ext, skew=side * skew, bimodal=bim, bim_above=bim_above,
                va_mig=va_mig, va_ovl=ovl, last_z=last_z, comp_z=comp_z, init_sgl=init_sgl,
                day_rng=(Hh.max() - Ll.min()) / atr)


def prereg(X, f):
    """事先登記的規則：1~5 月五分位中最差的一組刪掉。回傳 (說明, 保留遮罩)。"""
    IS = X[X.day < SPLIT]
    x = X[f]
    if x.nunique() <= 3:
        g = IS.groupby(f).pnl.mean(); worst = g.idxmin()
        return f"刪 {f}={worst:g}", x != worst
    e = np.unique(np.nanquantile(IS[f], [0, .2, .4, .6, .8, 1])); e[0], e[-1] = -np.inf, np.inf
    gi = pd.cut(IS[f], e); means = IS.groupby(gi, observed=True).pnl.mean()
    worst = means.idxmin()
    keep = ~((x > worst.left) & (x <= worst.right)) | x.isna()
    return f"刪 {f} ∈ ({worst.left:.2f}, {worst.right:.2f}]", keep


if __name__ == "__main__":
    out = open("results_part16.txt", "w")
    def P(s=""): print(s); out.write(s + "\n"); out.flush()
    for lab, kw in [("②+ 觸價", {}), ("②+ M5收盤+0.05ATR", dict(confirm="close5", buf=0.05))]:
        s, _ = signals(60, 60, **kw)
        X = exits(s).reset_index(drop=True)
        A = pd.DataFrame([adv(r) for r in X.itertuples()], index=X.index)
        B = pd.DataFrame([basic_features(r, 15) for r in X.itertuples()], index=X.index)[["tail"]]
        X = pd.concat([X, A, B], axis=1)
        X.to_csv(f"part16_trades_{'close5' if kw else 'touch'}.csv", index=False)
        t80 = X[X.day < SPLIT]["tail"].quantile(0.8)
        T = X["tail"] <= t80
        P("=" * 120)
        P(f"[{lab}]  基準 {both(X)}")
        P(f"           + tail {both(X[T])}")
        P("\n  各指標五分位（1~5 月 | 6~10 月 每筆，分界取 1~5 月）")
        for f in A.columns:
            x = X[f]
            if x.nunique() <= 3:
                cells = [f"{v:g}: {X.pnl[(x == v) & (X.day < SPLIT)].mean():+5.1f}|{X.pnl[(x == v) & (X.day >= SPLIT)].mean():+5.1f}"
                         f"({((x == v) & (X.day < SPLIT)).sum()}/{((x == v) & (X.day >= SPLIT)).sum()})" for v in sorted(x.dropna().unique())]
            else:
                e = np.unique(np.nanquantile(x[X.day < SPLIT], [0, .2, .4, .6, .8, 1])); e[0], e[-1] = -np.inf, np.inf
                g = pd.cut(x, e)
                cells = [f"≤{c.right:.2f}: {X.pnl[(g == c) & (X.day < SPLIT)].mean():+5.1f}|{X.pnl[(g == c) & (X.day >= SPLIT)].mean():+5.1f}"
                         for c in g.cat.categories]
            P(f"    {f:10s} " + "  ".join(cells))
        P("\n  事先登記的刪除規則（只看 1~5 月決定） → 6~10 月結果；以及疊在 tail 之上")
        for f in A.columns:
            desc, keep = prereg(X, f)
            o_all = X[keep & (X.day >= SPLIT)].pnl; o_base = X[X.day >= SPLIT].pnl
            o_t = X[keep & T & (X.day >= SPLIT)].pnl; o_tb = X[T & (X.day >= SPLIT)].pnl
            pf = lambda p: p[p > 0].sum() / -p[p <= 0].sum()
            P(f"    {desc:34s} OOS 每筆 {o_base.mean():+5.2f}→{o_all.mean():+5.2f} PF {pf(o_base):.2f}→{pf(o_all):.2f}"
              f" | 疊 tail：{o_tb.mean():+5.2f}→{o_t.mean():+5.2f} PF {pf(o_tb):.2f}→{pf(o_t):.2f}  剩 {len(o_t)}/{len(o_tb)} 筆")
    out.close()
