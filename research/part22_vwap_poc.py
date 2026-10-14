"""第二十二部分：VWAP 是不是均值回歸的中線？POC 能不能當支撐壓力？（tick volume）

資料：data/XAUUSD_M1_2023_2026.csv（2023/6 ~ 2026/10，M1，broker 時間 = 紐約 + 7）。
交易日 = broker 01:00 起算；VWAP 每天 01:00 重新錨定，權重 = tick volume，價格 = (H+L+C)/3。
尺度：ATR_D = 前 20 個交易日的日波幅平均（只用過去資料）。

A. 偏離 → 未來報酬回歸：fwd_h / ATR_D = a + b · (C − 中線) / ATR_D，b < 0 才是回歸。
   隨機漫步下對「任何」中線 b = 0（未來報酬與過去無關），所以 b 顯著 < 0 才是真的均值回歸。
   注意：不能用「偏離本身會縮小」來判斷 — VWAP 會自己往價格移動，隨機漫步也會看起來「回歸」。
   比較中線：VWAP、TWAP（等權）、當日開盤價、滾動 240 分 SMA、滾動 240 分 VWAP。
   虛無模型：每天把 M1 K 棒順序打亂（保留 K 棒形狀與 tick volume），重算全部統計。
B. 事件式：價格收盤第一次越過 ±T σ（VWAP 量加權標準差帶）→ 反向做，目標 = 碰到 VWAP，
   停損 = 等距（進場到 VWAP 的距離），當日收盤平倉。和打亂資料、TWAP 帶比較。
C. 前一日 tick volume 剖面的 POC / VAH / VAL 當支撐壓力：
   當日第一次碰到價位後，先反彈 X 還是先穿越 X（X = 0.15 / 0.3 ATR_D）。
   對照：前一日區間內的隨機價位（依「開盤到價位的距離」加權配對）、TPO POC、前高前低、前收、前日 VWAP。
"""
import numpy as np
import pandas as pd
from load import load
from tpo import tpo_profile, poc_va

rng = np.random.default_rng(22)
PATH = "data/XAUUSD_M1_2023_2026.csv"
HORIZONS = [15, 30, 60, 120]
OUT = []


def P(s=""):
    print(s)
    OUT.append(s)


# ---------------------------------------------------------------- 資料
def prep(d):
    d = d.copy()
    d["day"] = (d.index - pd.Timedelta(hours=1)).normalize()
    n = d.groupby("day").size()
    d = d[d.day.isin(n[n >= 600].index)]
    dr = d.groupby("day").agg(h=("high", "max"), l=("low", "min"))
    atrd = (dr.h - dr.l).rolling(20).mean().shift(1)
    d["atrd"] = d.day.map(atrd)
    d = d[d.atrd.notna()]
    return d


def shuffle_days(d):
    """每天內把 K 棒（相對前收的 O/H/L/C 位移 + tickvol + spread）順序打亂後重建價格。"""
    out = []
    for day, g in d.groupby("day", sort=True):
        c = g.close.values
        pc = np.r_[g.open.values[0], c[:-1]]
        rel = np.c_[g.open.values - pc, g.high.values - pc, g.low.values - pc, c - pc]
        idx = rng.permutation(len(g))
        rel = rel[idx]
        newc = pc[0] + np.cumsum(rel[:, 3])
        newpc = np.r_[pc[0], newc[:-1]]
        gg = pd.DataFrame({"open": newpc + rel[:, 0], "high": newpc + rel[:, 1], "low": newpc + rel[:, 2],
                           "close": newc, "tickvol": g.tickvol.values[idx], "spread": g.spread.values[idx],
                           "day": day, "atrd": g.atrd.values}, index=g.index)
        out.append(gg)
    return pd.concat(out)


def add_centers(d):
    d = d.copy()
    tp = (d.high + d.low + d.close) / 3
    v = d.tickvol.astype(float).clip(lower=1)
    g = d.day
    cv = v.groupby(g).cumsum()
    d["vwap"] = (tp * v).groupby(g).cumsum() / cv
    d["vsd"] = np.sqrt(((tp ** 2 * v).groupby(g).cumsum() / cv - d.vwap ** 2).clip(lower=0))
    cn = tp.groupby(g).cumcount() + 1
    d["twap"] = tp.groupby(g).cumsum() / cn
    d["tsd"] = np.sqrt(((tp ** 2).groupby(g).cumsum() / cn - d.twap ** 2).clip(lower=0))
    d["dopen"] = d.open.groupby(g).transform("first")
    d["sma240"] = tp.rolling(240).mean()
    d["rvwap240"] = (tp * v).rolling(240).sum() / v.rolling(240).sum()
    d["mins"] = cn - 1
    d["pos"] = np.arange(len(d))
    d["dend"] = d.pos.groupby(g).transform("max")
    d["year"] = d.index.year
    return d


# ---------------------------------------------------------------- A. 回歸係數
def clustered_slope(x, y, grp):
    xm, ym = x.mean(), y.mean()
    xc = x - xm
    sxx = (xc ** 2).sum()
    b = (xc * (y - ym)).sum() / sxx
    e = (y - ym) - b * xc
    sc = pd.Series(xc * e).groupby(grp).sum().values
    se = np.sqrt((sc ** 2).sum()) / sxx
    return b, b / se


def part_a(d, label, centers=("vwap", "twap", "dopen", "sma240", "rvwap240"), by_year=False):
    c = d.close.values
    res = {}
    samp = (d.index.minute % 15 == 0) & (d.mins.values >= 60)
    for h in HORIZONS:
        j = d.pos.values + h
        ok = samp & (j <= d.dend.values)
        i = np.where(ok)[0]
        fwd = (c[i + h] - c[i]) / d.atrd.values[i]
        grp = d.day.values[i]
        for cen in centers:
            x = (c[i] - d[cen].values[i]) / d.atrd.values[i]
            m = np.isfinite(x)
            res[(cen, h)] = clustered_slope(x[m], fwd[m], grp[m])
            if by_year and cen == "vwap" and h == 60:
                yr = d.year.values[i][m]
                res[("vwap_year", h)] = {y: clustered_slope(x[m][yr == y], fwd[m][yr == y], grp[m][yr == y])
                                         for y in np.unique(yr)}
    return res


def bucket_table(d):
    """VWAP 偏離（σ 帶）分組 → 未來 60 分報酬（ATR_D）。"""
    c = d.close.values
    samp = (d.index.minute % 15 == 0) & (d.mins.values >= 60)
    j = d.pos.values + 60
    i = np.where(samp & (j <= d.dend.values))[0]
    z = (c[i] - d.vwap.values[i]) / d.vsd.values[i]
    fwd = (c[i + 60] - c[i]) / d.atrd.values[i]
    edges = [-np.inf, -3, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 3, np.inf]
    cut = pd.cut(z, edges)
    t = pd.DataFrame({"b": cut, "f": fwd, "s": np.sign(z)}).groupby("b", observed=True).f.agg(["count", "mean", "std"])
    t["t"] = t["mean"] / (t["std"] / np.sqrt(t["count"] / 4))   # 60 分重疊 4 次 → 有效樣本 /4
    return t


# ---------------------------------------------------------------- B. 事件式回歸到 VWAP
def part_b(d, center="vwap", sd="vsd", thresholds=(1.0, 1.5, 2.0, 2.5, 3.0)):
    c, h, l = d.close.values, d.high.values, d.low.values
    cen, s = d[center].values, d[sd].values
    dend, mins = d.dend.values, d.mins.values
    spr = d.spread.values * 0.01
    z = (c - cen) / np.where(s > 0, s, np.nan)
    zp = np.r_[np.nan, z[:-1]]
    same_day = np.r_[False, d.day.values[1:] == d.day.values[:-1]]
    rows = []
    for T in thresholds:
        ev = np.where(same_day & (mins >= 60) & (d.index.hour < 22) & (np.abs(z) >= T) & (np.abs(zp) < T)
                      & (np.sign(z) == np.sign(zp)))[0]
        for i in ev:
            side = -np.sign(z[i])             # 反向做
            e = c[i]
            risk = abs(e - cen[i])
            if risk <= 0:
                continue
            stop = e - side * risk
            k = np.arange(i + 1, dend[i] + 1)
            if len(k) == 0:
                continue
            tgt = cen[k - 1]                  # 上一根結束時的 VWAP（當根開始時已知）
            if side > 0:
                hs = l[k] <= stop
                ht = h[k] >= tgt
            else:
                hs = h[k] >= stop
                ht = l[k] <= tgt
            fs = np.argmax(hs) if hs.any() else 10 ** 9
            ft = np.argmax(ht) if ht.any() else 10 ** 9
            if fs == 10 ** 9 and ft == 10 ** 9:
                pnl, out = (c[k[-1]] - e) * side, "eod"
            elif fs <= ft:
                pnl, out = -risk, "stop"
            else:
                pnl, out = (tgt[ft] - e) * side, "tgt"
            rows.append((T, d.year.values[i], pnl / risk, (pnl - spr[i]) / risk, out, risk / d.atrd.values[i]))
    return pd.DataFrame(rows, columns=["T", "year", "R", "Rnet", "out", "risk_atr"])


def summarize_b(r, label):
    P(f"  {label}")
    for T, g in r.groupby("T"):
        n = len(g)
        tr = (g.out == "tgt").mean()
        se = g.R.std() / np.sqrt(n)
        yrs = " ".join(f"{y}:{gy.R.mean():+.3f}" for y, gy in g.groupby("year"))
        P(f"    ±{T:.1f}σ  n={n:6d}  先到VWAP {tr:5.1%}  停損 {(g.out=='stop').mean():5.1%}  收盤 {(g.out=='eod').mean():5.1%}"
          f"  平均R {g.R.mean():+.3f} (t {g.R.mean()/se:+.1f})  扣點差 {g.Rnet.mean():+.3f}  風險中位 {g.risk_atr.median():.2f}ATR | {yrs}")


# ---------------------------------------------------------------- C. POC 支撐壓力
def volume_profile(g, step):
    lo = np.floor(g.low.min() / step) * step
    nb = int(np.floor((g.high.max() - lo) / step)) + 1
    edges = lo + step * np.arange(nb + 1)
    L, H, V = g.low.values, g.high.values, g.tickvol.values.astype(float)
    # 每根 K 棒的量平均分配到它 high~low 覆蓋的格子（按重疊長度比例）
    ov = np.clip(np.minimum(H[:, None], edges[None, 1:]) - np.maximum(L[:, None], edges[None, :-1]), 0, None)
    rng_ = (H - L)
    w = np.where(rng_[:, None] > 0, ov / np.where(rng_ > 0, rng_, 1)[:, None], 0)
    point = rng_ == 0
    if point.any():
        b = np.clip(((L[point] - lo) / step).astype(int), 0, nb - 1)
        w[np.where(point)[0], b] = 1
    vol = (w * V[:, None]).sum(0)
    return edges[:-1], vol


def poc_va_vol(prices, vol, step, pct=0.70):
    mx = vol.max()
    cand = np.where(vol >= mx - 1e-9)[0]
    poc = cand[np.argmin(np.abs(cand - (len(vol) - 1) / 2))]
    lo = hi = poc
    acc, tgt = vol[poc], pct * vol.sum()
    while acc < tgt and (lo > 0 or hi < len(vol) - 1):
        up = vol[hi + 1] if hi < len(vol) - 1 else -1
        dn = vol[lo - 1] if lo > 0 else -1
        if up >= dn:
            hi += 1; acc += vol[hi]
        else:
            lo -= 1; acc += vol[lo]
    return prices[poc] + step / 2, prices[hi] + step, prices[lo]


def daily_levels(d):
    rows = {}
    days = sorted(d.day.unique())
    for day, g in d.groupby("day", sort=True):
        step = max(0.1, round(g.atrd.iloc[0] / 50, 1))
        pr, vol = volume_profile(g, step)
        vpoc, vah, val = poc_va_vol(pr, vol, step)
        gg = g.copy()
        gg["period"] = ((gg.index - day - pd.Timedelta(hours=1)).total_seconds() // 1800).astype(int)
        tp_pr, tp_cnt, _ = tpo_profile(gg, tick=step)
        tpoc, _, _ = poc_va(tp_pr, tp_cnt)
        rows[day] = dict(vpoc=vpoc, vah=vah, val=val, tpoc=tpoc + step / 2, hi=g.high.max(), lo=g.low.min(),
                         cl=g.close.iloc[-1], vw=g.vwap.iloc[-1], mid=(g.high.max() + g.low.min()) / 2)
    L = pd.DataFrame(rows).T.astype(float)
    return L.shift(1).dropna()     # 前一日的價位用在今天


def touch_test(g, L, atr, ks=(0.15, 0.3)):
    """回傳 (touched, dist_atr, side, {k: outcome})；outcome 1 = 反彈、0 = 穿越、nan = 未決。"""
    o = g.open.values[0]
    dist = abs(o - L) / atr
    if dist < 0.1:
        return None
    side = 1 if o > L else -1          # 1 = 從上方測支撐，-1 = 從下方測壓力
    h, l = g.high.values, g.low.values
    hit = (l <= L) if side == 1 else (h >= L)
    if not hit.any():
        return (0, dist, side, {k: np.nan for k in ks})
    j = np.argmax(hit)
    res = {}
    for k in ks:
        X = k * atr
        brk_lvl, bnc_lvl = L - side * X, L + side * X
        if (side == 1 and l[j] <= brk_lvl) or (side == -1 and h[j] >= brk_lvl):
            res[k] = 0.0
            continue
        hh, ll = h[j + 1:], l[j + 1:]
        if side == 1:
            br, bn = ll <= brk_lvl, hh >= bnc_lvl
        else:
            br, bn = hh >= brk_lvl, ll <= bnc_lvl
        fb = np.argmax(br) if br.any() else 10 ** 9
        fn = np.argmax(bn) if bn.any() else 10 ** 9
        res[k] = np.nan if fb == fn == 10 ** 9 else (0.0 if fb <= fn else 1.0)
    return (1, dist, side, res)


def part_c(d, n_rand=20):
    lv = daily_levels(d)
    names = ["vpoc", "vah", "val", "tpoc", "hi", "lo", "cl", "vw", "mid"]
    rows = []
    for day, g in d.groupby("day", sort=True):
        if day not in lv.index:
            continue
        x = lv.loc[day]
        atr = g.atrd.iloc[0]
        lvls = [(n, x[n]) for n in names] + [("rand", v) for v in rng.uniform(x.lo, x.hi, n_rand)]
        for n, v in lvls:
            r = touch_test(g, v, atr)
            if r is None:
                continue
            rows.append(dict(day=day, year=day.year, lvl=n, touched=r[0], dist=r[1], side=r[2],
                             **{f"b{k}": r[3][k] for k in r[3]}))
    return pd.DataFrame(rows)


def matched_compare(R, name, col, nboot=1000):
    """價位 vs 隨機價位：依開盤距離分組加權，並以交易日 bootstrap 差異的 95% 信賴區間。"""
    bins = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, np.inf]
    R = R[R[col].notna()]
    days = np.sort(R.day.unique())
    di = {x: k for k, x in enumerate(days)}

    def mats(X):
        b = pd.cut(X.dist, bins, labels=False).values
        dd = X.day.map(di).values
        n = np.zeros((len(days), len(bins) - 1)); s = np.zeros_like(n)
        np.add.at(n, (dd, b), 1); np.add.at(s, (dd, b), X[col].values)
        return n, s

    An, As = mats(R[R.lvl == name]); Bn, Bs = mats(R[R.lvl == "rand"])

    def stat(c):
        an, as_, bn, bs = c @ An, c @ As, c @ Bn, c @ Bs
        m = (an > 0) & (bn > 0)
        w = an[m] / an[m].sum()
        return (w * as_[m] / an[m]).sum(), (w * bs[m] / bn[m]).sum()

    a, b = stat(np.ones(len(days)))
    C = rng.multinomial(len(days), np.ones(len(days)) / len(days), size=nboot)
    diffs = [np.subtract(*stat(c)) for c in C]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return int(An.sum()), a, b, lo, hi


# ---------------------------------------------------------------- main
def main():
    raw = prep(load(PATH))
    d = add_centers(raw)
    P(f"資料：{d.index[0]} ~ {d.index[-1]}，{d.day.nunique()} 個交易日，{len(d):,} 根 M1")
    sh = [add_centers(shuffle_days(raw)) for _ in range(3)]

    P("=" * 120)
    P("A. 偏離中線 → 未來報酬的回歸斜率 b（fwd/ATR_D 對 偏離/ATR_D；每 15 分取樣、開盤 60 分後；t 以交易日叢集）")
    P("   b < 0 = 均值回歸、b > 0 = 動能；隨機漫步下 b = 0。")
    names = {"vwap": "VWAP(tick量)", "twap": "TWAP(等權)", "dopen": "當日開盤價", "sma240": "SMA 240分", "rvwap240": "滾動VWAP 240分"}
    ra = part_a(d, "real", by_year=True)
    rs = [part_a(x, "shuf") for x in sh]
    P(f"   {'中線':<16}" + "".join(f"{f'{h}分 b (t)':>20}" for h in HORIZONS) + f"{'打亂資料 60分 b':>22}")
    for cen, nm in names.items():
        s = f"   {nm:<16}"
        for h in HORIZONS:
            b, t = ra[(cen, h)]
            s += f"{b:+10.4f} ({t:+5.1f})  "
        bs = np.mean([r[(cen, 60)][0] for r in rs]); ts = np.mean([r[(cen, 60)][1] for r in rs])
        s += f"  {bs:+.4f} ({ts:+.1f})"
        P(s)
    P("   VWAP 60 分 b 逐年：" + "  ".join(f"{y}: {b:+.4f} (t {t:+.1f})" for y, (b, t) in ra[("vwap_year", 60)].items()))

    P("")
    P("   VWAP σ 帶分組 → 未來 60 分平均報酬（ATR_D；t 已按重疊修正）")
    bt = bucket_table(d)
    bts = bucket_table(sh[0])
    for (b, r), (_, r2) in zip(bt.iterrows(), bts.iterrows()):
        P(f"     z ∈ {str(b):<14} n={int(r['count']):6d}  平均 {r['mean']:+.4f}  t {r['t']:+5.1f}   | 打亂 {r2['mean']:+.4f}")

    P("=" * 120)
    P("B. 事件式：收盤第一次越過 ±Tσ → 反向進場，目標 = 碰 VWAP，停損 = 等距，收盤平倉（R = 風險倍數）")
    P("   1:1 目標下，若 VWAP 是真的回歸中線，先到 VWAP 的比例應明顯 > 打亂資料。")
    rb = part_b(d)
    summarize_b(rb, "真實資料（VWAP 帶）")
    summarize_b(part_b(d, "twap", "tsd"), "真實資料（TWAP 等權帶）")
    for k, x in enumerate(sh[:2]):
        summarize_b(part_b(x), f"打亂資料 #{k+1}（VWAP 帶）")
    rb.to_csv("part22_vwap_events.csv", index=False)

    P("=" * 120)
    P("C. 前一日價位當支撐壓力：今天第一次碰到後，先反彈 X 還是先穿越 X（同根兩者都到 = 穿越）")
    P("   對照 = 前一日高低之間的隨機價位（每天 20 個），依「開盤到價位的距離」分組加權配對；[95% CI] = 交易日 bootstrap 的差異")
    R = part_c(d)
    R.to_csv("part22_levels.csv", index=False)
    lab = {"vpoc": "量 POC(tick)", "vah": "VAH", "val": "VAL", "tpoc": "TPO POC", "hi": "前日高", "lo": "前日低",
           "cl": "前日收", "vw": "前日 VWAP", "mid": "前日區間中點"}
    for col, title in [("touched", "當日碰到的比例（磁吸）"), ("b0.15", "反彈率 X=0.15 ATR"), ("b0.3", "反彈率 X=0.30 ATR")]:
        P(f"  {title}")
        for nm, txt in lab.items():
            n, a, b, lo, hi = matched_compare(R, nm, col)
            sig = " *" if lo > 0 or hi < 0 else ""
            P(f"    {txt:<14} n={n:5d}  價位 {a:6.1%}  隨機 {b:6.1%}  差 {a-b:+6.1%}  [{lo:+.1%}, {hi:+.1%}]{sig}")
    P("  逐年（反彈率 X=0.30，價位 − 配對隨機）")
    for nm in ["vpoc", "tpoc", "hi", "lo"]:
        s = f"    {lab[nm]:<14}"
        for y in sorted(R.year.unique()):
            n, a, b, lo, hi = matched_compare(R[R.year == y], nm, "b0.3", nboot=300)
            s += f"  {y}: {a-b:+5.1%} (n {n})"
        P(s)

    with open("results_part22.txt", "w") as f:
        f.write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
