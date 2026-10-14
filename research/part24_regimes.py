"""第二十四部分：切狀態找均值回歸。

方法（避免切太多狀態造成的假發現）：
  1. 2023-24 = 探索期，2025-26 = 確認期；狀態分組門檻（三分位）只用探索期決定。
  2. 一個狀態要算「有回歸」必須：探索期 b < 0 且 t ≤ −2.5，確認期 b < 0 且 t ≤ −2（方向與顯著性都重現）。
  3. 通過者再做事件交易（扣點差），分探索 / 確認期報告。

A. 與均線無關的檢查：
   A1. 每小時區塊內的變異數比（5 分報酬、縮尾 1%）；VR < 1 = 小時內回歸。
   A2. 時段反轉：每個半小時反向做前 30 分，持有 30 分，探索 / 確認期分開。
B. 偏離斜率 b（未來 h = n/2 根報酬 / ATR 對 偏離 / ATR，週叢集 t），中線 = EMA(n) 與當日 VWAP，
   M5 n = 6/12/24/48，M1 n = 10/30。狀態：
     sess   時段（broker = 紐約+7）：亞洲 01-09、倫敦 10-15、紐約 16-19、尾盤 20-24
     er     過去 n 根的效率比（三分位）
     vol    ATR(短)/ATR(長)（三分位，波動擴張 / 收縮）
     dtrend 過去 10 天日線效率比（三分位，前一天以前的資料）
     inrng  價格在前一日高低區間內 / 外
     vr     過去一天的滾動變異數比（三分位；「剛剛有在回歸」會不會延續）
     sess×er 時段與效率比交叉
C. 通過者：偏離越過 ±2σ（σ = 偏離滾動標準差 4n）→ 反向，目標碰中線、停損等距、最長 2n 根，扣點差。
"""
import numpy as np
import pandas as pd
from load import load

OUT = []


def P(s=""):
    print(s)
    OUT.append(s)


SESS_BINS = [0, 1, 10, 16, 20, 24]
SESS_LAB = ["尾盤", "亞洲", "倫敦", "紐約", "尾盤"]


def sess_of(idx):
    h = idx.hour
    return np.select([(h >= 1) & (h < 10), (h >= 10) & (h < 16), (h >= 16) & (h < 20)], ["亞洲", "倫敦", "紐約"], "尾盤")


# ---------------------------------------------------------------- A. 變異數比 / 時段反轉
def part_a(m1):
    """全部在同一個 60 分區塊內計算（分子、分母用同一批 K 棒），避免跨時段錯配。"""
    P("=" * 120)
    P("A1. 每小時區塊內的變異數比 VR = Var(60 分報酬) / (12 · Var(5 分報酬))，5 分報酬縮尾 1%（壓掉數據行情尖峰）")
    P("    < 1 = 小時內回歸、> 1 = 動能。iid 下標準誤約 ±0.07（每年約 ±0.13）。")
    c5 = m1.close.resample("5min", label="left", closed="left").last()
    r5 = c5.diff()
    r5[(c5.index.to_series().diff() != pd.Timedelta("5min")).values] = np.nan
    df = pd.DataFrame({"r": r5})
    df["blk"] = df.index.floor("60min"); df["h"] = df.index.hour; df["yr"] = df.index.year

    def vr(sub):
        lo, hi = sub.r.quantile([0.01, 0.99])
        r = sub.r.clip(lo, hi)
        g = pd.DataFrame({"r": r, "blk": sub.blk}).groupby("blk").r.agg(["sum", "count"])
        g = g[g["count"] == 12]
        return g["sum"].var() / (12 * r.var())
    for h in range(2, 23):
        sub = df[df.h == h]
        P(f"    {h:02d}:00  全部 {vr(sub):.2f}   " + " ".join(f"{y}:{vr(sub[sub.yr == y]):.2f}" for y in [2023, 2024, 2025, 2026]))

    P("")
    P("A2. 時段反轉：時間 = 第一個 30 分的起點；在其結束時反向做這 30 分的方向、再持有 30 分（單位 = 當日 ATR_D 的 %）")
    P("    探索 = 2023-24、確認 = 2025-26；共 42 個時段 → Bonferroni 門檻 |t| ≈ 3.2")
    c30 = m1.close.resample("30min", label="left", closed="left").last()
    sp30 = m1.spread.resample("30min", label="left", closed="left").mean() * 0.01
    day = (c30.index - pd.Timedelta(hours=1)).normalize()
    dr = m1.groupby((m1.index - pd.Timedelta(hours=1)).normalize()).agg(h=("high", "max"), l=("low", "min"))
    atrd = (dr.h - dr.l).rolling(20).mean().shift(1)
    x = pd.DataFrame({"prev": c30.diff(), "nxt": c30.shift(-1) - c30, "atr": day.map(atrd), "sp": sp30})
    ix = x.index.to_series()
    x = x[(ix - ix.shift(1) == pd.Timedelta("30min")) & (ix.shift(-1) - ix == pd.Timedelta("30min"))].dropna()
    x["pnl"] = -np.sign(x.prev) * x.nxt / x.atr * 100
    x["pnl_net"] = x.pnl - x.sp / x.atr * 100
    x["slot"] = x.index.strftime("%H:%M")
    x["per"] = np.where(x.index.year <= 2024, "IS", "OOS")
    rows = []
    for slot, g in x.groupby("slot"):
        r = {"slot": slot}
        for per, gp in g.groupby("per"):
            r[f"m_{per}"] = gp.pnl.mean(); r[f"t_{per}"] = gp.pnl.mean() / (gp.pnl.std() / np.sqrt(len(gp)))
            r[f"net_{per}"] = gp.pnl_net.mean(); r[f"n_{per}"] = len(gp)
        rows.append(r)
    T = pd.DataFrame(rows).set_index("slot")
    T = T[(T.index >= "01:30") & (T.index <= "22:30")]
    for slot, r in T.iterrows():
        mark = " ✓兩期都正" if r.t_IS >= 2 and r.t_OOS >= 2 else (" ✗兩期都負" if r.t_IS <= -2 and r.t_OOS <= -2 else "")
        P(f"    {slot}  探索 {r.m_IS:+5.2f}% (t {r.t_IS:+4.1f})  確認 {r.m_OOS:+5.2f}% (t {r.t_OOS:+4.1f})  扣點差 {r.net_IS:+5.2f}% | {r.net_OOS:+5.2f}%{mark}")
    T.to_csv("part24_slot_reversal.csv")


# ---------------------------------------------------------------- B. 狀態切分
def prep(d, n_atr, n_atr_long, bars_day):
    d = d.copy()
    pc = d.close.shift()
    tr = np.maximum(d.high - d.low, np.maximum((d.high - pc).abs(), (d.low - pc).abs()))
    d["atr"] = tr.rolling(n_atr).mean()
    d["vol"] = tr.rolling(n_atr // 8).mean() / tr.rolling(n_atr_long).mean()
    d["week"] = d.index.to_period("W").astype(str)
    d["day"] = (d.index - pd.Timedelta(hours=1)).normalize()
    d["sess"] = sess_of(d.index)
    d["period"] = np.where(d.index.year <= 2024, "IS", "OOS")
    # 前一日資料
    D = d.groupby("day").agg(h=("high", "max"), l=("low", "min"), c=("close", "last"))
    er10 = ((D.c - D.c.shift(10)).abs() / D.c.diff().abs().rolling(10).sum()).shift(1)
    d["dtrend"] = d.day.map(er10)
    d["inrng"] = np.where((d.close <= d.day.map(D.h.shift(1))) & (d.close >= d.day.map(D.l.shift(1))), "區間內", "區間外")
    d.loc[d.day.map(D.h.shift(1)).isna(), "inrng"] = np.nan
    # 當日 VWAP
    tp = (d.high + d.low + d.close) / 3
    v = d.tickvol.astype(float).clip(lower=1)
    d["vwap"] = (tp * v).groupby(d.day).cumsum() / v.groupby(d.day).cumsum()
    # 滾動變異數比（過去一天，q = bars_day/48）
    r = d.close.diff()
    q = max(2, bars_day // 48)
    d["vr"] = r.rolling(q).sum().rolling(bars_day).var() / (q * r.rolling(bars_day).var())
    d["pos"] = np.arange(len(d))
    d["dend"] = d.pos.groupby(d.day).transform("max")
    return d.iloc[n_atr_long:]


def ema(x, n):
    return pd.Series(x).ewm(span=n, adjust=False).mean().values


def er(c, n):
    s = pd.Series(c)
    return ((s - s.shift(n)).abs() / s.diff().abs().rolling(n).sum()).values


def tercile(x, is_mask, labels=("低", "中", "高")):
    q1, q2 = np.nanpercentile(x[is_mask], [33.3, 66.7])
    out = np.where(x <= q1, labels[0], np.where(x <= q2, labels[1], labels[2])).astype(object)
    out[~np.isfinite(x)] = None
    return out


def grouped_slopes(x, y, keys, cluster):
    df = pd.DataFrame({"x": x, "y": y, "k": keys, "c": cluster}).dropna()
    g = df.groupby("k")
    df["xc"] = df.x - g.x.transform("mean")
    df["yc"] = df.y - g.y.transform("mean")
    df["xy"] = df.xc * df.yc
    df["xx"] = df.xc ** 2
    s = df.groupby("k")[["xy", "xx"]].sum()
    s["b"] = s.xy / s.xx
    df["e"] = df.yc - df.k.map(s.b) * df.xc
    df["xe"] = df.xc * df.e
    sc = df.groupby(["k", "c"]).xe.sum() ** 2
    s["se"] = np.sqrt(sc.groupby(level=0).sum()) / s.xx
    s["t"] = s.b / s.se
    s["n"] = g.size()
    return s[["b", "t", "n"]]


def part_b(d, tf, lengths):
    c = d.close.values
    atr = d.atr.values
    is_m = (d.period.values == "IS")
    rows = []
    anchors = [(f"EMA{n}", ema(c, n), n) for n in lengths] + [("VWAP", d.vwap.values, lengths[1])]
    for name, f, n in anchors:
        h = max(1, n // 2)
        j = d.pos.values + h
        ok = j <= d.dend.values                      # 不跨交易日
        x = np.where(ok, (c - f) / atr, np.nan)
        y = np.where(ok, (np.r_[c[h:], np.full(h, np.nan)] - c) / atr, np.nan)
        st = {
            "sess": d.sess.values,
            "er": tercile(er(c, n), is_m),
            "vol": tercile(d.vol.values, is_m),
            "dtrend": tercile(d.dtrend.values, is_m),
            "inrng": d.inrng.values,
            "vr": tercile(d.vr.values, is_m),
        }
        st["sess×er"] = np.array([f"{a}/{b}" if b is not None else None for a, b in zip(st["sess"], st["er"])], dtype=object)
        st["全部"] = np.full(len(c), "全部", dtype=object)
        for sname, lab in st.items():
            keys = pd.Series(lab).astype(object) + "|" + pd.Series(d.period.values)
            keys = keys.where(pd.Series(lab).notna())
            res = grouped_slopes(x, y, keys.values, d.week.values)
            for k, r in res.iterrows():
                bucket, per = k.split("|")
                rows.append(dict(tf=tf, anchor=name, n=n, h=h, state=sname, bucket=bucket, per=per, b=r.b, t=r.t, N=r.n))
    R = pd.DataFrame(rows)
    W = R.pivot_table(index=["tf", "anchor", "n", "h", "state", "bucket"], columns="per", values=["b", "t", "N"])
    W.columns = [f"{a}_{b}" for a, b in W.columns]
    return W.reset_index()


# ---------------------------------------------------------------- C. 事件交易
def trade(d, anchor, n, mask, T=2.0):
    c, h, l = d.close.values, d.high.values, d.low.values
    f = d.vwap.values if anchor == "VWAP" else ema(c, n)
    spr = d.spread.values * 0.01
    dev = c - f
    sd = pd.Series(dev).rolling(4 * n).std().values
    z = dev / sd
    zp = np.r_[np.nan, z[:-1]]
    ev = np.where(mask & (np.abs(z) >= T) & (np.abs(zp) < T) & (np.sign(z) == np.sign(zp)))[0]
    out = []
    dend = d.dend.values
    for i in ev:
        side = -np.sign(dev[i]); e = c[i]; risk = abs(dev[i])
        k = np.arange(i + 1, min(i + 1 + 2 * n, dend[i] + 1))
        if len(k) == 0 or risk <= 0:
            continue
        stop = e - side * risk
        tgt = f[k - 1]
        hs, ht = ((l[k] <= stop), (h[k] >= tgt)) if side > 0 else ((h[k] >= stop), (l[k] <= tgt))
        fs = np.argmax(hs) if hs.any() else 10 ** 9
        ft = np.argmax(ht) if ht.any() else 10 ** 9
        if fs == ft == 10 ** 9:
            pnl = (c[k[-1]] - e) * side
        elif fs <= ft:
            pnl = -risk
        else:
            pnl = (tgt[ft] - e) * side
        out.append((d.period.values[i], pnl / risk, (pnl - spr[i]) / risk, risk / d.atr.values[i]))
    return pd.DataFrame(out, columns=["per", "R", "Rn", "risk_atr"])


def state_mask(d, state, bucket, n):
    c = d.close.values
    is_m = d.period.values == "IS"
    if state == "全部":
        return np.ones(len(d), bool)
    if state == "sess":
        return d.sess.values == bucket
    if state == "inrng":
        return d.inrng.values == bucket
    if state == "sess×er":
        s, e = bucket.split("/")
        return (d.sess.values == s) & (tercile(er(c, n), is_m) == e)
    src = {"er": er(c, n), "vol": d.vol.values, "dtrend": d.dtrend.values, "vr": d.vr.values}[state]
    return tercile(src, is_m) == bucket


# ---------------------------------------------------------------- main
def main():
    P(__doc__.strip().splitlines()[0])
    m1 = load("data/XAUUSD_M1_2023_2026.csv")
    part_a(m1)

    m5 = m1.resample("5min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "tickvol": "sum", "spread": "mean"}).dropna(subset=["close"])
    D5 = prep(m5, 96, 1152, 276)
    D1 = prep(m1, 480, 5760, 1380)
    W = pd.concat([part_b(D5, "M5", [6, 12, 24, 48]), part_b(D1, "M1", [10, 30])], ignore_index=True)
    W.to_csv("part24_state_slopes.csv", index=False)

    P("=" * 120)
    ntest = len(W)
    P(f"B. 狀態切分的偏離斜率：共 {ntest} 個（中線 × 長度 × 狀態 × 分組）檢定")
    base = W[W.state == "全部"]
    P("  不分狀態（基準）：")
    for _, r in base.iterrows():
        P(f"    {r.tf} {r.anchor:<7} h={r.h:<3} 探索 b {r.b_IS:+.4f} (t {r.t_IS:+5.1f}) | 確認 b {r.b_OOS:+.4f} (t {r.t_OOS:+5.1f})")
    neg_is = W[(W.b_IS < 0) & (W.t_IS <= -2.5)]
    surv = neg_is[(neg_is.b_OOS < 0) & (neg_is.t_OOS <= -2)]
    P(f"  探索期 t ≤ −2.5 的回歸狀態：{len(neg_is)} 個（若全是雜訊，預期約 {ntest*0.006:.0f} 個）；其中確認期也 t ≤ −2：{len(surv)} 個")
    pos_both = W[(W.t_IS >= 2.5) & (W.t_OOS >= 2)]
    P(f"  對照：兩期都顯著為「動能」（t ≥ 2.5 / ≥ 2）的狀態：{len(pos_both)} 個")
    P("")
    P("  探索期最強的 25 個回歸狀態（看確認期是否重現）：")
    for _, r in neg_is.sort_values("t_IS").head(25).iterrows():
        mark = " ✓" if (r.b_OOS < 0 and r.t_OOS <= -2) else ""
        P(f"    {r.tf} {r.anchor:<7} {r.state:<7} {r.bucket:<8} 探索 b {r.b_IS:+.4f} t {r.t_IS:+5.1f} (N {int(r.N_IS):7d}) | "
          f"確認 b {r.b_OOS:+.4f} t {r.t_OOS:+5.1f}{mark}")
    P("")
    P("  各狀態維度：回歸（b<0）的分組比例與平均 t（探索 | 確認），看哪個維度系統性地偏向回歸")
    for s, g in W[W.state != "全部"].groupby("state"):
        for bkt, gb in g.groupby("bucket"):
            if s == "sess×er":
                continue
            P(f"    {s:<7} {bkt:<6} 平均 t 探索 {gb.t_IS.mean():+5.2f} 確認 {gb.t_OOS.mean():+5.2f}   b<0 比例 {(gb.b_IS<0).mean():4.0%} | {(gb.b_OOS<0).mean():4.0%}  ({len(gb)} 個中線)")
    sx = W[W.state == "sess×er"].groupby("bucket")[["t_IS", "t_OOS"]].mean()
    P("    時段 × ER（平均 t，探索 | 確認）：")
    for k, r in sx.iterrows():
        P(f"      {k:<8} {r.t_IS:+5.2f} | {r.t_OOS:+5.2f}")

    P("=" * 120)
    P("C. 事件交易（±2σ 反向、目標中線、等距停損、最長 2n 根、不跨日；R 扣點差）")
    cands = surv.sort_values("t_IS").head(15)
    if len(cands) == 0:
        cands = neg_is.sort_values("t_IS").head(8)
        P("  （沒有通過者；改列探索期 t ≤ −2.5 的全部，僅供參考）")
    for _, r in cands.iterrows():
        d = D5 if r.tf == "M5" else D1
        tr = trade(d, r.anchor if r.anchor == "VWAP" else "EMA", int(r.n), state_mask(d, r.state, r.bucket, int(r.n)))
        s = f"  {r.tf} {r.anchor:<7} {r.state:<7} {r.bucket:<8}"
        for per in ["IS", "OOS"]:
            g = tr[tr.per == per]
            if len(g) < 5:
                s += f" | {per} n {len(g)}"
                continue
            s += (f" | {per} n {len(g):5d} R {g.R.mean():+.3f} (t {g.R.mean()/(g.R.std()/np.sqrt(len(g))):+4.1f}) "
                  f"扣點差 {g.Rn.mean():+.3f} 風險 {g.risk_atr.median():.1f}×K棒ATR")
        P(s)

    with open("results_part24.txt", "w") as fh:
        fh.write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
