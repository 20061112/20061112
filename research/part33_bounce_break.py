"""第三十三部分：碰到價位時，判斷「彈回」還是「突破」。

資料：data/XAUUSD_M1_2026.csv，執行 M5（2/2~10/8），拆半 2/2~5/31（前半）vs 6/1~10/8（後半）。

價位來源（每天開盤前決定，只用過去資料）：
  kde15   M15 插針 KDE，擴張視窗（第三十部分）
  kde60   H1 插針 KDE，滾動 5 天
  pdhl    前一天高點、低點
  r50     50 美元整數關卡；r10 10 美元整數關卡
  placebo kde15 價位隨機平移 3~6 個頻寬（看價位本身有沒有差別）
碰觸定義同第三十部分（先離開 1 ATR，再回到 0.1 ATR 內）。

標籤：bounce = 先往反彈方向走 1 ATR（碰觸棒之後），還是先往突破方向走 1 ATR（含碰觸棒）。

碰觸當下的特徵（只用碰觸棒之前已收盤的 K 棒 + 碰觸這件事本身）：
  run6 / run12   前 6 / 12 根往價位方向走了幾個 ATR（越大 = 衝過來越快）
  er12           前 12 根的效率比（越接近 1 = 走得越直）
  appr_bars      從離開價位 1 ATR 到碰到花了幾根（越少 = 越快）
  htf            碰觸方向與大週期趨勢：(close − EMA240) × dir / ATR，> 0 = 彈回方向順大趨勢
  htf_slope      EMA240 近 48 根斜率 × dir / ATR
  day_ext        今天開盤到現在往價位方向走了多少 ATR
  vol_ratio      ATR14 / 近一天平均 ATR（波動放大）
  tv_ratio       近 3 根 tick volume / 近 50 根平均
  ntouch         今天這條價位第幾次被碰
  session        亞洲 01-09、倫敦 09-15、紐約 15-20、尾盤 20-24（伺服器時間）
  strength       價位強度（KDE 峰值；其他來源為 1）

交易（一次一筆，依時間順序）：
  彈回單：在價位 ±0.1 ATR 限價反向進場，停損 1 ATR，停利 tp ATR；碰觸棒只算停損（保守）。
  突破單：同一價格順著來的方向進場（停損單），停損 1 ATR（反彈方向），停利 tp ATR；
          碰觸棒只算往突破方向的移動；額外扣 0.05 ATR 滑價。
  每筆扣點差。最多 24 根。
模型：邏輯迴歸，前半訓練 → 後半測試、後半訓練 → 前半測試（兩個方向都看）。
  門檻在訓練期挑（p_bounce > θ 做彈回、< 1−θ 做突破），測試期直接套用。
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from load import load
from td import atr
from kde_levels import find_pins
import part30_kde as P30
from part30_kde import touches, outcomes, build_levels, shift_levels, SPLIT

H, SLIP = 24, 0.05
START = pd.Timestamp("2026-02-02")
FEATS = ["run6", "run12", "er12", "appr_bars", "htf", "htf_slope", "day_ext", "vol_ratio",
         "tv_ratio", "ntouch", "strength"]
SESS = ["asia", "london", "ny", "late"]


def resample(m1, rule):
    return m1.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "tickvol": "sum",
         "spread": "max"}).dropna()


def day_slices(x):
    day = x.index.normalize()
    sl = {}
    for d in day.unique():
        idx = np.flatnonzero(day == d)
        sl[d] = (idx[0], idx[-1] + 1)
    return sl


def simple_levels(x, kind):
    """前日高低點 / 整數關卡。回傳 dict day -> DataFrame(price, strength, h)。"""
    daily = x.resample("1D").agg({"high": "max", "low": "min"}).dropna()
    out = {}
    prev = None
    for d, r in daily.iterrows():
        if prev is not None:
            if kind == "pdhl":
                p = [prev.high, prev.low]
            else:
                step = 50 if kind == "r50" else 10
                lo, hi = prev.low - 2 * step, prev.high + 2 * step
                p = list(np.arange(np.ceil(lo / step) * step, hi, step))
            out[d] = pd.DataFrame({"price": p, "strength": 1.0, "h": 2.0})
        prev = r
    return out


def features(x, A, ev):
    o, h, l, c = (x[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    tv = x.tickvol.to_numpy(float)
    ema = pd.Series(c).ewm(span=240, adjust=False).mean().to_numpy()
    a_day = pd.Series(A).rolling(288, min_periods=50).mean().to_numpy()
    tvm = pd.Series(tv).rolling(50, min_periods=10).mean().to_numpy()
    day = x.index.normalize()
    first = pd.Series(np.arange(len(c)), index=x.index).groupby(day).transform("min").to_numpy()
    t = ev.t.to_numpy()
    d = ev.dir.to_numpy()
    L = ev.L.to_numpy()
    a = ev.atr.to_numpy()
    p = t - 1
    f = pd.DataFrame(index=ev.index)
    f["run6"] = (c[np.maximum(p - 6, 0)] - c[p]) * d / a
    f["run12"] = (c[np.maximum(p - 12, 0)] - c[p]) * d / a
    path = np.array([np.abs(np.diff(c[max(i - 12, 0):i + 1])).sum() for i in p])
    f["er12"] = np.abs(c[np.maximum(p - 12, 0)] - c[p]) / np.where(path > 0, path, np.nan)
    ab = []
    for ti, di, Li, ai in zip(t, d, L, a):
        j = ti - 1
        while j > max(ti - 288, 0) and (c[j] - Li) * di < P30.ARM * ai:
            j -= 1
        ab.append(ti - j)
    f["appr_bars"] = ab
    f["htf"] = (c[p] - ema[p]) * d / a
    f["htf_slope"] = (ema[p] - ema[np.maximum(p - 48, 0)]) * d / a
    f["day_ext"] = (o[first[t]] - c[p]) * d / a
    f["vol_ratio"] = A[p] / a_day[p]
    f["tv_ratio"] = (tv[p] + tv[np.maximum(p - 1, 0)] + tv[np.maximum(p - 2, 0)]) / 3 / tvm[p]
    hr = x.index[t].hour
    f["session"] = np.select([hr < 9, hr < 15, hr < 20], ["asia", "london", "ny"], "late")
    key = pd.Series(list(zip(day[t], np.round(L, 2))), index=ev.index)
    f["ntouch"] = key.groupby(key).cumcount().to_numpy() + 1
    return f


def trade_R(x, ev, side, tp):
    """side = +1 彈回單、-1 突破單。回傳 (R, exit index)，未扣一次一筆的限制。"""
    h, l, c = (x[k].to_numpy(float) for k in ("high", "low", "close"))
    spr = x.spread.to_numpy(float) * 0.01
    n = len(c)
    R, EX = [], []
    for t, d, L, a in zip(ev.t.to_numpy(), ev.dir.to_numpy(), ev.L.to_numpy(), ev.atr.to_numpy()):
        ref = L + d * P30.TOL * a
        s = d * side  # 交易方向
        stop, tgt = ref - s * a, ref + s * tp * a
        end = min(t + H, n - 1)
        ex, px = end, c[end]
        for j in range(t, end + 1):
            hit_stop = (l[j] <= stop) if s == 1 else (h[j] >= stop)
            hit_tgt = (h[j] >= tgt) if s == 1 else (l[j] <= tgt)
            if j == t:  # 碰觸棒：只算「往原本行進方向」的那一邊
                hit_stop = hit_stop and side == 1
                hit_tgt = hit_tgt and side == -1
            if hit_stop:
                ex, px = j, stop
                break
            if hit_tgt:
                ex, px = j, tgt
                break
        cost = spr[t] + (SLIP * a if side == -1 else 0)
        R.append(((px - ref) * s - cost) / a)
        EX.append(ex)
    return np.array(R), np.array(EX)


def one_at_a_time(t, ex, take):
    keep = np.zeros(len(t), bool)
    busy = -1
    for i in np.argsort(t, kind="stable"):
        if take[i] and t[i] > busy:
            keep[i] = True
            busy = ex[i]
    return keep


def pf_stats(R, ndays):
    R = R[~np.isnan(R)]
    if len(R) < 5:
        return dict(n=len(R), per_day=np.nan, win=np.nan, avgR=np.nan, t=np.nan, PF=np.nan, totR=np.nan, maxDD=np.nan)
    eq = np.cumsum(R)
    dd = (np.maximum.accumulate(np.r_[0, eq]) - np.r_[0, eq]).max()
    return dict(n=len(R), per_day=len(R) / ndays, win=(R > 0).mean(), avgR=R.mean(),
                t=R.mean() / R.std(ddof=1) * np.sqrt(len(R)),
                PF=R[R > 0].sum() / -R[R < 0].sum(), totR=R.sum(), maxDD=dd)


def design(df, scaler=None):
    X = df[FEATS].copy()
    X["appr_bars"] = np.log1p(X.appr_bars)
    X["tv_ratio"] = np.log(X.tv_ratio.clip(0.05, 20))
    X["vol_ratio"] = np.log(X.vol_ratio.clip(0.05, 20))
    X["ntouch"] = np.minimum(X.ntouch, 4)
    for s in SESS[1:]:
        X["s_" + s] = (df.session == s).astype(float)
    for s in SOURCES[1:]:
        X["src_" + s] = (df.src == s).astype(float)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0).clip(-10, 10)
    if scaler is None:
        scaler = StandardScaler().fit(X)
    return scaler.transform(X), scaler, list(X.columns)


SOURCES = ["kde15", "kde60", "pdhl", "r50", "r10"]


def main():
    m1 = load("data/XAUUSD_M1_2026.csv")
    x = resample(m1, "5min")
    A = atr(x)
    sl = {d: v for d, v in day_slices(x).items() if d >= START}
    days = list(sl)
    P30.RNG = np.random.default_rng(3)
    d15 = resample(m1, "15min")
    d60 = resample(m1, "1h")
    lv = {
        "kde15": build_levels(find_pins(d15, atr(d15)), days, "expanding", None, 0.5),
        "kde60": build_levels(find_pins(d60, atr(d60)), days, "rolling", 5, 0.5),
        "pdhl": simple_levels(x, "pdhl"),
        "r50": simple_levels(x, "r50"),
        "r10": simple_levels(x, "r10"),
    }
    lv["placebo"] = shift_levels(lv["kde15"])
    evs = []
    for src, L in lv.items():
        ev = outcomes(x, touches(x, A, sl, L))
        ev["src"] = src
        evs.append(ev)
        print(src, len(ev), flush=True)
    ev = pd.concat(evs, ignore_index=True)
    ev = ev.join(features(x, A, ev))
    ev["time"] = x.index[ev.t]
    ev["half"] = np.where(ev.time < SPLIT, 1, 2)
    ev["bounce"] = ev.race1
    for side, nm in ((1, "bR"), (-1, "kR")):
        for tp in (1, 2):
            r, e = trade_R(x, ev, side, tp)
            ev[f"{nm}{tp}"] = r
            ev[f"{nm}{tp}_ex"] = e
    ev.to_pickle("/tmp/claude-0/-home-user-20061112/ec30ee8c-5c52-5218-87f6-11d2155d3637/scratchpad/p33_events.pkl")

    lines = []
    P = lines.append
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 400)
    nd = {0: len(days), 1: sum(d < SPLIT for d in days), 2: sum(d >= SPLIT for d in days)}

    P("=== 1. 各來源：碰觸數、彈回率（先反彈 1 ATR）、兩種單的平均 R（每次碰觸都做、未限制一次一筆）===")
    g = ev.groupby(["src", "half"]).agg(n=("t", "size"), bounce=("bounce", "mean"),
                                        bR2=("bR2", "mean"), kR2=("kR2", "mean"))
    P(g.round(3).to_string())

    P("\n=== 2. 單一特徵五分位的彈回率（真實價位 kde15+kde60+pdhl+r50+r10 合併；前半 / 後半；placebo）===")
    real = ev[ev.src != "placebo"]
    plc = ev[ev.src == "placebo"]
    for fcol in FEATS[:-1]:
        try:
            q, bins = pd.qcut(real[fcol], 5, retbins=True, duplicates="drop")
        except ValueError:
            continue
        lab = [f"{bins[i]:.2f}~{bins[i + 1]:.2f}" for i in range(len(bins) - 1)]
        qq = pd.cut(real[fcol], bins, labels=lab, include_lowest=True)
        qp = pd.cut(plc[fcol], bins, labels=lab, include_lowest=True)
        t1 = real.groupby([qq, "half"], observed=True).bounce.mean().unstack()
        t1["n"] = real.groupby(qq, observed=True).size()
        t1["placebo"] = plc.groupby(qp, observed=True).bounce.mean()
        P(f"\n[{fcol}]\n" + t1.round(3).to_string())
    for col in ("session", "src"):
        t1 = ev.groupby([col, "half"]).bounce.mean().unstack()
        t1["n"] = ev.groupby(col).size()
        P(f"\n[{col}]\n" + t1.round(3).to_string())

    P("\n=== 3. 邏輯迴歸：雙向樣本外（真實價位）===")
    data = real.copy()
    out_rows = []
    for tr_h, te_h in ((1, 2), (2, 1)):
        tr = data[data.half == tr_h].dropna(subset=["bounce"])
        te = data[data.half == te_h].copy()
        Xtr, sc, cols = design(tr)
        m = LogisticRegression(C=0.3, max_iter=500).fit(Xtr, tr.bounce.astype(int))
        coef = pd.Series(m.coef_[0], index=cols).sort_values()
        P(f"\n訓練 {tr_h} 半 → 測試 {te_h} 半；係數（標準化後，正 = 較容易彈回）\n" + coef.round(3).to_string())
        ptr = m.predict_proba(Xtr)[:, 1]
        Xte, _, _ = design(te, sc)
        pte = m.predict_proba(Xte)[:, 1]
        te["p"] = pte
        tr = tr.assign(p=ptr)
        # 校準：測試期依 p 分五組的實際彈回率
        qb = pd.qcut(te.p, 5, duplicates="drop")
        P("測試期依預測機率分組的實際彈回率\n" +
          te.groupby(qb, observed=True).agg(n=("bounce", "size"), bounce=("bounce", "mean"),
                                            bR2=("bR2", "mean"), kR2=("kR2", "mean")).round(3).to_string())
        # 門檻：訓練期挑
        for tp in (1, 2):
            best = None
            for th in np.arange(0.50, 0.66, 0.02):
                take_b = tr.p > th
                take_k = tr.p < 1 - th
                r = np.where(take_b, tr[f"bR{tp}"], np.where(take_k, tr[f"kR{tp}"], np.nan))
                ex = np.where(take_b, tr[f"bR{tp}_ex"], tr[f"kR{tp}_ex"])
                keep = one_at_a_time(tr.t.to_numpy(), ex, ~np.isnan(r))
                s = pf_stats(r[keep], nd[tr_h])
                if s["n"] >= 100 and (best is None or s["avgR"] > best[1]["avgR"]):
                    best = (th, s)
            th = best[0]
            for nm, tk_b, tk_k in (("model", te.p > th, te.p < 1 - th),
                                   ("model 只做彈回", te.p > th, te.p < -1),
                                   ("model 只做突破", te.p > 9, te.p < 1 - th),
                                   ("全部做彈回", te.p > -1, te.p < -1),
                                   ("全部做突破", te.p > 9, te.p < 9)):
                r = np.where(tk_b, te[f"bR{tp}"], np.where(tk_k, te[f"kR{tp}"], np.nan))
                ex = np.where(tk_b, te[f"bR{tp}_ex"], te[f"kR{tp}_ex"])
                keep = one_at_a_time(te.t.to_numpy(), ex, ~np.isnan(r))
                s = pf_stats(r[keep], nd[te_h])
                out_rows.append(dict(train=tr_h, test=te_h, tp=tp, th=round(th, 2), rule=nm, **s,
                                     train_avgR=best[1]["avgR"] if nm == "model" else np.nan))
    res = pd.DataFrame(out_rows)
    P("\n=== 4. 樣本外交易（一次一筆、扣點差；突破單另扣 0.05 ATR 滑價）===\n" + res.round(3).to_string())
    res.to_csv("part33_oos.csv", index=False, float_format="%.4f")
    open("results_part33.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
