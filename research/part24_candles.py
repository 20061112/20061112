"""第二十四部分：單根影線比例與經典 K 線型態（各種常見定義全部跑一遍），時框 M5 / M15 / M30 / H1。
- 每個型態都只寫「看漲版」，看跌版用鏡像價格（O,H,L,C → −O,−L,−H,−C）套同一個函式，定義完全對稱。
- 評估 1（收盤進場）：型態完成那根收盤進場，順型態方向持有 1/3/6/12 根，扣點差；並減去同期間同方向所有 K 棒的平均（扣掉黃金多頭漂移）。
- 評估 2（交易者實際做法）：下一根突破型態高點（看跌：低點）掛單進場，停損在型態另一端，目標 1R / 2R，最多 12 根，扣點差；
  同一根同時碰到停損與目標一律算停損。與「所有 K 棒用同樣方式進場」的基準相比。
- 發現期 2023/6~2024/12、驗證期 2025/1~2026/10；所有排名只用發現期。
"""
import numpy as np
import pandas as pd
from load import load

SPLIT = pd.Timestamp("2025-01-01")
TFS = [("M5", "5min", 5), ("M15", "15min", 15), ("M30", "30min", 30), ("H1", "1h", 60)]
AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "tickvol": "sum", "spread": "mean"}
HS = (1, 3, 6, 12)
MAXH = 12


def sh(a, k):
    """a[t-k]；前面補 nan。"""
    out = np.full_like(a, np.nan, dtype=float)
    out[k:] = a[:-k] if k else a
    return out


class P:
    """一組 OHLC（可能是鏡像）+ 衍生量。屬性 x1 / x2 = 前 1 / 2 根。"""

    def __init__(s, o, h, l, c, atr):
        s.o, s.h, s.l, s.c, s.atr = o, h, l, c, atr
        s.rng = h - l
        s.body = c - o                       # 有號：>0 陽線
        s.ab = np.abs(s.body)
        s.up = h - np.maximum(o, c)
        s.lo = np.minimum(o, c) - l
        with np.errstate(invalid="ignore", divide="ignore"):
            s.br = s.ab / s.rng               # 實體比例
            s.ur = s.up / s.rng
            s.lr = s.lo / s.rng
            s.cl = (c - l) / s.rng            # 收盤位置
        # 型態開始前的趨勢：前一根收盤 − 前 11 根收盤（單位 ATR）
        s.trend = lambda k: (sh(c, k + 1) - sh(c, k + 11)) / sh(atr, k + 1)
        s.ll20 = pd.Series(l).rolling(20).min().to_numpy()
        s.ll5p = sh(pd.Series(l).rolling(5).min().to_numpy(), 1)   # 前 5 根（不含這根）最低

    def prev(s, k, name):
        return sh(getattr(s, name), k)


def patterns(p):
    """回傳 {名稱: (看漲布林陣列, 型態根數, 類別)}。"""
    g = {}
    ATR = p.prev(1, "atr")                                  # 型態完成前的 ATR
    down0, down1, down2 = p.trend(0) <= -1.5, p.trend(1) <= -1.5, p.trend(2) <= -1.5
    up0 = p.trend(0) >= 1.5
    pos = p.rng > 0
    # ---------- 單根：錘子 / Pin bar ----------
    ham = pos & (p.lo >= 2 * p.ab) & (p.up <= 0.1 * p.rng)
    g["錘子A Nison（下影≥2×實體、上影≤10%）"] = (ham, 1, "Pin")
    g["錘子A＋下跌趨勢"] = (ham & down0, 1, "Pin")
    g["錘子B 下影≥3×實體、上影≤5%"] = (pos & (p.lo >= 3 * p.ab) & (p.up <= 0.05 * p.rng), 1, "Pin")
    pin = pos & (p.lr >= 2 / 3)
    g["PinC 下影≥2/3全長"] = (pin, 1, "Pin")
    g["PinC＋全長≥1ATR"] = (pin & (p.rng >= ATR), 1, "Pin")
    g["PinC＋陽線"] = (pin & (p.body > 0), 1, "Pin")
    g["PinC＋突出前5根低點（Nial Fuller）"] = (pin & (p.l < p.ll5p), 1, "Pin")
    g["PinC＋20根新低"] = (pin & (p.l <= p.ll20), 1, "Pin")
    g["PinC＋下跌趨勢"] = (pin & down0, 1, "Pin")
    g["PinD 收在上1/4、下影≥50%"] = (pos & (p.cl >= 0.75) & (p.lr >= 0.5), 1, "Pin")
    g["PinE 下影≥60%＋收在前一根範圍內"] = (pos & (p.lr >= 0.6) & (p.c <= p.prev(1, "h")) & (p.c >= p.prev(1, "l")), 1, "Pin")
    inv = pos & (p.up >= 2 * p.ab) & (p.lo <= 0.1 * p.rng)
    g["倒錘（＝鏡像吊人）＋下跌趨勢"] = (inv & down0, 1, "Pin")
    g["倒錘 不看趨勢"] = (inv, 1, "Pin")
    # ---------- 單根：光頭光腳（無影線）→ 順勢 ----------
    bull = p.body > 0
    g["光頭光腳A 實體≥95%"] = (bull & (p.br >= 0.95), 1, "無影線")
    g["光頭光腳B 實體≥90%"] = (bull & (p.br >= 0.90), 1, "無影線")
    g["光頭光腳C 實體≥90%＋實體≥1ATR"] = (bull & (p.br >= 0.90) & (p.ab >= ATR), 1, "無影線")
    g["光頭光腳D 完全無影線 O=L C=H"] = (bull & (p.up == 0) & (p.lo == 0), 1, "無影線")
    g["收盤光頭 C=H、實體≥50%"] = (bull & (p.up == 0) & (p.br >= 0.5), 1, "無影線")
    g["開盤光腳 O=L、實體≥50%"] = (bull & (p.lo == 0) & (p.br >= 0.5), 1, "無影線")
    g["光頭光腳B＋20根新高（突破）"] = (bull & (p.br >= 0.9) & (p.h >= pd.Series(p.h).rolling(20).max().to_numpy()), 1, "無影線")
    # ---------- 單根：十字線 / 陀螺 → 下跌後看漲 ----------
    g["十字A 實體≤10%＋下跌趨勢"] = (pos & (p.br <= 0.10) & down0, 1, "十字")
    g["十字B 實體≤5%＋下跌趨勢"] = (pos & (p.br <= 0.05) & down0, 1, "十字")
    drag = pos & (p.br <= 0.10) & (p.ur <= 0.10)
    g["蜻蜓十字 實體≤10%、上影≤10%"] = (drag, 1, "十字")
    g["蜻蜓十字＋下跌趨勢"] = (drag & down0, 1, "十字")
    g["長腳十字 兩影≥30%、全長≥1ATR＋下跌"] = (pos & (p.br <= 0.1) & (p.ur >= .3) & (p.lr >= .3) & (p.rng >= ATR) & down0, 1, "十字")
    g["陀螺 實體10~30%、兩影>實體＋下跌"] = (pos & (p.br > .1) & (p.br <= .3) & (p.up > p.ab) & (p.lo > p.ab) & down0, 1, "十字")
    # ---------- 兩根 ----------
    b1, o1, c1, h1, l1, ab1 = p.prev(1, "body"), p.prev(1, "o"), p.prev(1, "c"), p.prev(1, "h"), p.prev(1, "l"), p.prev(1, "ab")
    eng = (b1 < 0) & bull & (p.c >= o1) & (p.o <= c1) & (p.ab > ab1)
    g["吞噬A 實體吞噬"] = (eng, 2, "兩根")
    g["吞噬A＋下跌趨勢"] = (eng & down1, 2, "兩根")
    g["吞噬B 前根≥0.5ATR、這根≥1ATR"] = (eng & (ab1 >= 0.5 * ATR) & (p.ab >= ATR), 2, "兩根")
    g["吞噬C 全長吞噬（外包線）收在前高之上"] = ((b1 < 0) & bull & (p.h > h1) & (p.l < l1) & (p.c > h1), 2, "兩根")
    g["吞噬D 這根實體≥2倍前根"] = (eng & (p.ab >= 2 * ab1), 2, "兩根")
    har = (b1 < 0) & (ab1 >= 0.6 * sh(p.atr, 2)) & bull & (p.c <= o1) & (p.o >= c1) & (p.ab <= 0.5 * ab1)
    g["孕線 長陰後小陽在實體內"] = (har, 2, "兩根")
    g["孕線＋下跌趨勢"] = (har & down1, 2, "兩根")
    g["內包線 順母線方向（母線陽）"] = ((p.h < h1) & (p.l > l1) & (b1 > 0), 2, "兩根")
    g["貫穿線 長陰後陽線收過陰線中點"] = ((b1 < 0) & (ab1 >= 0.6 * sh(p.atr, 2)) & bull & (p.o <= c1) & (p.c > c1 + ab1 / 2) & (p.c < o1), 2, "兩根")
    g["平頭底 低點差≤0.05ATR＋下跌"] = ((b1 < 0) & bull & (np.abs(p.l - l1) <= 0.05 * ATR) & down1, 2, "兩根")
    # ---------- 三根：晨星 ----------
    bb1, oo1, cc1, aa1 = p.prev(2, "body"), p.prev(2, "o"), p.prev(2, "c"), p.prev(2, "ab")
    b2, a2, o2, c2, l2, h2, r2 = p.prev(1, "body"), p.prev(1, "ab"), p.prev(1, "o"), p.prev(1, "c"), p.prev(1, "l"), p.prev(1, "h"), p.prev(1, "rng")
    A3 = sh(p.atr, 3)
    mid1 = cc1 + aa1 / 2
    ms = (bb1 < 0) & (aa1 >= 0.6 * A3) & (a2 <= 0.3 * aa1) & bull & (p.c > mid1)
    g["晨星A 經典：長陰、小星≤30%、陽收過中點"] = (ms, 3, "晨星")
    g["晨星B 陽線收過第一根開盤（完全收復）"] = (ms & (p.c >= oo1), 3, "晨星")
    g["晨星C 星是三根最低"] = (ms & (l2 <= np.minimum(p.prev(2, "l"), p.l)), 3, "晨星")
    g["晨星D 十字星（星實體≤10%全長）"] = (ms & (a2 <= 0.1 * r2), 3, "晨星")
    g["晨星E ＋下跌趨勢"] = (ms & down2, 3, "晨星")
    g["晨星F 寬鬆：星實體<兩側的一半"] = ((bb1 < 0) & bull & (a2 < 0.5 * aa1) & (a2 < 0.5 * p.ab) & (p.c > mid1), 3, "晨星")
    g["晨星G 星實體在第一根收盤之下（跳空近似）"] = (ms & (np.maximum(o2, c2) <= cc1), 3, "晨星")
    g["晨星H 第三根≥1ATR"] = (ms & (p.ab >= ATR), 3, "晨星")
    # ---------- 三根：其他 ----------
    har_prev = (bb1 < 0) & (aa1 >= 0.6 * A3) & (b2 > 0) & (c2 <= oo1) & (o2 >= cc1) & (a2 <= 0.5 * aa1)
    g["三內升（孕線＋第三根收過第一根開盤）"] = (har_prev & bull & (p.c > oo1), 3, "三根")
    eng_prev = (bb1 < 0) & (b2 > 0) & (c2 >= oo1) & (o2 <= cc1) & (a2 > aa1)
    g["三外升（吞噬＋第三根再收高）"] = (eng_prev & bull & (p.c > c2), 3, "三根")
    sold = (bb1 > 0) & (b2 > 0) & bull & (c2 > cc1) & (p.c > c2)
    big3 = (aa1 >= 0.5 * A3) & (a2 >= 0.5 * sh(p.atr, 2)) & (p.ab >= 0.5 * ATR)
    g["紅三兵A 三根陽線≥0.5ATR、收在上1/4"] = (sold & big3 & (p.prev(2, "cl") >= .75) & (p.prev(1, "cl") >= .75) & (p.cl >= .75), 3, "三根")
    g["紅三兵B 開盤在前一根實體內"] = (sold & big3 & (o2 >= oo1) & (o2 <= cc1) & (p.o >= o2) & (p.o <= c2), 3, "三根")
    g["紅三兵C 寬鬆：三根陽線收盤遞增"] = (sold, 3, "三根")
    return g


def fwd_matrix(a, idx, k):
    """a[idx+1 .. idx+k]，超出範圍補 nan。"""
    j = idx[:, None] + np.arange(1, k + 1)[None, :]
    out = np.full(j.shape, np.nan)
    m = j < len(a)
    out[m] = a[j[m]]
    return out


def trade_r(p, idx, cost, valid, tgt, mp, st, chunk=4000):
    """分批呼叫 _trade_r，避免記憶體爆掉。"""
    if len(idx) <= chunk:
        return _trade_r(p, idx, cost, valid, tgt, mp, st)
    return np.concatenate([_trade_r(p, idx[i:i + chunk], cost[i:i + chunk], valid[i:i + chunk], tgt, mp, st)
                           for i in range(0, len(idx), chunk)])


def _trade_r(p, idx, cost, valid, tgt, mp, st):
    """下一根突破型態高點進場、停損型態低點、目標 tgt R、最多 MAXH 根。用 M1 路徑判斷先後。
    mp = (o, h, l, c) M1 陣列（與 p 同方向，鏡像時也鏡像）；st = 每根 TF K 棒第一根 M1 的位置（多一格結尾）。
    同一分鐘同時碰到停損與目標算停損。回傳每筆 R（未觸發 = nan）。"""
    mo, mh, ml, mc = mp
    out = np.full(len(idx), np.nan)
    if len(idx) == 0:
        return out
    a, b1 = st[idx + 1], st[np.minimum(idx + 2, len(st) - 1)]
    e = st[np.minimum(idx + 1 + MAXH, len(st) - 1)]
    W = int((e - a).max())
    j = a[:, None] + np.arange(W)[None, :]
    inwin = j < e[:, None]
    j = np.minimum(j, len(mh) - 1)
    H, L, O, C = mh[j], ml[j], mo[j], mc[j]
    ph, pl = p.h[idx], p.l[idx]
    entry_zone = j < b1[:, None]
    trig_m = (H > ph[:, None]) & entry_zone & inwin
    trig = trig_m.any(1) & valid
    k0 = trig_m.argmax(1)
    entry = np.maximum(O[np.arange(len(idx)), k0], ph)
    R = entry - pl
    target = entry + tgt * R
    after = (np.arange(W)[None, :] >= k0[:, None]) & inwin
    hit_s = (L <= pl[:, None]) & after
    hit_t = (H >= target[:, None]) & after & ~hit_s
    fs = np.where(hit_s.any(1), hit_s.argmax(1), 10 ** 6)
    ft = np.where(hit_t.any(1), hit_t.argmax(1), 10 ** 6)
    last = C[np.arange(len(idx)), np.maximum((inwin.sum(1) - 1), 0)]
    res = np.where((fs == 10 ** 6) & (ft == 10 ** 6), (last - entry) / np.where(R > 0, R, np.nan), np.where(fs <= ft, -1.0, float(tgt)))
    res = res - cost / np.where(R > 0, R, np.nan)
    ok = trig & (R > 0)
    out[ok] = res[ok]
    return out


def evaluate(name, tf, mins, df, m1):
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    rng_ = h - l
    atr = pd.Series(rng_).rolling(20).mean().to_numpy()
    cost = df.spread.to_numpy(float) * 0.01
    t = df.index
    gap = np.r_[np.inf, np.diff(t.values).astype("timedelta64[m]").astype(float)]
    contig = gap <= 2 * mins                                     # 這根與前一根相連
    contig3 = contig & sh(contig.astype(float), 1).astype(bool) & sh(contig.astype(float), 2).astype(bool)
    fut_ok = {}
    tm = t.values.astype("datetime64[m]").astype(np.int64)
    for hh in list(HS) + [MAXH]:
        span = np.full(len(t), np.inf)
        span[:-hh] = tm[hh:] - tm[:-hh]
        fut_ok[hh] = span <= hh * mins + 65
    st = np.r_[np.searchsorted(m1.index.values, t.values), len(m1)]
    M = [m1[k].to_numpy(float) for k in ("open", "high", "low", "close")]
    per = np.where(t < SPLIT, 0, 1)
    year = t.year.to_numpy()
    rows, trades = [], {}
    for side in (1, -1):
        if side == 1:
            p = P(o, h, l, c, atr)
            mp = (M[0], M[1], M[2], M[3])
        else:
            p = P(-o, -l, -h, -c, atr)
            mp = (-M[0], -M[2], -M[1], -M[3])
        pats = patterns(p)
        # 漂移基準：同方向所有 K 棒持有 h 根的平均（分期間）
        base = {hh: np.where(fut_ok[hh], (np.r_[c[hh:], [np.nan] * hh] - c) * side / atr, np.nan) for hh in HS}
        drift = {(hh, q): np.nanmean(base[hh][per == q]) for hh in HS for q in (0, 1)}
        all_idx = np.where(contig & ~np.isnan(atr) & fut_ok[MAXH])[0]
        for tg in (1, 2):
            br = trade_r(p, all_idx, cost[all_idx], np.ones(len(all_idx), bool), tg, mp, st)
            for q in (0, 1):
                m = per[all_idx] == q
                trades[("基準", side, tg, q)] = np.nanmean(br[m])
        for pn, (mask, k, fam) in pats.items():
            ok = np.nan_to_num(mask, nan=0).astype(bool) & ~np.isnan(atr) & (contig3 if k == 3 else contig if k == 2 else True)
            idx = np.where(ok)[0]
            rec = dict(tf=tf, pattern=pn, family=fam, side=side, n=len(idx), per_day=len(idx) / (len(df) * mins / 1380))
            for hh in HS:
                r = base[hh][idx] - cost[idx] / atr[idx]
                for q in (0, 1):
                    m = per[idx] == q
                    rec[f"r{hh}_{q}"] = np.nanmean(r[m]) if m.sum() else np.nan
                    rec[f"x{hh}_{q}"] = rec[f"r{hh}_{q}"] - drift[(hh, q)] + np.nanmean((cost[idx] / atr[idx])[m]) if m.sum() else np.nan   # 未扣成本的超額
                    rec[f"n_{q}"] = int(m.sum())
                    rr = r[m]
                    rr = rr[~np.isnan(rr)]
                    rec[f"t{hh}_{q}"] = rr.mean() / rr.std() * np.sqrt(len(rr)) if len(rr) > 2 else np.nan
            nb = base[1][idx]
            rec["win1"] = np.nanmean(nb[~np.isnan(nb)] > 0)
            idx2 = idx[fut_ok[MAXH][idx]]
            for tg in (1, 2):
                rr = trade_r(p, idx2, cost[idx2], np.ones(len(idx2), bool), tg, mp, st)
                for q in (0, 1):
                    m = per[idx2] == q
                    v = rr[m]
                    v = v[~np.isnan(v)]
                    rec[f"R{tg}_{q}"] = v.mean() if len(v) else np.nan
                    rec[f"Rn{tg}_{q}"] = len(v)
                    rec[f"Rb{tg}_{q}"] = trades[("基準", side, tg, q)]
                    rec[f"Rt{tg}_{q}"] = v.mean() / v.std() * np.sqrt(len(v)) if len(v) > 2 else np.nan
                yy = rr[~np.isnan(rr)]
                rec[f"Ryear{tg}"] = " ".join(f"{y}:{np.mean(yy_):+.2f}" for y in sorted(set(year[idx2]))
                                            for yy_ in [rr[(year[idx2] == y) & ~np.isnan(rr)]] if len(yy_))
            rows.append(rec)
    return rows


def main():
    m1 = load("data/XAUUSD_M1_2023_2026.csv")
    rows = []
    for tf, rule, mins in TFS:
        df = m1.resample(rule, label="left", closed="left").agg(AGG).dropna()
        rows += evaluate(tf, tf, mins, df, m1)
    R = pd.DataFrame(rows)
    R.to_csv("part24_patterns.csv", index=False)
    print(f"資料 {m1.index[0]} ~ {m1.index[-1]}；發現期 < {SPLIT.date()} ≤ 驗證期")
    print(f"型態定義 {R.pattern.nunique()} 種 × 時框 4 × 多/空 = {len(R)} 列（明細：part24_patterns.csv）")


if __name__ == "__main__":
    main()
