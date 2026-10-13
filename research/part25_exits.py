"""第二十五部分：(1) 把第二十四部分的 49 種型態搬到 M1 / M5 / M15（M30 對照）；
(2) 改變停損方式（型態端點 / 0.5~3 ATR）、盈虧比（0.5~3R、不設目標）、最長持有（6 / 24 根）能不能讓型態變賺。
進場：型態收盤價。路徑：一律用 M1 判斷先碰停損還是目標（同一分鐘都碰到算停損）。成本：當根點差。
基準：隨機抽 K 棒、多空都做，套同一組出場 → 「沒有進場優勢時，出場參數能做出什麼」。
"""
import numpy as np
import pandas as pd
from load import load
from part24_candles import P, patterns, sh, AGG, SPLIT

TFS = [("M1", "1min", 1), ("M5", "5min", 5), ("M15", "15min", 15), ("M30", "30min", 30)]
STOPS = ["型態端點", 0.5, 1.0, 1.5, 2.0, 3.0]
RRS = [0.5, 1.0, 1.5, 2.0, 3.0, None]          # None = 不設目標，只靠停損 / 時間出場
HOLDS = [6, 24]
rng = np.random.default_rng(1)


def simulate(idx, side_p, entry, atr, cost, plow, st, M, mins, hold):
    """回傳 dict[(stop, rr)] = 每筆 R 陣列（nan = 無效）。M = 鏡像後的 M1 (h, l, c)。"""
    mh, ml, mc = M
    W = hold * mins
    if len(idx) == 0:
        return {}
    a = st[np.minimum(idx + 1, len(st) - 1)]
    e = st[np.minimum(idx + 1 + hold, len(st) - 1)]
    full = (e - a) >= W * 0.5                         # 視窗至少要有一半的分鐘（排除週末前）
    j = a[:, None] + np.arange(W)[None, :]
    inwin = j < e[:, None]
    j = np.minimum(j, len(mh) - 1)
    A = atr[idx][:, None]
    up = np.where(inwin, (mh[j] - entry[:, None]) / A, -np.inf)
    dn = np.where(inwin, (entry[:, None] - ml[j]) / A, -np.inf)
    cup = np.maximum.accumulate(up, axis=1)
    cdn = np.maximum.accumulate(dn, axis=1)
    last = (mc[np.minimum(e - 1, len(mc) - 1)] - entry) / atr[idx]
    c_atr = cost[idx] / atr[idx]
    out = {}
    for s in STOPS:
        sd = np.maximum((entry - plow) / atr[idx], 0.05) if s == "型態端點" else np.full(len(idx), s)
        fs = (cdn < sd[:, None]).sum(1)              # 第一次碰停損的分鐘（W = 沒碰）
        for rr in RRS:
            if rr is None:
                ft = np.full(len(idx), W + 1)
            else:
                ft = (cup < (rr * sd)[:, None]).sum(1)
            r = np.where(fs <= ft, np.where(fs < W, -1.0, np.nan), rr if rr is not None else np.nan)
            timeout = (fs >= W) & (ft >= W)
            r = np.where(timeout, np.clip(last / sd, -1, None), r)
            r = r - c_atr / sd
            r[~full] = np.nan
            out[(s, rr)] = (r, sd)
    return out


def run_tf(m1, tf, rule, mins, rows):
    df = m1 if rule == "1min" else m1.resample(rule, label="left", closed="left").agg(AGG).dropna()
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    atr = pd.Series(h - l).rolling(20).mean().to_numpy()
    cost = df.spread.to_numpy(float) * 0.01
    t = df.index
    gap = np.r_[np.inf, np.diff(t.values).astype("timedelta64[m]").astype(float)]
    contig = gap <= 2 * mins
    contig3 = contig & sh(contig.astype(float), 1).astype(bool)
    st = np.r_[np.searchsorted(m1.index.values, t.values), len(m1)]
    MH, ML, MC = (m1[k].to_numpy(float) for k in ("high", "low", "close"))
    per = (t >= SPLIT).astype(int)
    base_pool = np.where(contig & ~np.isnan(atr))[0]
    base_idx = np.sort(rng.choice(base_pool, size=min(60000, len(base_pool)), replace=False))
    for side in (1, -1):
        if side == 1:
            p, M = P(o, h, l, c, atr), (MH, ML, MC)
        else:
            p, M = P(-o, -l, -h, -c, atr), (-ML, -MH, -MC)
        pats = patterns(p)
        pats["隨機基準（所有K棒抽樣）"] = (np.isin(np.arange(len(c)), base_idx), 1, "基準")
        for pn, (mask, k, fam) in pats.items():
            ok = np.nan_to_num(mask, nan=0).astype(bool) & ~np.isnan(atr) & (contig3 if k == 3 else contig)
            idx = np.where(ok)[0]
            if len(idx) == 0:
                continue
            for hold in HOLDS:
                res = {}
                for i0 in range(0, len(idx), 3000 if mins >= 15 else 20000):
                    ii = idx[i0:i0 + (3000 if mins >= 15 else 20000)]
                    part = simulate(ii, side, p.c[ii], atr, cost, p.lowk(k)[ii], st, M, mins, hold)
                    for key, (r, sd) in part.items():
                        res.setdefault(key, []).append((r, sd))
                pp = per[idx]
                for (s, rr), lst in res.items():
                    r = np.concatenate([x[0] for x in lst])
                    sd = np.concatenate([x[1] for x in lst])
                    for q in (0, 1):
                        v = r[(pp == q) & ~np.isnan(r)]
                        va = (r * sd)[(pp == q) & ~np.isnan(r)]
                        rows.append(dict(tf=tf, pattern=pn, family=fam, side=side, stop=str(s), rr="無" if rr is None else rr, hold=hold, per=q,
                                         n=len(v), sumR=v.sum(), sumR2=(v ** 2).sum(), sumA=va.sum(), win=(v > 0).sum()))
    print(f"  {tf} 完成", flush=True)


def main():
    m1 = load("data/XAUUSD_M1_2023_2026.csv")
    rows = []
    for tf, rule, mins in TFS:
        run_tf(m1, tf, rule, mins, rows)
    pd.DataFrame(rows).to_csv("part25_exits_raw.csv.gz", index=False)


if __name__ == "__main__":
    main()
