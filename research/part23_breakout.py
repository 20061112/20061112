"""第二十三部分：反過來用 — 位能障礙高的時候順勢追突破；M1 / M3 / M5 / M15 各時框。

第二十二部分發現字面 ΔU/D = ln(p_max/p_邊緣) 越大，碰到邊緣後越「不」反彈。這裡把它當突破過濾：
  事件：該時框 K 棒收盤落在當天區間上/下 20% 內（開盤 120 分鐘後、22:00 前，同邊冷卻 30 分鐘，區間 ≥ 4S）
  方向：往區間外（上緣做多、下緣做空）
停留時間剖面：在該時框的 K 棒上累積（close / range / tickvol），格寬一律 = c × 前一天 M1 收盤差標準差，
  所以各時框的價位解析度相同，只差在「時間解析度」與 D 的估計。
D：每分鐘變異數（該時框收盤差平方 / 時框分鐘）。S = √(D_eq·60)。
結果（該時框收盤路徑；鞅虛無假設 = 0.5）：esc05 / esc10 = 先往外 0.5S / 1S 還是先往內同距離；fwd60 / fwd120 往外報酬（S）。
交易在 M1 上逐根模擬（part23_trade.py）。輸出 part23_events_M{tf}.pkl
"""
import numpy as np
import pandas as pd
from multiprocessing import Pool
from part22_kramers import load_all, occupancy, KERN, DAY_START_H

TFS = [1, 3, 5, 15]
OCCS = ["close", "range", "tickvol"]
CS = [0.5, 1.0, 2.0]
HS = [0, 1, 3]
WARMUP, LAST_MIN, COOL, HORIZON = 120, (22 - DAY_START_H) * 60, 30, 240
ZONE = 0.20


def resample(g, tf):
    if tf == 1:
        return g
    r = g.resample(f"{tf}min", label="left", closed="left")
    out = pd.DataFrame({"open": r.open.first(), "high": r.high.max(), "low": r.low.min(),
                        "close": r.close.last(), "tickvol": r.tickvol.sum(), "spread": r.spread.last()})
    return out.dropna()


def day_events(args):
    g1, sig_prev, tf = args
    if len(g1) < 600 or not np.isfinite(sig_prev) or sig_prev <= 0:
        return []
    g = resample(g1, tf)
    T = len(g)
    o, h, l, c = (g[k].to_numpy() for k in ("open", "high", "low", "close"))
    tv = g.tickvol.to_numpy().astype(float)
    sp = g.spread.to_numpy() * 0.01
    mins = ((g.index.hour - DAY_START_H) * 60 + g.index.minute).to_numpy()
    runH, runL = np.maximum.accumulate(h), np.minimum.accumulate(l)
    r2 = np.diff(c, prepend=o[0]) ** 2 / tf
    cr2 = np.cumsum(r2)
    Deq = cr2 / np.arange(1, T + 1)
    N = max(2, 30 // tf)
    D30 = np.empty(T)
    D30[:N] = cr2[:N] / np.arange(1, N + 1)
    D30[N:] = (cr2[N:] - cr2[:-N]) / N

    ev, last = [], {1: -10 ** 9, -1: -10 ** 9}
    for t in range(1, T):
        m_close = mins[t] + tf                       # 這根收盤的時間
        if m_close < WARMUP or m_close >= LAST_MIN:
            continue
        rng, S = runH[t] - runL[t], np.sqrt(Deq[t] * 60)
        if S <= 0 or rng < 4 * S:
            continue
        pos = (c[t] - runL[t]) / rng
        side = 1 if pos >= 1 - ZONE else (-1 if pos <= ZONE else 0)
        if side == 0 or m_close - last[side] < COOL:
            continue
        last[side] = m_close
        ev.append((t, side, pos, S))
    if not ev:
        return []

    base = l.min()
    rows = [dict(side=s, pos=pos, S=S) for t, s, pos, S in ev]
    for occ in OCCS:
        for cw in CS:
            bw = cw * sig_prev
            lo_b = np.floor((l - base) / bw).astype(int)
            hi_b = np.floor((h - base) / bw).astype(int)
            cl_b = np.floor((c - base) / bw).astype(int)
            C = occupancy(lo_b, hi_b, cl_b, tv, occ, hi_b.max() + 2)
            for k, (t, s, pos, S) in enumerate(ev):
                a, b = lo_b[:t + 1].min(), hi_b[:t + 1].max()
                raw = C[t, a:b + 1]
                for hh in HS:
                    p = raw if KERN[hh] is None else np.convolve(raw, KERN[hh])[3 * hh:3 * hh + len(raw)]
                    p = np.maximum(p / p.sum(), 1e-4 / len(p)); p = p / p.sum()
                    i_bot = int(np.argmax(p))
                    i_edge = (b - a) if s == 1 else 0
                    x = base + (np.arange(a, b + 1) + 0.5) * bw
                    mu = (p * x).sum(); sd = np.sqrt((p * (x - mu) ** 2).sum())
                    x_edge = runH[t] if s == 1 else runL[t]
                    key = f"{occ}_c{cw}_h{hh}"
                    rows[k][f"{key}_sh"] = np.log(p[i_bot] / p[i_edge])
                    rows[k][f"{key}_harm"] = (x_edge - mu) ** 2 / (2 * sd ** 2) if sd > 0 else np.nan
                    rows[k][f"{key}_poc"] = x[i_bot]
    hb = HORIZON // tf
    for k, (t, s, pos, S) in enumerate(ev):
        R = rows[k]
        R.update(time_close=g.index[t] + pd.Timedelta(minutes=tf), min=mins[t] + tf, close=c[t],
                 edge=runH[t] if s == 1 else runL[t], spread=sp[t], Deq=Deq[t], D30=D30[t],
                 newext=int((h[t] >= runH[t - 1]) if s == 1 else (l[t] <= runL[t - 1])))
        path = s * (c[t + 1:min(T, t + 1 + hb)] - c[t])          # 往外為正
        for mm in (60, 120):
            j = mm // tf - 1
            R[f"fwd{mm}"] = path[j] / S if len(path) > j else np.nan
        for name, dist in (("esc05", 0.5 * S), ("esc10", 1.0 * S)):
            up, dn = np.nonzero(path >= dist)[0], np.nonzero(path <= -dist)[0]
            iu = up[0] if len(up) else 10 ** 9
            idn = dn[0] if len(dn) else 10 ** 9
            R[name] = np.nan if iu == idn else float(iu < idn)
    return rows


def main():
    d = load_all()
    days = [g for _, g in d.groupby("day")]
    sig = [g.close.diff().std() for g in days]
    for tf in TFS:
        tasks = [(days[i], sig[i - 1], tf) for i in range(1, len(days))]
        with Pool(4) as pool:
            res = pool.map(day_events, tasks, chunksize=8)
        E = pd.DataFrame([r for rr in res for r in rr])
        E["day"] = (E.time_close - pd.Timedelta(minutes=1) - pd.Timedelta(hours=DAY_START_H)).dt.normalize()
        E.to_pickle(f"part23_events_M{tf}.pkl")
        print(f"M{tf}: {len(E)} events, {E.day.nunique()} days")


if __name__ == "__main__":
    main()
