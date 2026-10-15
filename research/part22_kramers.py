"""第二十二部分：位能井 + Kramers 逃逸率（盤中區間邊緣的反向訊號）。

物理圖像：當天價格 = 位能井裡的布朗粒子。
  p(x)  ：當天到目前為止，價格在每個價位停留的時間比例（只用到當根收盤以前的資料）
  U(x)  ：Boltzmann 反演 U = -D_eq · ln p(x)
  D     ：擴散係數 = 每分鐘實現變異數
  ΔU    ：井底（停留最久的價位）到目前區間邊緣的位能差
  Kramers：k ∝ exp(-ΔU / D)

注意：若 U 與分母用同一個 D，ΔU/D = ln(p_max/p_edge)，D 完全消掉（K_shape）。
要讓「溫度升高」有意義，井是用「當天到目前的溫度 D_eq」形成的，現在的噪音是「最近 N 分鐘的 D_N」：
  K_temp = (D_eq / D_N) · ln(p_max / p_edge)
另外測：
  K_cur  ：p 取目前價格所在格子而不是區間邊緣
  K_harm ：井用二次（諧振子）近似，ΔU/D_eq = (x_edge-μ)² / (2σ_prof²)，再乘 D_eq/D_N
  基準（非物理）：vol_ratio = D_eq/D_N、z_poc = |close-POC| / sqrt(D_N·60)

停留時間算法（occ）：close（每分鐘收盤所在格 +1）、range（每分鐘 high~low 均分 1，類似 TPO）、
  tickvol（range 均分 tickvol，類似 volume profile）。
格子寬度 = c × 前一交易日 1 分鐘收盤差的標準差（c = 0.5 / 1 / 2）；平滑 = 高斯核 h 格（0 / 1 / 3）。
D 估計：cc（收盤差平方）、pk（Parkinson，(H-L)²/(4 ln2)）；N = 15 / 30 / 60。

事件：開盤後 120 分鐘起、22:00 前，收盤落在當天區間上/下 20% 內（同一邊冷卻 30 分鐘），
  區間 ≥ 4·S（S = sqrt(D_eq·60)，當天一小時的擴散尺度）。
結果（只用收盤路徑、同一交易日、最長 240 分鐘；鞅的虛無假設下都是 0.5）：
  reb05 / reb10：先往區間內走 0.5S / 1S，還是先往外走同樣距離
  rebpoc       ：先回到參考 POC（range、c=1、h=1），還是先往外走同樣距離
  fwd60        ：60 分鐘後往區間內的報酬（S 單位）
輸出：part22_events.pkl
"""
import sys
import numpy as np
import pandas as pd
from multiprocessing import Pool
from load import load

DAY_START_H = 1
WARMUP = 120
LAST_MIN = (22 - DAY_START_H) * 60
COOL = 30
ZONE = 0.20
HORIZON = 240
OCCS = ["close", "range", "tickvol"]
CS = [0.5, 1.0, 2.0]
HS = [0, 1, 3]
NS = [15, 30, 60]
REF = ("range", 1.0, 1)


def gauss_kernel(h):
    if h == 0:
        return None
    x = np.arange(-3 * h, 3 * h + 1)
    k = np.exp(-0.5 * (x / h) ** 2)
    return k / k.sum()


KERN = {h: gauss_kernel(h) for h in HS}


def occupancy(lo_b, hi_b, cl_b, tv, occ, nb):
    """每根 K 棒對各格子的停留時間（T × nb），回傳累積（含當根）。"""
    T = len(lo_b)
    M = np.zeros((T, nb + 1))
    r = np.arange(T)
    if occ == "close":
        M[r, cl_b] += 1.0
    else:
        w = 1.0 / (hi_b - lo_b + 1)
        if occ == "tickvol":
            w = w * tv
        np.add.at(M, (r, lo_b), w)
        np.add.at(M, (r, hi_b + 1), -w)
        M = np.cumsum(M, axis=1)
    return np.cumsum(M[:, :nb], axis=0)


def day_events(args):
    g, sig_prev = args
    T = len(g)
    if T < 600 or not np.isfinite(sig_prev) or sig_prev <= 0:
        return []
    o, h, l, c = (g[k].to_numpy() for k in ("open", "high", "low", "close"))
    tv = g.tickvol.to_numpy().astype(float)
    sp = g.spread.to_numpy() * 0.01
    mins = ((g.index.hour - DAY_START_H) * 60 + g.index.minute).to_numpy()
    runH = np.maximum.accumulate(h)
    runL = np.minimum.accumulate(l)
    dc = np.diff(c, prepend=o[0])
    r2 = dc ** 2
    pk = (h - l) ** 2 / (4 * np.log(2))
    cr2, cpk = np.cumsum(r2), np.cumsum(pk)
    Deq_cc = cr2 / np.arange(1, T + 1)
    Deq_pk = cpk / np.arange(1, T + 1)

    def Drec(cs, N):
        out = np.empty(T)
        out[:N] = cs[:N] / np.arange(1, N + 1)
        out[N:] = (cs[N:] - cs[:-N]) / N
        return out

    DN = {(e, N): Drec(cs, N) for e, cs in (("cc", cr2), ("pk", cpk)) for N in NS}

    # ---- 事件 ----
    ev = []
    last = {1: -10 ** 9, -1: -10 ** 9}
    for t in range(T):
        if mins[t] < WARMUP or mins[t] >= LAST_MIN:
            continue
        rng = runH[t] - runL[t]
        S = np.sqrt(Deq_cc[t] * 60)
        if rng < 4 * S or S <= 0:
            continue
        pos = (c[t] - runL[t]) / rng
        side = 1 if pos >= 1 - ZONE else (-1 if pos <= ZONE else 0)   # +1 = 上緣（反向做空）
        if side == 0 or t - last[side] < COOL:
            continue
        last[side] = t
        ev.append((t, side, pos, S))
    if not ev:
        return []

    # ---- 停留時間剖面 ----
    base = l.min()
    rows = [dict(t=t, side=s, pos=pos, S=S) for t, s, pos, S in ev]
    tt = np.array([e[0] for e in ev])
    for occ in OCCS:
        for cw in CS:
            bw = cw * sig_prev
            lo_b = np.floor((l - base) / bw).astype(int)
            hi_b = np.floor((h - base) / bw).astype(int)
            cl_b = np.floor((c - base) / bw).astype(int)
            nb = hi_b.max() + 2
            C = occupancy(lo_b, hi_b, cl_b, tv, occ, nb)
            for k, (t, s, pos, S) in enumerate(ev):
                a, b = lo_b[:t + 1].min(), hi_b[:t + 1].max()
                raw = C[t, a:b + 1]
                for hh in HS:
                    key = f"{occ}_c{cw}_h{hh}"
                    p = raw if KERN[hh] is None else np.convolve(raw, KERN[hh])[3 * hh:3 * hh + len(raw)]
                    p = p / p.sum()
                    floor_ = 1e-4 / len(p)
                    p = np.maximum(p, floor_)
                    p = p / p.sum()
                    i_bot = int(np.argmax(p))
                    i_edge = (b - a) if s == 1 else 0
                    i_cur = int(np.clip(cl_b[t] - a, 0, b - a))
                    x = base + (np.arange(a, b + 1) + 0.5) * bw
                    mu = (p * x).sum()
                    sd = np.sqrt((p * (x - mu) ** 2).sum())
                    x_edge = runH[t] if s == 1 else runL[t]
                    R = rows[k]
                    R[f"{key}_sh"] = np.log(p[i_bot] / p[i_edge])
                    R[f"{key}_cur"] = np.log(p[i_bot] / p[i_cur])
                    R[f"{key}_harm"] = (x_edge - mu) ** 2 / (2 * sd ** 2) if sd > 0 else np.nan
                    R[f"{key}_poc"] = x[i_bot]
    # ---- 溫度與結果 ----
    for k, (t, s, pos, S) in enumerate(ev):
        R = rows[k]
        R["time"] = g.index[t]
        R["min"] = mins[t]
        R["close"] = c[t]
        R["edge"] = runH[t] if s == 1 else runL[t]
        R["spread"] = sp[t]
        R["Deq_cc"], R["Deq_pk"] = Deq_cc[t], Deq_pk[t]
        for (e, N), arr in DN.items():
            R[f"D_{e}{N}"] = arr[t]
        R["newext"] = int((h[t] >= runH[t - 1]) if s == 1 else (l[t] <= runL[t - 1]))
        # 往區間內為正
        end = min(T, t + 1 + HORIZON)
        path = -s * (c[t + 1:end] - c[t])
        R["fwd30"] = path[29] / S if len(path) >= 30 else np.nan
        R["fwd60"] = path[59] / S if len(path) >= 60 else np.nan
        R["fwd120"] = path[119] / S if len(path) >= 120 else np.nan
        for name, dist in (("reb05", 0.5 * S), ("reb10", 1.0 * S),
                           ("rebpoc", abs(c[t] - R["%s_c%s_h%d_poc" % REF]))):
            if dist <= 0:
                R[name] = np.nan
                continue
            up = np.nonzero(path >= dist)[0]
            dn = np.nonzero(path <= -dist)[0]
            iu = up[0] if len(up) else 10 ** 9
            idn = dn[0] if len(dn) else 10 ** 9
            R[name] = np.nan if iu == idn else float(iu < idn)
        # 交易用：OHLC 路徑（之後模擬）
        R["t_idx"] = t
    return rows


def load_all():
    d = load("data/XAUUSD_M1_2023_2026.csv")
    d = d[~d.index.duplicated()]
    d["day"] = (d.index - pd.Timedelta(hours=DAY_START_H)).normalize()
    d = d[d.index.hour >= DAY_START_H]
    return d


def main():
    d = load_all()
    days = [g for _, g in d.groupby("day")]
    sig = [g.close.diff().std() for g in days]
    tasks = [(days[i], sig[i - 1]) for i in range(1, len(days))]
    with Pool(4) as pool:
        res = pool.map(day_events, tasks, chunksize=8)
    E = pd.DataFrame([r for rr in res for r in rr])
    E["day"] = (E.time - pd.Timedelta(hours=DAY_START_H)).dt.normalize()
    E.to_pickle("part22_events.pkl")
    print(len(E), "events,", E.day.nunique(), "days")


if __name__ == "__main__":
    main()
