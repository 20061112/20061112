"""NAS100 IB 核心引擎（第三十七部分起）—— 以 XAU `core_ib.py` 的 IB 突破核心為雛形，改成可掃描任意時段 / 長度 / 方向。

資料：data/NAS100_M1_2023_2024.csv + 2024_2025 + 2025_2026（2023/10/2 ~ 2026/10/16），broker 時間 = 紐約 + 7
     （16:30 = 美股現貨開盤，23:00 = 現貨收盤）。重疊日以較新的檔為準。
交易日 = broker 01:00 ~ 23:59（M1 ≥ 1000 根才算）；ATR10 = 前 10 天日線真實波幅平均（不含當天）。
點差：<SPREAD> × 0.1 指數點，每筆扣進場那根 M1 的點差（+ 可選額外滑價 SLIP 點）。
結果一律以 R（停損距離）與 ATR 倍數表示 —— 三年間指數從 14,800 漲到 29,000，絕對點數不可比。

期間：研究期 2023/10 ~ 2025/09（四個半年，用來挑參數）；保留期 2025/10 ~ 2026/10（最後才打開）。
"""
import os
import numpy as np
import pandas as pd
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = [("data/NAS100_M1_2023_2024.csv", "2024-10-01 01:00"), ("data/NAS100_M1_2024_2025.csv", "2025-10-01 01:00"),
         ("data/NAS100_M1_2025_2026.csv", None)]
PT = 0.1                      # 1 spread 單位 = 0.1 指數點
NMIN = 1500                   # 每天的分鐘格（從 broker 00:00 起算，交易日 01:00 ~ 次日 00:59 以內）
HALVES = [("23Q4-24Q1", "2023-10-01", "2024-04-01"), ("24Q2-Q3", "2024-04-01", "2024-10-01"),
          ("24Q4-25Q1", "2024-10-01", "2025-04-01"), ("25Q2-Q3", "2025-04-01", "2025-10-01")]
RESEARCH = ("2023-10-01", "2025-10-01")
HOLDOUT = ("2025-10-01", "2026-11-01")
CACHE = os.path.join(HERE, "data", "nas_core_cache.npz")


def _read(path):
    df = pd.read_csv(os.path.join(HERE, path), sep="\t")
    df.columns = [c.strip("<>").lower() for c in df.columns]
    df.index = pd.to_datetime(df["date"] + " " + df["time"], format="%Y.%m.%d %H:%M:%S")
    return df[["open", "high", "low", "close", "spread"]]


def build():
    parts = []
    for path, end in FILES:
        d = _read(path)
        if end:
            d = d[d.index < pd.Timestamp(end)]
        parts.append(d)
    m = pd.concat(parts)
    m = m[~m.index.duplicated(keep="last")].sort_index()
    m["day"] = (m.index - pd.Timedelta(hours=1)).normalize()
    days = []
    for day, g in m.groupby("day"):
        if len(g) >= 1000:
            days.append(day)
    days = pd.DatetimeIndex(days)
    m = m[m.day.isin(days)]
    di = np.searchsorted(days.values, m.day.values)
    mi = ((m.index.values - m.day.values) // np.timedelta64(1, "m")).astype(int)
    shape = (len(days), NMIN)
    O, H, L, C, S = (np.full(shape, np.nan) for _ in range(5))
    O[di, mi], H[di, mi], L[di, mi], C[di, mi] = m.open, m.high, m.low, m.close
    S[di, mi] = m.spread.to_numpy() * PT
    np.savez_compressed(CACHE, days=days.values.astype("datetime64[ns]").astype("int64"), O=O, H=H, L=L, C=C, S=S)


if not os.path.exists(CACHE):
    build()
_z = np.load(CACHE)
DAYS = pd.DatetimeIndex(_z["days"].astype("datetime64[ns]"))
O, H, L, C, S = (_z[k] for k in ("O", "H", "L", "C", "S"))


def _daily():
    rows = []
    for i in range(len(DAYS)):
        ok = np.isfinite(C[i])
        idx = np.flatnonzero(ok)
        rows.append(dict(o=O[i, idx[0]], h=np.nanmax(H[i]), l=np.nanmin(L[i]), c=C[i, idx[-1]], first=idx[0], last=idx[-1]))
    D = pd.DataFrame(rows, index=DAYS)
    pc = D.c.shift()
    D["tr"] = np.maximum(D.h - D.l, np.maximum((D.h - pc).abs(), (D.l - pc).abs()))
    D["atr10"] = D.tr.rolling(10).mean().shift()
    D["atr60"] = D.tr.rolling(60).mean().shift()
    eff = (D.c - D.o).abs() / (D.h - D.l)
    D["eff"] = eff
    D["peff"] = eff.shift()
    D["pdir"] = np.sign(D.c - D.o).shift()
    D["pret_atr"] = ((D.c - D.o) / D.atr10).shift()
    D["trend20"] = ((D.c - D.c.shift(20)) / (D.tr.rolling(20).mean() * np.sqrt(20))).shift()
    D["volr"] = D.atr10 / D.atr60
    return D


D = _daily()
ATR = D.atr10.to_numpy()


@njit(cache=True)
def _run(O, H, L, C, S, ATR, s0, ib, k, buf, win, mode, stop_kind, stop_k, exit_min, slip, touch, tp, hold, rng):
    """一個 IB 起點 s0（分鐘）、長度 ib 的全部交易。
    mode +1 = 突破順勢；-1 = 反向（同一個確認點做反方向，停損距離 = 順勢版的風險，對稱）。
    stop_kind 0 = IB 另一側；1 = IB 中點；2 = k×ATR；3 = IB 另一側但最多 k×ATR
    exit_min：當天第幾分鐘強制平倉（<=0 表示當天最後一根）。tp：R 倍數停利（<=0 不用）。hold：最長持有分鐘（<=0 不限）。
    rng 0 = IB 高低；1 = IB 期間 1 分鐘 TPO 的 70% 價值區（VAH/VAL）當突破線與停損；2 = 價值區當突破線、停損仍在 IB 另一端
    回傳每筆：day, side, 進場分鐘, R, pnl(點), risk(點), 出場原因(0 時間 1 停損 2 停利), IB 寬(點), 進場價"""
    nd, nm = C.shape
    out = np.full((nd, 9), np.nan)
    for d in range(nd):
        atr = ATR[d]
        if not np.isfinite(atr):
            continue
        cnt = 0
        hi = -1e18
        lo = 1e18
        for m in range(s0, s0 + ib):
            if m < nm and np.isfinite(C[d, m]):
                cnt += 1
                hi = max(hi, H[d, m])
                lo = min(lo, L[d, m])
        if cnt < 0.8 * ib:
            continue
        ibh = hi
        ibl = lo
        if rng > 0:
            tick = atr / 100.0
            nb = int((hi - lo) / tick) + 1
            cntb = np.zeros(nb)
            for m in range(s0, s0 + ib):
                if m < nm and np.isfinite(C[d, m]):
                    a0 = int((L[d, m] - lo) / tick)
                    a1 = int((H[d, m] - lo) / tick)
                    for b in range(max(a0, 0), min(a1, nb - 1) + 1):
                        cntb[b] += 1
            tot = cntb.sum()
            p = int(np.argmax(cntb))
            va0 = p
            va1 = p
            acc = cntb[p]
            while acc < 0.7 * tot:
                up = cntb[va1 + 1] if va1 + 1 < nb else -1.0
                dn = cntb[va0 - 1] if va0 - 1 >= 0 else -1.0
                if up < 0 and dn < 0:
                    break
                if up >= dn:
                    va1 += 1
                    acc += up
                else:
                    va0 -= 1
                    acc += dn
            hi = lo + (va1 + 1) * tick
            lo = lo + va0 * tick
        upl = hi + buf * atr
        dnl = lo - buf * atr
        endw = min(nm, s0 + ib + win)
        side = 0
        ent = -1
        lvl = 0.0
        for m in range(s0 + ib, endw):
            if not np.isfinite(C[d, m]):
                continue
            if touch:
                if H[d, m] > upl:
                    side = 1
                    lvl = max(upl, O[d, m])
                    ent = m
                    break
                if L[d, m] < dnl:
                    side = -1
                    lvl = min(dnl, O[d, m])
                    ent = m
                    break
            elif (m + 1) % k == 0:
                if C[d, m] > upl:
                    side, lvl, ent = 1, C[d, m], m
                    break
                if C[d, m] < dnl:
                    side, lvl, ent = -1, C[d, m], m
                    break
        if side == 0:
            continue
        # 順勢版停損
        if rng == 2:
            hi = ibh
            lo = ibl
        if stop_kind == 0:
            st = lo if side == 1 else hi
        elif stop_kind == 1:
            st = 0.5 * (hi + lo)
        elif stop_kind == 2:
            st = lvl - side * stop_k * atr
        else:
            st = lo if side == 1 else hi
            if side * (lvl - st) > stop_k * atr:
                st = lvl - side * stop_k * atr
        risk = side * (lvl - st)
        if not risk > 0:
            continue
        s = side * mode
        st = lvl - s * risk
        tgt = lvl + s * tp * risk if tp > 0 else np.nan
        start = ent if touch else ent + 1
        last = nm - 1
        while last > 0 and not np.isfinite(C[d, last]):
            last -= 1
        if hold > 0:
            last = min(last, ent + hold)
        if exit_min > 0:
            last = min(last, exit_min)
        while last > start and not np.isfinite(C[d, last]):
            last -= 1
        if start > last:
            continue
        px = C[d, last]
        why = 0
        for m in range(start, last + 1):
            if not np.isfinite(C[d, m]):
                continue
            if (L[d, m] <= st) if s == 1 else (H[d, m] >= st):
                if m > start:
                    px = min(O[d, m], st) if s == 1 else max(O[d, m], st)
                else:
                    px = st
                why = 1
                break
            if tp > 0 and ((H[d, m] >= tgt) if s == 1 else (L[d, m] <= tgt)):
                px = tgt
                why = 2
                break
        cost = S[d, ent] if np.isfinite(S[d, ent]) else 0.0
        pnl = s * (px - lvl) - cost - slip
        out[d, 0] = d
        out[d, 1] = s
        out[d, 2] = ent
        out[d, 3] = pnl / risk
        out[d, 4] = pnl
        out[d, 5] = risk
        out[d, 6] = why
        out[d, 7] = ibh - ibl
        out[d, 8] = lvl
    return out


def run(s0, ib=60, k=5, buf=0.05, win=180, mode=1, stop="ib", exit_min=0, slip=0.0, touch=False, tp=0.0, hold=0, rng=0):
    """回傳 DataFrame（一個 IB 起點）。s0、exit_min 用 broker 分鐘（例如 16:30 = 990）。"""
    sk, skk = {"ib": (0, 0.0), "mid": (1, 0.0)}.get(stop, (None, None)) if isinstance(stop, str) else \
        ({"atr": 2, "cap": 3}[stop[0]], float(stop[1]))
    a = _run(O, H, L, C, S, ATR, int(s0), int(ib), int(k), float(buf), int(win), int(mode), sk, skk,
             int(exit_min), float(slip), bool(touch), float(tp), int(hold), int(rng))
    a = a[np.isfinite(a[:, 0])]
    X = pd.DataFrame(a, columns=["d", "side", "ent", "R", "pnl", "risk", "why", "ibw", "lvl"])
    X["d"] = X.d.astype(int)
    X["day"] = DAYS[X.d.values]
    X["s0"] = s0
    X["atr"] = ATR[X.d.values]
    return X


def run_many(starts, **kw):
    """多個 IB 起點合併（同一天同進場分鐘同方向只留一筆，比照 core_ib 的去重）。"""
    X = pd.concat([run(s, **kw) for s in starts], ignore_index=True)
    if len(X) == 0:
        return X
    return X.sort_values(["day", "ent", "s0"]).drop_duplicates(["day", "ent", "side"]).reset_index(drop=True)


def period(X, a, b):
    return X[(X.day >= a) & (X.day < b)]


def stats(X):
    if len(X) < 5:
        return dict(n=len(X))
    r = X.R
    dR = X.groupby("day").R.sum()
    nd = X.day.nunique()
    out = dict(n=len(X), mR=r.mean(), PF=r[r > 0].sum() / max(1e-9, -r[r <= 0].sum()),
               win=(r > 0).mean(), dt=dR.mean() / dR.std(ddof=1) * np.sqrt(len(dR)) if len(dR) > 2 else np.nan,
               totR=r.sum(), per_day=len(X) / nd)
    eq = X.sort_values(["day", "ent"]).R.cumsum()
    out["mddR"] = (eq - eq.cummax()).min()
    return out


def halves(X):
    """四個研究半年的每筆 R。"""
    return [period(X, a, b).R.mean() if len(period(X, a, b)) >= 5 else np.nan for _, a, b in HALVES]


def hm(s):
    return f"{s // 60:02d}:{s % 60:02d}"
