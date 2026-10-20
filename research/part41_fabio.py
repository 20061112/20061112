"""第四十一部分：Fabio Valentini 三套策略（IVB 開盤區間突破 / 趨勢模型 + AAA / 倫敦均值回歸）在 XAUUSD 與 NAS100 上的回測。

時間：broker = 紐約 + 7 → 16:30 = 紐約 9:30 開盤，23:00 = 紐約 16:00 收盤，10:00 = 倫敦 8:00。
成交量：只有 tick volume（CFD 沒有真實成交、沒有主動買賣方向），成交量分佈用 tick volume 攤到每根 M1 的高低之間。
訂單流「大單 / 吸收」無法重現，用價格 + tick volume 近似（見各策略說明），這是本回測最大的限制。
成本：每筆扣進場那根 M1 的點差（XAU spread×0.01、NAS spread×0.1）。同一根同時碰到停損與停利算停損。

重現：python3 part41_fabio.py → results_part41.txt、part41_trades.csv
"""
import numpy as np
import pandas as pd

SYMBOLS = {
    "XAUUSD": dict(files=[("data/XAUUSD_M1_2023_2026.csv", None)], pt=0.01),
    "NAS100": dict(files=[("data/NAS100_M1_2023_2024.csv", "2024-10-01 01:00"),
                          ("data/NAS100_M1_2024_2025.csv", "2025-10-01 01:00"),
                          ("data/NAS100_M1_2025_2026.csv", None)], pt=0.1),
}
NY_OPEN, NY_CLOSE, LDN_OPEN = 16 * 60 + 30, 23 * 60, 10 * 60
NB = 50            # 成交量分佈格數（依區間自動定格寬，兩個商品通用）
BIGQ = 0.90        # 「大單」近似：該根 M1 tick volume / 前 20 根中位數，高於過去 1000 根此比值的 90 分位


def load_symbol(sym):
    cfg = SYMBOLS[sym]
    parts = []
    for path, end in cfg["files"]:
        d = pd.read_csv(path, sep="\t")
        d.columns = [c.strip("<>").lower() for c in d.columns]
        d.index = pd.to_datetime(d["date"] + " " + d["time"], format="%Y.%m.%d %H:%M:%S")
        if end:
            d = d[d.index < pd.Timestamp(end)]
        parts.append(d[["open", "high", "low", "close", "tickvol", "spread"]])
    m = pd.concat(parts)
    m = m[~m.index.duplicated(keep="last")].sort_index()
    ratio = m.tickvol / m.tickvol.rolling(20).median().shift(1)
    m["big"] = ratio >= ratio.rolling(1000).quantile(BIGQ).shift(1)
    m["sp"] = m.spread * cfg["pt"]
    days = {}
    for day, g in m.groupby(m.index.normalize()):
        if len(g) < 1000:
            continue
        days[day] = dict(O=g.open.values, H=g.high.values, L=g.low.values, C=g.close.values,
                         V=g.tickvol.values.astype(float), BIG=g.big.values, S=g.sp.values,
                         M=(g.index.hour * 60 + g.index.minute).values)
    return days


# ---------- 成交量分佈 ----------
def profile(H, L, V, nb=NB):
    lo, hi = L.min(), H.max()
    if not hi > lo:
        return None
    w = (hi - lo) / nb
    cnt = np.zeros(nb)
    a = np.clip(((L - lo) / w).astype(int), 0, nb - 1)
    b = np.clip(((H - lo) / w).astype(int), 0, nb - 1)
    for i in range(len(H)):
        cnt[a[i]:b[i] + 1] += V[i] / (b[i] - a[i] + 1)
    poc = int(np.argmax(cnt))
    lo_i = hi_i = poc
    acc, tgt = cnt[poc], 0.7 * cnt.sum()
    while acc < tgt:
        up = cnt[hi_i + 1] if hi_i < nb - 1 else -1
        dn = cnt[lo_i - 1] if lo_i > 0 else -1
        if up >= dn:
            hi_i += 1; acc += cnt[hi_i]
        else:
            lo_i -= 1; acc += cnt[lo_i]
    ctr = lo + (np.arange(nb) + 0.5) * w
    return dict(lo=lo, hi=hi, w=w, cnt=cnt, ctr=ctr, poc=ctr[poc], vah=lo + (hi_i + 1) * w, val=lo + lo_i * w)


def lvn(p, f0=0.2, f1=0.8):
    """區間中段（20%~80%）平滑後成交最少的一格 = 低成交量節點。"""
    s = np.convolve(p["cnt"], np.ones(3) / 3, "same")
    i0, i1 = int(NB * f0), int(NB * f1)
    i = i0 + int(np.argmin(s[i0:i1]))
    return p["ctr"][i]


# ---------- 出場模擬 ----------
def run_trade(d, i, side, entry, stop, target, last_m, same_bar=False, be_at=None):
    """i = 成交那根。same_bar=True：限價在第 i 根內成交，第 i 根就檢查停損。be_at：到達此價後停損移到保本。"""
    H, L, C, M = d["H"], d["L"], d["C"], d["M"]
    R = abs(entry - stop)
    st = stop
    k0 = i if same_bar else i + 1
    if same_bar and (side > 0 and L[i] <= stop or side < 0 and H[i] >= stop):
        return -1.0, i, "stop"
    for k in range(k0, len(H)):
        if M[k] >= last_m:
            return side * (C[k - 1] - entry) / R, k, "time"
        if side > 0:
            if L[k] <= st:
                return (st - entry) / R, k, "stop" if st < entry else "be"
            if H[k] >= target:
                return (target - entry) / R, k, "target"
            if be_at is not None and H[k] >= be_at:
                st = max(st, entry)
        else:
            if H[k] >= st:
                return (entry - st) / R, k, "stop" if st > entry else "be"
            if L[k] <= target:
                return (entry - target) / R, k, "target"
            if be_at is not None and L[k] <= be_at:
                st = min(st, entry)
    return side * (C[-1] - entry) / R, len(H) - 1, "eod"


def mk(day, strat, d, i, side, entry, stop, target, res):
    r, k, why = res
    R = abs(entry - stop)
    cost = d["S"][i] / R
    return dict(day=day, strat=strat, side=side, t_in=d["M"][i], t_out=d["M"][k], entry=entry, stop=stop,
                target=target, R_pts=R, R_gross=r, cost_R=cost, R=r - cost, exit=why)


def big_bar(d, i, side):
    """訂單流近似：同方向的大量 K 棒（tick volume 相對量在過去 1000 根的前 10%、收盤在 K 棒 60% 以上的位置）。"""
    O, H, L, C = d["O"], d["H"], d["L"], d["C"]
    rng = H[i] - L[i]
    if not (d["BIG"][i] and rng > 0):
        return False
    pos = (C[i] - L[i]) / rng
    return (C[i] > O[i] and pos >= 0.6) if side > 0 else (C[i] < O[i] and pos <= 0.4)


def absorb_bar(d, i, side):
    """吸收近似：大量 K 棒，但反方向推不動（做多：下影線 ≥ 50%，收盤在上半部）。"""
    H, L, C = d["H"], d["L"], d["C"]
    rng = H[i] - L[i]
    if not (d["BIG"][i] and rng > 0):
        return False
    pos = (C[i] - L[i]) / rng
    return pos >= 0.5 if side > 0 else pos <= 0.5


# ---------- 第一套：IVB ----------
def ivb(day, d, mode, rr=2.0, shorts=True, or_len=30, buf_f=0.02, last_entry=21 * 60, through=0.0):
    """紐約開盤 30 分鐘區間 → 5 分 K 收在區間外 = 突破 → 回踩 [POC, VAH] 框 → 進場；停損框外，目標 rr 倍。
    mode: limit  = 價格碰到 VAH 直接限價買（純價格、最積極）
          candle = 碰過框之後，1 分 K 整根實體收在 VAH 之上才進（他直播沒用訂單流時的做法）
          flow   = 框內出現吸收 K 棒（大量 + 下影線）才進（大單被吸收的近似）
    through：限價成交要穿價多少（預設 0 = 碰到就成交）。
    1 分 K 收回 VAL 之下 → 作廢。突破要在 19:30 前（紐約 12:30），進場要在 21:00 前，23:00 平倉。"""
    O, H, L, C, M = d["O"], d["H"], d["L"], d["C"], d["M"]
    orm = (M >= NY_OPEN) & (M < NY_OPEN + or_len)
    if orm.sum() < 0.8 * or_len:
        return []
    oh, ol = H[orm].max(), L[orm].min()
    p = profile(H[orm], L[orm], d["V"][orm])
    if p is None:
        return []
    buf = buf_f * (oh - ol)
    side = 0
    touched = False
    for i in np.where(M >= NY_OPEN + or_len)[0]:
        if M[i] >= NY_CLOSE:
            break
        if side == 0:
            if M[i] >= 19 * 60 + 30:
                break
            if (M[i] + 1) % 5 == 0:
                if C[i] > oh:
                    side = 1
                elif C[i] < ol and shorts:
                    side = -1
            continue
        if M[i] >= last_entry:
            break
        top, bot, inval = (p["vah"], p["poc"], p["val"]) if side > 0 else (p["val"], p["poc"], p["vah"])
        stop = bot - side * buf
        if side * (C[i] - inval) < 0:
            break
        hit = L[i] <= top if side > 0 else H[i] >= top
        # through：限價單要價格穿過掛單價這麼多才算成交（排隊沒成交的保守假設）
        if mode == "limit" and (L[i] <= top - through if side > 0 else H[i] >= top + through):
            entry = min(O[i], top) if side > 0 else max(O[i], top)
            if side * (entry - stop) <= 0:
                break
            tgt = entry + side * rr * abs(entry - stop)
            return [mk(day, "IVB-" + mode, d, i, side, entry, stop, tgt,
                       run_trade(d, i, side, entry, stop, tgt, NY_CLOSE, same_bar=True))]
        touched |= hit
        if not touched:
            continue
        ok = (min(O[i], C[i]) > top if side > 0 else max(O[i], C[i]) < top) if mode == "candle" else \
             (absorb_bar(d, i, side) and side * (C[i] - bot) > 0)
        if ok:
            entry = C[i]
            if side * (entry - stop) <= 0:
                continue
            tgt = entry + side * rr * abs(entry - stop)
            return [mk(day, "IVB-" + mode, d, i, side, entry, stop, tgt, run_trade(d, i, side, entry, stop, tgt, NY_CLOSE))]
    return []


# ---------- 第二套：趨勢模型（+ AAA） ----------
def trend(day, d, prev, mode, shorts=True, maxn=3):
    """前一天紐約盤成交量分佈 = 平衡區；5 分 K 收在 VA 外 = 失衡，只做那個方向（收回 VA 內就重置）。
    第一波不追：等回檔 ≥ 該波 30% 後再創新高（第二次往外推），從這波起漲低點到高點畫分佈，取低成交量節點。
    價格回到節點 → 1 分 K 出現同方向大單近似（big_bar）→ 收盤進場；停損放在這根 K 棒後面。
    mode: wave  = 目標 = 這一波高點，盈虧比 < 2.5 不做
          fixed = 固定 2.5R，價格再創新高後停損移到保本
          aaa   = AAA：從當天這波起漲處畫分佈、POC 在上半部，回到該分佈 VAL + 吸收 K 棒 → 進場，停損吸收段低點下，目標 VAH
    只在紐約盤：16:50（開盤 20 分鐘後）~ 22:30 進場，23:00 平倉。一天最多 maxn 筆，最多 3 筆停損。"""
    O, H, L, C, M, V = d["O"], d["H"], d["L"], d["C"], d["M"], d["V"]
    pm = (prev["M"] >= NY_OPEN) & (prev["M"] < NY_CLOSE)
    if pm.sum() < 300:
        return []
    pp = profile(prev["H"][pm], prev["L"][pm], prev["V"][pm])
    if pp is None:
        return []
    trades, losses = [], 0
    state, j0 = 0, None           # 失衡方向、這段失衡開始的 bar
    wave_ext = wave_start = None  # 這一波的極值與起點 bar
    pulled, pushes = False, 0
    arm = None                    # (節點價, 波段極值, 起點 bar)
    i = int(np.searchsorted(M, LDN_OPEN))
    n = len(M)
    while i < n:
        if M[i] >= NY_CLOSE or len(trades) >= maxn or losses >= 3:
            break
        # 失衡狀態（只用收完的 5 分 K）
        if (M[i] + 1) % 5 == 0:
            s = 1 if C[i] > pp["vah"] else (-1 if C[i] < pp["val"] else 0)
            if s == -1 and not shorts:
                s = 0
            if s != state:
                state, j0 = s, i
                wave_start, wave_ext, pulled, pushes, arm = i, (H[i] if s > 0 else L[i]), False, 0, None
        if state == 0:
            i += 1
            continue
        sd = state
        ext_now = H[i] if sd > 0 else L[i]
        if sd * (ext_now - wave_ext) > 0:
            if pulled:            # 回檔後再創新高 = 新的一波往外推
                pushes += 1
                pulled = False
                seg = slice(wave_start, i + 1)
                lo_i = wave_start + (int(np.argmin(L[seg])) if sd > 0 else int(np.argmax(H[seg])))
                wave_start = lo_i
            wave_ext = ext_now
            if pushes >= 1:
                seg = slice(wave_start, i + 1)
                p = profile(H[seg], L[seg], V[seg])
                if p is not None:
                    arm = (lvn(p), wave_ext, wave_start, p)
        else:
            base = L[wave_start:i + 1].min() if sd > 0 else H[wave_start:i + 1].max()
            if abs(wave_ext - (L[i] if sd > 0 else H[i])) >= 0.3 * abs(wave_ext - base):
                pulled = True
        in_win = NY_OPEN + 20 <= M[i] < 22 * 60 + 30
        if mode == "aaa":
            if in_win:
                seg = slice(j0, i + 1)
                lo_k = j0 + (int(np.argmin(L[seg])) if sd > 0 else int(np.argmax(H[seg])))
                if i - lo_k >= 15:
                    p = profile(H[lo_k:i + 1], L[lo_k:i + 1], V[lo_k:i + 1])
                    if p is not None:
                        upper = (p["poc"] - p["lo"]) / (p["hi"] - p["lo"]) >= 0.5 if sd > 0 else \
                                (p["hi"] - p["poc"]) / (p["hi"] - p["lo"]) >= 0.5
                        edge = p["val"] if sd > 0 else p["vah"]
                        near = abs(C[i] - edge) <= 0.1 * (p["hi"] - p["lo"])
                        if upper and near and absorb_bar(d, i, sd):
                            a0 = max(i - 5, lo_k)
                            ext = L[a0:i + 1].min() if sd > 0 else H[a0:i + 1].max()
                            entry, stop = C[i], ext - sd * 0.02 * (p["hi"] - p["lo"])
                            tgt = p["vah"] if sd > 0 else p["val"]
                            if sd * (entry - stop) > 0 and sd * (tgt - entry) >= 1.0 * abs(entry - stop):
                                t = mk(day, "TREND-aaa", d, i, sd, entry, stop, tgt,
                                       run_trade(d, i, sd, entry, stop, tgt, NY_CLOSE))
                                trades.append(t)
                                losses += t["R_gross"] < 0
                                i = int(np.searchsorted(M, t["t_out"])) + 1
                                state = 0
                                continue
            i += 1
            continue
        if arm is not None and in_win:
            node, ext, ws, p = arm
            at_node = L[i] <= node if sd > 0 else H[i] >= node
            if at_node and big_bar(d, i, sd):
                entry = C[i]
                stop = (L[i] if sd > 0 else H[i]) - sd * 0.05 * (p["hi"] - p["lo"])
                R = abs(entry - stop)
                if mode == "wave":
                    tgt, ok, be = ext, sd * (ext - entry) >= 2.5 * R, None
                else:
                    tgt, ok, be = entry + sd * 2.5 * R, True, ext
                if ok and R > 0:
                    t = mk(day, "TREND-" + mode, d, i, sd, entry, stop, tgt,
                           run_trade(d, i, sd, entry, stop, tgt, NY_CLOSE, be_at=be))
                    trades.append(t)
                    losses += t["R_gross"] < 0
                    arm = None
                    i = int(np.searchsorted(M, t["t_out"])) + 1
                    continue
            if sd * (C[i] - (L[ws] if sd > 0 else H[ws])) < 0:
                arm = None          # 跌破這波起漲點 → 這波作廢
        i += 1
    return trades


# ---------- 第三套：倫敦盤均值回歸 ----------
def meanrev(day, d, prev, end_m=NY_OPEN):
    """前一天整天的成交量分佈 VA = 區間。5 分 K 先收到區間外，再收回區間內 = 假突破。
    從衝出去的那根到收回那根畫分佈，取低成交量節點；價格回到節點且出現往區間中間的大單近似 → 進場。
    停損在大單 K 棒後面；若後面就是假突破極值，停損放在極值內 2 跳。目標 = 前一天 POC。倫敦 10:00~16:15 進場，end_m 平倉。"""
    O, H, L, C, M, V = d["O"], d["H"], d["L"], d["C"], d["M"], d["V"]
    pp = profile(prev["H"], prev["L"], prev["V"])
    if pp is None:
        return []
    tick = pp["w"] / 10
    trades = []
    out_side, out_i, arm = 0, None, None
    for i in np.where((M >= LDN_OPEN) & (M < end_m))[0]:
        if len(trades) >= 2:
            break
        if (M[i] + 1) % 5 == 0:
            s = 1 if C[i] > pp["vah"] else (-1 if C[i] < pp["val"] else 0)
            if s != 0 and out_side == 0:
                out_side, out_i, arm = s, i - 4, None
            elif s == 0 and out_side != 0:
                seg = slice(out_i, i + 1)
                p = profile(H[seg], L[seg], V[seg])
                ext = H[seg].max() if out_side > 0 else L[seg].min()
                if p is not None:
                    arm = (-out_side, lvn(p), ext)
                out_side = 0
        if arm is None or M[i] >= end_m - 15:
            continue
        sd, node, ext = arm
        if sd * (C[i] - ext) < 0:          # 價格又越過極值 → 作廢
            arm = None
            continue
        at = H[i] >= node if sd < 0 else L[i] <= node
        if at and big_bar(d, i, sd):
            entry = C[i]
            stop = (H[i] if sd < 0 else L[i]) - sd * 2 * tick
            lim = ext + sd * 2 * tick       # 極值內 2 跳
            if sd * (stop - lim) < 0:
                stop = lim
            tgt = pp["poc"]
            R = abs(entry - stop)
            if R > 0 and sd * (entry - stop) > 0 and sd * (tgt - entry) >= R:
                t = mk(day, "MR-london", d, i, sd, entry, stop, tgt, run_trade(d, i, sd, entry, stop, tgt, end_m))
                trades.append(t)
            arm = None
    return trades


def stats(r):
    r = np.asarray(r)
    if len(r) == 0:
        return dict(n=0)
    eq = np.cumsum(r)
    dd = (np.maximum.accumulate(np.r_[0, eq])[1:] - eq).max()
    pf = r[r > 0].sum() / -r[r < 0].sum() if (r < 0).any() else np.inf
    t = r.mean() / r.std(ddof=1) * np.sqrt(len(r)) if len(r) > 2 else np.nan
    return dict(n=len(r), win=(r > 0).mean(), avg=r.mean(), t=t, pf=pf, tot=eq[-1], dd=dd)


def fmt(s):
    if s["n"] == 0:
        return "n=0"
    return (f"n={s['n']:4d} 勝率={s['win']:.0%} 期望={s['avg']:+.3f}R t={s['t']:+.2f} "
            f"PF={s['pf']:.2f} 累計={s['tot']:+.1f}R 回撤={s['dd']:.1f}R")


def main():
    out, rows = [], []
    for sym in SYMBOLS:
        days = load_symbol(sym)
        keys = sorted(days)
        for k_i, day in enumerate(keys):
            d = days[day]
            prev = days[keys[k_i - 1]] if k_i else None
            for mode in ("limit", "candle", "flow"):
                rows += [dict(t, sym=sym) for t in ivb(day, d, mode)]
            rows += [dict(t, sym=sym, strat="IVB-candle-2.5R") for t in ivb(day, d, "candle", rr=2.5)]
            if prev is not None:
                for mode in ("wave", "fixed", "aaa"):
                    rows += [dict(t, sym=sym) for t in trend(day, d, prev, mode)]
                rows += [dict(t, sym=sym) for t in meanrev(day, d, prev)]
                rows += [dict(t, sym=sym, strat="MR-london-to-21") for t in meanrev(day, d, prev, end_m=21 * 60)]
        print(sym, "done", flush=True)
    df = pd.DataFrame(rows)
    df["year"] = pd.to_datetime(df.day).dt.year
    df.to_csv("part41_trades.csv", index=False)
    for sym, g0 in df.groupby("sym"):
        out.append(f"\n===== {sym} =====")
        for strat, g in g0.groupby("strat"):
            out.append(f"\n[{strat}]")
            out.append("  全部   淨 " + fmt(stats(g.R)) + f"  | 毛期望 {g.R_gross.mean():+.3f}R，成本 {g.cost_R.mean():.3f}R/筆")
            for sd, gs in g.groupby("side"):
                out.append(f"  {'做多' if sd > 0 else '做空'}   淨 " + fmt(stats(gs.R)))
            for y, gy in g.groupby("year"):
                out.append(f"  {y}   淨 " + fmt(stats(gy.R)))
            out.append("  出場：" + ", ".join(f"{k} {v:.0%}" for k, v in g.exit.value_counts(normalize=True).items()))
    txt = "\n".join(out)
    print(txt)
    open("results_part41.txt", "w").write(txt + "\n")


if __name__ == "__main__":
    main()
