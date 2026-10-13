"""第二十二部分：大 K 推動段 → 小 K 震盪（旗形）→ 順勢突破。

想法（使用者提出）：一段趨勢會先出現幾根大 K（推動段 / 旗桿），接著 K 棒變小、小幅震盪（旗面），
之後往某一方向走出去。用物理量 / 積分 / 布林帶描述「震盪」，記錄 HH / LL 當突破位，做順勢倉位。

事件定義（做多；做空鏡像），全部只用 K 棒 t 收盤前已知的資料：
  1. HH：e = 過去 Mmax 根（t-Mmax ~ t-1）最高點所在位置，且 high[e] 也是 [e-K, e] 的最高 → 真正的波段高點
  2. 旗桿（推動段）：s = [e-K, e] 內最低點；pole = high[e] - low[s]
  3. 旗面（震盪）：e+1 ~ t-1，根數 m = t-e-1，需 Mmin <= m；箱體 = [min low(e..t-1), HH]
  4. 突破：close[t] > HH（第一根收盤站上記錄的 HH）→ 以 close[t] + 點差 進場
特徵（物理 / 微積分 / 布林帶）：
  pole_atr     旗桿長度 / ATR（推動段位移）
  pole_er      旗桿 Kaufman ER（位移 / 路徑長 = |∫v dt| / ∫|v| dt）
  body_ratio   旗面平均實體 / 旗桿平均實體（大 K → 小 K）
  ke_ratio     動能比 = 旗面 mean(v²) / 旗桿 mean(v²)，v = Δclose / ATR（能量耗散 / 阻尼）
  zc_rate      旗面速度 v 的正負號變換次數 / 根數（震盪頻率）
  area         ∫(close − 箱體中點)dt / (m × 箱體半寬)，順突破方向為正（旗面「黏」在突破側 → +1）
  retr         旗面回檔深度 / 旗桿（0 = 沒回、1 = 全吐回）
  box_atr      箱體高 / ATR
  bbw_ratio    突破前一根的布林帶寬 / 旗桿頂點那根的布林帶寬（帶寬收縮）
  bbw_pct      布林帶寬在過去 500 根的百分位
  vol_ratio    旗面平均 tick volume / 旗桿平均 tick volume
  brk_body     突破 K 棒實體 / ATR（突破力道）
  htf          更大週期方向（close − EMA(4 天)）是否與突破同向
  day_er       當天開盤到突破前的 ER（使用者第二十一部分的因子）
  hour, dow    時段、星期
出場（全部在同週期 K 棒上模擬；同一根同時碰停損與停利 → 算停損；每筆扣進場點差）：
  停損 = 箱體另一側（box）或箱體中點（mid）；停利 = 1R / 2R / 3R / 量測目標（旗桿長度）/ 不設；最長 H 根以收盤出場
"""
import numpy as np
import pandas as pd


def _atr(h, l, c, n):
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return pd.Series(tr).rolling(n, min_periods=n).mean().to_numpy()


def events(df, K=24, Mmin=3, Mmax=24, atr_n=96, ema_days=4):
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    v = df.tickvol.to_numpy(float)
    sp = df.spread.to_numpy(float) * 0.01
    idx = df.index
    n = len(c)
    atr = _atr(h, l, c, atr_n)
    std20 = pd.Series(c).rolling(20).std().to_numpy()
    bbw = 4 * std20 / atr
    bbw_pct = pd.Series(bbw).rolling(500, min_periods=200).rank(pct=True).to_numpy()
    ema = pd.Series(c).ewm(span=ema_days * atr_n, adjust=False).mean().to_numpy()
    day = idx.normalize()
    # 當天開盤價、當天到目前的路徑長（ER 用）
    day_id = pd.factorize(day)[0]
    first = np.r_[True, day_id[1:] != day_id[:-1]]
    day_open = pd.Series(np.where(first, o, np.nan)).ffill().to_numpy()
    dc = np.abs(np.diff(c, prepend=c[0]))
    dc[first] = np.abs(c[first] - o[first])
    path = pd.Series(dc).groupby(day_id).cumsum().to_numpy()
    body = np.abs(c - o)
    rows = []
    start = max(K + Mmax + 2, atr_n + 2)
    for t in range(start, n):
        A = atr[t - 1]
        if not A > 0:
            continue
        for d in (1, -1):
            seg = h[t - Mmax:t] if d == 1 else -l[t - Mmax:t]
            # 最後一次出現的極值（同價時取最近）
            j = Mmax - 1 - int(np.argmax(seg[::-1]))
            e = t - Mmax + j
            ext = seg[j]
            m = t - e - 1
            if m < Mmin:
                continue
            cc = c[t] if d == 1 else -c[t]
            if not cc > ext:
                continue
            # 必須是第一根收盤突破（前一根收盤沒站上，seg 是最高價，所以一定成立）
            ph = h[e - K:e + 1] if d == 1 else -l[e - K:e + 1]
            if ph.max() > ext + 1e-9:
                continue  # e 不是 K 根內的波段頂點
            pl = l[e - K:e + 1] if d == 1 else -h[e - K:e + 1]
            s = e - K + int(np.argmin(pl))
            if s >= e:
                continue
            pole = ext - pl[s - (e - K)]
            box_lo = (l[e:t].min() if d == 1 else -h[e:t].max())
            box = ext - box_lo
            if not box > 0:
                continue
            # 旗桿 / 旗面的 K 線量
            cp = c[s:e + 1]
            vel_p = np.diff(cp) / A
            vel_f = np.diff(c[e:t]) / A
            er_p = abs(cp[-1] - cp[0]) / max(np.abs(np.diff(cp)).sum(), 1e-9)
            body_p = body[s + 1:e + 1].mean() if e > s else np.nan
            body_f = body[e + 1:t].mean()
            ke_p = (vel_p ** 2).mean() if len(vel_p) else np.nan
            ke_f = (vel_f ** 2).mean() if len(vel_f) else np.nan
            sgn = np.sign(vel_f[vel_f != 0])
            zc = (np.sum(sgn[1:] != sgn[:-1]) / len(vel_f)) if len(sgn) > 1 else 0.0
            mid = ext - box / 2
            cf = c[e + 1:t] if d == 1 else -c[e + 1:t]
            area = ((cf - mid) / (box / 2)).mean()
            vol_p = v[s:e + 1].mean()
            vol_f = v[e + 1:t].mean()
            entry = c[t]
            rows.append(dict(
                t=t, time=idx[t], dir=d, entry=entry, hh=ext * d, box_lo=box_lo * d,
                mid=mid * d, sp=sp[t], atr=A, m=m, pole_bars=e - s,
                pole_atr=pole / A, pole_er=er_p, body_ratio=body_f / body_p if body_p > 0 else np.nan,
                ke_ratio=ke_f / ke_p if ke_p > 0 else np.nan, zc_rate=zc, area=area,
                retr=box / pole if pole > 0 else np.nan, box_atr=box / A,
                bbw_ratio=bbw[t - 1] / bbw[e] if bbw[e] > 0 else np.nan, bbw_pct=bbw_pct[t - 1],
                vol_ratio=vol_f / vol_p if vol_p > 0 else np.nan,
                brk_body=(c[t] - o[t]) * d / A, brk_dist=(cc - ext) / A,
                htf=float(np.sign(c[t - 1] - ema[t - 1]) == d),
                day_er=abs(c[t - 1] - day_open[t - 1]) / max(path[t - 1], 1e-9),
                day_dir=float(np.sign(c[t - 1] - day_open[t - 1]) == d),
                hour=idx[t].hour, dow=idx[t].dayofweek,
            ))
    return pd.DataFrame(rows)


def simulate(df, E, stop="box", tp=2.0, H=32, measured=False):
    """回傳每個事件的 R 結果（不處理重疊）。tp=None → 不設停利，H 根收盤出場。"""
    h, l, c = (df[k].to_numpy(float) for k in ("high", "low", "close"))
    n = len(c)
    t = E.t.to_numpy()
    d = E.dir.to_numpy()
    entry = E.entry.to_numpy()
    sl = (E.box_lo if stop == "box" else E.mid).to_numpy()
    risk = (entry - sl) * d
    if measured:
        tgt = entry + d * E.pole_atr.to_numpy() * E.atr.to_numpy()
    elif tp is not None:
        tgt = entry + d * tp * risk
    else:
        tgt = entry + d * 1e9
    off = np.arange(1, H + 1)
    ix = np.minimum(t[:, None] + off[None, :], n - 1)
    hh, ll, cc = h[ix], l[ix], c[ix]
    fav = np.where(d[:, None] == 1, hh, -ll)
    adv = np.where(d[:, None] == 1, ll, -hh)
    hit_sl = adv <= (sl * d)[:, None]
    hit_tp = fav >= (tgt * d)[:, None]
    big = H + 10
    fsl = np.where(hit_sl.any(1), hit_sl.argmax(1), big)
    ftp = np.where(hit_tp.any(1), hit_tp.argmax(1), big)
    exit_px = cc[:, -1]
    exit_px = np.where(ftp < fsl, tgt, exit_px)
    exit_px = np.where(fsl <= ftp, np.where(fsl < big, sl, exit_px), exit_px)
    bars = np.minimum(np.minimum(fsl, ftp), H - 1) + 1
    pnl = (exit_px - entry) * d - E.sp.to_numpy()
    R = pnl / risk
    out = pd.DataFrame({"R": R, "pnl_atr": pnl / E.atr.to_numpy(), "bars": bars, "risk_atr": risk / E.atr.to_numpy()},
                       index=E.index)
    return out


def non_overlap(E, bars):
    """同一時間只持有一筆（依時間順序，前一筆出場後才接下一筆）。"""
    keep, busy = [], -1
    for i, (t, b) in enumerate(zip(E.t.to_numpy(), bars)):
        if t > busy:
            keep.append(i)
            busy = t + b
    return np.array(keep, dtype=int)


def summ(r):
    r = np.asarray(r, float)
    r = r[~np.isnan(r)]
    if len(r) < 2:
        return dict(n=len(r), mean=np.nan, t=np.nan, win=np.nan, pf=np.nan)
    gp, gl = r[r > 0].sum(), -r[r < 0].sum()
    return dict(n=len(r), mean=r.mean(), t=r.mean() / r.std(ddof=1) * np.sqrt(len(r)),
                win=(r > 0).mean(), pf=gp / gl if gl > 0 else np.inf)


def formations(df, K=24, Mmin=4, Mmax=24, atr_n=96, F=48):
    """測試 A：旗面剛成形（HH 後已有 Mmin 根、還沒突破）時凍結箱體，往後看先收盤突破哪一邊。
    隨機漫步下先碰上緣的機率 ≈ (現價 − 箱底) / 箱高，用它當基準，比較實際「順旗桿方向」的突破率。"""
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    n = len(c)
    atr = _atr(h, l, c, atr_n)
    std20 = pd.Series(c).rolling(20).std().to_numpy()
    bbw = 4 * std20 / atr
    body = np.abs(c - o)
    rows = []
    for t0 in range(max(K + Mmax + 2, atr_n + 2), n - F - 1):
        A = atr[t0]
        if not A > 0:
            continue
        for d in (1, -1):
            e = t0 - Mmin
            hi_ = h if d == 1 else -l
            lo_ = l if d == 1 else -h
            ext = hi_[e]
            # e 必須是 [e-K, t0] 的最高點（之後 Mmin 根都沒有創高）
            if hi_[e - K:t0 + 1].max() > ext + 1e-9 or hi_[e + 1:t0 + 1].max() >= ext:
                continue
            s = e - K + int(np.argmin(lo_[e - K:e + 1]))
            if s >= e:
                continue
            pole = ext - lo_[s]
            blo = lo_[e:t0 + 1].min()
            box = ext - blo
            c0 = c[t0] * d
            if not (box > 0 and blo < c0 <= ext):
                continue
            fut = c[t0 + 1:t0 + 1 + F] * d
            up = np.flatnonzero(fut > ext)
            dn = np.flatnonzero(fut < blo)
            fu = up[0] if len(up) else 10 ** 6
            fd = dn[0] if len(dn) else 10 ** 6
            res = 1 if fu < fd else (0 if fd < fu else np.nan)
            vel_p = np.diff(c[s:e + 1]) / A
            vel_f = np.diff(c[e:t0 + 1]) / A
            cp = c[s:e + 1]
            rows.append(dict(
                t=t0, time=df.index[t0], dir=d, up_first=res, exp_up=(c0 - blo) / box,
                pole_atr=pole / A, box_atr=box / A, retr=box / pole,
                pole_er=abs(cp[-1] - cp[0]) / max(np.abs(np.diff(cp)).sum(), 1e-9),
                body_ratio=body[e + 1:t0 + 1].mean() / max(body[s + 1:e + 1].mean(), 1e-9),
                ke_ratio=(vel_f ** 2).mean() / max((vel_p ** 2).mean(), 1e-12),
                area=(((c[e + 1:t0 + 1] * d) - (ext - box / 2)) / (box / 2)).mean(),
                bbw_ratio=bbw[t0] / bbw[e] if bbw[e] > 0 else np.nan,
            ))
    return pd.DataFrame(rows)


def placebo_boxes(df, L=5, F=48, step=1):
    """安慰劑：任意時間點、最近 L 根 K 棒的箱體（不要求旗桿）。方向隨意取「上緣」，
    回傳先收盤突破上緣的結果、隨機漫步基準與現價在箱體中的位置，用來和旗面比較。"""
    h, l, c = (df[k].to_numpy(float) for k in ("high", "low", "close"))
    n = len(c)
    rows = []
    for t0 in range(L + 100, n - F - 1, step):
        hi, lo = h[t0 - L + 1:t0 + 1].max(), l[t0 - L + 1:t0 + 1].min()
        c0 = c[t0]
        if not (hi > lo and lo < c0 <= hi):
            continue
        fut = c[t0 + 1:t0 + 1 + F]
        up = np.flatnonzero(fut > hi)
        dn = np.flatnonzero(fut < lo)
        fu = up[0] if len(up) else 10 ** 6
        fd = dn[0] if len(dn) else 10 ** 6
        if fu == fd:
            continue
        rows.append((df.index[t0], float(fu < fd), (c0 - lo) / (hi - lo)))
    return pd.DataFrame(rows, columns=["time", "up_first", "exp_up"])
