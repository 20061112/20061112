"""第二十二部分：Opening Range（OR）× Session × Auction Market Theory（AMT）— 共用結構。

時間：broker 時間（= 紐約 + 7），交易日 = broker 01:00 ~ 23:59。
資料：data/XAUUSD_M1_2023_2026.csv（2023/6/1 ~ 2026/10/8，M1）。

Session（開盤時間 / 結束）
  ASIA  01:00 ~ 09:59   （= 18:00 ET，Globex 重新開盤，交易日開頭）
  LDN   10:00 ~ 15:29   （= 08:00 倫敦）
  NY    15:30 ~ 23:54   （= 08:30 ET，美國數據 / 紐約盤）
OR：每個 session 開盤後前 OR_MIN 分鐘的高 / 低 / 中點；ORW = 高 − 低。

AMT 參考價（全部只用「過去」的資料）
  前日剖面（30 分 TPO、70% 價值區）：pPOC / pVAH / pVAL；前日高低 PDH / PDL
  前一個 session 的高低：LDN 用 ASIA、NY 用 LDN、ASIA 用前一天 NY
  開盤位置（open_loc）：session 開盤價在前日價值區之上 / 之內 / 之下
  開盤型態（Dalton）：由 OR 期間的 M1 走勢分類（見 open_type）
單位：價格距離一律除以 ATR10（前 10 天日 ATR，shift 1），跨年度（金價 1,800 → 4,300）可比較。
"""
import numpy as np
import pandas as pd
from load import load
from tpo import poc_va

DATA = "data/XAUUSD_M1_2023_2026.csv"
SESSIONS = {"ASIA": (1 * 60, 9 * 60 + 59), "LDN": (10 * 60, 15 * 60 + 29), "NY": (15 * 60 + 30, 23 * 60 + 54)}
SESS_ORDER = ["ASIA", "LDN", "NY"]
OR_MIN = 30


def load_m1(path=DATA):
    d = load(path)
    d["day"] = (d.index - pd.Timedelta(hours=1)).normalize()
    d["mod"] = d.index.hour * 60 + d.index.minute          # minute of day（broker）
    return d


def tick_for(price):
    return max(0.25, 0.25 * round(price / 1000))          # 2,000 → 0.5、4,000 → 1.0


def day_profile(g):
    """30 分 TPO 剖面 → POC / VAH / VAL。"""
    tick = tick_for(g.close.iloc[-1])
    lo = np.floor(g.low.min() / tick) * tick
    n = int(round((np.floor(g.high.max() / tick) * tick - lo) / tick)) + 1
    counts = np.zeros(n, int)
    per = (g["mod"].values - 60) // 30
    H, L = g.high.values, g.low.values
    for p in np.unique(per):
        m = per == p
        a = int(round((np.floor(L[m].min() / tick) * tick - lo) / tick))
        b = int(round((np.floor(H[m].max() / tick) * tick - lo) / tick))
        counts[a:b + 1] += 1
    prices = lo + tick * np.arange(n)
    poc, vah, val = poc_va(prices, counts)
    return poc + tick / 2, vah + tick, val


def open_type(o, h, l, c, orh, orl):
    """Dalton 開盤型態（以 OR 期間的 M1 為準，簡化版）。
    OD  Open Drive：開盤價在 OR 一端 15% 內，收在另一端 25% 內（一路單邊）
    OTD Open Test Drive：先往反方向測試 15~50% 再反向收在另一端 25% 內
    ORR Open Rejection Reverse：先往一邊走 > 50% 後反轉，收在另一端 25% 內
    OA  Open Auction：其他（在區間內來回）
    回傳 (型態, 方向 +1/−1/0)。"""
    w = orh - orl
    if w <= 0:
        return "OA", 0
    po, pc = (o - orl) / w, (c - orl) / w
    for d, adverse, done in ((1, po, pc >= 0.75), (-1, 1 - po, pc <= 0.25)):
        if done:                                            # adverse = 開盤後往反方向走了多少（OR 寬比例）
            if adverse <= 0.15: return "OD", d
            if adverse <= 0.50: return "OTD", d
            return "ORR", d
    return "OA", 0


def build(path=DATA, or_min=OR_MIN):
    d = load_m1(path)
    days = []
    for day, g in d.groupby("day"):
        if len(g) < 1000:
            continue
        poc, vah, val = day_profile(g)
        days.append(dict(day=day, o=g.open.iloc[0], h=g.high.max(), l=g.low.min(), c=g.close.iloc[-1],
                         poc=poc, vah=vah, val=val))
    D = pd.DataFrame(days).set_index("day")
    D["tr"] = np.maximum(D.h - D.l, np.maximum((D.h - D.c.shift()).abs(), (D.l - D.c.shift()).abs()))
    D["atr"] = D.tr.rolling(10).mean().shift(1)
    for k in ["poc", "vah", "val", "h", "l", "c"]:
        D["p" + k] = D[k].shift(1)

    rows = []
    gd = dict(tuple(d.groupby("day")))
    prev_sess_hl = None
    for day in D.index:
        g = gd[day]
        r0 = D.loc[day]
        for s in SESS_ORDER:
            a, b = SESSIONS[s]
            gs = g[(g["mod"] >= a) & (g["mod"] <= b)]
            if len(gs) < 120:
                prev_sess_hl = None
                continue
            orb = gs[gs["mod"] < a + or_min]
            if len(orb) < or_min * 0.6 or gs["mod"].iloc[0] > a + 5:     # 開盤缺資料（假日 / 提早收盤）
                prev_sess_hl = None
                continue
            orh, orl = orb.high.max(), orb.low.min()
            o = orb.open.iloc[0]
            ot, od = open_type(o, orh, orl, orb.close.iloc[-1], orh, orl)
            rows.append(dict(day=day, sess=s, o=o, orh=orh, orl=orl, orm=(orh + orl) / 2, orw=orh - orl,
                             or_c=orb.close.iloc[-1], otype=ot, odir=od,
                             sh=gs.high.max(), sl=gs.low.min(), sc=gs.close.iloc[-1],
                             atr=r0.atr, ppoc=r0.ppoc, pvah=r0.pvah, pval=r0.pval, pdh=r0.ph, pdl=r0.pl, pdc=r0.pc,
                             psh=prev_sess_hl[0] if prev_sess_hl else np.nan,
                             psl=prev_sess_hl[1] if prev_sess_hl else np.nan,
                             i0=d.index.get_loc(gs.index[0]), i_or=d.index.get_loc(orb.index[-1]) + 1,
                             i1=d.index.get_loc(gs.index[-1])))
            prev_sess_hl = (gs.high.max(), gs.low.min())
    S = pd.DataFrame(rows)
    S = S[S.atr.notna() & S.pvah.notna()].reset_index(drop=True)
    S["year"] = pd.to_datetime(S.day).dt.year
    S["orw_atr"] = S.orw / S.atr
    # OR 大小對過去 20 個同 session 的 z（log）
    S["or_z"] = S.groupby("sess").orw.transform(
        lambda x: (np.log(x) - np.log(x).rolling(20).mean().shift(1)) / np.log(x).rolling(20).std().shift(1))
    S["open_loc"] = np.where(S.o > S.pvah, "above", np.where(S.o < S.pval, "below", "inside"))
    S["va_w_atr"] = (S.pvah - S.pval) / S.atr
    return d, D, S


def arrays(d):
    return d.high.values, d.low.values, d.close.values, d.open.values, d.spread.values * 0.01


def first_touch(H, L, C, i_start, i_end, level):
    """i_start ~ i_end 之間第一根碰到 level 的 K 棒（含 high/low），回傳 (索引, 來自上方 +1 / 下方 −1)。
    來自哪一邊用前一根收盤判斷；若第一根就跨越（開在另一側）以前一根收盤為準。"""
    if i_start >= i_end:
        return -1, 0
    hit = np.where((L[i_start:i_end] <= level) & (H[i_start:i_end] >= level))[0]
    if len(hit) == 0:
        return -1, 0
    j = i_start + hit[0]
    side = 1 if C[j - 1] > level else -1
    return j, side


def barrier(H, L, j, level, direction, b, i_end):
    """從 level 進場（第 j 根），順 direction 走 b 先到 → +1；反向走 b 先到 → −1；同一根兩邊都碰 / 都沒碰 → 0。"""
    if j < 0 or j >= i_end:
        return 0
    hs, ls = H[j:i_end] >= level + b, L[j:i_end] <= level - b
    iu = np.argmax(hs) if hs.any() else 10 ** 9
    idn = np.argmax(ls) if ls.any() else 10 ** 9
    if iu == idn:
        return 0
    return direction if iu < idn else -direction


def race(H, L, j, up, dn, i_end):
    """第 j 根之後先碰 up → +1、先碰 dn → −1、同根或都沒碰 → 0。"""
    if j < 0 or j >= i_end:
        return 0
    hs, ls = H[j:i_end] >= up, L[j:i_end] <= dn
    iu = np.argmax(hs) if hs.any() else 10 ** 9
    idn = np.argmax(ls) if ls.any() else 10 ** 9
    if iu == idn:
        return 0
    return 1 if iu < idn else -1


def tstat(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 3:
        return np.nan
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x)) + 1e-12)
