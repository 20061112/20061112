"""Part 22：ICT / SMC 價位被觸碰後，到底有沒有反應？

資料：data/XAUUSD_M1_2023_2026.csv（2023/6 ~ 2026/10，MT5 bid K 棒 + 點差）。
偵測週期：M15、H1（由 M1 重取樣）；觸碰後的成交與出場一律在 M1 上模擬（像實盤掛限價單）。

偵測的價位（全部只用「當下已收盤」的資料，沒有未來函數）：
  swing  : 碎形擺動點（左右各 n 根），在 s+n 根收盤才確認；依前一個同向擺動點標 HH / LH / HL / LL。
           區域 = 擺動 K 棒的影線（實體端 ~ 極值），寬度限制在 0.1~1 ATR。
  BOS / CHoCH : 收盤突破最近一個未被突破的擺動點；與目前結構方向同向 = BOS，反向 = CHoCH。
           retest 區域 = 被突破價位往回 0.25 ATR（突破上方 → 回測當支撐）。
  OB     : 造成 BOS/CHoCH 的那段走勢起點（擺動點~突破棒之間的最低/最高點）前最後一根反向 K 棒，區域 = 該 K 棒高~低。
  breaker: OB 之後被收盤完全跌破/突破 → 角色互換（多方 OB 變壓力）。
  FVG    : 三根 K 棒缺口 low[i] > high[i-2]（多）/ high[i] < low[i-2]（空），在 i 收盤成立。
  IFVG   : FVG 之後被收盤完全穿越 → 角色互換。
  POC    : 前一個交易日（伺服器日）以 M1 tick volume 計算的成交量最大價位（$0.5 一格），±0.1 ATR。
           新的一天開盤在 POC 上方 → 視為支撐，下方 → 壓力。

觸碰與交易（多方區域；空方鏡像）：
  - 區域成立後，價格第一次回到區域近端（hi）= 第一次觸碰（最多等 1000 根）。成立當下若價格已在區域內則略過。
  - 在 M1 上找實際觸價的那一分鐘，以 min(該分鐘開盤, hi) 成交（限價單）。
  - 反應測試 A（與區域寬度無關）：從成交價起，先 +1 ATR 還是先 −1 ATR（ATR = 偵測週期 ATR14）。
  - 交易測試 B：停損 = 遠端 lo − 0.1 ATR，停利 1R / 2R，最多持有 48 根偵測週期 K 棒，到期以收盤平倉；
    扣除當下點差。成交那一分鐘只檢查停損（不知道高點發生在成交前或後，保守），之後同一分鐘同時碰到停損停利算停損。

對照組（placebo）：
  每個真實區域複製 3 個「假區域」：同一成立時間、同寬度、同方向、同樣在價格的同一側，
  只是把價位隨機平移 0.5~3 ATR。若 ICT/SMC 價位真的有意義，真實區域的反應率應該明顯高於假區域。
"""
import sys
import numpy as np
import pandas as pd
from load import load

RNG = np.random.default_rng(22)
MAX_AGE = 1000
N_PLACEBO = 3
HOLD = 48


def resample(m1, rule):
    return m1.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "tickvol": "sum", "spread": "max"}).dropna()


def atr14(h, l, c):
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))
    return pd.Series(tr).rolling(14).mean().to_numpy()


# ---------------------------------------------------------------- 價位偵測
def detect(df, n_sw=3):
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    A = atr14(h, l, c)
    N = len(df)
    zones = []          # (kind, dir, lo, hi, created_idx)

    def add(kind, d, lo, hi, i):
        if i < N - 1 and A[i] > 0 and hi > lo:
            zones.append((kind, d, lo, hi, i))

    # 結構方向序列（給順/逆勢分組用）
    trend_arr = np.zeros(N, int)
    trend = 0
    last_sh = last_sl = None        # [idx, level, broken]
    prev_sh_lvl = prev_sl_lvl = None
    obs = []                        # (kind, dir, lo, hi, idx)
    for i in range(2 * n_sw + 1, N):
        s = i - n_sw
        a = A[i]
        if a > 0:
            win_h = h[s - n_sw:i + 1]
            win_l = l[s - n_sw:i + 1]
            if h[s] >= win_h.max() and h[s] > h[s - n_sw:s].max():
                lab = "HH" if prev_sh_lvl is not None and h[s] > prev_sh_lvl else "LH"
                top, bot = h[s], max(o[s], c[s])
                bot = min(max(bot, top - a), top - 0.1 * a)
                add("swing_" + lab, -1, bot, top, i)
                prev_sh_lvl = h[s]
                last_sh = [s, h[s], False]
            if l[s] <= win_l.min() and l[s] < l[s - n_sw:s].min():
                lab = "LL" if prev_sl_lvl is not None and l[s] < prev_sl_lvl else "HL"
                bot, top = l[s], min(o[s], c[s])
                top = max(min(top, bot + a), bot + 0.1 * a)
                add("swing_" + lab, 1, bot, top, i)
                prev_sl_lvl = l[s]
                last_sl = [s, l[s], False]
            # 突破
            if last_sh is not None and not last_sh[2] and c[i] > last_sh[1]:
                ev = "CHoCH" if trend == -1 else "BOS"
                trend = 1
                last_sh[2] = True
                L = last_sh[1]
                add(ev + "_retest", 1, L - 0.25 * a, L, i)
                seg = slice(last_sh[0], i + 1)
                m = last_sh[0] + int(np.argmin(l[seg]))
                k = m
                while k > max(m - 5, last_sh[0]) and not c[k] < o[k]:
                    k -= 1
                if not c[k] < o[k]:
                    k = m
                if h[k] < c[i]:
                    obs.append(("OB_" + ev, 1, l[k], h[k], i))
            if last_sl is not None and not last_sl[2] and c[i] < last_sl[1]:
                ev = "CHoCH" if trend == 1 else "BOS"
                trend = -1
                last_sl[2] = True
                L = last_sl[1]
                add(ev + "_retest", -1, L, L + 0.25 * a, i)
                seg = slice(last_sl[0], i + 1)
                m = last_sl[0] + int(np.argmax(h[seg]))
                k = m
                while k > max(m - 5, last_sl[0]) and not c[k] > o[k]:
                    k -= 1
                if not c[k] > o[k]:
                    k = m
                if l[k] > c[i]:
                    obs.append(("OB_" + ev, -1, l[k], h[k], i))
        trend_arr[i] = trend

    for kind, d, lo, hi, i in obs:
        add(kind, d, lo, hi, i)
        j = flip_idx(c, d, lo, hi, i)
        if j is not None:
            add("breaker", -d, lo, hi, j)

    # FVG / IFVG
    for i in range(2, N):
        if l[i] > h[i - 2]:
            lo, hi, d = h[i - 2], l[i], 1
        elif h[i] < l[i - 2]:
            lo, hi, d = h[i], l[i - 2], -1
        else:
            continue
        add("FVG", d, lo, hi, i)
        j = flip_idx(c, d, lo, hi, i)
        if j is not None:
            add("IFVG", -d, lo, hi, j)
    return zones, A, trend_arr


def flip_idx(c, d, lo, hi, i):
    """區域成立後第一根收盤完全穿越遠端的 K 棒。"""
    seg = c[i + 1:i + 1 + MAX_AGE]
    hit = seg < lo if d == 1 else seg > hi
    if not hit.any():
        return None
    return i + 1 + int(np.argmax(hit))


def poc_zones(m1, df, A):
    """前一日 POC（tick volume）。"""
    zones = []
    day = m1.index.normalize()
    t_idx = df.index
    for dt, g in m1.groupby(day):
        px = np.round(g.close.to_numpy() * 2) / 2
        vol = pd.Series(g.tickvol.to_numpy(float)).groupby(px).sum()
        poc = float(vol.idxmax())
        nxt = dt + pd.Timedelta(days=1)
        i = int(t_idx.searchsorted(nxt))        # 下一個交易日第一根
        if i >= len(df) - 1 or not A[i] > 0:
            continue
        w = 0.1 * A[i]
        op = df.open.iloc[i]
        d = 1 if op > poc + w else -1 if op < poc - w else 0
        if d:
            zones.append(("POC", d, poc - w, poc + w, i))
    return zones


# ---------------------------------------------------------------- 觸碰 + M1 模擬
def make_placebos(zones, c):
    out = []
    for z in zones:
        kind, d, lo, hi, i = z[:5]
        for _ in range(N_PLACEBO):
            for _try in range(10):
                sh = RNG.uniform(0.5, 3.0) * RNG.choice([-1, 1]) * A_glob[i]
                nlo, nhi = lo + sh, hi + sh
                if (d == 1 and nhi < c[i]) or (d == -1 and nlo > c[i]):
                    out.append((kind, d, nlo, nhi, i) + tuple(z[5:]))
                    break
    return out


def run(zones, df, A, trend_arr, m1a, m1_start, tf_min, placebo, rev=False):
    """rev=True：觸碰時改做「穿越」方向（停損單在同一價位成交），停損 1 ATR、停利 1/2 ATR。"""
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    mo, mh, ml, mc, msp = m1a
    N = len(df)
    W = HOLD * tf_min
    rows = []
    for z in zones:
        kind, d, lo, hi, i = z[:5]
        a = A[i]
        # 成立當下必須在區域外
        if (d == 1 and c[i] <= hi) or (d == -1 and c[i] >= lo):
            continue
        seg = slice(i + 1, min(i + 1 + MAX_AGE, N))
        hit = l[seg] <= hi if d == 1 else h[seg] >= lo
        if not hit.any():
            continue
        j = i + 1 + int(np.argmax(hit))
        k0, k1 = m1_start[j], m1_start[j + 1] if j + 1 < N else len(mo)
        ms = ml[k0:k1] <= hi if d == 1 else mh[k0:k1] >= lo
        if not ms.any():
            continue
        k = k0 + int(np.argmax(ms))
        entry = min(mo[k], hi) if d == 1 else max(mo[k], lo)
        sl = lo - 0.1 * a if d == 1 else hi + 0.1 * a
        if rev:
            d = -d
            sl = entry - d * a
        risk = d * (entry - sl)
        if risk <= 0.02 * a:
            continue
        end = min(k + W, len(mo))
        HH, LL = mh[k + 1:end], ml[k + 1:end]
        spr = msp[k]
        res = dict(kind=kind, d=d, t=j, age=j - i, atr=a, width=(hi - lo) / a, risk_atr=risk / a,
                   aligned=int(trend_arr[j - 1] == d), placebo=placebo,
                   year=df.index[j].year, zid=z[5] if len(z) > 5 else -1)
        # A：±1 ATR 誰先到（也做 ±0.5 ATR）
        for m, tag in ((1.0, "b1"), (0.5, "b05")):
            up = entry + d * m * a
            dn = entry - d * m * a
            first_bad = ml[k] <= dn if d == 1 else mh[k] >= dn
            if first_bad:
                res[tag] = 0.0
                continue
            g = HH >= up if d == 1 else LL <= up
            b = LL <= dn if d == 1 else HH >= dn
            gi = int(np.argmax(g)) if g.any() else 10**9
            bi = int(np.argmax(b)) if b.any() else 10**9
            res[tag] = np.nan if gi == bi == 10**9 else float(gi < bi)
        # B：停損在區域外，1R/2R
        stop_now = ml[k] <= sl if d == 1 else mh[k] >= sl
        for R, tag in ((1.0, "r1"), (2.0, "r2")):
            tp = entry + d * R * risk
            if stop_now:
                pnl = -risk
            else:
                g = HH >= tp if d == 1 else LL <= tp
                b = LL <= sl if d == 1 else HH >= sl
                gi = int(np.argmax(g)) if g.any() else 10**9
                bi = int(np.argmax(b)) if b.any() else 10**9
                if gi == bi == 10**9:
                    pnl = d * (mc[end - 1] - entry)
                elif bi <= gi:
                    pnl = -risk
                else:
                    pnl = R * risk
            res[tag] = (pnl - spr) / risk
        # C：觸碰後 8 根偵測週期內的有利/不利最大幅度（ATR）
        e8 = min(k + 8 * tf_min, len(mo))
        res["mfe8"] = (mh[k:e8].max() - entry) / a if d == 1 else (entry - ml[k:e8].min()) / a
        res["mae8"] = (entry - ml[k:e8].min()) / a if d == 1 else (mh[k:e8].max() - entry) / a
        rows.append(res)
    return rows


def summarize(ev, by):
    def agg(g):
        r, p = g[g.placebo == 0], g[g.placebo == 1]
        out = {"N": len(r)}
        for col in ("b1", "b05", "r2"):
            rv, pv = r[col].dropna(), p[col].dropna()
            out[col] = rv.mean()
            out[col + "_pl"] = pv.mean()
            se = np.sqrt(rv.var() / max(len(rv), 1) + pv.var() / max(len(pv), 1))
            out[col + "_t"] = (rv.mean() - pv.mean()) / se if se > 0 else np.nan
        out["mfe-mae"] = (r.mfe8 - r.mae8).mean()
        out["mfe-mae_pl"] = (p.mfe8 - p.mae8).mean()
        return pd.Series(out)
    return ev.groupby(by).apply(agg, include_groups=False)


A_glob = None


def main():
    global A_glob
    m1 = load("data/XAUUSD_M1_2023_2026.csv")
    m1a = tuple(m1[k].to_numpy(float) for k in ("open", "high", "low", "close")) + \
        (m1.spread.to_numpy(float) * 0.01,)
    all_ev = []
    for rule, tf_min in (("15min", 15), ("1h", 60)):
        df = resample(m1, rule)
        zones, A, tr = detect(df)
        zones += poc_zones(m1, df, A)
        A_glob = A
        c = df.close.to_numpy(float)
        m1_start = m1.index.searchsorted(df.index)
        pl = make_placebos(zones, c)
        ev = pd.DataFrame(run(zones, df, A, tr, m1a, m1_start, tf_min, 0) +
                          run(pl, df, A, tr, m1a, m1_start, tf_min, 1))
        ev["tf"] = rule
        all_ev.append(ev)
        print(rule, "zones", len(zones), "events", len(ev), file=sys.stderr)
    ev = pd.concat(all_ev, ignore_index=True)
    ev.to_pickle("part22_events.pkl")


if __name__ == "__main__":
    main()
