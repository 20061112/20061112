"""一段行情的 Fib 回撤進場（不偷看未來）

波段：波段 TF 上的 ZigZag（k×ATR14）確認一個低點 L 後，追蹤 L 之後的「目前最高價」H
      （做空相反）。H-L ≥ MINLEG×ATR 才算一段行情。
進場：回撤到 Fib 位掛限價（H 用到前一根 M1 為止的最高價，不用當根）。
      0.382 進場 → 止損 0.5；0.5 → 0.618；0.618 → 0.786；0.786 → 1.0（L）
      每段行情、每個進場位最多一筆；下一個反向轉折被確認後，這段行情的掛單取消。
停利：TP0 = 回到 H（0 位）；TP127 = 1.272 延伸。
撮合：M1。同一根碰到止損和停利 → 算止損；進場那根碰到止損 → 算止損。扣點差。
趨勢確認（進場時已知的資訊）：
  none  不過濾
  hhhl  結構：L 高於前一個確認低點（HL），且 H 高於前一個確認高點（HH）；空單相反
  ema   H1 EMA50 > EMA200 且收盤 > EMA200（用已收的 H1 K 棒）；空單相反
  strong 這段行情 ≥ 5×ATR 且 ER（淨位移/路徑長）≥ 0.5（用波段 TF 收盤算到進場前）
  hhhl+ema 兩者都要
"""
import sys
import numpy as np
import pandas as pd
from load import load
from harmonic import atr

LEVELS = [(0.382, 0.5), (0.5, 0.618), (0.618, 0.786), (0.786, 1.0)]
MINLEG = 3.0
MAXHOLD = 60 * 24 * 10   # 最長持有 10 天（M1 根數）
TFLEN = {"M1": "1min", "M5": "5min", "M15": "15min", "H1": "1h", "H4": "4h"}


def zigzag_conf(h, l, a, k):
    piv, d = [], 0
    ext_i, ext_p = 0, h[0]
    lo_i, lo_p, hi_i, hi_p = 0, l[0], 0, h[0]
    for i in range(1, len(h)):
        th = k * a[i]
        if d == 0:
            if h[i] > hi_p: hi_i, hi_p = i, h[i]
            if l[i] < lo_p: lo_i, lo_p = i, l[i]
            if hi_p - lo_p > th:
                if hi_i > lo_i:
                    piv.append((lo_i, lo_p, -1, i)); d, ext_i, ext_p = 1, hi_i, hi_p
                else:
                    piv.append((hi_i, hi_p, 1, i)); d, ext_i, ext_p = -1, lo_i, lo_p
        elif d == 1:
            if h[i] > ext_p: ext_i, ext_p = i, h[i]
            elif ext_p - l[i] > th:
                piv.append((ext_i, ext_p, 1, i)); d, ext_i, ext_p = -1, i, l[i]
        else:
            if l[i] < ext_p: ext_i, ext_p = i, l[i]
            elif h[i] - ext_p > th:
                piv.append((ext_i, ext_p, -1, i)); d, ext_i, ext_p = 1, i, h[i]
    return piv


def resample(df, rule):
    return df.resample(rule).agg(dict(open="first", high="max", low="min", close="last",
                                      tickvol="sum", spread="mean")).dropna()


class Data:
    def __init__(self):
        m1 = load("data/XAUUSD_M1_2023_2026.csv")
        self.m1 = m1
        self.t = m1.index.values
        self.o, self.h, self.l, self.c = (m1[x].values for x in ("open", "high", "low", "close"))
        self.sp = m1["spread"].values * 0.01
        h1 = resample(m1, "1h")
        e50 = h1["close"].ewm(span=50).mean(); e200 = h1["close"].ewm(span=200).mean()
        up = (e50 > e200) & (h1["close"] > e200)
        dn = (e50 < e200) & (h1["close"] < e200)
        # H1 K 棒在開盤時間 +1h 才收完
        self.ema_t = (h1.index + pd.Timedelta("1h")).values
        self.ema_up, self.ema_dn = up.values, dn.values
        self.tf = {}

    def legs(self, tf, k):
        df = resample(self.m1, TFLEN[tf])
        a = atr(df).bfill().values
        piv = zigzag_conf(df["high"].values, df["low"].values, a, k)
        end = (df.index + pd.Timedelta(TFLEN[tf])).values  # 該 K 棒收完的時間
        out = []
        for j in range(2, len(piv) - 1):
            P, prevE, L = piv[j - 2], piv[j - 1], piv[j]   # L = 起點；P 同向前一轉折；prevE 反向前一轉折
            nxt = piv[j + 1]
            # 下一個與 L 同類的轉折（新的起點）確認時，取消
            cancel = piv[j + 2][3] if j + 2 < len(piv) else len(df) - 1
            out.append(dict(s=-L[2], L=L[1], tL=df.index.values[L[0]], t_act=end[L[3]],
                            t_cancel=end[cancel], atr=a[L[3]],
                            step=int(pd.Timedelta(TFLEN[tf]).total_seconds() // 60),
                            prevL=P[1], prevH=prevE[1]))
        return out


def ema_ok(D, t, s):
    i = np.searchsorted(D.ema_t, t, side="right") - 1
    if i < 0:
        return False
    return bool(D.ema_up[i] if s == 1 else D.ema_dn[i])


def run_leg(D, lg, filt, tp_mode):
    """回傳這段行情在 4 個進場位的交易（list）。"""
    s, L = lg["s"], lg["L"]
    i_start = np.searchsorted(D.t, lg["tL"])            # L 那根 K 棒開始（追蹤 H 從這裡起）
    i0 = np.searchsorted(D.t, lg["t_act"])              # 可以開始掛單
    i1 = np.searchsorted(D.t, lg["t_cancel"])
    if i0 >= i1 or i1 > len(D.t):
        return []
    hh = D.h if s == 1 else -D.l
    ll = D.l if s == 1 else -D.h
    Ls = L * s
    # 「到前一根為止」的最佳價 H
    ext = np.maximum.accumulate(hh[i_start:i1])
    Hprev = np.concatenate([[ext[0]], ext[:-1]])[i0 - i_start:]
    lows = ll[i0:i1]
    rng = Hprev - Ls
    legok = rng >= MINLEG * lg["atr"]
    if filt in ("hhhl", "hhhl+ema"):
        hl_ok = (L > lg["prevL"]) if s == 1 else (L < lg["prevL"])
        if not hl_ok:
            return []
        legok &= Hprev > lg["prevH"] * s
    if filt == "strong":
        legok &= rng >= 5 * lg["atr"]
    trades = []
    for ent_f, sl_f in LEVELS:
        lvl = Hprev - ent_f * rng
        hit = legok & (lows <= lvl)
        if not hit.any():
            continue
        k = int(np.argmax(hit)); i = i0 + k
        t_fill = D.t[i]
        if filt in ("ema", "hhhl+ema") and not ema_ok(D, t_fill, s):
            continue
        H, R0 = Hprev[k], rng[k]
        if filt == "strong":
            seg = D.c[i_start:i][::lg["step"]]   # 用波段 TF 的收盤算 ER
            if len(seg) < 3:
                continue
            er = abs(seg[-1] - seg[0]) / (np.abs(np.diff(seg)).sum() + 1e-9)
            if er < 0.5:
                continue
        entry = min(lvl[k], (D.o[i] if s == 1 else -D.o[i]))   # 跳空低開 → 用開盤價
        sl = H - sl_f * R0
        tp = H if tp_mode == "TP0" else H + 0.272 * R0
        if entry <= sl:
            continue
        risk = entry - sl
        cost = D.sp[i]
        # 進場那根
        if ll[i] <= sl:
            trades.append(dict(lv=ent_f, t=t_fill, R=(-risk - cost) / risk, out="SL"))
            continue
        j1 = min(i + 1 + MAXHOLD, len(D.t))
        hs, ls_ = hh[i + 1:j1], ll[i + 1:j1]
        stop = ls_ <= sl; win = hs >= tp
        js = int(np.argmax(stop)) if stop.any() else 10**9
        jw = int(np.argmax(win)) if win.any() else 10**9
        if js == 10**9 and jw == 10**9:
            px = D.c[j1 - 1] * s
            trades.append(dict(lv=ent_f, t=t_fill, R=(px - entry - cost) / risk, out="時間"))
        elif js <= jw:
            px = min(sl, (D.o[i + 1 + js] if s == 1 else -D.o[i + 1 + js]))
            trades.append(dict(lv=ent_f, t=t_fill, R=(px - entry - cost) / risk, out="SL"))
        else:
            trades.append(dict(lv=ent_f, t=t_fill, R=(tp - entry - cost) / risk, out="TP"))
    for x in trades:
        x["s"] = s
    return trades


def stats(x):
    R = x["R"].values
    n = len(R)
    if n < 2:
        return dict(筆數=n)
    h = n // 2
    return dict(筆數=n, 勝率=(R > 0).mean(), 平均R=R.mean(),
                t值=R.mean() / (R.std(ddof=1) / np.sqrt(n)), 總R=R.sum(),
                前半R=R[:h].mean(), 後半R=R[h:].mean())


def main():
    tfs = sys.argv[1].split(",") if len(sys.argv) > 1 else ["M15", "H1"]
    k = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
    D = Data()
    pd.set_option("display.width", 250); pd.set_option("display.float_format", "{:.3f}".format)
    pd.set_option("display.max_rows", 500)
    rows, keep = [], {}
    for tf in tfs:
        legs = D.legs(tf, k)
        for filt in ["none", "hhhl", "ema", "strong", "hhhl+ema"]:
            for tp in ["TP0", "TP127"]:
                tr = [x for lg in legs for x in run_leg(D, lg, filt, tp)]
                if not tr:
                    continue
                tr = pd.DataFrame(tr).sort_values("t")
                keep[(tf, filt, tp)] = tr
                for lv, x in tr.groupby("lv"):
                    rows.append(dict(TF=tf, 趨勢=filt, 停利=tp, 進場=lv, **stats(x)))
    res = pd.DataFrame(rows)
    print(f"資料 {D.m1.index[0]} ~ {D.m1.index[-1]}，ZigZag {k}×ATR，波段 ≥ {MINLEG}×ATR\n")
    print(res.to_string(index=False))
    return res, keep


if __name__ == "__main__":
    main()
