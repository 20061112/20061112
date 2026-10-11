"""Fib 回撤 + 確認後進場（不偷看未來）

波段同 fib_pullback.py：ZigZag 確認低點 L 後追蹤目前最高價 H，H-L ≥ 3×ATR。
觸發：確認 TF 的 K 棒碰到 0.382 → 錨定 Fib（H、L 固定）。
      之後回撤碰到的最深 Fib 位 = 「區」，止損 = 下一個 Fib 位
      （0.382→0.5、0.5→0.618、0.618→0.786、0.786→1.0）；跌破 L 取消。
確認進場（每段行情最多一筆）：
  A 收盤確認：確認 K 棒收陽，且收盤 > 前一根高點 → 收盤價進場
  B 突破確認：價格突破前一根確認 K 棒的高點 → 在突破價進場（用 M1 判斷先後）
停利：TP0 = H；TP127 = 1.272 延伸。出場用 M1，同根碰止損/停利算止損，扣點差。
確認 TF：M15 波段用 M5，H1 波段用 M15。
"""
import numpy as np
import pandas as pd
from fib_pullback import Data, resample, ema_ok, stats, TFLEN, MINLEG, MAXHOLD

LV = [0.382, 0.5, 0.618, 0.786, 1.0]
CONF_TF = {"M1": "1min", "M5": "1min", "M15": "5min", "H1": "15min"}


def exit_sim(D, i, s, entry, sl, tp, cost):
    """從 M1 第 i 根（含）開始管理部位。"""
    hh = D.h if s == 1 else -D.l
    ll = D.l if s == 1 else -D.h
    oo = D.o * s
    j1 = min(i + MAXHOLD, len(D.t))
    stop = ll[i:j1] <= sl
    win = hh[i:j1] >= tp
    js = int(np.argmax(stop)) if stop.any() else 10**9
    jw = int(np.argmax(win)) if win.any() else 10**9
    risk = entry - sl
    if js == 10**9 and jw == 10**9:
        return (D.c[j1 - 1] * s - entry - cost) / risk, "時間"
    if js <= jw:
        px = min(sl, oo[i + js])
        return (px - entry - cost) / risk, "SL"
    return (tp - entry - cost) / risk, "TP"


class Conf:
    def __init__(self, D, rule):
        c = resample(D.m1, rule)
        self.start = c.index.values
        self.end = (c.index + pd.Timedelta(rule)).values
        self.o, self.h, self.l, self.c = (c[x].values for x in ("open", "high", "low", "close"))


def run_leg(D, C, lg, filt, tp_mode, mode, arm=0.382):
    s, L = lg["s"], lg["L"]
    Ls = L * s
    ch, cl = (C.h, C.l) if s == 1 else (-C.l, -C.h)
    co, cc = C.o * s, C.c * s
    # 錨定前的 H：L 那根到可掛單時間為止
    a0 = np.searchsorted(D.t, lg["tL"]); a1 = np.searchsorted(D.t, lg["t_act"])
    if a1 <= a0:
        return None
    H = (D.h[a0:a1].max() if s == 1 else -D.l[a0:a1].min())
    b0 = np.searchsorted(C.start, lg["t_act"])
    b1 = np.searchsorted(C.start, lg["t_cancel"])
    if filt in ("hhhl", "hhhl+ema"):
        if not ((L > lg["prevL"]) if s == 1 else (L < lg["prevL"])):
            return None
    anchored, deep = False, 0.0
    for b in range(b0, min(b1, len(C.start))):
        if not anchored:
            rng = H - Ls
            if rng >= MINLEG * lg["atr"] and cl[b] <= H - arm * rng:
                if filt in ("hhhl", "hhhl+ema") and not (H > lg["prevH"] * s):
                    return None
                anchored = True
                deep = (H - cl[b]) / rng
                if deep >= 1.0:
                    return None
                continue        # 碰到的這根不當確認棒
            H = max(H, ch[b])
            continue
        prev_hi = ch[b - 1]
        if mode == "A":
            deep = max(deep, (H - cl[b]) / rng)
            if deep >= 1.0:
                return None
            if not (cc[b] > co[b] and cc[b] > prev_hi):
                continue
            entry, t_fill = cc[b], C.end[b]
            i_next = np.searchsorted(D.t, C.end[b])
            cost_i = i_next - 1
        else:
            if ch[b] <= prev_hi:
                deep = max(deep, (H - cl[b]) / rng)
                if deep >= 1.0:
                    return None
                continue
            m0 = np.searchsorted(D.t, C.start[b]); m1 = np.searchsorted(D.t, C.end[b])
            mh = (D.h if s == 1 else -D.l)[m0:m1]
            ml = (D.l if s == 1 else -D.h)[m0:m1]
            k = int(np.argmax(mh > prev_hi))
            deep = max(deep, (H - ml[:k + 1].min()) / rng)   # 同一根 M1 假設先低後高
            if deep >= 1.0:
                return None
            entry = max(prev_hi, D.o[m0 + k] * s)
            t_fill = D.t[m0 + k]
            i_next = m0 + k + 1
            cost_i = m0 + k
        z = max(i for i in range(4) if deep >= LV[i] - 1e-9)
        sl = H - LV[z + 1] * rng
        tp = H if tp_mode == "TP0" else H + 0.272 * rng
        if entry >= tp or entry <= sl:
            return None
        if filt in ("ema", "hhhl+ema") and not ema_ok(D, t_fill, s):
            return None
        R, out = exit_sim(D, i_next, s, entry, sl, tp, D.sp[cost_i])
        return dict(t=t_fill, s=s, 區=LV[z], R=R, out=out,
                    進場位=(H - entry) / rng, 風險佔段=(entry - sl) / rng,
                    風險=entry - sl, 點差=D.sp[cost_i])
    return None


def main():
    D = Data()
    pd.set_option("display.width", 250); pd.set_option("display.float_format", "{:.3f}".format)
    pd.set_option("display.max_rows", 500)
    rows, allt = [], []
    for tf in ["M15", "H1"]:
        legs = D.legs(tf, 3)
        C = Conf(D, CONF_TF[tf])
        for mode in ["A", "B"]:
            for filt in ["none", "ema", "hhhl+ema"]:
                for tp in ["TP0", "TP127"]:
                    tr = [r for lg in legs if (r := run_leg(D, C, lg, filt, tp, mode))]
                    tr = pd.DataFrame(tr).sort_values("t")
                    tr["TF"], tr["確認"], tr["趨勢"], tr["停利"] = tf, mode, filt, tp
                    allt.append(tr)
                    for z, x in [("全部", tr)] + list(tr.groupby("區")):
                        rows.append(dict(TF=tf, 確認=mode, 趨勢=filt, 停利=tp, 區=z,
                                         **stats(x),
                                         多R=x.loc[x.s == 1, "R"].mean(),
                                         空R=x.loc[x.s == -1, "R"].mean(),
                                         平均風險佔段=x["風險佔段"].mean()))
    res = pd.DataFrame(rows)
    print(f"資料 {D.m1.index[0]} ~ {D.m1.index[-1]}\n")
    print(res.to_string(index=False))
    pd.concat(allt).to_csv("fib_confirm_trades.csv", index=False)


if __name__ == "__main__":
    main()
