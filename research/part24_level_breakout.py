"""第二十四部分：既然前高前低容易被穿越 → 直接做突破單。

規則：session 內第一次碰到參考高點（由下往上）→ 在該價位買進（突破單）；碰到參考低點（由上往下）→ 賣出。
      參考 = 前一天高低（D）、前一個 session 高低（S）。若那根 K 棒跳空開在價位外，用開盤價成交（滑價）。
出場：停損 s × ATR、停利 t × ATR（t = None → 只有停損，session 結束平倉）；
      進場那根若碰到停損就算停損（保守），同一根同時碰到停損與停利算停損。每筆扣一次點差。
對照：同樣規則用在 ±0.2 / ±0.4 ATR 的假價位（安慰劑）→ 真價位要比假價位好才算「價位本身」有優勢。
篩選（事前，依第二十三部分的發現）：尖頂（極值只有 1 個 30 分時段碰到）、順價值移動方向、session。
選擇只看 2023~24，2025~26 驗證。
"""
import pickle
import numpy as np
import pandas as pd
from part22_core import load_m1, arrays, first_touch, tstat, SESS_ORDER

YRS = (2023, 2024, 2025, 2026)
EXITS = [(0.1, 0.1), (0.1, 0.2), (0.1, 0.3), (0.2, 0.2), (0.2, 0.4), (0.2, None), (0.3, 0.6), (0.3, None)]
SHIFTS = (0, -0.4, -0.2, 0.2, 0.4)


def sim(H, L, C, O, SP, j, lv, d, atr, i1):
    px = lv if (O[j] - lv) * d <= 0 else O[j]                 # 跳空 → 開盤價成交
    res = {}
    h, l = H[j:i1 + 1], L[j:i1 + 1]
    for s, t in EXITS:
        stop = px - d * s * atr
        hs = np.where((l <= stop) if d == 1 else (h >= stop))[0]
        ks = hs[0] if len(hs) else 10 ** 9
        if t is not None:
            tgt = px + d * t * atr
            ht = np.where((h >= tgt) if d == 1 else (l <= tgt))[0]
            kt = ht[0] if len(ht) else 10 ** 9
            if kt == 0:
                kt = 10 ** 9 if ks == 0 else kt                  # 進場那根：同時碰停損 → 停損
        else:
            kt = 10 ** 9
        if ks <= kt and ks < 10 ** 9:
            pnl = -s * atr
        elif kt < 10 ** 9:
            pnl = t * atr
        else:
            pnl = (C[i1] - px) * d
        res[f"{s}/{t}"] = (pnl - SP[j]) / (s * atr)            # 以 R 計（R = 停損距離）
    return res


def run():
    d = load_m1()
    H, L, C, O, SP = arrays(d)
    S = pickle.load(open("part23_refs.pkl", "rb"))
    D, _ = pickle.load(open("part22_sessions.pkl", "rb"))
    # 價值移動（同 part23_regime）
    lo_ov = np.maximum(D.val, D.val.shift(1)); hi_ov = np.minimum(D.vah, D.vah.shift(1))
    narrow = np.minimum(D.vah - D.val, (D.vah - D.val).shift(1))
    ov = (hi_ov - lo_ov).clip(lower=0) / narrow
    mid = (D.vah + D.val) / 2
    mig = pd.Series(np.where(ov >= 0.5, 0, np.where(mid > mid.shift(1), 1, -1)), index=D.index).shift(1)
    rows = []
    for r in S.itertuples():
        m = mig.get(pd.Timestamp(r.day), np.nan)
        for p in ("D", "S"):
            for e, dr in (("h", 1), ("l", -1)):
                lev = getattr(r, f"{p}_{e}")
                ntop = getattr(r, f"{p}_ntop_{e}")
                for sh in SHIFTS:
                    lv = lev + sh * r.atr
                    j, side = first_touch(H, L, C, r.i0 + 1, r.i1, lv)
                    if j < 0 or -side != dr or j >= r.i1 - 5:      # 只做「往外」穿越
                        continue
                    rows.append(dict(day=r.day, year=r.year, sess=r.sess, ref=p, dir=dr, kind="real" if sh == 0 else "placebo",
                                     sharp=ntop == 1, with_mig=(m == dr), mig=m, mins=j - r.i0,
                                     spread_atr=SP[j] / r.atr, **sim(H, L, C, O, SP, j, lv, dr, r.atr, r.i1)))
    T = pd.DataFrame(rows)
    T.to_csv("part24_trades.csv", index=False)
    return T


def pf(x):
    return x[x > 0].sum() / -x[x < 0].sum() if (x < 0).any() else np.nan


def summarize(T):
    out = ["=" * 125, "第二十四部分：前高前低突破單（R = 停損距離，已扣點差；真價位 vs 假價位用同一規則）", "=" * 125]
    keys = [f"{s}/{t}" for s, t in EXITS]

    def grid(g, label):
        out.append(f"\n  [{label}]  真 n={(g.kind == 'real').sum()}  假 n={(g.kind == 'placebo').sum()}")
        out.append(f"    {'停損/停利(ATR)':<12}{'真 每筆R':>10}{'t':>6}{'PF':>6}{'假 每筆R':>10}{'真−假':>8}{'發現 真−假':>12}{'驗證 真−假':>12}   逐年 真 每筆R")
        for k in keys:
            re, pl = g[g.kind == "real"][k], g[g.kind == "placebo"][k]
            gr, gp = g[g.kind == "real"], g[g.kind == "placebo"]
            dif = lambda a, b: a.mean() - b.mean()
            disc = dif(gr[gr.year <= 2024][k], gp[gp.year <= 2024][k])
            val = dif(gr[gr.year >= 2025][k], gp[gp.year >= 2025][k])
            yr = " ".join(f"{gr[gr.year == y][k].mean():+.3f}" for y in YRS)
            out.append(f"    {k:<12}{re.mean():+10.3f}{tstat(re):+6.1f}{pf(re.values):6.2f}{pl.mean():+10.3f}{re.mean()-pl.mean():+8.3f}"
                       f"{disc:+12.3f}{val:+12.3f}   {yr}")

    grid(T, "全部（前一天 + 前一 session，三個 session）")
    grid(T[T.sharp], "尖頂（極值只有 1 個時段碰到）")
    grid(T[T.with_mig], "順價值移動方向")
    grid(T[T.sharp & T.with_mig], "尖頂 + 順價值移動")
    for s in SESS_ORDER:
        grid(T[T.sess == s], f"{s}")
    grid(T[(T.sess == "LDN") & T.sharp], "LDN 尖頂")
    out.append("\n  點差（ATR）中位數逐年: " + " ".join(f"{y}:{T[T.year == y].spread_atr.median():.4f}" for y in YRS))
    return out


if __name__ == "__main__":
    T = run()
    txt = "\n".join(summarize(T))
    print(txt)
    open("results_part24.txt", "w").write(txt + "\n")
