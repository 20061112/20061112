"""第二十四部分 (2)：同樣的 ER 減速訊號，反過來「順原趨勢」做（把減速當回檔）。

進場：訊號 K 棒下一根開盤，方向 = 前段方向 d
停損：前 M 根（中 ER 長度）反向極值外 0.1 ATR（做多 = 前 M 根最低點）
出場：tp 1R / 2R / 3R，最長 2L 根；其餘同 part24_er_cascade.py
→ part24_cont.csv
"""
import itertools
import sys
import numpy as np
import pandas as pd
from part23_ma_cycle import load_m1, resample, st, SPLIT
from part24_er_cascade import TFS, TRIPLES, A_, B_, C_, ser, atr, signals


def sim(o, h, l, c, spr, A, t, side, LB, hold, tp, buf=0.1):
    n = len(c)
    res = np.full(len(t), np.nan)
    for i, (k, s) in enumerate(zip(t, side)):
        e = k + 1
        if e >= n or k < LB:
            continue
        ext = l[k - LB + 1:k + 1].min() if s == 1 else h[k - LB + 1:k + 1].max()
        stop = ext - s * buf * A[k]
        ep = o[e]
        R = s * (ep - stop)
        if not R > 0:
            continue
        tgt = ep + s * tp * R
        end = min(e + hold, n)
        hs = (l[e:end] <= stop) if s == 1 else (h[e:end] >= stop)
        ht = (h[e:end] >= tgt) if s == 1 else (l[e:end] <= tgt)
        ks = hs.argmax() if hs.any() else 10 ** 9
        kt = ht.argmax() if ht.any() else 10 ** 9
        if ks == kt == 10 ** 9:
            xp = c[end - 1]
        elif ks <= kt:
            xp = stop if ks == 0 else (min(stop, o[e + ks]) if s == 1 else max(stop, o[e + ks]))
        else:
            xp = tgt
        res[i] = (s * (xp - ep) - spr[e]) / R
    return res


def main():
    m1 = load_m1()
    rng = np.random.default_rng(1)
    rows = []
    for tf in TFS:
        df = resample(m1, tf)
        o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        spr = df.spread.to_numpy(float) * 0.01
        A = atr(h, l, c)
        half = np.where(df.index < SPLIT, 1, 2)
        E = {n: ser(c, n) for n in {x for tri in TRIPLES for x in tri}}
        for L, M, S in TRIPLES:
            sets = [("random", np.nan, np.nan, np.nan)] + \
                   [(k, a, b, cc) for k, a, b, cc in itertools.product(["full", "noM"], A_, B_, C_)
                    if not (k == "noM" and b != B_[0])]
            for kind, a, b, cc in sets:
                if kind == "random":
                    t = np.sort(rng.integers(L + 20, len(c) - 2 * L - 2, size=min(20000, len(c) // 3)))
                    side = rng.choice([-1, 1], size=len(t))
                else:
                    t, rev = signals(E, L, M, S, a, b, cc, kind)
                    keep = (t > 30) & (t < len(c) - 2)
                    t, side = t[keep], -rev[keep]
                if len(t) < 10:
                    continue
                for tp in (1, 2, 3):
                    R = sim(o, h, l, c, spr, A, t, side, M, 2 * L, tp)
                    G = sim(o, h, l, c, np.zeros_like(spr), A, t, side, M, 2 * L, tp)
                    n_, mu, tt, win, pf = st(R)
                    rows.append(dict(tf=tf, L=L, M=M, S=S, kind=kind, a=a, b=b, c=cc, tp=tp, n=n_,
                                     gross=np.nanmean(G), avgR=mu, t=tt, win=win, pf=pf,
                                     h1=st(R[half[t] == 1])[1], h2=st(R[half[t] == 2])[1]))
            print(tf, L, file=sys.stderr, flush=True)
    pd.DataFrame(rows).to_csv("part24_cont.csv", index=False)


if __name__ == "__main__":
    main()
