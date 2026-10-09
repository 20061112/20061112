"""第二十四部分：多長度 ER 依序減速 → 抓動能開始反轉的點。

帶方向的 ER：ER_n(t) = (c_t − c_{t−n}) / Σ|c_i − c_{i−1}|（i = t−n+1..t），範圍 −1 ~ +1
前段方向 d = sign(ER_L)

訊號（K 棒 t 收盤判定，下一根開盤反向進場 = −d）
  長：d·ER_L ≥ a          前一段有明顯、效率高的趨勢
  中：d·ER_M ≥ b 且 d·ER_M(t) < d·ER_M(t−S)   中期速度還在，但已經開始變慢
  短：d·ER_S ≤ c          最近 S 根已經走平或反向
  同一段趨勢只取第一個訊號（M 根內不重複）

消融（看每一層有沒有用）
  full   : 長 + 中(減速) + 短
  noM    : 長 + 短（不看中）
  accel  : 長 + 中「加速」(d·ER_M(t) > d·ER_M(t−S)) + 短  ← 對照
  onlyL  : 長 + 短 > 0（還在走，沒有減速）← 對照：順勢不反轉的點

結果
  前瞻報酬：往反轉方向 k 根（ATR14 單位）
  交易：下一根開盤進，停損 = 前 L 根極值外 0.1 ATR，tp 1R / 2R，最長 L 根；同根停損停利算停損；扣點差
  隨機基準：同週期隨機 K 棒、隨機方向、停損用前 L 根極值
資料：data/XAUUSD_M1_2026.csv（+ 上傳檔補 4/3~4/6），拆半 1/2~5/31 vs 6/1~10/8
"""
import itertools
import sys
import numpy as np
import pandas as pd
from part23_ma_cycle import load_m1, resample, st, SPLIT

TFS = ["1min", "5min", "15min", "30min", "1h"]
TRIPLES = [(20, 10, 2), (20, 10, 3), (14, 7, 2), (30, 15, 3), (40, 20, 5), (60, 30, 5)]
A_ = [0.3, 0.5]
B_ = [0.0, 0.3]
C_ = [0.0, -0.3]
KINDS = ["full", "noM", "accel", "onlyL"]
KS = [1, 3, 5, 10, 20]
TPS = [1, 2]


def ser(c, n):
    d = np.abs(np.diff(c, prepend=np.nan))
    path = pd.Series(d).rolling(n).sum().to_numpy()
    disp = c - np.r_[np.full(n, np.nan), c[:-n]]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(path > 0, disp / path, 0.0)


def atr(h, l, c, n=14):
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))
    return pd.Series(tr).rolling(n).mean().to_numpy()


def signals(E, L, M, S, a, b, cc, kind):
    eL, eM, eS = E[L], E[M], E[S]
    d = np.sign(eL)
    eMp = np.r_[np.full(S, np.nan), eM[:-S]]
    with np.errstate(invalid="ignore"):
        cond = d * eL >= a
        if kind == "full":
            cond &= (d * eM >= b) & (d * eM < d * eMp) & (d * eS <= cc)
        elif kind == "noM":
            cond &= d * eS <= cc
        elif kind == "accel":
            cond &= (d * eM >= b) & (d * eM > d * eMp) & (d * eS <= cc)
        elif kind == "onlyL":
            cond &= (d * eM >= b) & (d * eS > 0.3)
    idx = np.flatnonzero(cond)
    out, last = [], -10 ** 9
    for t in idx:
        if t - last > M:
            out.append(t)
        last = t
    t = np.array(out, int)
    return t, -d[t].astype(int)          # 反轉方向


def sim(o, h, l, c, spr, A, t, side, L, tp, buf=0.1):
    n = len(c)
    res = np.full(len(t), np.nan)
    for i, (k, s) in enumerate(zip(t, side)):
        e = k + 1
        if e >= n or k < L:
            continue
        ext = h[k - L + 1:k + 1].max() if s == -1 else l[k - L + 1:k + 1].min()
        stop = ext - s * buf * A[k]
        ep = o[e]
        R = s * (ep - stop)
        if not R > 0:
            continue
        tgt = ep + s * tp * R
        end = min(e + L, n)
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
    rng = np.random.default_rng(0)
    fw, tr = [], []
    for tf in TFS:
        df = resample(m1, tf)
        o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        spr = df.spread.to_numpy(float) * 0.01
        A = atr(h, l, c)
        half = np.where(df.index < SPLIT, 1, 2)
        E = {n: ser(c, n) for n in {x for tri in TRIPLES for x in tri}}
        ndays = len(np.unique(df.index.date))
        for L, M, S in TRIPLES:
            # 隨機基準
            rt = rng.integers(L + 20, len(c) - L - 2, size=min(20000, len(c) // 3))
            rs = rng.choice([-1, 1], size=len(rt))
            for tp in TPS:
                R = sim(o, h, l, c, spr, A, rt, rs, L, tp)
                n_, mu, tt, win, pf = st(R)
                gross = st(sim(o, h, l, c, np.zeros_like(spr), A, rt, rs, L, tp))[1]
                tr.append(dict(tf=tf, L=L, M=M, S=S, kind="random", a=np.nan, b=np.nan, c=np.nan, tp=tp, n=n_,
                               per_day=np.nan, gross=gross, avgR=mu, t=tt, win=win, pf=pf,
                               h1=st(R[half[rt] == 1])[1], h2=st(R[half[rt] == 2])[1]))
            for kind, a, b, cc in itertools.product(KINDS, A_, B_, C_):
                if kind in ("noM",) and b != B_[0]:
                    continue
                if kind == "onlyL" and cc != C_[0]:
                    continue
                t, side = signals(E, L, M, S, a, b, cc, kind)
                t, side = t[(t > 30) & (t < len(c) - 2)], side[(t > 30) & (t < len(c) - 2)]
                if len(t) < 10:
                    continue
                for k in KS:
                    ok = t + k < len(c)
                    v = side[ok] * (c[t[ok] + k] - c[t[ok]]) / A[t[ok]]
                    hh = half[t[ok]]
                    fw.append(dict(tf=tf, L=L, M=M, S=S, kind=kind, a=a, b=b, c=cc, k=k, n=len(v),
                                   fwd=np.nanmean(v), t=st(v)[2], h1=st(v[hh == 1])[1], h2=st(v[hh == 2])[1]))
                for tp in TPS:
                    R = sim(o, h, l, c, spr, A, t, side, L, tp)
                    n_, mu, tt, win, pf = st(R)
                    gross = st(sim(o, h, l, c, np.zeros_like(spr), A, t, side, L, tp))[1]
                    tr.append(dict(tf=tf, L=L, M=M, S=S, kind=kind, a=a, b=b, c=cc, tp=tp, n=n_,
                                   per_day=n_ / ndays, gross=gross, avgR=mu, t=tt, win=win, pf=pf,
                                   h1=st(R[half[t] == 1])[1], h2=st(R[half[t] == 2])[1]))
            print(tf, L, M, S, file=sys.stderr, flush=True)
    pd.DataFrame(fw).to_csv("part24_fwd.csv", index=False)
    pd.DataFrame(tr).to_csv("part24_trades.csv", index=False)


if __name__ == "__main__":
    main()
