"""第二十二部分：TD Sequential（9 轉 13 展）在黃金上是否有作用。

資料：data/XAUUSD_M1_2026.csv（2026/1/2 ~ 10/8，4/3~4/14 缺資料），重採樣成 M1~H4。
1. 前瞻報酬：訊號 K 棒收盤後 k 根，往訊號方向的報酬（ATR14 單位）。
2. 交易模擬：下一根開盤進場；停損 = setup / countdown 極值外 0.1 ATR；
   停利 tp × R；超過 H 根收盤平倉；同一根同時碰停損停利算停損；每筆扣點差一次。
3. 樣本拆半：1/2~5/31 vs 6/1~10/8。
"""
import numpy as np
import pandas as pd
from load import load
from td import resample, atr, td_signals

TFS = ["1min", "5min", "15min", "30min", "1h", "4h"]
KS = [1, 3, 5, 10, 20, 40]
SPLIT = pd.Timestamp("2026-06-01")


def fwd_table(df, sig, A):
    c = df.close.to_numpy()
    n = len(c)
    rows = {}
    for k in KS:
        v = np.full(len(sig), np.nan)
        ok = sig.i.to_numpy() + k < n
        i = sig.i.to_numpy()[ok]
        v[ok] = (c[i + k] - c[i]) / A[i] * sig.dir.to_numpy()[ok]
        rows[k] = v
    return pd.DataFrame(rows, index=sig.index)


def simulate(df, sig, A, tp, H, buf=0.1):
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    spr = df.spread.to_numpy(float) * 0.01
    n = len(c)
    res = []
    busy = -1
    for r in sig.itertuples():
        t, d = r.i, r.dir
        if t + 1 >= n or t + 1 <= busy:
            res.append(np.nan)
            continue
        e = o[t + 1]
        stop = r.ext - d * buf * A[t]
        risk = (e - stop) * d
        if not risk > 0:
            res.append(np.nan)
            continue
        tgt = e + d * tp * risk
        out = None
        for j in range(t + 1, min(t + 1 + H, n)):
            if (l[j] <= stop) if d == 1 else (h[j] >= stop):
                out, x = j, stop
                break
            if (h[j] >= tgt) if d == 1 else (l[j] <= tgt):
                out, x = j, tgt
                break
        if out is None:
            out = min(t + H, n - 1)
            x = c[out]
        busy = out
        res.append(((x - e) * d - spr[t + 1]) / risk)
    return np.array(res)


def stat(v):
    v = v[~np.isnan(v)]
    if len(v) < 5:
        return dict(n=len(v), mean=np.nan, t=np.nan)
    return dict(n=len(v), mean=v.mean(), t=v.mean() / v.std(ddof=1) * np.sqrt(len(v)))


def main():
    m1 = load("data/XAUUSD_M1_2026.csv")
    lines = []
    P = lines.append
    summary = []
    for tf in TFS:
        df = resample(m1, tf)
        A = atr(df)
        sig = td_signals(df)
        sig = sig[~np.isnan(A[sig.i])].reset_index(drop=True)
        sig["time"] = df.index[sig.i]
        sig["half"] = np.where(sig.time < SPLIT, "H1", "H2")
        F = fwd_table(df, sig, A)
        # 同週期所有 K 棒的無條件報酬（多空各自），用來扣掉漂移
        c = df.close.to_numpy()
        P(f"\n=== {tf}  bars={len(df)}  S9={int((sig.kind=='S9').sum())}  C13={int((sig.kind=='C13').sum())} ===")
        P("前瞻報酬（往訊號方向，ATR 單位；t 為 t 值）")
        P(f"{'group':<18}" + "".join(f"{'k='+str(k):>16}" for k in KS))
        groups = {
            "S9 all": sig.kind == "S9",
            "S9 perfected": (sig.kind == "S9") & sig.perf,
            "S9 not perf": (sig.kind == "S9") & ~sig.perf,
            "S9 buy": (sig.kind == "S9") & (sig.dir == 1),
            "S9 sell": (sig.kind == "S9") & (sig.dir == -1),
            "C13 all": sig.kind == "C13",
            "C13 buy": (sig.kind == "C13") & (sig.dir == 1),
            "C13 sell": (sig.kind == "C13") & (sig.dir == -1),
        }
        for g, m in groups.items():
            for half in ("all", "H1", "H2"):
                mm = m if half == "all" else m & (sig.half == half)
                s = [stat(F.loc[mm, k].to_numpy()) for k in KS]
                P(f"{g + ' ' + half:<18}" + "".join(f"{x['mean']:>+8.3f}({x['t']:>+5.1f})" for x in s)
                  + f"  n={s[0]['n']}")
                if half == "all":
                    for k, x in zip(KS, s):
                        summary.append(dict(tf=tf, group=g, k=k, n=x["n"], mean=x["mean"], t=x["t"]))
        # 隨機基準：所有 K 棒、隨機方向的 k 根報酬標準差，用來判斷訊號均值的大小
        P("交易模擬（R，已扣點差）：tp × R / 最多持有 H 根")
        for g in ("S9 all", "S9 perfected", "C13 all"):
            m = groups[g]
            sub = sig[m].reset_index(drop=True)
            for tp in (1.0, 2.0, 3.0):
                for H in (10, 20, 40):
                    r = simulate(df, sub, A, tp, H)
                    ok = ~np.isnan(r)
                    parts = []
                    for half in ("H1", "H2"):
                        rr = r[ok & (sub.half == half).to_numpy()]
                        pf = rr[rr > 0].sum() / -rr[rr < 0].sum() if (rr < 0).any() else np.nan
                        parts.append(f"{half}: n={len(rr):4d} avg={rr.mean():+.3f}R win={np.mean(rr > 0):.0%} PF={pf:.2f}")
                    rr = r[ok]
                    P(f"  {g:<13} tp={tp:.0f} H={H:<3} n={len(rr):5d} avg={rr.mean():+.3f}R  sum={rr.sum():+7.1f}R | "
                      + " | ".join(parts))
    txt = "\n".join(lines)
    print(txt)
    open("results_part22.txt", "w").write(txt + "\n")
    pd.DataFrame(summary).to_csv("part22_fwd_summary.csv", index=False)


if __name__ == "__main__":
    main()
