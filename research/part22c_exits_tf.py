"""第二十二部分 c：M15 順勢突破（A）的停損 / RR 敏感度，以及放到 M5、M1 的結果。
全部 M1 逐根模擬（2023/6 ~ 2026/10），扣點差。

一、停損 ATR × RR：推薦版本（M15、k=3/4/5、N=480、單一持倉），最長 48 根。
    R 的大小隨停損不同，所以另外列「每筆 ATR」（= R × 停損 ATR）以及「每月 R / 回撤」比較風險報酬。
二、不設停利，只用時間出場。
三、更小週期：M5、M1（由 M1 組成），ZigZag k 依時框換算成相近的實際擺幅，N 換算成相同的 5 天。
    ATR 約與 √時間 成比例：M15 的 3 ATR ≈ M5 的 5 ATR ≈ M1 的 11~12 ATR。
"""
import numpy as np
import pandas as pd
import part22_zigzag as z
import part22_m1check as m

MON = (m.M1.index[-1] - m.M1.index[0]).days / 30.4


def signals(d, k, N, tf_min, stop, rr, hold=48):
    z.STOP_ATR, z.TARGET_R, z.MAX_HOLD = stop, rr, hold
    P = z.zigzag(d, k)
    E = z.classify(z.events(d, P, N))
    E["time"] = d.index[E.i.values]
    E["year"] = E.time.dt.year
    S = E[(E.setup == "A") & (E.time >= m.M1.index[0])].copy()
    R, X = m.sim_m1(d, S, tf_min, False, return_exit=True)
    S["R"], S["exit"], S["k"] = R, X, k
    S["entry"] = S.time + pd.Timedelta(minutes=tf_min)
    return S.dropna(subset=["R"])


def single_position(S):
    S = S.sort_values(["entry", "k"])
    keep, busy = [], pd.Timestamp.min
    for idx, e, x in zip(S.index, S.entry, S.exit):
        if e < busy:
            continue
        keep.append(idx)
        busy = x
    return S.loc[keep]


def row(S, **kw):
    r = S.R.values
    eq = np.cumsum(r)
    dd = (np.maximum.accumulate(eq) - eq).max()
    yr = S.groupby("year").R.mean()
    gp, gl = r[r > 0].sum(), -r[r < 0].sum()
    return dict(**kw, 每月筆數=round(len(r) / MON, 1), 期望R=round(r.mean(), 3),
                t=round(r.mean() / r.std() * np.sqrt(len(r)), 2), 勝率=round((r > 0).mean(), 2), PF=round(gp / gl, 2),
                每月R=round(r.sum() / MON, 2), 回撤R=round(dd, 1), 月R除回撤=round(r.sum() / MON / dd, 3),
                正年=f"{(yr > 0).sum()}/{len(yr)}", 最差年=round(yr.min(), 2))


def combo(d, ks, N, tf_min, stop, rr, hold=48):
    U = pd.concat([signals(d, k, N, tf_min, stop, rr, hold) for k in ks])
    return single_position(U)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    D15 = z.prep(z.M15)

    print("一、停損 ATR × RR（M15、k=3/4/5、N=480、單一持倉、最長 48 根）")
    rows = []
    for stop in (0.75, 1.0, 1.5, 2.0, 3.0):
        for rr in (1.0, 1.5, 2.0, 3.0, 4.0):
            S = combo(D15, (3, 4, 5), 480, 15, stop, rr)
            r = row(S, 停損ATR=stop, RR=rr)
            r["每筆ATR"] = round(r["期望R"] * stop, 3)
            rows.append(r)
            print(pd.DataFrame([r]).to_string(index=False, header=len(rows) == 1))
    G = pd.DataFrame(rows)
    G.to_csv("part22c_stop_rr.csv", index=False)

    print("\n  最長持有時間（停損 1.5、RR 2）")
    for hold in (16, 48, 96, 192):
        print(pd.DataFrame([row(combo(D15, (3, 4, 5), 480, 15, 1.5, 2.0, hold), 最長根數=hold)]).to_string(index=False))

    print("\n二、不設停利，只用時間出場（停損 1.5 ATR）")
    for hold in (16, 48, 96):
        print(pd.DataFrame([row(combo(D15, (3, 4, 5), 480, 15, 1.5, 99.0, hold), 最長根數=hold)]).to_string(index=False))

    print("\n三、更小週期（單一持倉；停損 1.5 ATR、RR 2、最長 48 根）")
    M5 = z.prep(z.resample(m.M1, "5min"))
    M1d = z.prep(m.M1.copy())
    rows = []
    for tf, d, tfm, N, ks in (("M15", D15, 15, 480, (2, 3, 4, 5)),
                              ("M5", M5, 5, 1440, (3, 4, 5, 6, 8, 10)),
                              ("M1", M1d, 1, 7200, (6, 8, 12, 16, 20))):
        for k in ks:
            S = single_position(signals(d, k, N, tfm, 1.5, 2.0))
            rows.append(row(S, 時框=tf, k=k))
            print(pd.DataFrame([rows[-1]]).to_string(index=False, header=len(rows) == 1))
    T = pd.DataFrame(rows)
    T.to_csv("part22c_timeframes.csv", index=False)

    print("\n  小週期的點差負擔：中位點差 / 停損距離")
    for tf, d in (("M15", D15), ("M5", M5), ("M1", M1d)):
        x = (d.sp / (1.5 * d.atr)).groupby(d.index.year).median()
        print(f"  {tf}: " + "  ".join(f"{y}:{v:.1%}" for y, v in x.items()))

    print("\n  小週期多 k 合併（單一持倉）")
    for tf, d, tfm, N, ks in (("M5", M5, 5, 1440, (5, 8, 10)), ("M1", M1d, 1, 7200, (12, 16, 20))):
        print(pd.DataFrame([row(combo(d, ks, N, tfm, 1.5, 2.0), 時框=tf, k="/".join(map(str, ks)))]).to_string(index=False))
