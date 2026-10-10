"""第五十一部分：ER 回檔順勢樣本外的補充分析 —— 成本拆解、對照組（接續第五十部分）→ results_part51.txt

逐年績效與第五十部分（part50_er_oos.py）相同，這裡另外回答：虧損是成本造成還是訊號本身沒用？回檔條件是否仍勝過隨機方向？

資料：data/XAUUSD_M15_2023_2026.csv（2023/1/3 ~ 2026/10/9，M15，broker 時間）
規則：FINAL_STRATEGY_ER.md 整合版，一個參數都不改
  A M15 ER40/20/5 a0.3 冷卻5 tp1.5 最長80根；B M30 ER20/10/3 a0.3 冷卻3 tp2 最長20根；C M15 ER20/10/3 a0.4 冷卻10 tp1.5 最長40根
  不做週一、週五 20 點後不開新單、週五最後一根平倉、停損上限 $100（拉近）、掉期 $0.7/盎司/晚（週三 ×3）
  加碼：H1+H4 ER20 同向且到 +1R 加 1 單位（保守處理：觸發那根也碰到原進場價 → 加碼單停損）
與 M1 版的差別：只有 M15，點差用 M15 K 棒的 spread 欄位（M1 版用進場那根第一分鐘）。
2026 年同時用 M15 版重跑，和 M1 版結果比對，確認換資料來源沒有造成差異。
對照組（同樣出場規則）：
  趨勢中沒回檔就進（d·ER長 ≥ a 且 d·ER短 > 0.3）
  隨機方向（同樣的訊號時間點，方向隨機）
"""
import numpy as np
import pandas as pd
from load import load
from part25_exits import prep, er
from part25_ext import tf_er_done
from part27_sizing import sim
from part28_final import nights
from part29_optimize import LEGS

EXIT = {"A": (1.5, 80), "B": (2, 20), "C": (1.5, 40)}
DATA = "data/XAUUSD_M15_2023_2026.csv"


def gen(B, L, M, S, a, cool, mode="pull"):
    eL, eM, eS = er(B, L), er(B, M), er(B, S)
    d = np.sign(eL)
    eMp = np.r_[np.full(S, np.nan), eM[:-S]]
    with np.errstate(invalid="ignore"):
        if mode == "pull":
            cond = (d * eL >= a) & (d * eM >= 0) & (d * eM < d * eMp) & (d * eS <= 0)
        else:                                            # 趨勢中、沒回檔
            cond = (d * eL >= a) & (d * eM >= 0) & (d * eS > 0.3)
    tt, last = [], -10 ** 9
    for t in np.flatnonzero(cond):
        if t - last > cool:
            tt.append(t)
        last = t
    t = np.array(tt, int)
    t = t[(t > 60) & (t < len(eL) - 2)]
    return t, d[t].astype(int)


def add_conservative(B, k, s, r):
    if not r["added"]:
        return r
    o, h, l = B["o"], B["h"], B["l"]
    ep, R = o[k + 1], r["Rusd"]
    for j in range(k + 1, r["j"] + 1):
        if (h[j] >= ep + R) if s == 1 else (l[j] <= ep - R):
            if (l[j] <= ep) if s == 1 else (h[j] >= ep):
                cost = B["spr"][k + 1]
                r = dict(r, add_usd=-R - cost, addR=(-R - cost) / R)
            break
    return r


def run(m, mode="pull", rand=None, swap=0.7, cap=100):
    cache, rows = {}, []
    for leg, (tp, hold) in EXIT.items():
        tf, L, M, S, a, cool = LEGS[leg]
        B = cache.get(tf)
        if B is None:
            B = cache[tf] = prep(m, tf)
            B["h1"], B["h4"] = tf_er_done(m, B, "1h"), tf_er_done(m, B, "4h")
        step = pd.Timedelta(tf)
        t, side = gen(B, L, M, S, a, cool, mode)
        if rand is not None:
            side = rand.choice([-1, 1], len(side))
        for k, s in zip(t, side):
            ts = B["idx"][k] + step
            if ts.dayofweek == 0 or (ts.dayofweek == 4 and ts.hour >= 20):
                continue
            r = sim(B, k, s, M, hold // 2, tp=tp, cap_usd=cap, cap_mode="cap" if cap else None, flat="week")
            if not r:
                continue
            r = add_conservative(B, k, s, r)
            sw = swap * nights(B["idx"], k + 1, r["j"])
            htf = bool((s * B["h1"][k] > 0) and (s * B["h4"][k] > 0))
            add = r["added"] and htf
            rows.append(dict(leg=leg, time=B["idx"][k], xtime=B["idx"][r["j"]], side=int(s), htf=htf, added=add,
                             reason=r["reason"], stop_usd=r["Rusd"], R1=r["R"] - sw / r["Rusd"],
                             Radd=r["R"] - sw / r["Rusd"] + ((r["addR"] - sw / r["Rusd"]) if add else 0.0),
                             usd1=r["usd"] - sw, usdadd=r["usd"] - sw + ((r["add_usd"] - sw) if add else 0.0)))
    return pd.DataFrame(rows).sort_values("time").reset_index(drop=True)


def stats(T, col, days):
    if len(T) < 5:
        return {}
    v = T[col].to_numpy()
    dr = T.groupby(T.xtime.dt.date)[col].sum().reindex(days, fill_value=0)
    eq = dr.cumsum().to_numpy()
    mdd_d = -(eq - np.maximum.accumulate(np.r_[0, eq])[1:]).min()
    eqt = np.cumsum(T.sort_values("xtime")[col].to_numpy())
    mdd_t = -(eqt - np.maximum.accumulate(np.r_[0, eqt])[1:]).min()
    g, b = v[v > 0].sum(), -v[v < 0].sum()
    return dict(n=len(v), 每天=round(len(v) / len(days), 2), 勝率=round(float((v > 0).mean()), 2), PF=round(float(g / b), 2),
                每筆R=round(float(v.mean()), 3), 總R=round(float(v.sum()), 1), MDD=round(float(mdd_t), 1),
                P_MDD=round(float(v.sum() / mdd_t), 2) if mdd_t > 0 else np.nan,
                Sharpe=round(float(dr.mean() / dr.std() * np.sqrt(252)), 2),
                t值=round(float(v.mean() / v.std(ddof=1) * np.sqrt(len(v))), 2))


def main():
    m = load(DATA)
    out = []
    P = out.append
    P(f"資料 {m.index[0]} ~ {m.index[-1]}，{len(m)} 根 M15")
    T = run(m)
    T["year"] = T.time.dt.year
    alldays = pd.Series(sorted(set(m.index.date)))
    yd = {y: list(alldays[pd.to_datetime(alldays).dt.year == y]) for y in (2023, 2024, 2025, 2026)}
    yd["2023-2025"] = yd[2023] + yd[2024] + yd[2025]

    def sel(D, y):
        return D[D.year.isin([2023, 2024, 2025])] if y == "2023-2025" else D[D.year == y]

    for col, lab in (("R1", "1 單位"), ("Radd", "加碼版（+1R 且 H1/H4 同向加 1）")):
        P(f"\n=== 1. 逐年（{lab}，掉期與點差已扣）===")
        P(pd.DataFrame({y: stats(sel(T, y), col, yd[y]) for y in yd}).to_string())
    P("\n每筆美元（每單位 1 盎司）：" + "、".join(
        f"{y} 1單位 {sel(T, y).usd1.mean():+.2f} / 加碼 {sel(T, y).usdadd.mean():+.2f}（總 {sel(T, y).usdadd.sum():+.0f}）"
        for y in yd))
    P("停損距離中位數（美元）：" + "、".join(f"{y} {sel(T, y).stop_usd.median():.1f}" for y in (2023, 2024, 2025, 2026)))

    P("\n=== 2. 各組逐年（1 單位，每筆 R / 筆數）===")
    P(T.pivot_table(index="leg", columns="year", values="R1", aggfunc=["mean", "count"]).round(3).to_string())

    P("\n=== 3. 逐月（1 單位 R）===")
    mo = T.assign(m=T.xtime.dt.month).pivot_table(index="year", columns="m", values="R1", aggfunc="sum").round(1)
    P(mo.to_string())
    neg = (mo < 0).sum().sum()
    tot = mo.notna().sum().sum()
    P(f"虧損月份 {neg} / {tot}")

    P("\n=== 4. 對照組（同樣出場規則，1 單位）===")
    C1 = run(m, mode="run")
    C1["year"] = C1.time.dt.year
    rng = np.random.default_rng(11)
    rnd = []
    for i in range(5):
        Rr = run(m, rand=rng)
        Rr["year"] = Rr.time.dt.year
        rnd.append({y: sel(Rr, y).R1.mean() for y in yd})
    rows = []
    for y in yd:
        rows.append(dict(年=y, 回檔進場=round(float(sel(T, y).R1.mean()), 3),
                         趨勢中沒回檔就進=round(float(sel(C1, y).R1.mean()), 3),
                         同時點隨機方向_5次平均=round(float(np.mean([r[y] for r in rnd])), 3)))
    P(pd.DataFrame(rows).to_string(index=False))

    P("\n=== 5. 2026 年：M15 版 vs M1 版（part29_final_trades.csv）===")
    try:
        M1T = pd.read_csv("part29_final_trades.csv", parse_dates=["time", "xtime"])
        a = T[(T.time >= "2026-01-02") & (T.time < "2026-10-08")]
        P(f"M15 版 {len(a)} 筆、每筆 {a.R1.mean():+.3f}R、總 {a.R1.sum():.1f}R；M1 版 {len(M1T)} 筆、每筆 {M1T.R.mean():+.3f}R、總 {M1T.R.sum():.1f}R")
        both = a.merge(M1T, on=["leg", "time"], how="inner", suffixes=("", "_m1"))
        P(f"相同訊號時間 {len(both)} 筆（M15 {len(a)} / M1 {len(M1T)}），方向一致 {(both.side == both.side_m1).mean():.0%}")
    except FileNotFoundError:
        pass
    P("\n=== 6. 成本拆解（1 單位，每筆 R）：毛利 / 只扣點差 / 點差+掉期 ===")
    G0 = run(m.assign(spread=0), swap=0.0)
    G1 = run(m, swap=0.0)
    rows = []
    for D, lab in ((G0, "毛利（無點差、無掉期）"), (G1, "只扣點差"), (T, "點差 + 掉期 0.7")):
        D = D.assign(year=D.time.dt.year)
        rows.append(dict(項目=lab, **{str(y): round(float(sel(D, y).R1.mean()), 3) for y in yd}))
    P(pd.DataFrame(rows).to_string(index=False))
    P("平均點差（美元）：" + "、".join(f"{y} {m.spread[m.index.year == y].mean() * 0.01:.2f}" for y in (2023, 2024, 2025, 2026)))
    P("\n=== 7. 停損距離分組（2023-2025，毛利 R / 扣成本 R）===")
    G0y = G0.assign(year=G0.time.dt.year)
    a = sel(G0y, "2023-2025").reset_index(drop=True)
    b = sel(T, "2023-2025").reset_index(drop=True)
    q = pd.qcut(a.stop_usd, 4)
    P(pd.DataFrame(dict(筆數=a.groupby(q, observed=True).size(), 毛利R=a.groupby(q, observed=True).R1.mean().round(3),
                        扣成本R=b.R1.groupby(pd.qcut(b.stop_usd, 4), observed=True).mean().round(3).to_numpy())).to_string())
    open("results_part51.txt", "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
