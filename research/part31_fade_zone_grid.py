"""第三十一部分：反過來做 —— IB 高上方做空網格、IB 低下方做多網格（逆勢，賭價格回到 IB）。日內、23:50 全平。

格線從 IB 高往上（IB 高 + j·g）、IB 低往下鏡像；k = 最後碰到的線（IB 高 = 0）。
  FG 逆勢網格：IB 高上方第 k 條線持有 k 單位空單（每往上一格多賣 1、回落一格買回 1）；IB 低下方鏡像做多。
     = 固定錨點在 IB 邊緣、只在 IB 外運作的中性網格。
  FB 逆勢底倉：在 IB 高上方持有 1 空、跌回 IB 高下一格出場（= 第三十部分 ZB 的反向）。
停損兩種：不停損（收盤平）；離 IB 邊緣 1 ATR 時該邊全平、今天不再做該邊。
IB 起點 01:00 / 10:00 / 16:30；格距 0.05 / 0.10 ATR。成交模型、鏡像基準、樣本切分同第二十八 ~ 三十部分。
"""
import numpy as np, pandas as pd
from multiprocessing import Pool
import part29_va_grid as V
from part28_ib_grid import DAYS, mirrored, CUT, t, FLAT

SEEDS, IB_MIN = 10, 60


def s3_day(mins, ohlc, sp, atr, s0, g, stop_atr):
    ib = (mins >= s0) & (mins < s0 + IB_MIN)
    if ib.sum() < IB_MIN * 0.8: return None
    ibh, ibl = ohlc[ib, 1].max(), ohlc[ib, 2].min()
    idx = np.where((mins >= s0 + IB_MIN) & (mins < FLAT))[0]
    if len(idx) < 2: return None
    B = V.Book(("FG", "FB"))
    kstop = int(round(stop_atr / (g / atr))) if stop_atr else 10 ** 9
    Z = {1: dict(ref=ibh), -1: dict(ref=ibl)}       # 1 = 上方區（做空）、−1 = 下方區（做多）

    def target():
        tg = dict(FG=0, FB=0)
        for sd, z in Z.items():
            if z["dead"]: continue
            tg["FG"] += -sd * max(0, z["k"])
            tg["FB"] += -sd * (1 if z["on"] else 0)
        return tg

    j0 = idx[0]; x0 = ohlc[j0, 0]
    for sd, z in Z.items():
        z["k"] = int(np.floor(sd * (x0 - z["ref"]) / g)); z["on"] = z["k"] >= 0; z["dead"] = False
    B.to(target(), x0, sp[j0])
    for j in idx:
        for x in V.path(ohlc, j)[(1 if j == j0 else 0):]:
            for sd, z in Z.items():
                if z["dead"]: continue
                u = sd * (x - z["ref"]) / g
                while u >= z["k"] + 1 or u <= z["k"] - 1:
                    z["k"] += 1 if u >= z["k"] + 1 else -1
                    if z["k"] == 0: z["on"] = True
                    elif z["k"] == -1: z["on"] = False
                    if z["k"] >= kstop: z["dead"] = True
                    if z["k"] >= -1: B.to(target(), z["ref"] + sd * z["k"] * g, sp[j])
                    if z["dead"]: break
    B.flat(ohlc[idx[-1], 3], sp[idx[-1]])
    return dict(ibw=(ibh - ibl) / atr, **B.cash)


def job(args):
    s0, m, st, seed = args
    rng = np.random.default_rng(seed) if seed is not None else None
    rows = []
    for k, a, mins, ohlc, sp in DAYS:
        oh = mirrored(ohlc, rng) if rng is not None else ohlc
        r = s3_day(mins, oh, sp, a, s0, m * a, st)
        if r: rows.append(dict(day=k, atr=a, **r))
    return args, pd.DataFrame(rows)


if __name__ == "__main__":
    cfg = [(s0, m, st) for s0 in V.STARTS.values() for m in V.MS for st in (None, 1.0)]
    jobs = [c + (sd,) for c in cfg for sd in [None] + list(range(SEEDS))]
    with Pool(4) as pool:
        res = dict(pool.map(job, jobs, chunksize=1))
    out = open("results_part31.txt", "w")
    def P(s=""): print(s); out.write(s + "\n")
    P("第三十一部分：IB 高上方做空網格、IB 低下方做多網格（逆勢）")
    P("單位：每日損益 ÷ ATR10；括號 = t；基準 = 鏡像 K 棒同模型 10 次平均；樣本內 2023~24 / 樣本外 2025~26。")
    name = {v: k for k, v in V.STARTS.items()}; keep = []
    for c in cfg:
        s0, m, st = c
        real = res[c + (None,)]; base = pd.concat([res[c + (sd,)] for sd in range(SEEDS)])
        IS = (real.day.dt.year < CUT).values
        P(f"\nIB {name[s0]}｜格距 {m} ATR｜{'不停損、收盤平' if st is None else '離 IB 邊緣 1 ATR 停損'}｜{len(real)} 天")
        P(f"  {'組合':18s} | {'樣本內':>13s} {'樣本外':>13s} | {'勝率':>5s} {'最差日':>7s} | {'基準':>6s} | 逐年")
        for nm, cols in (("FG 逆勢網格", ["FG"]), ("FB 逆勢底倉", ["FB"]), ("FG + FB", ["FG", "FB"])):
            a = real[cols].sum(axis=1) / real.atr; b = base[cols].sum(axis=1) / base.atr
            yr = " ".join(f"{y % 100}:{a[(real.day.dt.year == y).values].mean():+.3f}" for y in range(2023, 2027))
            P(f"  {nm:18s} | {a[IS].mean():+.3f}({t(a[IS]):+5.2f}) {a[~IS].mean():+.3f}({t(a[~IS]):+5.2f}) | "
              f"{(a > 0).mean():5.0%} {a.min():+7.2f} | {b.mean():+6.3f} | {yr}")
            keep.append(dict(ib=name[s0], m=m, stop=st, combo=nm, IS=a[IS].mean(), t_IS=t(a[IS]), OOS=a[~IS].mean(),
                             t_OOS=t(a[~IS]), win=(a > 0).mean(), worst=a.min(), base=b.mean()))
    pd.DataFrame(keep).to_csv("part31_summary.csv", index=False, float_format="%.4f")
    out.close()
