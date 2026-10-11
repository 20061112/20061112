"""第三十部分：(A) 第二十九部分改用 IB 高 / 低（取代 VAH / VAL）；(B) 不管 IB，多網格 + 空網格同時開。

(A) 規則與元件完全同 part29_va_grid.py，只把 VAH → IB 高、VAL → IB 低。
    S1：IB 結束時已在 IB 外 → 立刻順向；否則 IB 高買進停止單 / IB 低賣出停止單，先到先做；停損 IB 另一側。
    S2：IB 高上方做多區、IB 低下方做空區，跌回 IB 高下一格離開（鏡像），一天可來回。

(B) 雙向網格（不看方向）：錨點 = 時段第一根開盤，格線 錨點 + j·g。
    LG 多網格：每條線掛買單、反彈一格賣（一般的網格機器人：碰過上一條線的線才掛買單 → 持倉 = M − k，M = 到過的最高線）
    SG 空網格：每條線掛賣單、回落一格買回（持倉 = −(k − m)，m = 到過的最低線）
    LG + SG 同時開（多空單分開記帳、各自付點差）；對照：中性網格 N（固定錨點，持倉 = −k，第二十三部分）。
    分解：總損益 = 完成來回的利潤（每完成一格 +g）＋ 收盤時剩餘庫存的損益 − 點差。
    兩種時段：每天（01:00 起、23:50 全平）、每週（週一第一根起、週五最後一根全平）。
單位：損益 ÷ ATR10（每天 / 每週）。基準：鏡像 K 棒，同模型 10 次平均。
"""
import numpy as np, pandas as pd
from multiprocessing import Pool
import part29_va_grid as V
from part28_ib_grid import DAYS, mirrored, CUT, t, FLAT

V.ib_va = lambda ohlc, atr: (ohlc[:, 1].max(), ohlc[:, 2].min())     # (A) VAH/VAL → IB 高/低
SEEDS, MS = 10, (0.05, 0.10)


def dual(ohlc, sp, g):
    E = ohlc[0, 0]; k = M = m = 0
    B = V.Book(("LG", "SG", "N")); rt = 0
    for j in range(len(ohlc)):
        for x in V.path(ohlc, j)[(1 if j == 0 else 0):]:
            while x >= E + (k + 1) * g or x <= E + (k - 1) * g:
                up = x >= E + (k + 1) * g
                k += 1 if up else -1
                M, m = max(M, k), min(m, k)
                before = (B.pos["LG"], B.pos["SG"])
                B.to(dict(LG=M - k, SG=-(k - m), N=-k), E + k * g, sp[j])
                rt += max(0, before[0] - B.pos["LG"]) + max(0, B.pos["SG"] - before[1])
    inv = {nm: B.pos[nm] for nm in B.pos}
    B.flat(ohlc[-1, 3], sp[-1])
    return dict(LG=B.cash["LG"], SG=B.cash["SG"], N=B.cash["N"], rt=rt * g, inv=inv["LG"] + inv["SG"],
                gross=inv["LG"] - inv["SG"])


# 每週切段（用第二十八部分的日資料拼週）
WEEKS = {}
for k, a, mins, ohlc, sp in DAYS:
    w = k.to_period("W-SUN")
    WEEKS.setdefault(w, []).append((k, a, mins, ohlc, sp))


def jobB(args):
    per, mlt, seed = args
    rng = np.random.default_rng(seed) if seed is not None else None
    rows = []
    if per == "day":
        segs = [(k, a, ohlc[mins < FLAT], sp[mins < FLAT]) for k, a, mins, ohlc, sp in DAYS]
    else:
        segs = [(v[0][0], v[0][1], np.vstack([x[3] for x in v]), np.concatenate([x[4] for x in v]))
                for v in WEEKS.values() if len(v) >= 4]
    for k, a, ohlc, sp in segs:
        oh = mirrored(ohlc, rng) if rng is not None else ohlc
        r = dual(oh, sp, mlt * a)
        rows.append(dict(day=k, atr=a, net=abs(oh[-1, 3] - oh[0, 0]) / a, **r))
    return args, pd.DataFrame(rows)


if __name__ == "__main__":
    cfgA = [("S1", s0, m, "VA") for s0 in V.STARTS.values() for m in V.MS] + \
           [("S2", s0, m, "-") for s0 in V.STARTS.values() for m in V.MS]
    jobsA = [c + (sd,) for c in cfgA for sd in [None] + list(range(SEEDS))]
    jobsB = [(p, m, sd) for p in ("day", "week") for m in MS for sd in [None] + list(range(SEEDS))]
    with Pool(4) as pool:
        resA = dict(pool.map(V.job, jobsA, chunksize=1))
        resB = dict(pool.map(jobB, jobsB, chunksize=1))
    out = open("results_part30.txt", "w")
    def P(s=""): print(s); out.write(s + "\n")
    P("第三十部分 (A)：第二十九部分的兩個想法，VAH / VAL 改成 IB 高 / 低")
    P("單位：每日損益 ÷ ATR10；括號 = t；基準 = 鏡像 K 棒同模型 10 次平均；樣本內 2023~24 / 樣本外 2025~26。")
    name = {v: k for k, v in V.STARTS.items()}
    keep = []
    for c in cfgA:
        kind, s0, m, sk = c
        real = resA[c + (None,)]; base = pd.concat([resA[c + (sd,)] for sd in range(SEEDS)])
        IS = (real.day.dt.year < CUT).values
        hdr = "S1：IB 邊緣底倉 + 順向網格（停損 IB 另一側）" if kind == "S1" else "S2：IB 高上做多區 / IB 低下做空區"
        P(f"\n{hdr}｜IB {name[s0]}｜格距 {m} ATR｜{len(real)} 天，IB 寬中位 {real.va.median():.2f} ATR")
        P(f"  {'組合':22s} | {'樣本內':>13s} {'樣本外':>13s} | {'最差日':>7s} | {'基準':>6s} | 逐年")
        for nm, cols in V.COMBOS[kind].items():
            if "固定錨點" in nm or nm == "BASE + L": continue
            a = real[cols].sum(axis=1) / real.atr; b = base[cols].sum(axis=1) / base.atr
            yr = " ".join(f"{y % 100}:{a[(real.day.dt.year == y).values].mean():+.3f}" for y in range(2023, 2027))
            P(f"  {nm:22s} | {a[IS].mean():+.3f}({t(a[IS]):+5.2f}) {a[~IS].mean():+.3f}({t(a[~IS]):+5.2f}) | "
              f"{a.min():+7.2f} | {b.mean():+6.3f} | {yr}")
            keep.append(dict(part="A", kind=kind, ib=name[s0], m=m, combo=nm, IS=a[IS].mean(), t_IS=t(a[IS]),
                             OOS=a[~IS].mean(), t_OOS=t(a[~IS]), worst=a.min(), base=b.mean()))

    P("\n\n第三十部分 (B)：不管 IB，多網格 + 空網格同時開")
    P("單位：損益 ÷ ATR10 / 每段；「來回利潤」= 完成的格數 × g；「庫存」= 收盤時剩餘部位的損益；點差 = 總 − 來回 − 庫存。")
    for per in ("day", "week"):
        for m in MS:
            real = resB[(per, m, None)]; base = pd.concat([resB[(per, m, sd)] for sd in range(SEEDS)])
            IS = (real.day.dt.year < CUT).values
            real["both"] = real.LG + real.SG; base["both"] = base.LG + base.SG
            P(f"\n{'每天 01:00~23:50' if per == 'day' else '每週'}｜格距 {m} ATR｜{len(real)} 段")
            P(f"  {'組合':18s} | {'樣本內':>13s} {'樣本外':>13s} | {'勝率':>5s} {'最差':>7s} | {'基準':>6s} | 逐年")
            for nm, col in (("多網格 LG", "LG"), ("空網格 SG", "SG"), ("LG + SG 同時開", "both"), ("中性網格 N（參考）", "N")):
                a = real[col] / real.atr; b = base[col] / base.atr
                yr = " ".join(f"{y % 100}:{a[(real.day.dt.year == y).values].mean():+.3f}" for y in range(2023, 2027))
                P(f"  {nm:18s} | {a[IS].mean():+.3f}({t(a[IS]):+5.2f}) {a[~IS].mean():+.3f}({t(a[~IS]):+5.2f}) | "
                  f"{(a > 0).mean():5.0%} {a.min():+7.2f} | {b.mean():+6.3f} | {yr}")
                keep.append(dict(part="B", kind=per, m=m, combo=nm, IS=a[IS].mean(), t_IS=t(a[IS]), OOS=a[~IS].mean(),
                                 t_OOS=t(a[~IS]), worst=a.min(), base=b.mean()))
            a = real.both / real.atr; rt = real.rt / real.atr
            inv = (real.both - real.rt) / real.atr        # 含點差
            P(f"  分解（LG+SG）：來回利潤 {rt.mean():+.3f}、庫存 + 點差 {inv.mean():+.3f} → 合計 {a.mean():+.3f}；"
              f"收盤剩餘多+空單位 中位 {real.gross.median():.0f}、P90 {real.gross.quantile(.9):.0f}")
            real["bin"] = pd.cut(real.net, [0, 0.5, 1, 2, 3, 99], labels=["<0.5", "0.5~1", "1~2", "2~3", ">3"])
            P("  依該段淨漲跌（ATR）：" + "  ".join(f"{b}: {a[real.bin == b].mean():+.2f}（{(real.bin == b).sum()}）"
                                              for b in real.bin.cat.categories))
    pd.DataFrame(keep).to_csv("part30_summary.csv", index=False, float_format="%.4f")
    out.close()
