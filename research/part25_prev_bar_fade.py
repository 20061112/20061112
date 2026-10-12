"""第二十五部分：前一根 K 棒高/低點「逆勢」—— 碰到前高做空、碰到前低做多（D1 / H4 / H1 / M30 / M15）。

第二十四部分：小週期的突破單大虧（M15 −0.5R），但這不代表反過來做就會賺 ——
突破單的停損很近、停利 2R，加上點差，兩邊都會被成本與括號形狀吃掉，所以要直接測逆勢。

規則：
  - 當前 P 週期 K 棒開盤在前一根高點下方，期間內第一次碰到前一根高點 → 限價在高點做空；低點對稱做多。
  - 成交條件兩種：
      touch   ：價格碰到點位就算成交（樂觀）
      through ：價格要穿過點位至少 max(點差, 0.03·ATR_P) 才算成交，但仍以點位價成交（排隊較後面也成交得到，保守）
  - 停損 = 點位外側 s·ATR_P；停利 = 點位往回 t × 停損距離；最長持有同第二十四部分。
  - 碰觸那根 K 棒若也碰到停損 → 停損（從下方碰到前高後再往上，順序是確定的）；
    碰觸那根的停利不算（無法確定是在碰觸之前或之後）；之後同一根兩邊都碰 → 停損。每筆扣點差。
  - 對照：點位平移 ±0.15 / ±0.3 ATR_P 的假點位。
"""
import numpy as np
import pandas as pd
import part24_prev_bar_breakout as B

H, Lo, C, SP, mins, dayv, day_end, N = B.H, B.Lo, B.C, B.SP, B.mins, B.dayv, B.day_end, B.N
OUT = "results_part25.txt"
lines = []


def p(s=""):
    print(s)
    lines.append(s)


def prep(P):
    pid = B.period_ids(P)
    g = pd.DataFrame({"pid": pid, "H": H, "L": Lo, "O": B.d.open.values})
    agg = g.groupby("pid").agg(H=("H", "max"), L=("L", "min"), O=("O", "first"))
    atrP = (agg.H - agg.L).rolling(10 if P >= 1440 else 20).mean().shift(1)
    se = pd.Series(np.arange(N)).groupby(pid).agg(["first", "last"])
    return dict(agg=agg, atr=atrP.values, pH=agg.H.shift(1).values, pL=agg.L.shift(1).values, O=agg.O.values,
                f=se["first"].values, l=se["last"].values, hold=240 if P >= 1440 else max(60, min(P, 240)))


def events(Q, offs=(0.0,), fill="touch"):
    """回傳事件：(k, i, L, dir, a, off)；dir = 逆勢方向（前高 → −1 做空，前低 → +1 做多）。"""
    ev = []
    for k in range(1, len(Q["O"])):
        a = Q["atr"][k]
        if not np.isfinite(a) or a <= 0:
            continue
        i0, i1 = Q["f"][k], Q["l"][k]
        if mins[i0] >= 21 * 60:
            continue
        for side, base in ((1, Q["pH"][k]), (-1, Q["pL"][k])):  # side = 點位在上方(+1)/下方(−1)
            for off in offs:
                for sg in ((1,) if off == 0 else (1, -1)):
                    L = base + sg * off * a
                    if (Q["O"][k] - L) * side >= 0:
                        continue
                    hh, ll = H[i0:i1 + 1], Lo[i0:i1 + 1]
                    ok = mins[i0:i1 + 1] < 22 * 60
                    if fill == "touch":
                        seg = (ll <= L) & (hh >= L)
                    else:
                        thr = np.maximum(SP[i0:i1 + 1], 0.03 * a)
                        seg = (hh >= L + thr) if side > 0 else (ll <= L - thr)
                    seg &= ok
                    if not seg.any():
                        continue
                    i = i0 + int(np.argmax(seg))
                    if i == i0 and i0 > 0 and (C[i0 - 1] - L) * side > 0:
                        continue
                    ev.append((i, L, -side, a, off))
    return ev


def trade(ev, s, t, hold):
    out = []
    for i, L, dir_, a, off in ev:
        risk = s * a
        stop, tgt = L - dir_ * risk, L + dir_ * t * risk
        last = min(i + hold, day_end[i])
        if (Lo[i] <= stop) if dir_ > 0 else (H[i] >= stop):
            pnl = -risk
        elif last <= i:
            pnl = (C[i] - L) * dir_
        else:
            hh, ll = H[i + 1:last + 1], Lo[i + 1:last + 1]
            hs, ht = ((ll <= stop), (hh >= tgt)) if dir_ > 0 else ((hh >= stop), (ll <= tgt))
            js = int(np.argmax(hs)) if hs.any() else 10 ** 9
            jt = int(np.argmax(ht)) if ht.any() else 10 ** 9
            if js == jt == 10 ** 9:
                pnl = (C[last] - L) * dir_
            elif js <= jt:
                pnl = -risk
            else:
                pnl = t * risk
        out.append((dayv[i], dir_, off, (pnl - SP[i]) / risk, SP[i] / risk, B.UP50.iat[i]))
    T = pd.DataFrame(out, columns=["day", "dir", "off", "R", "cost", "up50"])
    T["year"] = pd.DatetimeIndex(T.day).year
    return T


def st(r):
    r = pd.Series(r)
    if len(r) < 5:
        return f"n={len(r)}"
    gp, gl = r[r > 0].sum(), -r[r < 0].sum()
    return f"n={len(r):6d} {r.mean():+.3f}R t={r.mean() / r.std() * np.sqrt(len(r)):+6.2f} 勝率 {(r > 0).mean():.0%} PF {gp / max(gl, 1e-9):.2f}"


def yearly(T):
    return " / ".join(f"{T.R[T.year == y].mean():+.2f}" for y in (2023, 2024, 2025, 2026))


p("第二十五部分：前一根 K 棒高/低點逆勢（碰前高做空、碰前低做多），XAUUSD M1 2023/6~2026/10")
p("每筆已扣點差；逐年 = 2023 / 2024 / 2025 / 2026 平均 R\n")
PER = [(1440, "D1"), (240, "H4"), (60, "H1"), (30, "M30"), (15, "M15")]
GRID = [(s, t) for s in (0.2, 0.5, 1.0) for t in (0.5, 1.0, 2.0)]
best = {}
for P, nm in PER:
    Q = prep(P)
    p(f"=== {nm} ===")
    for fill in ("touch", "through"):
        ev = events(Q, offs=(0.0, 0.15, 0.3), fill=fill)
        p(f"  [{'碰到即成交' if fill == 'touch' else '需穿過才成交（保守）'}]  每天約 {sum(e[4] == 0 for e in ev) / len(B.DAYS):.1f} 筆")
        p(f"  {'停損/停利':10s} {'真（多空合計）':48s} {'假點位':>8s}  {'做空前高':>9s} {'做多前低':>9s} | 逐年")
        for s, t in GRID:
            T = trade(ev, s, t, Q["hold"])
            r, f = T[T.off == 0], T[T.off != 0]
            p(f"  s={s:<4} t={t:<4} {st(r.R):48s} {f.R.mean():+8.3f}  {r[r.dir < 0].R.mean():+9.3f} {r[r.dir > 0].R.mean():+9.3f} | {yearly(r)}")
            best[(nm, fill, s, t)] = r
    p()

p("=== 趨勢狀態（through 成交，s=0.5, t=1）：逆勢做多前低 / 做空前高，依 50 日均線分 ===")
for P, nm in PER:
    r = best[(nm, "through", 0.5, 1.0)]
    for dir_, lab in ((-1, "做空前高"), (1, "做多前低")):
        x = r[r.dir == dir_]
        p(f"  {nm:4s} {lab}  均線上方 {st(x[x.up50 == True].R)} | 下方 {st(x[x.up50 == False].R)}")
p("\n=== 未扣點差（碰到即成交）：逆勢 vs 第二十四部分的突破，看價位本身有沒有方向性 ===")
for P, nm in PER:
    for (s_, t_) in ((0.5, 1.0), (1.0, 1.0)):
        r = best[(nm, "touch", s_, t_)]
        p(f"  {nm:4s} s={s_} t={t_}  逆勢未扣點差 {(r.R + r.cost).mean():+.3f}R（點差 {r.cost.mean():.3f}R）  做空前高 {(r.R + r.cost)[r.dir < 0].mean():+.3f}  做多前低 {(r.R + r.cost)[r.dir > 0].mean():+.3f}")
open(OUT, "w").write("\n".join(lines) + "\n")
