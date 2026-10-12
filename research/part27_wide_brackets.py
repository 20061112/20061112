"""第二十七部分：前一根 K 棒高/低點，停損停利放大後的突破 / 逆勢（M15 / M30 / H1 / H4 / D1）。

停損 s × ATR_P（s = 0.5 / 1 / 2 / 3），停利 t × 停損（t = 1 / 2 / 3）；持有到停損、停利或當天收盤。
同一根 M1 內的先後順序一律用 K 棒顏色推估（陽線 開→低→高→收、陰線 開→高→低→收），
進場那根也照這個路徑判斷（第二十六部分顯示保守處理會把小週期突破的虧損放大）。每筆扣點差。
對照：同一筆單換成「隨機交易日、同一分鐘、同方向、同停損停利」的隨機進場 → 真點位 − 隨機 = 點位本身的效果。
"""
import numpy as np
import pandas as pd
import part24_prev_bar_breakout as B

H, Lo, C, SP, mins, dayv, day_end, N = B.H, B.Lo, B.C, B.SP, B.mins, B.dayv, B.day_end, B.N
O1 = B.d.open.values
OUT = "results_part27.txt"
lines = []
rng = np.random.default_rng(1)
SG = [0.5, 1.0, 2.0, 3.0]
TG = [1.0, 2.0, 3.0]


def p(s=""):
    print(s)
    lines.append(s)


def path(j):
    return (O1[j], Lo[j], H[j], C[j]) if C[j] >= O1[j] else (O1[j], H[j], Lo[j], C[j])


def walk(points, dir_, stop, tgt):
    """依序走過路徑點，回傳 'S'、'T' 或 None。"""
    for v in points:
        if (v <= stop) if dir_ > 0 else (v >= stop):
            return "S"
        if (v >= tgt) if dir_ > 0 else (v <= tgt):
            return "T"
    return None


def sim_full(i, entry, dir_, stop, tgt, entry_from_level=True):
    """包裝：處理進場那根直接停利的情況。"""
    if entry_from_level:
        pts = path(i)
        for a in range(3):
            lo_, hi_ = min(pts[a], pts[a + 1]), max(pts[a], pts[a + 1])
            if lo_ <= entry <= hi_:
                r = walk(pts[a + 1:], dir_, stop, tgt)
                if r == "S":
                    return (stop - entry) * dir_
                if r == "T":
                    return (tgt - entry) * dir_
                break
        else:
            return None
    last = day_end[i]
    if last <= i:
        return (C[i] - entry) * dir_
    hh, ll = H[i + 1:last + 1], Lo[i + 1:last + 1]
    hs, ht = ((ll <= stop), (hh >= tgt)) if dir_ > 0 else ((hh >= stop), (ll <= tgt))
    js = int(np.argmax(hs)) if hs.any() else 10 ** 9
    jt = int(np.argmax(ht)) if ht.any() else 10 ** 9
    if js == jt == 10 ** 9:
        return (C[last] - entry) * dir_
    if js == jt:
        r = walk(path(i + 1 + js), dir_, stop, tgt)
        return (stop - entry) * dir_ if r == "S" else (tgt - entry) * dir_
    return (stop - entry) * dir_ if js < jt else (tgt - entry) * dir_


def events(P):
    pid = B.period_ids(P)
    g = pd.DataFrame({"pid": pid, "H": H, "L": Lo, "O": O1})
    agg = g.groupby("pid").agg(H=("H", "max"), L=("L", "min"), O=("O", "first"))
    atr = (agg.H - agg.L).rolling(10 if P >= 1440 else 20).mean().shift(1).values
    pH, pL, O = agg.H.shift(1).values, agg.L.shift(1).values, agg.O.values
    se = pd.Series(np.arange(N)).groupby(pid).agg(["first", "last"])
    f, l = se["first"].values, se["last"].values
    ev = []
    for k in range(1, len(O)):
        a = atr[k]
        if not np.isfinite(a) or a <= 0 or mins[f[k]] >= 21 * 60:
            continue
        for side, L in ((1, pH[k]), (-1, pL[k])):
            if (O[k] - L) * side >= 0:
                continue
            seg = (Lo[f[k]:l[k] + 1] <= L) & (H[f[k]:l[k] + 1] >= L) & (mins[f[k]:l[k] + 1] < 22 * 60)
            if not seg.any():
                continue
            i = f[k] + int(np.argmax(seg))
            if i == f[k] and i > 0 and (C[i - 1] - L) * side > 0:
                continue
            ev.append((i, L, side, a))
    return ev


# 隨機進場用：每個「分鐘」對應的所有 bar 索引
by_min = pd.Series(np.arange(N)).groupby(mins).apply(lambda x: x.values).to_dict()


def run(ev, s, t, mode):
    """mode: 'brk' 突破（順 side）、'fade' 逆勢、'rnd_brk' / 'rnd_fade' 隨機進場同方向。"""
    R, cost = [], []
    for i, L, side, a in ev:
        dir_ = side if mode.endswith("brk") else -side
        risk = s * a
        if mode.startswith("rnd"):
            j = int(rng.choice(by_min[mins[i]]))
            entry = C[j]
            pnl = sim_full(j, entry, dir_, entry - dir_ * risk, entry + dir_ * t * risk, False)
            sp = SP[j]
        else:
            pnl = sim_full(i, L, dir_, L - dir_ * risk, L + dir_ * t * risk, True)
            sp = SP[i]
        if pnl is None:
            continue
        R.append((pnl - sp) / risk)
        cost.append(sp / risk)
    return np.array(R), np.array(cost)


def t_(x):
    return x.mean() / x.std() * np.sqrt(len(x))


if __name__ == "__main__":
    p("第二十七部分：前一根 K 棒高/低點，放大停損停利（K 棒顏色推估同根順序；持有到當天收盤；每筆扣點差）")
    p("格子內：每筆 R（扣點差）[t]；括號 = 真點位 − 同方向隨機進場（點位本身的效果）\n")
    for P, nm in ((15, "M15"), (30, "M30"), (60, "H1"), (240, "H4"), (1440, "D1")):
        ev = events(P)
        if P < 240:  # 小週期抽樣 1/3 以節省時間（仍有上萬筆）
            ev = ev[::3]
        p(f"=== {nm}（{len(ev)} 筆）===")
        for mode, lab in (("brk", "突破"), ("fade", "逆勢")):
            p(f"  [{lab}]  {'':6s}" + "".join(f"{'t=' + str(t):>26s}" for t in TG))
            for s in SG:
                cells = []
                for t in TG:
                    r, c = run(ev, s, t, mode)
                    rr, _ = run(ev, s, t, "rnd_" + mode)
                    cells.append(f"{r.mean():+.3f}[{t_(r):+.1f}]({r.mean() - rr.mean():+.3f})")
                p(f"  s={s:<4} 點差{c.mean():.3f}R " + "".join(f"{x:>26s}" for x in cells))
        p()
    open(OUT, "w").write("\n".join(lines) + "\n")
