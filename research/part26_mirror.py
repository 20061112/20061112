"""第二十六部分：M15 前高/前低突破 −0.52R / −0.57R，「完全反過來做」為什麼不會變成 +0.5R？

完全鏡像：同一個進場價、方向相反，停損放在原本的停利（0.4 ATR_P 外），停利放在原本的停損（0.2 ATR_P）。
如果虧損來自「方向」，鏡像未扣點差應該 ≈ +原本的虧損。
拆解三個來源：
  1. 點差：兩個方向都要付，反過來不會變成收入。
  2. 同一根 M1 K 棒的先後順序不明：停損只有 0.2 ATR_P（M15 約 0.8 美元），一根 M1 常常同時碰到停損與停利 / 進場與停損。
     原本一律算停損（保守）。這裡比較三種處理：
       保守：不明一律算停損（第二十四/二十五部分的做法）
       中性：不明算一半（0.5 停損 + 0.5 停利；進場那根的停損碰觸算一半機率發生在進場前）
       樂觀：不明一律算停利
       路徑：用 M1 K 棒顏色推估順序（陰線 = 開→高→低→收，陽線 = 開→低→高→收），突破與鏡像用同一個推估
  3. 剩下的才是價位本身的方向性。
"""
import numpy as np
import pandas as pd
import part24_prev_bar_breakout as B

H, Lo, C, SP, mins, dayv, day_end = B.H, B.Lo, B.C, B.SP, B.mins, B.dayv, B.day_end
O1 = B.d.open.values
OUT = "results_part26.txt"
lines = []


def p(s=""):
    print(s)
    lines.append(s)


def events(P):
    pid = B.period_ids(P)
    g = pd.DataFrame({"pid": pid, "H": H, "L": Lo, "O": B.d.open.values})
    agg = g.groupby("pid").agg(H=("H", "max"), L=("L", "min"), O=("O", "first"))
    atr = (agg.H - agg.L).rolling(10 if P >= 1440 else 20).mean().shift(1).values
    pH, pL, O = agg.H.shift(1).values, agg.L.shift(1).values, agg.O.values
    se = pd.Series(np.arange(len(H))).groupby(pid).agg(["first", "last"])
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


def bracket(i, L, dir_, stop_d, tgt_d, hold, stop_first_on_entry_bar_certain):
    """回傳 (確定的結果, 不明的情況)。結果以價格計。
    不明情況：'entry'（進場那根也碰到停損，順序不明）或 'same'（之後同一根兩邊都碰）。"""
    stop, tgt = L - dir_ * stop_d, L + dir_ * tgt_d
    last = min(i + hold, day_end[i])
    hit_stop_i = (Lo[i] <= stop) if dir_ > 0 else (H[i] >= stop)
    if hit_stop_i:
        if stop_first_on_entry_bar_certain:
            return -stop_d, None
        amb = "entry"
    else:
        amb = None
    hh, ll = H[i + 1:last + 1], Lo[i + 1:last + 1]
    hs, ht = ((ll <= stop), (hh >= tgt)) if dir_ > 0 else ((hh >= stop), (ll <= tgt))
    js = int(np.argmax(hs)) if hs.any() else 10 ** 9
    jt = int(np.argmax(ht)) if ht.any() else 10 ** 9
    if js == jt == 10 ** 9:
        rest = (C[last] - L) * dir_ if last > i else (C[i] - L) * dir_
        same = False
    elif js < jt:
        rest, same = -stop_d, False
    elif jt < js:
        rest, same = tgt_d, False
    else:
        rest, same = None, True
    return rest, (amb, same)


def evaluate(ev, mirror, P):
    hold = max(60, min(P, 240))
    out = []
    for i, L, side, a in ev:
        if not mirror:  # 原本的突破單：順 side，停損 0.2a，停利 0.4a
            dir_, sd, td = side, 0.2 * a, 0.4 * a
            certain = False  # 從點位下方碰上來，進場那根的低點可能在碰觸之前 → 順序不明
        else:  # 鏡像：逆 side，停損 0.4a（原停利），停利 0.2a（原停損）
            dir_, sd, td = -side, 0.4 * a, 0.2 * a
            certain = True  # 鏡像的停損在點位外側，進場那根碰到一定是碰觸之後
        rest, info = bracket(i, L, dir_, sd, td, hold, certain)
        amb_entry = info is not None and info[0] == "entry"
        same = info is not None and info[1]
        res = {}
        for mode, w in (("保守", 0.0), ("中性", 0.5), ("樂觀", 1.0)):
            later = rest if not same else (w * td - (1 - w) * sd)
            if amb_entry:
                # 進場那根也碰到停損：w = 停損發生在進場前（不算數，結果看之後）的機率
                v = w * later + (1 - w) * (-sd)
            else:
                v = later
            res[mode] = v / (0.2 * a)  # 一律以原本突破單的 R（0.2 ATR_P）計，方便直接比較
        # 路徑推估：進場那根同時跨過點位與「點位反側 0.2a」時，K 棒顏色決定先碰哪一個
        far = L - side * 0.2 * a
        spans = (Lo[i] <= far) if side > 0 else (H[i] >= far)
        touch_first = (C[i] - O1[i]) * side < 0
        if spans and touch_first:
            res["路徑"] = -1.0 if not mirror else 1.0  # 先碰點位再走到反側：突破停損、鏡像停利
        elif spans:
            # 先到反側才碰點位：突破的停損、鏡像的停利都發生在進場前，不算，從下一根開始
            res["路徑"] = (rest if not same else -0.5 * sd + 0.5 * td) / (0.2 * a)
        else:
            res["路徑"] = res["中性"]
        out.append(dict(res, sp=SP[i] / (0.2 * a), amb_entry=amb_entry, same=same))
    return pd.DataFrame(out)


p("第二十六部分：M15 / M30 / H1 前高前低突破的「完全鏡像」（單位一律 = 原突破單的 R = 0.2 ATR_P）\n")
for P, nm in ((15, "M15"), (30, "M30"), (60, "H1")):
    ev = events(P)
    A = evaluate(ev, False, P)
    M = evaluate(ev, True, P)
    p(f"=== {nm}：{len(ev)} 筆，點差平均 {A.sp.mean():.3f}R ===")
    p(f"  突破單：進場那根就碰到停損（順序不明）{A.amb_entry.mean():.0%}；之後同一根兩邊都碰 {A.same.mean():.0%}")
    p(f"  鏡像  ：之後同一根兩邊都碰 {M.same.mean():.0%}（進場那根碰到停損是確定的，不算不明）")
    p(f"  {'不明的處理':8s} {'突破 未扣點差':>14s} {'突破 扣點差':>12s} {'鏡像 未扣點差':>14s} {'鏡像 扣點差':>12s} {'兩者相加(未扣)':>15s}")
    for mode in ("保守", "中性", "樂觀", "路徑"):
        a, m = A[mode].mean(), M[mode].mean()
        p(f"  {mode:10s} {a:+14.3f} {a - A.sp.mean():+12.3f} {m:+14.3f} {m - M.sp.mean():+12.3f} {a + m:+15.3f}")
    p()
open(OUT, "w").write("\n".join(lines) + "\n")
