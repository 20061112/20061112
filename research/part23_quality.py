"""第二十三部分 第 1+2 步：參考價品質（poor / excess）與用法（接受 / 拒絕、80% 法則）。

事前登記的預測（寫程式前就定好，2023~24 = 發現、2025~26 = 驗證，全部和假價位比較）：
  Q1  poor high/low（平頂）碰到後比較容易被突破：反彈率 < 假價位
  Q2  excess high/low（有尾巴）比較會撐住：反彈率 ≥ 假價位，且 > poor
  A1  越過參考價後「被接受」（30 分後仍在外面、且 ≥ 一半時間在外面）→ 之後往外延續 > 假價位
  A2  越過後「被拒絕」（30 分後收回、且確實越過 ≥ 0.05 ATR）→ 之後往回走 > 假價位
  V1  80% 法則：session 開在參考價值區外，連續 2 個 30 分時段收盤在價值區內
      → session 結束前走到價值區另一側的比例 > 假價值區（同寬、平移 ±0.3 ATR）
參考價：前一天（D_）與前一個 session（S_）的 高 / 低 / VAH / VAL / POC。
假價位：真價位 ±0.2 / ±0.4 ATR，沿用真價位的品質標籤（控制「什麼樣的行情才會留下 poor high」）。
"""
import pickle
import numpy as np
import pandas as pd
from part22_core import load_m1, arrays, first_touch, barrier, race, tstat, SESS_ORDER
from part23_core import quality, TAIL_EXCESS

SHIFTS = (0, -0.4, -0.2, 0.2, 0.4)
YRS = (2023, 2024, 2025, 2026)


def run():
    d = load_m1()
    H, L, C, O, SP = arrays(d)
    S = pickle.load(open("part23_refs.pkl", "rb"))
    ev, va = [], []
    for r in S.itertuples():
        i_lim = min(r.i1 + 60, len(C) - 1)
        base = dict(day=r.day, year=r.year, sess=r.sess, atr=r.atr)
        for p in ("D", "S"):
            g = lambda k: getattr(r, f"{p}_{k}")
            levels = [("H", g("h"), quality(g("ntop_h"), g("tail_h")), g("tail_h"), g("ntop_h")),
                      ("L", g("l"), quality(g("ntop_l"), g("tail_l")), g("tail_l"), g("ntop_l")),
                      ("VAH", g("vah"), "", np.nan, np.nan), ("VAL", g("val"), "", np.nan, np.nan),
                      ("POC", g("poc"), "", np.nan, np.nan)]
            for name, lev, q, tail, ntop in levels:
                for sh in SHIFTS:
                    lv = lev + sh * r.atr
                    j, side = first_touch(H, L, C, r.i0 + 1, r.i1, lv)
                    if j < 0:
                        continue
                    e = dict(base, ref=p, level=name, q=q, tail=tail, ntop=ntop, kind="real" if sh == 0 else "placebo",
                             side=side, b10=barrier(H, L, j, lv, side, 0.10 * r.atr, min(j + 240, i_lim)),
                             r60=(C[min(j + 60, r.i1)] - lv) * side / r.atr)
                    # ---- 接受 / 拒絕（決定點 = 碰到後 30 分；之後的結果都從決定點起算，往「外面」為正）
                    k = j + 30
                    if k < r.i1 - 15:
                        out_dir = -side                                  # 越過價位的方向
                        beyond = (C[j + 1:k + 1] - lv) * out_dir > 0
                        probe = ((H[j:k + 1].max() - lv) if out_dir == 1 else (lv - L[j:k + 1].min())) / r.atr
                        if (C[k] - lv) * out_dir > 0 and beyond.mean() >= 0.5:
                            st = "accept"
                        elif (C[k] - lv) * out_dir < 0 and probe >= 0.05:
                            st = "reject"
                        elif probe < 0.05:
                            st = "held"                                  # 幾乎沒有越過
                        else:
                            st = "mixed"
                        px = C[k]
                        e.update(state=st, probe=probe,
                                 o60=((C[min(k + 60, r.i1)] - px) * out_dir - SP[k]) / r.atr,
                                 oS=((C[r.i1] - px) * out_dir - SP[k]) / r.atr,
                                 orace=race(H, L, k + 1, px + 0.2 * r.atr, px - 0.2 * r.atr, i_lim) * out_dir)
                    ev.append(e)
            # ---- V1：80% 法則
            vah, val = g("vah"), g("val")
            for sh in (0, -0.3, 0.3):
                hi, lo = vah + sh * r.atr, val + sh * r.atr
                if lo <= r.o <= hi:
                    continue
                frm = 1 if r.o > hi else -1                              # 從上方（+1）或下方開盤
                pcl = C[r.i0 + 29:r.i1 + 1:30]                            # 每個 30 分時段的收盤
                inside = (pcl <= hi) & (pcl >= lo)
                hit = np.where(inside[:-1] & inside[1:])[0]
                if len(hit) == 0:
                    va.append(dict(base, ref=p, kind="real" if sh == 0 else "placebo", trig=False))
                    continue
                k = r.i0 + 29 + 30 * (hit[0] + 1)
                if k >= r.i1 - 15:
                    continue
                tgt, back = (lo, hi) if frm == 1 else (hi, lo)
                rc = race(H, L, k + 1, max(tgt, back), min(tgt, back), r.i1)
                reach_other = (rc == -1) if frm == 1 else (rc == 1)      # 先到另一側（而不是先回到開盤那一側外面）
                seg_l, seg_h = L[k + 1:r.i1 + 1], H[k + 1:r.i1 + 1]
                reached = (seg_l.min() <= tgt) if frm == 1 else (seg_h.max() >= tgt)
                va.append(dict(base, ref=p, kind="real" if sh == 0 else "placebo", trig=True,
                               reach=reached, first=reach_other,
                               rS=((C[r.i1] - C[k]) * -frm - SP[k]) / r.atr, w=(hi - lo) / r.atr))
    E, V = pd.DataFrame(ev), pd.DataFrame(va)
    for c in ("reach", "first"):
        V[c] = V[c].astype(float)
    E.to_csv("part23_events.csv", index=False)
    V.to_csv("part23_va80.csv", index=False)
    return E, V


def br(g):
    x = g.b10[g.b10 != 0]
    return (x > 0).mean() * 100 if len(x) else np.nan


def diff_line(g, label):
    re, pl = g[g.kind == "real"], g[g.kind == "placebo"]
    yr = [br(re[re.year == y]) - br(pl[pl.year == y]) for y in YRS]
    a = br(re[re.year <= 2024]) - br(pl[pl.year <= 2024])
    b = br(re[re.year >= 2025]) - br(pl[pl.year >= 2025])
    return (f"  {label:<30} n={len(re):5d}  真 {br(re):5.1f}%  假 {br(pl):5.1f}%  差 {br(re)-br(pl):+5.1f}   "
            f"發現 {a:+5.1f} / 驗證 {b:+5.1f}   逐年 " + " ".join(f"{v:+5.1f}" for v in yr))


def summarize(E, V):
    out = ["=" * 120, "第 1 步：參考價品質 — 碰到後先反彈 0.10 ATR 的比例（真 vs 假價位，假價位沿用真價位的品質標籤）",
           "   預測：poor（平頂）差 < 0、excess（有尾巴）差 ≥ 0 且 > poor", "=" * 120]
    X = E[E.level.isin(["H", "L"])]
    for p, pn in (("D", "前一天"), ("S", "前一個 session")):
        out.append(f"\n  [{pn}的高低點]")
        for q in ("poor", "normal", "excess"):
            out.append(diff_line(X[(X.ref == p) & (X.q == q)], f"{q}"))
        for s in SESS_ORDER:
            for q in ("poor", "excess"):
                out.append(diff_line(X[(X.ref == p) & (X.q == q) & (X.sess == s)], f"  {s} {q}"))
    # 敏感度：尾巴門檻、ntop 連續值
    out.append("\n  敏感度（前一天 + 前一 session 合併）：")
    for thr in (0.03, 0.05, 0.10):
        q = np.where(X.ntop >= 2, "poor", np.where(X["tail"] >= thr, "excess", "normal"))
        out.append(diff_line(X[q == "excess"], f"excess 尾巴 ≥ {thr}"))
    for n in (1, 2, 3):
        out.append(diff_line(X[(X.ntop == n) if n < 3 else (X.ntop >= 3)], f"ntop = {n}{'+' if n == 3 else ''}"))

    out += ["", "=" * 120, "第 2 步：接受 / 拒絕 — 碰到後 30 分的狀態，之後往「外面」（越過價位的方向）的結果",
            "   orace = 決定點起先走 +0.2 ATR（往外）的比例；o60 / oS = 60 分 / session 結束報酬（ATR，往外為正，已扣點差）",
            "   預測：accept → 往外 > 假價位；reject → 往外 < 假價位（往回走）", "=" * 120]
    Z = E[E.state.notna()]

    def st_line(g, label):
        re, pl = g[g.kind == "real"], g[g.kind == "placebo"]
        rr = lambda x: (x.orace[x.orace != 0] > 0).mean() * 100
        yr = " ".join(f"{re[re.year == y].oS.mean() - pl[pl.year == y].oS.mean():+.3f}" for y in YRS)
        a = re[re.year <= 2024].oS.mean() - pl[pl.year <= 2024].oS.mean()
        b = re[re.year >= 2025].oS.mean() - pl[pl.year >= 2025].oS.mean()
        return (f"  {label:<30} n={len(re):5d}  往外% 真 {rr(re):5.1f} / 假 {rr(pl):5.1f}  "
                f"oS 真 {re.oS.mean():+.3f} (t {tstat(re.oS):+4.1f}) / 假 {pl.oS.mean():+.3f}  "
                f"差 發現 {a:+.3f} / 驗證 {b:+.3f}  逐年差 {yr}")
    for lvset, nm in ((["H", "L"], "高低點"), (["VAH", "VAL"], "價值區邊緣"), (["POC"], "POC")):
        out.append(f"\n  [{nm}]")
        for st in ("accept", "reject", "held", "mixed"):
            out.append(st_line(Z[Z.level.isin(lvset) & (Z.state == st)], st))
        if nm == "高低點":
            for q in ("poor", "excess"):
                for st in ("accept", "reject"):
                    out.append(st_line(Z[Z.level.isin(lvset) & (Z.state == st) & (Z.q == q)], f"{q} + {st}"))
        for s in SESS_ORDER:
            for st in ("accept", "reject"):
                out.append(st_line(Z[Z.level.isin(lvset) & (Z.state == st) & (Z.sess == s)], f"{s} {st}"))

    out += ["", "=" * 120, "V1：80% 法則（session 開在參考價值區外，連續 2 個 30 分時段收在價值區內 → 走到另一側）", "=" * 120]
    for p, pn in (("D", "前一天價值區"), ("S", "前一 session 價值區")):
        for s in SESS_ORDER + ["ALL"]:
            g = V[(V.ref == p) & ((V.sess == s) | (s == "ALL"))]
            cells = []
            for k in ("real", "placebo"):
                gg = g[(g.kind == k) & (g.trig == True)]
                trig = (g[g.kind == k].trig == True).mean() * 100
                cells.append(f"{'真' if k == 'real' else '假'} 觸發 {trig:4.1f}% n={len(gg):4d} 到另一側 {gg.reach.mean()*100:5.1f}% "
                             f"先到 {gg['first'].mean()*100:5.1f}% rS {gg.rS.mean():+.3f}")
            re = g[(g.kind == "real") & (g.trig == True)]
            yr = " ".join(f"{re[re.year == y].reach.mean()*100:4.0f}" for y in YRS)
            out.append(f"  {pn:<12}{s:<5} " + " | ".join(cells) + f" | 真逐年到另一側% {yr}")
    return out


if __name__ == "__main__":
    E, V = run()
    txt = "\n".join(summarize(E, V))
    print(txt)
    open("results_part23_step12.txt", "w").write(txt + "\n")
