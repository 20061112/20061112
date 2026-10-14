"""因子一（區間內：OR 高 / 低 / 中點）＋ AMT 參考價：碰到價位時有沒有反應？

事件：價位在 session 內第一次被碰到（OR 價位從 OR 結束後開始算；AMT 價位從 session 開盤後開始算）。
反應指標（以 ATR10 為單位）：
  bounce：從價位起，先往「來的方向」走 b 還是先穿越 b（b = 0.05 / 0.10 ATR），隨機漫步 = 50%
  r30   ：30 分後收盤相對價位，往反彈方向為正
對照組（安慰劑）：
  OR 價位 → 同一 session 內「其他時間的 30 分區間」（開盤後 +60 / +120 / +180 分起算），問「開盤區間是否特別」
  AMT 價位 → 同一價位 ±0.2 / ±0.4 ATR 的假價位
輸出：results_part22.txt 的一部分（part22_levels.csv 為逐筆事件）。
"""
import pickle
import numpy as np
import pandas as pd
from part22_core import load_m1, arrays, first_touch, barrier, tstat, SESS_ORDER

HORIZON = 240                     # 結果最多看 240 分（且不超過 session 結束後 60 分）
TOUCH_WIN = 240                   # OR 價位：區間結束後 240 分內的第一次觸碰


def react(H, L, C, j, level, side, atr, i_lim, i1):
    """side = 來的方向（+1 從上方來 → 反彈向上）。r60 / rS：60 分後 / session 結束時往反彈方向的報酬。"""
    end = min(j + HORIZON, i_lim)
    out = {}
    for b in (0.05, 0.10):
        out[f"b{int(b*100):02d}"] = barrier(H, L, j, level, side, b * atr, end)
    k = min(j + 30, len(C) - 1)
    out["r30"] = (C[k] - level) * side / atr
    out["r60"] = (C[min(j + 60, i1)] - level) * side / atr
    out["rS"] = (C[i1] - level) * side / atr
    return out


def run():
    d = load_m1()
    H, L, C, O, SP = arrays(d)
    _, S = pickle.load(open("part22_sessions.pkl", "rb"))
    ev = []
    for r in S.itertuples():
        i_lim = min(r.i1 + 60, len(C) - 1)
        base = dict(day=r.day, year=r.year, sess=r.sess, open_loc=r.open_loc, otype=r.otype, atr=r.atr)
        # ---- OR 價位：真的（offset 0）與其他時間的 30 分區間（安慰劑）
        for off in (0, 60, 120, 180):
            a = r.i0 + off
            e = a + 30
            if e + 30 >= r.i1:
                continue
            hh, ll = H[a:e].max(), L[a:e].min()
            for name, lev in (("ORH", hh), ("ORL", ll), ("ORM", (hh + ll) / 2)):
                j, side = first_touch(H, L, C, e, min(e + TOUCH_WIN, r.i1), lev)
                if j < 0:
                    continue
                ev.append(dict(base, level=name, kind="real" if off == 0 else "placebo", off=off, side=side,
                               mins=j - e, **react(H, L, C, j, lev, side, r.atr, i_lim, r.i1)))
        # ---- AMT 參考價（前日 POC/VAH/VAL、前日高低、前一個 session 高低）
        for name, lev in (("pPOC", r.ppoc), ("pVAH", r.pvah), ("pVAL", r.pval),
                          ("PDH", r.pdh), ("PDL", r.pdl), ("pSH", r.psh), ("pSL", r.psl)):
            if np.isnan(lev):
                continue
            for k, shift in enumerate((0, -0.4, -0.2, 0.2, 0.4)):
                lv = lev + shift * r.atr
                j, side = first_touch(H, L, C, r.i0 + 1, r.i1, lv)
                if j < 0:
                    continue
                if j < r.i_or:
                    orpos = "during_or"                 # OR 還沒形成完
                elif r.orl <= lv <= r.orh:
                    orpos = "inside_or"
                else:
                    orpos = "outside_or"                # 價位在 OR 外 → 碰到時價格已突破 OR
                ev.append(dict(base, level=name, kind="real" if shift == 0 else "placebo", off=shift, side=side,
                               mins=j - r.i0, orpos=orpos, pvah=r.pvah, pval=r.pval, lv=lv, **react(H, L, C, j, lv, side, r.atr, i_lim, r.i1)))
    E = pd.DataFrame(ev)
    E.to_csv("part22_levels.csv", index=False)
    return E


def summarize(E, out):
    def stats(g, col="b10"):
        x = g[col]
        x = x[x != 0]
        return pd.Series(dict(n=len(g), bounce=(x > 0).mean() * 100, r30=g.r30.mean(), t30=tstat(g.r30)))

    out.append("=" * 100)
    out.append("A. 因子一：OR 價位（區間內）的反應 — 真 OR vs 其他時間 30 分區間（安慰劑）")
    out.append("   bounce = 碰到後先反彈 0.10 ATR 的比例（%）；r30 = 30 分後往反彈方向的報酬（ATR）")
    out.append("=" * 100)
    O = E[E.level.isin(["ORH", "ORL", "ORM"])]
    for lv in ["ORH", "ORL", "ORM"]:
        out.append(f"\n[{lv}]   " + ("（從 OR 內往上碰高點：反彈 = 回到區間內）" if lv == "ORH" else
                                     "（從 OR 內往下碰低點：反彈 = 回到區間內）" if lv == "ORL" else "（中點：反彈 = 不穿越中點）"))
        rows = []
        for s in SESS_ORDER:
            g = O[(O.level == lv) & (O.sess == s)]
            re, pl = stats(g[g.kind == "real"]), stats(g[g.kind == "placebo"])
            yr = []
            for y in sorted(E.year.unique()):
                gy = g[g.year == y]
                a = stats(gy[gy.kind == "real"]).bounce - stats(gy[gy.kind == "placebo"]).bounce
                yr.append(f"{a:+5.1f}")
            rows.append(f"  {s:<5} 真 n={re.n:4.0f} 反彈 {re.bounce:5.1f}% r30 {re.r30:+.3f} (t {re.t30:+.1f}) | "
                        f"安慰劑 n={pl.n:4.0f} 反彈 {pl.bounce:5.1f}% r30 {pl.r30:+.3f} | 差 {re.bounce-pl.bounce:+5.1f}  逐年差 {' '.join(yr)}")
        out += rows

    out.append("\n" + "=" * 100)
    out.append("B. AMT 參考價第一次被碰到的反應 — 真價位 vs ±0.2/±0.4 ATR 假價位")
    out.append("=" * 100)
    A = E[~E.level.isin(["ORH", "ORL", "ORM"])]
    yrs = sorted(E.year.unique())
    out.append(f"  {'價位':<5}{'session':<6}{'n':>5}{'真反彈%':>9}{'假反彈%':>9}{'差':>7}{'真r30':>8}{'t':>6}{'假r30':>8}   逐年差(反彈%) " + " ".join(str(y) for y in yrs))
    for lv in ["pPOC", "pVAH", "pVAL", "PDH", "PDL", "pSH", "pSL"]:
        for s in SESS_ORDER + ["ALL"]:
            g = A[(A.level == lv) & ((A.sess == s) | (s == "ALL"))]
            re, pl = stats(g[g.kind == "real"]), stats(g[g.kind == "placebo"])
            yr = []
            for y in yrs:
                gy = g[g.year == y]
                yr.append(f"{stats(gy[gy.kind == 'real']).bounce - stats(gy[gy.kind == 'placebo']).bounce:+5.1f}")
            out.append(f"  {lv:<5}{s:<6}{re.n:5.0f}{re.bounce:9.1f}{pl.bounce:9.1f}{re.bounce-pl.bounce:+7.1f}"
                       f"{re.r30:+8.3f}{re.t30:+6.1f}{pl.r30:+8.3f}   {' '.join(yr)}")
        out.append("")
    # b = 0.05 的簡表
    out.append("  （b = 0.05 ATR 的真 − 假反彈差，ALL session）: " + ", ".join(
        f"{lv} {stats(A[(A.level==lv)&(A.kind=='real')],'b05').bounce - stats(A[(A.level==lv)&(A.kind=='placebo')],'b05').bounce:+.1f}"
        for lv in ["pPOC", "pVAH", "pVAL", "PDH", "PDL", "pSH", "pSL"]))
    return out


if __name__ == "__main__":
    E = run()
    out = summarize(E, [])
    txt = "\n".join(out)
    print(txt)
    open("results_part22_levels.txt", "w").write(txt + "\n")
