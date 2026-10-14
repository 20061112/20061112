"""因子二（突破 OR）× 因子三（session）× AMT 情境。

事件：OR 結束後、session 內第一根 M1 收盤在 OR 外（高點之上 / 低點之下）→ 以該收盤價為進場點，方向 = 突破方向。
結果（全部扣一次點差，以 ATR10 為單位，往突破方向為正）：
  r30 / r60 / r120 ：之後 30 / 60 / 120 分的報酬
  rS               ：持有到 session 結束
  ext              ：先到「OR 邊 + 1×OR 寬」(+1) 還是先回到 OR 中點 (−1)（隨機漫步約 33~40%）
  fail             ：突破後 30 分內收盤回到 OR 內（= 拒絕 / failed auction）
  fade_opp         ：失敗之後，先到 OR 另一邊 (+1) 還是先創突破新極值 (−1)
對照組：同一 session 其他時間的 30 分區間（+60 / +120 / +180 分）用一樣的規則。
AMT 情境：開盤相對前日價值區（open_loc）、突破方向相對價值（away / toward / in_va）、
          開盤型態（OD/OTD/ORR/OA）與方向是否一致、突破時是否已超出 PDH/PDL、OR 大小 z、
          30 分後是否「被接受」（仍在 OR 外）。
"""
import pickle
import numpy as np
import pandas as pd
from part22_core import load_m1, arrays, race, tstat, SESS_ORDER


def run():
    d = load_m1()
    H, L, C, O, SP = arrays(d)
    _, S = pickle.load(open("part22_sessions.pkl", "rb"))
    ev = []
    for r in S.itertuples():
        for off in (0, 60, 120, 180):
            a, e = r.i0 + off, r.i0 + off + 30
            if e + 30 >= r.i1:
                continue
            hh, ll = H[a:e].max(), L[a:e].min()
            w, m = hh - ll, (hh + ll) / 2
            seg = C[e:r.i1]
            up = np.where(seg > hh)[0]
            dn = np.where(seg < ll)[0]
            ju = e + up[0] if len(up) else 10 ** 9
            jd = e + dn[0] if len(dn) else 10 ** 9
            if ju == jd:
                continue
            j, dr = (ju, 1) if ju < jd else (jd, -1)
            px, cost = C[j], SP[j]
            edge = hh if dr == 1 else ll

            def fr(k):
                k = min(j + k, r.i1)
                return ((C[k] - px) * dr - cost) / r.atr

            tgt = edge + dr * w
            ext = race(H, L, j + 1, tgt, m, r.i1) if dr == 1 else -race(H, L, j + 1, m, tgt, r.i1)
            # 30 分內是否收回 OR 內
            k30 = min(j + 30, r.i1)
            back = np.where((C[j + 1:k30 + 1] <= hh) & (C[j + 1:k30 + 1] >= ll))[0]
            fail = len(back) > 0
            fade_opp, rfade = np.nan, np.nan
            if fail:
                jb = j + 1 + back[0]
                ext_px = H[j:jb + 1].max() if dr == 1 else L[j:jb + 1].min()
                opp = ll if dr == 1 else hh
                fade_opp = race(H, L, jb + 1, ext_px + 1e-9, opp, r.i1) * -dr if dr == 1 else \
                    race(H, L, jb + 1, opp, ext_px - 1e-9, r.i1)
                rfade = ((C[r.i1] - C[jb]) * -dr - SP[jb]) / r.atr
            # 接受：30 分後收盤仍在 OR 外（同方向）→ 從那時起抱到 session 結束
            acc = (C[k30] - edge) * dr > 0
            racc = ((C[r.i1] - C[k30]) * dr - SP[k30]) / r.atr
            # 突破相對前日價值
            if r.open_loc == "inside":
                rel = "in_va"
            else:
                rel = "away" if (r.open_loc == "above") == (dr == 1) else "toward"
            beyond_pd = (px > r.pdh) if dr == 1 else (px < r.pdl)
            beyond_va = (px > r.pvah) if dr == 1 else (px < r.pval)
            ev.append(dict(day=r.day, year=r.year, sess=r.sess, off=off, kind="real" if off == 0 else "placebo",
                           dir=dr, delay=j - e, px=px, atr=r.atr,
                           beyond_ps=((px > r.psh) if dr == 1 else (px < r.psl)) if not np.isnan(r.psh) else False, orw_atr=w / r.atr, or_z=r.or_z if off == 0 else np.nan,
                           open_loc=r.open_loc, rel=rel, otype=r.otype if off == 0 else "",
                           oalign=(r.odir == dr) if off == 0 and r.odir != 0 else np.nan,
                           beyond_pd=beyond_pd, beyond_va=beyond_va,
                           r30=fr(30), r60=fr(60), r120=fr(120), rS=fr(10 ** 6), ext=ext,
                           fail=fail, fade_opp=fade_opp, rfade=rfade, acc=acc, racc=racc))
    E = pd.DataFrame(ev)
    E.to_csv("part22_breakout.csv", index=False)
    return E


def line(g, label, cols=("r60", "rS")):
    if len(g) == 0:
        return f"  {label:<34} n=   0"
    ex = g.ext[g.ext != 0]
    s = f"  {label:<34} n={len(g):5d}  ext {(ex > 0).mean()*100:5.1f}%"
    for c in cols:
        s += f"  {c} {g[c].mean():+.3f} (t {tstat(g[c]):+4.1f})"
    s += f"  失敗 {g.fail.mean()*100:4.1f}%"
    yrs = " ".join(f"{g[g.year == y].rS.mean():+.2f}" if (g.year == y).sum() >= 5 else "   . " for y in (2023, 2024, 2025, 2026))
    return s + f"  | 逐年 rS {yrs}"


def summarize(E):
    out = ["=" * 120, "C. 因子二 × 因子三：OR 突破後是延續還是失敗？（真 OR vs 其他時間 30 分區間）",
           "   ext = 先到 1×OR 延伸 vs 先回中點；r60 / rS = 60 分 / 持有到 session 結束（ATR，已扣點差）；逐年 = 2023 2024 2025 2026",
           "=" * 120]
    R = E[E.kind == "real"]
    for s in SESS_ORDER:
        g = E[E.sess == s]
        out.append(line(g[g.kind == "real"], f"{s} 真 OR"))
        out.append(line(g[g.kind == "placebo"], f"{s} 安慰劑（其他 30 分區間）"))
    out.append("")
    out.append("  失敗突破（30 分內收回 OR）→ 反向：先到 OR 另一邊的比例、反向抱到 session 結束")
    for s in SESS_ORDER:
        for k in ("real", "placebo"):
            g = E[(E.sess == s) & (E.kind == k) & E.fail]
            fo = g.fade_opp[g.fade_opp != 0]
            yrs = " ".join(f"{g[g.year == y].rfade.mean():+.2f}" for y in (2023, 2024, 2025, 2026))
            out.append(f"  {s:<5}{k:<8} n={len(g):4d}  到另一邊 {(fo > 0).mean()*100:5.1f}%  反向 rS {g.rfade.mean():+.3f} "
                       f"(t {tstat(g.rfade):+4.1f})  逐年 {yrs}")

    out += ["", "=" * 120, "D. AMT 情境下的 OR 突破（只看真 OR）", "=" * 120]
    for s in SESS_ORDER + ["ALL"]:
        g = R if s == "ALL" else R[R.sess == s]
        out.append(f"\n  [{s}]")
        out.append("  -- 開盤相對前日價值區 × 突破方向（away = 往遠離價值方向突破 = initiative；toward = 回價值方向 = responsive）")
        for rel in ("away", "toward", "in_va"):
            out.append(line(g[g.rel == rel], f"  {rel}"))
        out.append("  -- 開盤型態（Dalton）與突破方向")
        for ot in ("OD", "OTD", "ORR"):
            out.append(line(g[(g.otype == ot) & (g.oalign == True)], f"  {ot} 順開盤方向突破"))
            out.append(line(g[(g.otype == ot) & (g.oalign == False)], f"  {ot} 逆開盤方向突破"))
        out.append(line(g[g.otype == "OA"], "  OA（開盤競價）"))
        out.append("  -- 突破時已超出前日價值區 / 前日高低")
        out.append(line(g[g.beyond_va], "  已在前日 VA 外"))
        out.append(line(g[~g.beyond_va], "  仍在前日 VA 內"))
        out.append(line(g[g.beyond_pd], "  已超出 PDH/PDL"))
        out.append("  -- OR 大小（z，對過去 20 天同 session）")
        out.append(line(g[g.or_z < -0.5], "  小 OR（z < −0.5）"))
        out.append(line(g[(g.or_z >= -0.5) & (g.or_z <= 0.5)], "  中 OR"))
        out.append(line(g[g.or_z > 0.5], "  大 OR（z > 0.5）"))
        out.append("  -- 接受 vs 拒絕：突破 30 分後仍在 OR 外 → 從那時起抱到結束（racc）")
        for a in (True, False):
            gg = g[g.acc == a]
            yrs = " ".join(f"{gg[gg.year == y].racc.mean():+.2f}" for y in (2023, 2024, 2025, 2026))
            out.append(f"    {'接受' if a else '未接受'} n={len(gg):4d}  racc {gg.racc.mean():+.3f} (t {tstat(gg.racc):+4.1f})  逐年 {yrs}")
    return out


if __name__ == "__main__":
    E = run()
    txt = "\n".join(summarize(E))
    print(txt)
    open("results_part22_breakout.txt", "w").write(txt + "\n")
